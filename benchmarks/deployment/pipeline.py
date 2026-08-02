"""Shared RGB-D-to-PointRAFT inference pipeline."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from models import PointRAFT

from .pointcloud import PointCloudUtils

CONVEYOR_DEPTH = {"2023": 0.345, "2024": 0.460, "2025": 0.465}
PIPELINE_SCOPE = (
    "Precomputed instance mask and RGB-D images through point-cloud creation, "
    "DBSCAN filtering, PointRAFT preprocessing (including host-to-device "
    "transfer when applicable), and final regression. Instance segmentation "
    "and one-time model loading are excluded."
)


def checkpoint_sha256(path: Path) -> str:
    """Calculate a checkpoint's SHA-256 digest.

    Parameters
    ----------
    path
        Checkpoint file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested CPU or CUDA execution device.

    Parameters
    ----------
    requested
        ``auto``, ``cpu``, or ``cuda``.

    Returns
    -------
    torch.device
        Validated execution device.

    Raises
    ------
    RuntimeError
        If CUDA was requested but is unavailable.
    """
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


class InferencePipeline:
    """Run the project RGB-D preprocessing and PointRAFT model.

    Parameters
    ----------
    device
        CPU or CUDA device used for PointRAFT regression.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dataset_root = PROJECT_ROOT / "data" / "3DPotatoTwin"
        self.metadata = pd.read_csv(
            self.dataset_root / "ground_truth.csv", dtype={"label": str}
        ).set_index("label")
        self.point_cloud_utils = PointCloudUtils(
            intrinsics=self.dataset_root
            / "1_rgbd"
            / "0_camera_intrinsics"
            / "realsense_d405_camera_intrinsic.json",
            depth_scale=1000.0,
            depth_trunc=0.5,
        )
        self.checkpoint_path = PROJECT_ROOT / "weights" / "pointraft.pth"
        self.checkpoint_sha256 = checkpoint_sha256(self.checkpoint_path)
        self.model = PointRAFT().to(device)
        torch_version = torch.__version__.split("+", maxsplit=1)[0].split(".")
        load_options = (
            {"weights_only": True}
            if (int(torch_version[0]), int(torch_version[1])) >= (2, 6)
            else {}
        )
        state_dict = torch.load(
            self.checkpoint_path, map_location=device, **load_options
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.synchronize()

    def synchronize(self) -> None:
        """Wait for pending CUDA work when the pipeline uses a GPU."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def test_samples(self, limit: int = 0) -> List[Path]:
        """Return deterministic RGB image paths from the test split.

        Parameters
        ----------
        limit
            Maximum samples to return. Zero selects the complete test set.

        Returns
        -------
        list of pathlib.Path
            Sorted masked RGB image paths.
        """
        splits = pd.read_csv(
            self.dataset_root / "splits.csv", dtype={"label": str}
        )
        test_ids = set(splits.loc[splits["split"] == "test", "label"])
        image_root = self.dataset_root / "1_rgbd" / "1_image"
        samples = [
            path
            for path in sorted(image_root.rglob("*_rgb_*"))
            if path.parent.name in test_ids
        ]
        if not samples:
            raise RuntimeError(f"No test RGB images found under {image_root}")
        return samples[:limit] if limit else samples

    def create_point_cloud(
        self, rgb_file: Path
    ) -> Tuple[o3d.geometry.PointCloud, str]:
        """Create an unfiltered point cloud from one masked RGB-D pair.

        Parameters
        ----------
        rgb_file
            RGB image whose alpha channel contains the precomputed mask.

        Returns
        -------
        tuple
            Unfiltered Open3D point cloud and tuber identifier.
        """
        depth_file = Path(str(rgb_file).replace("_rgb_", "_depth_"))
        rgba = cv2.imread(str(rgb_file), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if rgba is None or depth is None:
            raise RuntimeError(f"Failed to read RGB-D pair for {rgb_file}")
        if rgba.ndim != 3 or rgba.shape[2] < 4:
            raise RuntimeError(f"RGB image does not contain an alpha mask: {rgb_file}")

        mask = rgba[:, :, -1]
        x, y, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            raise RuntimeError(f"Empty instance mask: {rgb_file}")
        point_cloud = self.point_cloud_utils.create_pcd(
            depth, mask, x, y, x + width, y + height
        )
        return point_cloud, rgb_file.parent.name

    def filter_point_cloud(
        self, point_cloud: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """Apply the project's DBSCAN filtering configuration.

        Parameters
        ----------
        point_cloud
            Unfiltered point cloud.

        Returns
        -------
        open3d.geometry.PointCloud
            Filtered point cloud.
        """
        return self.point_cloud_utils.remove_distant_clusters(
            point_cloud, eps=0.01, min_points=200, distance_threshold=1
        )

    def prepare_model_input(
        self, point_cloud: o3d.geometry.PointCloud, tuber_id: str
    ) -> Tuple[Data, float, int]:
        """Create centered, sampled PointRAFT input data.

        Parameters
        ----------
        point_cloud
            Filtered point cloud.
        tuber_id
            Dataset label used to retrieve the growing season.

        Returns
        -------
        tuple
            PyG model input, object height in metres, and filtered point count.
        """
        points = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
        if points.numel() == 0:
            raise RuntimeError(f"DBSCAN produced an empty point cloud: {tuber_id}")
        season = str(self.metadata.loc[tuber_id, "growing_season"])
        if season not in CONVEYOR_DEPTH:
            raise RuntimeError(
                f"No conveyor depth is configured for season {season}: {tuber_id}"
            )
        object_height = torch.clamp(
            torch.tensor(CONVEYOR_DEPTH[season]) - points[:, 2].min(),
            min=0.0,
            max=0.08,
        )
        points = points - points.mean(dim=0, keepdim=True)
        filtered_points = points.size(0)
        if points.size(0) > 1024:
            points, _ = torch_fpsample.sample(points, 1024, start_idx=0)
        data = Data(
            pos=points,
            height=object_height.reshape(1),
            batch=torch.zeros(points.size(0), dtype=torch.long),
        ).to(self.device)
        self.synchronize()
        return data, object_height.item(), filtered_points

    def predict(self, data: Data) -> torch.Tensor:
        """Run one PointRAFT regression.

        Parameters
        ----------
        data
            Prepared PyG model input.

        Returns
        -------
        torch.Tensor
            Predicted tuber weight.
        """
        with torch.inference_mode():
            prediction = self.model(data)
        self.synchronize()
        return prediction

    def warmup(self, data: Data, runs: int) -> None:
        """Run unmeasured model warm-up iterations.

        Parameters
        ----------
        data
            Prepared PyG model input.
        runs
            Number of warm-up regressions.
        """
        with torch.inference_mode():
            for _ in range(runs):
                self.model(data)
        self.synchronize()

    def run_sample(self, rgb_file: Path) -> Dict[str, Any]:
        """Run one complete sample without performance measurement.

        Parameters
        ----------
        rgb_file
            Masked RGB image path.

        Returns
        -------
        dict
            Point counts, object height, and prediction.
        """
        unfiltered, tuber_id = self.create_point_cloud(rgb_file)
        filtered = self.filter_point_cloud(unfiltered)
        data, object_height, filtered_points = self.prepare_model_input(
            filtered, tuber_id
        )
        prediction = self.predict(data)
        return {
            "rgb_file": rgb_file,
            "tuber_id": tuber_id,
            "points_unfiltered": len(unfiltered.points),
            "points_filtered": filtered_points,
            "points_model_input": data.pos.size(0),
            "height_m": object_height,
            "prediction_g": prediction.item(),
        }
