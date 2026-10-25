"""Export the paper point-cloud dataset across the isolated PTv3 boundary."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_sample(paths: list[Path], maximum: int | None) -> list[Path]:
    """Select an evenly spaced bounded subset while preserving order."""
    if maximum is None or len(paths) <= maximum:
        return paths
    if maximum == 1:
        return paths[:1]
    indices = [round(i * (len(paths) - 1) / (maximum - 1)) for i in range(maximum)]
    return [paths[index] for index in indices]


def export_split(
    split: str,
    data_root: Path,
    split_ids: set[str],
    targets: pd.DataFrame,
    output_path: Path,
    *,
    num_points: int,
    maximum: int | None,
) -> dict[str, object]:
    """Export one split as concatenated arrays without Torch serialization."""
    paths = [
        path
        for path in sorted(data_root.rglob("*.ply"))
        if path.parent.name in split_ids
    ]
    paths = evenly_sample(paths, maximum)
    if not paths:
        raise RuntimeError(f"No point clouds found for the {split} split")

    coordinates: list[np.ndarray] = []
    offsets = [0]
    target_values: list[float] = []
    classes: list[int] = []
    file_names: list[str] = []
    unique_ids: list[str] = []

    for index, path in enumerate(paths, start=1):
        unique_id = path.parent.name
        points = torch.as_tensor(
            np.asarray(o3d.io.read_point_cloud(str(path)).points),
            dtype=torch.float32,
        )
        if points.numel() == 0:
            raise RuntimeError(f"Point cloud is empty: {path}")
        points = points - points.mean(dim=0, keepdim=True)
        if points.size(0) > num_points:
            points, _ = torch_fpsample.sample(points, num_points, start_idx=0)

        points_array = points.numpy().astype(np.float32, copy=False)
        coordinates.append(points_array)
        offsets.append(offsets[-1] + len(points_array))
        target_values.append(float(targets.loc[unique_id, "weight_g_inctack"]))
        classes.append(
            int(targets.loc[unique_id, "weight_class"]) if split == "train" else -1
        )
        file_names.append(str(path.resolve()))
        unique_ids.append(unique_id)
        if index % 1000 == 0 or index == len(paths):
            print(f"Exported {split}: {index}/{len(paths)}", flush=True)

    np.savez_compressed(
        output_path,
        coordinates=np.concatenate(coordinates, axis=0),
        offsets=np.asarray(offsets, dtype=np.int64),
        targets=np.asarray(target_values, dtype=np.float32),
        classes=np.asarray(classes, dtype=np.int64),
        file_names=np.asarray(file_names),
        unique_ids=np.asarray(unique_ids),
    )
    return {
        "file": output_path.name,
        "point_clouds": len(paths),
        "points": offsets[-1],
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    """Parse export controls and prepare train, validation, and test arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-files-per-split", type=int)
    parser.add_argument("--num-points", type=int, default=1024)
    args = parser.parse_args()
    if args.max_files_per_split is not None and args.max_files_per_split < 1:
        parser.error("--max-files-per-split must be positive")
    if args.num_points < 32:
        parser.error("--num-points must be at least 32")

    project_root = Path(__file__).resolve().parents[4]
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / ".data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_root / "data" / "3DPotatoTwin"
    data_root = data_dir / "1_rgbd" / "2_pcd"
    splits_path = data_dir / "splits.csv"
    targets_path = data_dir / "ground_truth.csv"

    desired = {
        "ground_truth_sha256": sha256_file(targets_path),
        "max_files_per_split": args.max_files_per_split,
        "num_points": args.num_points,
        "splits_sha256": sha256_file(splits_path),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached = all(manifest.get(key) == value for key, value in desired.items())
        cached = cached and all(
            (output_dir / f"{split}.npz").exists()
            and sha256_file(output_dir / f"{split}.npz")
            == manifest["splits"][split]["sha256"]
            for split in ("train", "val", "test")
        )
        if cached:
            print(f"Reusing verified PTv3 export: {manifest_path}")
            return

    split_frame = pd.read_csv(splits_path)
    targets = pd.read_csv(targets_path).set_index("label")
    split_records: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        split_ids = set(
            split_frame.loc[split_frame["split"] == split, "label"].astype(str)
        )
        split_records[split] = export_split(
            split,
            data_root,
            split_ids,
            targets,
            output_dir / f"{split}.npz",
            num_points=args.num_points,
            maximum=args.max_files_per_split,
        )

    manifest = {
        **desired,
        "dataset": str(data_dir.resolve()),
        "export_command": [sys.executable, *sys.argv],
        "format": "pointraft-ptv3-npz-v1",
        "splits": split_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote PTv3 dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
