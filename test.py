"""Evaluate PointRAFT with the original repository protocol."""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import os
import timeit
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_fpsample
import torch_geometric.transforms as T
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_CLUSTER
from tqdm import tqdm

from models import PointRAFT
from utils.provenance import (
    git_commit,
    git_source_dirty,
    runtime_versions,
    sha256_file,
    write_json,
)
from utils.reproducibility import seed_everything

CHECKPOINT_PATH = Path("weights/pointraft.pth")
PREDICTIONS_PATH = Path("results/pointraft/predictions.csv")
TEST_METRICS_PATH = Path("results/pointraft/test_metrics.txt")
TEST_PROVENANCE_PATH = Path("results/pointraft/test.provenance.json")


def main() -> None:
    """Evaluate PointRAFT with the original repository protocol."""
    if not WITH_TORCH_CLUSTER:
        raise SystemExit("This code requires 'torch-cluster'")

    parser = argparse.ArgumentParser(description="Test PointRAFT.")
    parser.add_argument(
        "--max-test-files",
        type=int,
        help="Bound test point clouds for a prerun.",
    )
    args = parser.parse_args()

    if args.max_test_files is not None and args.max_test_files < 1:
        parser.error("--max-test-files must be positive")

    seed_everything()

    ## set the relevant paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    datafolder = project_root / "data" / "3DPotatoTwin"

    data_root = os.path.join(datafolder, "1_rgbd", "2_pcd")
    splits_csv = os.path.join(datafolder, "splits.csv")
    target_csv = os.path.join(datafolder, "ground_truth.csv")

    resultsfolder = project_root / "results"
    resultsfolder.mkdir(parents=True, exist_ok=True)
    model_name = "pointraft"
    checkpoint_path = project_root / CHECKPOINT_PATH
    result_path = project_root / PREDICTIONS_PATH
    log_path = project_root / TEST_METRICS_PATH
    provenance_path = project_root / TEST_PROVENANCE_PATH
    print(f"Dataset: {datafolder}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Result: {result_path}")

    ## process the data
    splits_df = pd.read_csv(splits_csv, delimiter=",")
    test_ids = set(splits_df.loc[splits_df["split"] == "test", "label"].astype(str))
    all_files = sorted(Path(data_root).rglob("*.ply"))
    ply_files = [str(f) for f in all_files if f.parent.name in test_ids]
    if args.max_test_files is not None:
        sample_count = min(len(ply_files), args.max_test_files)
        if sample_count == 1:
            ply_files = ply_files[:1]
        else:
            sample_indices = [
                round(index * (len(ply_files) - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
            ply_files = [ply_files[index] for index in sample_indices]
    df = pd.read_csv(target_csv, delimiter=",").set_index("label")

    ## initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize = False
    model = PointRAFT().to(device)
    best_model_weights = torch.load(checkpoint_path)
    model.load_state_dict(best_model_weights)
    model.eval()

    ## define pre-transform and number of points sampled
    pre_transform = T.Center()
    num_points = 1024
    conveyor_depth = {"2023": 0.345, "2024": 0.460, "2025": 0.465}
    search_col = "growing_season"
    max_height = 0.08

    ## run the model once on dummy data
    data = Data(
        pos=torch.rand(num_points, 3), height=torch.tensor([0.0], dtype=torch.float32)
    )
    data.batch = torch.zeros(num_points, dtype=torch.int64)
    data = data.to(device)
    output = model(data)

    ## define the arrays
    file_names = np.array([])
    unique_ids = np.array([])
    predictions = np.array([], dtype=np.float32)
    ground_truths = np.array([], dtype=np.float32)
    exec_times = np.array([], dtype=np.float32)

    ## loop over the .ply files
    for ply_file in tqdm(ply_files):
        starttime = timeit.default_timer()
        file_name = os.path.join(data_root, ply_file)
        unique_id = os.path.basename(os.path.dirname(file_name))

        file_names = np.append(file_names, file_name)
        unique_ids = np.append(unique_ids, unique_id)

        gt = df.loc[unique_id, "weight_g_inctack"].item()

        pcd = o3d.io.read_point_cloud(file_name)
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

        ## height embedding
        height_key = df.loc[unique_id, search_col]
        z_max = conveyor_depth[str(height_key)]
        z_min = torch.min(points[:, 2])
        height = z_max - z_min
        height = torch.clamp(height, min=0.0, max=max_height)

        data = Data(pos=points)
        data = pre_transform(data)
        points = data.pos

        if points.size(0) > num_points:
            points, _ = torch_fpsample.sample(points, num_points, start_idx=0)

        data = Data(pos=points, height=torch.tensor([height], dtype=torch.float32))
        data.batch = torch.zeros(points.size(0), dtype=torch.int64)
        data = data.to(device)

        with torch.no_grad():
            output = model(data)
            predictions = np.append(predictions, output.cpu().numpy())
            ground_truths = np.append(ground_truths, gt)

        if visualize:
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            window_name = f"FILE: {ply_file}, GT: {gt:.1f} g, PRED: {output.cpu().numpy().squeeze():.1f} g"
            o3d.visualization.draw_geometries([pcd], window_name=window_name)

        endtime = timeit.default_timer()
        exec_times = np.append(exec_times, (endtime - starttime) * 1e3)

    tuber_count = len(np.unique(unique_ids))
    point_cloud_count = len(file_names)
    mae = float(mean_absolute_error(ground_truths, predictions))
    rmse = float(root_mean_squared_error(ground_truths, predictions))
    r2 = float(r2_score(ground_truths, predictions))
    prediction_std = float(np.std(predictions))
    average_exec_time = float(np.average(exec_times))
    print(f"Tubers: {tuber_count}")
    print(f"Point clouds: {point_cloud_count}")
    print(f"MAE weight: {mae:.1f} g")
    print(f"RMSE weight: {rmse:.1f} g")
    print(f"R2: {r2:.2f}")
    print(f"Prediction SD: {prediction_std:.3f} g")
    print(f"Execution time: {average_exec_time:.1f} ms \r\n")
    df_output = pd.DataFrame(
        {
            "file_name": file_names,
            "unique_id": unique_ids,
            "gt": ground_truths,
            "pred": predictions,
            "diff": abs(ground_truths - predictions),
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(result_path, index=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"tubers={tuber_count}",
                f"point_clouds={point_cloud_count}",
                f"mae_g={mae:.8f}",
                f"rmse_g={rmse:.8f}",
                f"r2={r2:.8f}",
                f"prediction_std_g={prediction_std:.8f}",
                f"average_execution_ms={average_exec_time:.8f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        provenance_path,
        {
            "artifact": model_name,
            "code_commit": git_commit(project_root),
            "code_dirty": git_source_dirty(project_root),
            "command": [sys.executable, *sys.argv],
            "dataset": str(datafolder.resolve()),
            "device": str(device),
            "ground_truth_sha256": sha256_file(Path(target_csv)),
            "mae_g": mae,
            "max_test_files": args.max_test_files,
            "model": model_name,
            "point_clouds": point_cloud_count,
            "prediction_sha256": sha256_file(result_path),
            "prediction_std_g": prediction_std,
            "r2": r2,
            "rmse_g": rmse,
            "splits_sha256": sha256_file(Path(splits_csv)),
            "status": "completed",
            "tubers": tuber_count,
            "versions": runtime_versions(),
            "weight_sha256": sha256_file(checkpoint_path),
        },
    )

    try:
        cultivar_map = dict(zip(df.index, df["cultivar"]))
        print("=== Sub-analyses per cultivar ===")
        unique_cultivars = df["cultivar"].unique()
        for cultivar in unique_cultivars:
            indices = [
                i for i, pid in enumerate(unique_ids) if cultivar_map[pid] == cultivar
            ]
            if len(indices) == 0:
                continue

            gt_sel = ground_truths[indices]
            pred_sel = predictions[indices]
            unique_tubers = [
                i for i, pid in enumerate(test_ids) if cultivar_map[pid] == cultivar
            ]

            print(f"Cultivar: {cultivar}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(gt_sel)}")
            print(f"MAE weight: {mean_absolute_error(gt_sel, pred_sel):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(gt_sel, pred_sel):.1f} g")
            print(f"R2: {r2_score(gt_sel, pred_sel):.2f}\r\n")

        season_map = dict(zip(df.index, df["growing_season"]))
        print("=== Sub-analyses per growing season ===")
        unique_seasons = df["growing_season"].unique()
        for season in unique_seasons:
            indices = [
                i for i, pid in enumerate(unique_ids) if season_map[pid] == season
            ]
            if len(indices) == 0:
                continue

            gt_sel = ground_truths[indices]
            pred_sel = predictions[indices]
            unique_tubers = [
                i for i, pid in enumerate(test_ids) if season_map[pid] == season
            ]

            print(f"Season: {season}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(gt_sel)}")
            print(f"MAE weight: {mean_absolute_error(gt_sel, pred_sel):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(gt_sel, pred_sel):.1f} g")
            print(f"R2: {r2_score(gt_sel, pred_sel):.2f}\r\n")
    except KeyError as error:
        print(f"Skipped subgroup analyses because metadata is missing: {error}")


if __name__ == "__main__":
    main()
