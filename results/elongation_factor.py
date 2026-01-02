import os
from pathlib import Path
import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_ply_files(root_dir):
    """Recursively collect all .ply files under root_dir."""
    ply_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ply"):
                ply_files.append(os.path.join(root, file))
    return ply_files


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    data_root = os.path.join(datafolder, '1_rgbd', '2_pcd')
    splits_csv = os.path.join(datafolder, 'splits.csv')
    target_csv = os.path.join(datafolder, 'ground_truth.csv')

    conveyor_depth = {'2023': 0.345, '2024': 0.460, '2025': 0.465}
    search_col = "growing_season"
    max_height = 0.08

    df_splits = pd.read_csv(splits_csv)
    split_map = dict(zip(df_splits['label'], df_splits['split']))

    df_gt = pd.read_csv(target_csv).set_index('label')
    gt_map = dict(zip(df_gt.index, df_gt['weight_g_inctack']))

    ply_files = collect_ply_files(data_root)

    elongation_factors = []
    test_files = []
    test_ids = []

    for ply_file in tqdm(ply_files):
        unique_id = os.path.splitext(os.path.basename(ply_file))[0].split("_")[0]

        if unique_id not in split_map or unique_id not in gt_map:
            continue

        if split_map[unique_id] == "test":
            test_files.append(ply_file)
            test_ids.append(unique_id)
            pcd = o3d.io.read_point_cloud(ply_file)

            aabb = pcd.get_axis_aligned_bounding_box()
            min_bound = aabb.min_bound.copy()
            max_bound = aabb.max_bound.copy()

            height_key = df_gt.loc[unique_id, search_col]
            z_belt = conveyor_depth[str(height_key)]
            max_bound[2] = z_belt

            aabb_expanded = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound,
                max_bound=max_bound
            )
            aabb_expanded.color = (1, 0, 0) 

            extents = aabb_expanded.get_extent()
            length = extents[0]
            width = extents[1]
            if length >= width:
                elongation_ratio = length / width
            else:
                elongation_ratio = width / length

            elongation_factors.append(elongation_ratio)


    print(f"Tubers: {len(np.unique(test_ids))}")
    print(f"Point clouds: {len(test_files)}")
    print(f"Elongation factor: {np.average(elongation_factors):.2f}\r\n")

    try:
        cultivar_map = dict(zip(df_gt.index, df_gt['cultivar']))
        print("=== Sub-analyses per cultivar ===")
        unique_cultivars = df_gt['cultivar'].unique()
        for cultivar in unique_cultivars:
            indices = [i for i, pid in enumerate(test_ids) if cultivar_map[pid] == cultivar]
            if len(indices) == 0:
                continue

            pcds = [test_files[i] for i in indices]
            ef = [elongation_factors[i] for i in indices]
            unique_tubers = set([test_ids[i] for i in indices])

            print(f"Cultivar: {cultivar}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(pcds)}")
            print(f"Elongation factor: {np.average(ef):.2f}\r\n")

        season_map = dict(zip(df_gt.index, df_gt['growing_season']))
        print("=== Sub-analyses per growing season ===")
        unique_seasons = df_gt['growing_season'].unique()
        for season in unique_seasons:
            indices = [i for i, pid in enumerate(test_ids) if season_map[pid] == season]
            if len(indices) == 0:
                continue

            pcds = [test_files[i] for i in indices]
            ef = [elongation_factors[i] for i in indices]
            unique_tubers = set([test_ids[i] for i in indices])

            print(f"Season: {season}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(pcds)}")
            print(f"Elongation factor: {np.average(ef):.2f}\r\n")
    except:
        pass