import os
from pathlib import Path
import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
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
    resultsfolder = os.path.join(project_root, 'results')

    conveyor_depth = {'2023': 0.345, '2024': 0.460, '2025': 0.465}
    search_col = "growing_season"
    max_height = 0.08

    df_splits = pd.read_csv(splits_csv)
    split_map = dict(zip(df_splits['label'], df_splits['split']))

    df_gt = pd.read_csv(target_csv).set_index('label')
    gt_map = dict(zip(df_gt.index, df_gt['weight_g_inctack']))

    ply_files = collect_ply_files(data_root)

    file_names = np.array([])
    unique_ids = np.array([])
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_ids = []

    for ply_file in tqdm(ply_files):
        file_name = os.path.join(data_root, ply_file)
        unique_id = os.path.basename(os.path.dirname(file_name))

        if unique_id not in split_map or unique_id not in gt_map:
            continue

        weight_g = gt_map[unique_id]

        pcd = o3d.io.read_point_cloud(ply_file)

        try:
            bb = pcd.get_oriented_bounding_box()
            bb.color = np.array([1.0, 0.0, 0.0])
            length, width, _ = bb.extent * 1e3

            ## height calculation
            search_key = df_gt.loc[unique_id, search_col]
            z_max = conveyor_depth[str(search_key)]
            z_min = np.min(np.asarray(pcd.points)[:,2])
            height = z_max - z_min
            height = np.clip(height, 0.0, max_height) * 1e3

            # o3d.visualization.draw_geometries([pcd, bb])

            features = [length, width, height]

            if split_map[unique_id] == "train":
                X_train.append(features)
                y_train.append(weight_g)
            elif split_map[unique_id] == "test":
                file_names = np.append(file_names, file_name)
                unique_ids = np.append(unique_ids, unique_id)
                X_test.append(features)
                y_test.append(weight_g)
                test_ids.append(unique_id)
        except:
            continue

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Tubers: {len(np.unique(test_ids))}")
    print(f"Point clouds: {len(y_test)}")
    print(f"MAE weight: {mean_absolute_error(y_test, y_pred):.1f} g")
    print(f"RMSE weight: {root_mean_squared_error(y_test, y_pred):.1f} g")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    intercept = model.intercept_
    coef = model.coef_
    w0 = intercept
    w1, w2, w3 = coef
    print(f"Linear regression formula:\nweight_g_inctack = {w0:.4f} + {w1:.4f}*length_mm + {w2:.4f}*width_mm + {w3:.4f}*height_mm")
    
    df_output = pd.DataFrame({'file_name': file_names, 'unique_id': unique_ids, 'gt': y_test, 'pred': y_pred, 'diff': abs(y_test-y_pred)})
    df_output.to_csv(os.path.join(resultsfolder, 'linear_regression.csv'), index=False)  

    try:
        cultivar_map = dict(zip(df_gt.index, df_gt['cultivar']))
        print("=== Sub-analyses per cultivar ===")
        unique_cultivars = df_gt['cultivar'].unique()
        for cultivar in unique_cultivars:
            indices = [i for i, pid in enumerate(test_ids) if cultivar_map[pid] == cultivar]
            if len(indices) == 0:
                continue

            X_sub = X_test[indices]
            y_sub = y_test[indices]
            y_sub_pred = model.predict(X_sub)
            unique_tubers = set([test_ids[i] for i in indices])

            print(f"Cultivar: {cultivar}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(y_sub)}")
            print(f"MAE weight: {mean_absolute_error(y_sub, y_sub_pred):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(y_sub, y_sub_pred):.1f} g")
            print(f"R2: {r2_score(y_sub, y_sub_pred):.2f}\r\n")

        season_map = dict(zip(df_gt.index, df_gt['growing_season']))
        print("=== Sub-analyses per growing season ===")
        unique_seasons = df_gt['growing_season'].unique()
        for season in unique_seasons:
            indices = [i for i, pid in enumerate(test_ids) if season_map[pid] == season]
            if len(indices) == 0:
                continue

            X_sub = X_test[indices]
            y_sub = y_test[indices]
            y_sub_pred = model.predict(X_sub)
            unique_tubers = set([test_ids[i] for i in indices])

            print(f"Season: {season}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(y_sub)}")
            print(f"MAE weight: {mean_absolute_error(y_sub, y_sub_pred):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(y_sub, y_sub_pred):.1f} g")
            print(f"R2: {r2_score(y_sub, y_sub_pred):.2f}\r\n")
    except:
        pass