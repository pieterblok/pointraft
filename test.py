import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import open3d as o3d
import pandas as pd
import random
from tqdm import tqdm
import timeit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.typing import WITH_TORCH_CLUSTER
import torch_fpsample

from models import *
model_dict = {'pointraft': PointRAFT}


if not WITH_TORCH_CLUSTER:
    quit("This code requires 'torch-cluster'")


if __name__ == '__main__':
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)

    ## set the relevant paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    data_root = os.path.join(datafolder, '1_rgbd', '2_pcd')
    splits_csv = os.path.join(datafolder, 'splits.csv')
    target_csv = os.path.join(datafolder, 'ground_truth.csv')

    weightsfolder = os.path.join(project_root, 'weights')
    os.makedirs(weightsfolder, exist_ok=True)
    resultsfolder = os.path.join(project_root, 'results')


    ## process the data
    splits_df = pd.read_csv(splits_csv, delimiter=',') 
    test_ids = set(splits_df.loc[splits_df['split'] == 'test', 'label'].astype(str))
    all_files = list(Path(data_root).rglob("*.ply"))
    ply_files = [str(f) for f in all_files if f.parent.name in test_ids]
    df = pd.read_csv(target_csv, delimiter=',').set_index('label')


    ## initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'pointraft'
    visualize = False
    model = model_dict[model_name]().to(device)
    best_model_weights = torch.load(os.path.join(weightsfolder, model_name + '.pth'))
    model.load_state_dict(best_model_weights)
    model.eval()


    ## define pre-transform and number of points sampled
    pre_transform = T.Center()
    num_points = 1024
    conveyor_depth = {'2023': 0.345, '2024': 0.460, '2025': 0.465}
    search_col = "growing_season"
    max_height = 0.08


    ## run the model once on dummy data
    data = Data(pos=torch.rand(num_points, 3), height=torch.tensor([0.0], dtype=torch.float32))
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

        gt = df.loc[unique_id, 'weight_g_inctack'].item()

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
            points, _ = torch_fpsample.sample(points, num_points)

        data = Data(pos=points, height=torch.tensor([height], dtype=torch.float32))
        data.batch = torch.zeros(points.size(0), dtype=torch.int64)
        data = data.to(device)

        with torch.no_grad():
            output = model(data)
            predictions = np.append(predictions, output.cpu().numpy())
            ground_truths = np.append(ground_truths, gt)
            
        if visualize:
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])    
            window_name = f'FILE: {ply_file}, GT: {gt:.1f} g, PRED: {output.cpu().numpy().squeeze():.1f} g'
            o3d.visualization.draw_geometries([pcd], window_name=window_name)

        endtime = timeit.default_timer()
        exec_times = np.append(exec_times, (endtime - starttime)*1e3)
 

    print(f"Tubers: {len(np.unique(unique_ids))}")
    print(f"Point clouds: {len(file_names)}")
    print(f"MAE weight: {mean_absolute_error(ground_truths, predictions):.1f} g") 
    print(f"RMSE weight: {root_mean_squared_error(ground_truths, predictions):.1f} g")
    print(f"R2: {r2_score(ground_truths, predictions):.2f}")
    print(f"Execution time: {np.average(exec_times):.1f} ms \r\n")
    df_output = pd.DataFrame({'file_name': file_names, 'unique_id': unique_ids, 'gt': ground_truths, 'pred': predictions, 'diff': abs(ground_truths-predictions)})
    df_output.to_csv(os.path.join(resultsfolder, model_name + '.csv'), index=False)  


    try:
        cultivar_map = dict(zip(df.index, df['cultivar']))
        print("=== Sub-analyses per cultivar ===")
        unique_cultivars = df['cultivar'].unique()
        for cultivar in unique_cultivars:
            indices = [i for i, pid in enumerate(unique_ids) if cultivar_map[pid] == cultivar]
            if len(indices) == 0:
                continue

            gt_sel = ground_truths[indices]
            pred_sel = predictions[indices]
            unique_tubers = [i for i, pid in enumerate(test_ids) if cultivar_map[pid] == cultivar]

            print(f"Cultivar: {cultivar}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(gt_sel)}")
            print(f"MAE weight: {mean_absolute_error(gt_sel, pred_sel):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(gt_sel, pred_sel):.1f} g")
            print(f"R2: {r2_score(gt_sel, pred_sel):.2f}\r\n")

        season_map = dict(zip(df.index, df['growing_season']))
        print("=== Sub-analyses per growing season ===")
        unique_seasons = df['growing_season'].unique()
        for season in unique_seasons:
            indices = [i for i, pid in enumerate(unique_ids) if season_map[pid] == season]
            if len(indices) == 0:
                continue

            gt_sel = ground_truths[indices]
            pred_sel = predictions[indices]
            unique_tubers = [i for i, pid in enumerate(test_ids) if season_map[pid] == season]

            print(f"Season: {season}")
            print(f"Tubers: {len(unique_tubers)}")
            print(f"Point clouds: {len(gt_sel)}")
            print(f"MAE weight: {mean_absolute_error(gt_sel, pred_sel):.1f} g")
            print(f"RMSE weight: {root_mean_squared_error(gt_sel, pred_sel):.1f} g")
            print(f"R2: {r2_score(gt_sel, pred_sel):.2f}\r\n")
    except:
        pass