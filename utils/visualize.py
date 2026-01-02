import torch
import os
import random
import numpy as np
import open3d as o3d
from torch_geometric.data import Data


def visualize_point_cloud(data_pos, window_name=""):
    points = data_pos.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1]) 

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_point_clouds(data_pos1, data_pos2, window_name=""):
    points1 = data_pos1.cpu().numpy()
    points2 = data_pos2.cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)

    pcd1.paint_uniform_color([0, 0, 1]) 
    pcd2.paint_uniform_color([1, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()


def visualize_batch(data_loader):
    for batch_data in data_loader:
        y_vals = batch_data.y
        smallest_idx = torch.argmin(y_vals).item()
        largest_idx = torch.argmax(y_vals).item()

        smallest_value = batch_data.y.min().cpu().numpy()
        largest_value = batch_data.y.max().cpu().numpy()
        name_smallest = os.path.splitext(os.path.basename(batch_data.file_name[smallest_idx]))[0].split("_")[0]
        name_largest = os.path.splitext(os.path.basename(batch_data.file_name[largest_idx]))[0].split("_")[0]
        
        smallest_pcd = batch_data[smallest_idx].pos
        largest_pcd = batch_data[largest_idx].pos

        window_name = (f"blue: {name_smallest} (w={smallest_value:.1f}g), "
                        f"red: {name_largest} (w={largest_value:.1f}g)"
                        )
        visualize_point_clouds(smallest_pcd, largest_pcd, window_name)


def pcd_to_pyg(pcd, pre_transform):
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

    if pre_transform is not None:
        data = Data(pos=points)
        data = pre_transform(data)
        points = data.pos
    
    data = Data(pos=points, y=torch.tensor([0.0], dtype=torch.float32))

    return data


def visualize_augmentation(data_loader, pre_transform):
    for batch_data in data_loader:
        file_name = random.choice(batch_data.file_name)
        idx = batch_data.file_name.index(file_name)

        pcd = o3d.io.read_point_cloud(file_name)
        transformed_pcd = pcd_to_pyg(pcd, pre_transform)
        augmented_pcd = batch_data[idx].pos

        name = os.path.splitext(os.path.basename(file_name))[0].split("_")[0]

        window_name = (f"blue: {name} (original), "
                       f"red: {name} (augmented)"
                        )
        visualize_point_clouds(transformed_pcd.pos, augmented_pcd, window_name)