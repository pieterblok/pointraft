import os
from pathlib import Path
import torch
import random
import pandas as pd
import numpy as np
import open3d as o3d
from torch_geometric.data import Data, InMemoryDataset
import torch_fpsample
from tqdm import tqdm


class PointCloudDataset(InMemoryDataset):
    def __init__(self, root, splits_csv, target_csv, target_col, class_col="", split="train", conveyor_depth={}, search_col="", max_height=0.0, pre_transform=None, transform=None, num_points=1024, apply_augmentation=True):
        """
        In-memory dataset for loading point cloud data using a split definition from a CSV file and target ground truth.
        """
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        self.split = split
        
        # Load split information
        self.splits_df = pd.read_csv(splits_csv)
        self.ids = set(self.splits_df[self.splits_df['split'] == split]['label'].astype(str))
        
        # Load target ground truth
        self.target_df = pd.read_csv(target_csv).set_index('label')
        self.target_col = target_col
        
        self.class_col = class_col
        if self.class_col != "":
            self.set_class = True
        else:
            self.set_class = False

        self.conveyor_depth = conveyor_depth
        self.search_col = search_col
        self.max_height = max_height
        
        super().__init__(root, transform, pre_transform)
        
        # Load preprocessed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # List all .ply files that belong to an unique ID of the current split. This procedure assumes a nested folder structure:
        # -- 2_pcd
        # ---- 2R1-11
        # ---- 2R1-12

        all_files = list(Path(self.root).rglob("*.ply"))
        random.shuffle(all_files)
        return [str(f) for f in all_files if f.parent.name in self.ids]

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def download(self):
        # No downloading required; data is assumed to be present locally.
        pass

    def process(self):
        """
        Processes the raw data files and saves them into a single file for in-memory loading.
        """
        data_list = []
        
        for file_name in tqdm(self.raw_file_names, desc=f"Processing {self.split} split"):
            pcd = o3d.io.read_point_cloud(file_name)
            
            # Extract unique_id from file name
            unique_id = os.path.basename(os.path.dirname(file_name))
            
            # Get target value
            if unique_id in self.target_df.index:
                target_value = self.target_df.loc[unique_id, self.target_col]
            else:
                continue  # Skip if no target value is found

            # Get classification value
            if self.set_class:
                if unique_id in self.target_df.index:
                    target_class = self.target_df.loc[unique_id, self.class_col]
                else:
                    continue  # Skip if no target value is found
            
            # Load points
            points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

            # Add the approximate height of the object
            try:
                search_key = self.target_df.loc[unique_id, self.search_col]
                z_max = self.conveyor_depth[str(search_key)]
                z_min = torch.min(points[:, 2])
                height = z_max - z_min
                height = torch.clamp(height, min=0.0, max=self.max_height)
            except:
                height = torch.tensor([0.0], dtype=torch.float32)

            # Apply pre-transform (e.g., centering)
            if self.pre_transform is not None:
                data = Data(pos=points)
                data = self.pre_transform(data)
                points = data.pos

            # Downsample points using FPS
            if points.size(0) > self.num_points:
                points, _ = torch_fpsample.sample(points, self.num_points)

            # Create the data object
            if self.set_class:
                data = Data(pos=points, 
                            height=torch.tensor([height], dtype=torch.float32), 
                            y=torch.tensor([target_value], dtype=torch.float32), 
                            cls=torch.tensor([target_class], dtype=torch.long))
            else:
                data = Data(pos=points,
                            height=torch.tensor([height], dtype=torch.float32), 
                            y=torch.tensor([target_value], dtype=torch.float32))

            # Apply augmentation if needed
            if self.apply_augmentation and self.transform is not None:
                data = self.transform(data)

            data.file_name = file_name
            data_list.append(data)

        # Save processed dataset
        torch.save(self.collate(data_list), self.processed_paths[0])