## thanks to: https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing#scrollTo=hUbRw_BhLuXr

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.typing import WITH_TORCH_CLUSTER

from data import *
from utils import *
from models import *
model_dict = {'pointraft': PointRAFT}


if not WITH_TORCH_CLUSTER:
    quit("This code requires 'torch-cluster'")



if __name__ == '__main__':
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)


    ## File paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    data_root = os.path.join(datafolder, '1_rgbd', '2_pcd')
    splits_csv = os.path.join(datafolder, 'splits.csv')
    target_csv = os.path.join(datafolder, 'ground_truth.csv')

    weightsfolder = os.path.join(project_root, 'weights')
    os.makedirs(weightsfolder, exist_ok=True)


    ## Data preprocessing and augmentation
    pre_transform = T.Center()
    transform = T.Compose([T.RandomJitter(0.0005), 
                           T.RandomRotate(2, axis=0), 
                           T.RandomRotate(2, axis=1), 
                           T.RandomRotate(2, axis=2),
                           T.RandomFlip(axis=0, p=0.5),
                           T.RandomFlip(axis=1, p=0.5),
                           T.RandomShear(0.2)])


    ## Create/load InMemoryDatasets
    ## Please remove the "processed" folder in your data_root if you want to redo the data augmentation!
    train_dataset = PointCloudDataset(data_root, 
                                      splits_csv, 
                                      target_csv, 
                                      target_col="weight_g_inctack", 
                                      class_col="weight_class", 
                                      split="train",
                                      conveyor_depth={'2023': 0.345, '2024': 0.460, '2025': 0.465},
                                      search_col="growing_season",
                                      max_height=0.08,
                                      pre_transform=pre_transform, 
                                      transform=transform, 
                                      num_points=1024, 
                                      apply_augmentation=True)
    
    val_dataset = PointCloudDataset(data_root, 
                                    splits_csv, 
                                    target_csv, 
                                    target_col="weight_g_inctack", 
                                    split="val",
                                    conveyor_depth={'2023': 0.345, '2024': 0.460, '2025': 0.465},
                                    search_col="growing_season",
                                    max_height=0.08,
                                    pre_transform=pre_transform, 
                                    transform=transform, 
                                    num_points=1024, 
                                    apply_augmentation=False)


    ## Dataloaders
    train_sampler = ImbalancedSampler(train_dataset.cls)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6)


    ## Initialize the training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'pointraft'
    visualize = False
    model = model_dict[model_name]().to(device)

    criterion = nn.SmoothL1Loss(beta=20)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    if visualize:
        visualize_batch(train_loader)
        visualize_augmentation(train_loader, pre_transform)


    ## Training
    best_loss = float('inf')
    for epoch in range(1, 51):
        model.train()
        train_loss = 0
        for train_data in train_loader:
            train_data = train_data.to(device)
            optimizer.zero_grad()
            pred = model(train_data)
            target = train_data.y
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        for val_data in val_loader:
            val_data = val_data.to(device)
            with torch.no_grad():
                pred = model(val_data)
                target = val_data.y
                loss = criterion(pred, target)
                val_loss  += loss.item()
                
        val_loss /= len(val_loader)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weightsfolder, model_name + '.pth'))
            print('Saved best model!')
        scheduler.step()