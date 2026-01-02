## thanks to: https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing#scrollTo=hUbRw_BhLuXr

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.typing import WITH_TORCH_CLUSTER
from sklearn.metrics import root_mean_squared_error
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data import *
from utils import *
from models import *
model_dict = {'pointraft': PointRAFT}


if not WITH_TORCH_CLUSTER:
    quit("This code requires 'torch-cluster'")



def objective(trial):
    torch.manual_seed(133)
    random.seed(133)
    np.random.seed(133)


    ## File paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    data_root = os.path.join(datafolder, '1_rgbd', '2_pcd')
    splits_csv = os.path.join(datafolder, 'splits.csv')
    target_csv = os.path.join(datafolder, 'ground_truth.csv')

    weightsfolder = os.path.join(project_root, 'weights', 'optuna_hyperparameters')
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
    train_dataset = PointCloudDataset(data_root, splits_csv, target_csv, target_col="weight_g_inctack", class_col="weight_class", split="train", pre_transform=pre_transform, transform=transform, num_points=1024, apply_augmentation=True)
    val_dataset = PointCloudDataset(data_root, splits_csv, target_csv, target_col="weight_g_inctack", split="val", pre_transform=pre_transform, transform=transform, num_points=1024, apply_augmentation=False)


    ## Initialize the training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'pointraft'
    model = model_dict[model_name]().to(device)


    ## Optuna choices
    lr = trial.suggest_categorical('lr', [0.01, 0.005, 0.001, 0.0005, 0.0001])
    weight_decay = trial.suggest_categorical('weight_decay', [0.01, 0.001, 0.0001, 0.00001])
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    loss_function_choice = trial.suggest_categorical('loss_function', ['MSELoss', 'L1Loss', 'SmoothL1Loss'])

    if loss_function_choice == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_function_choice == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_function_choice == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()


    ## Prepare dataloaders
    train_sampler = ImbalancedSampler(train_dataset.cls)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    

    ## Hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    best_loss = float('inf')

    
    ## Training
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
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f'Trial: {trial.number:03d}, Epoch: {epoch:03d}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weightsfolder, f"trial_{trial.number:03d}.pth"))
            print('Saved best model!')
        scheduler.step()


    ## Independent testing
    best_model_weights = torch.load(os.path.join(weightsfolder, f"trial_{trial.number:03d}.pth"))
    model.load_state_dict(best_model_weights)
    model.eval()

    predictions = np.array([])
    ground_truths = np.array([])
    for val_data in val_loader:
        val_data = val_data.to(device)
        with torch.no_grad():
            pred = model(val_data)
            pred_weight = pred
            predictions = np.append(predictions, pred_weight.cpu().numpy())
            target = val_data.y
            ground_truths = np.append(ground_truths, target.cpu().numpy())
            
    rmse_weight = root_mean_squared_error(ground_truths, predictions)

    new_row = pd.DataFrame({"Trial": [trial.number],
                            "lr": [lr],
                            "weight_decay": [weight_decay],
                            "batch_size": [batch_size],
                            "loss_function": [loss_function_choice],
                            "rmse_weight": [rmse_weight]})
    
    csv_file = os.path.join(weightsfolder, "optuna_results.csv")
    if os.path.exists(csv_file):
        df_csv = pd.read_csv(csv_file)
        df_csv = pd.concat([df_csv, new_row], ignore_index=True)
    else:
        df_csv = new_row
    df_csv.to_csv(csv_file, index=False)
        
    return rmse_weight


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25)

    print("Best hyperparameters: ", study.best_params)
    print("Best RMSE weight: ", study.best_value)