"""Train one PointRAFT ablation with the repository protocol."""

## thanks to: https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing#scrollTo=hUbRw_BhLuXr

import argparse
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER

from data import PointCloudDataset
from models import PointRAFT
from utils.provenance import (
    git_commit,
    git_source_dirty,
    runtime_versions,
    sha256_file,
    write_json,
)
from utils.reproducibility import (
    RANDOM_SEED,
    make_generator,
    seed_everything,
    seed_worker,
)
from utils.visualize import visualize_augmentation, visualize_batch

from .ablations import ABLATION_REGISTRY

# Written into a rebuilt "processed" folder so default-data ablations can
# refuse to train on data cached with another ablation's settings.
PROCESSED_MARKER = "ablation_source.txt"


def main() -> None:
    """Train one registered ablation."""
    if not WITH_TORCH_CLUSTER:
        raise SystemExit("This code requires 'torch-cluster'")

    parser = argparse.ArgumentParser(description="Train a PointRAFT ablation.")
    parser.add_argument(
        "--ablation",
        choices=ABLATION_REGISTRY,
        required=True,
        help="Ablation to train.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing model weight.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Adam learning rate (default: 0.0005).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default: the ablation's batch size, normally 32).",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Bound training batches per epoch for a prerun.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Bound validation batches per epoch for a prerun.",
    )
    args = parser.parse_args()
    config = ABLATION_REGISTRY[args.ablation]
    if args.batch_size is None:
        args.batch_size = config.batch_size

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for batch-normalized models")
    for argument in ("max_train_batches", "max_val_batches"):
        value = getattr(args, argument)
        if value is not None and value < 1:
            parser.error(f"--{argument.replace('_', '-')} must be positive")

    seed_everything()

    ## File paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    datafolder = project_root / "data" / "3DPotatoTwin"

    data_root = os.path.join(datafolder, "1_rgbd", "2_pcd")
    splits_csv = os.path.join(datafolder, "splits.csv")
    target_csv = os.path.join(datafolder, "ground_truth.csv")

    weightsfolder = project_root / "weights" / "ablation_study"
    weightsfolder.mkdir(parents=True, exist_ok=True)
    model_name = "pointraft"
    checkpoint_path = weightsfolder / f"{config.name}.pth"
    if checkpoint_path.exists() and not args.overwrite:
        parser.error(f"{checkpoint_path} already exists; use --overwrite to replace it")

    print(f"Dataset: {datafolder}")
    print(f"Model: {model_name}")
    print(f"Ablation: {config.name} ({config.description})")
    print(f"Checkpoint: {checkpoint_path}")

    resultsfolder = project_root / "results" / "ablation_study" / config.name
    log_path = resultsfolder / "train_metrics.csv"
    provenance_path = resultsfolder / "train.provenance.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "epoch,train_loss,val_loss,val_prediction_std,learning_rate\n",
        encoding="utf-8",
    )

    ## Cached PointCloudDataset tensors live in <data_root>/processed
    processed_dir = Path(data_root) / "processed"
    processed_marker = processed_dir / PROCESSED_MARKER
    if config.rebuild_processed:
        if processed_dir.exists():
            print(f"Deleting {processed_dir} so it is rebuilt for {config.name}")
            shutil.rmtree(processed_dir)
    elif processed_marker.exists():
        source = processed_marker.read_text(encoding="utf-8").strip()
        parser.error(
            f"{processed_dir} was rebuilt by the '{source}' ablation; restore the "
            f"default processed folder before training '{config.name}'"
        )

    ## Data preprocessing and augmentation
    pre_transform = T.Center()
    transform = None
    if config.augmentation:
        transform = T.Compose(
            [
                T.RandomJitter(0.0005),
                T.RandomRotate(2, axis=0),
                T.RandomRotate(2, axis=1),
                T.RandomRotate(2, axis=2),
                T.RandomFlip(axis=0, p=0.5),
                T.RandomFlip(axis=1, p=0.5),
                T.RandomShear(0.2),
            ]
        )

    ## Create/load InMemoryDatasets
    ## Please remove the "processed" folder in your data_root if you want to redo the data augmentation!
    train_dataset = PointCloudDataset(
        data_root,
        splits_csv,
        target_csv,
        target_col="weight_g_inctack",
        class_col="weight_class",
        split="train",
        conveyor_depth={"2023": 0.345, "2024": 0.460, "2025": 0.465},
        search_col="growing_season",
        max_height=0.08,
        pre_transform=pre_transform,
        transform=transform,
        num_points=config.num_points,
        apply_augmentation=config.augmentation,
    )

    val_dataset = PointCloudDataset(
        data_root,
        splits_csv,
        target_csv,
        target_col="weight_g_inctack",
        split="val",
        conveyor_depth={"2023": 0.345, "2024": 0.460, "2025": 0.465},
        search_col="growing_season",
        max_height=0.08,
        pre_transform=pre_transform,
        transform=transform,
        num_points=config.num_points,
        apply_augmentation=False,
    )

    if config.rebuild_processed:
        processed_marker.write_text(f"{config.name}\n", encoding="utf-8")

    ## Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        worker_init_fn=seed_worker,
        generator=make_generator(RANDOM_SEED + 1),
    )
    val_loader_dataset = val_dataset
    if args.max_val_batches is not None:
        sample_count = min(
            len(val_dataset),
            args.max_val_batches * args.batch_size,
        )
        if sample_count == 1:
            val_indices = [0]
        else:
            val_indices = [
                round(index * (len(val_dataset) - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        val_loader_dataset = val_dataset.index_select(val_indices)
    val_loader = DataLoader(
        val_loader_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        worker_init_fn=seed_worker,
        generator=make_generator(RANDOM_SEED + 2),
    )

    ## Initialize the training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize = False
    model = PointRAFT(**config.model_kwargs).to(device)

    criterion = config.loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    if visualize:
        visualize_batch(train_loader)
        visualize_augmentation(train_loader, pre_transform)

    ## Training
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_batches = 0
        for batch_index, train_data in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            train_data = train_data.to(device)
            optimizer.zero_grad()
            pred = model(train_data)
            target = train_data.y
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(target.shape)}"
                )
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        model.eval()
        val_loss = 0
        val_batches = 0
        val_predictions = []
        for batch_index, val_data in enumerate(val_loader):
            if args.max_val_batches is not None and batch_index >= args.max_val_batches:
                break
            val_data = val_data.to(device)
            with torch.no_grad():
                pred = model(val_data)
                target = val_data.y
                if pred.shape != target.shape:
                    raise RuntimeError(
                        f"Prediction shape {tuple(pred.shape)} does not match "
                        f"target shape {tuple(target.shape)}"
                    )
                loss = criterion(pred, target)
                val_loss += loss.item()
                val_batches += 1
                val_predictions.append(pred.detach().cpu())

        val_loss /= val_batches
        val_prediction_std = torch.cat(val_predictions).std(unbiased=False).item()

        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.5f}, "
            f"Val Loss: {val_loss:.5f}, Val Prediction SD: {val_prediction_std:.5f}"
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"{epoch},{train_loss:.8f},{val_loss:.8f},"
                f"{val_prediction_std:.8f},"
                f"{optimizer.param_groups[0]['lr']:.12g}\n"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model!")
        scheduler.step()

    write_json(
        provenance_path,
        {
            "ablation": config.name,
            "ablation_description": config.description,
            "artifact": config.name,
            "augmentation": config.augmentation,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "code_commit": git_commit(project_root),
            "code_dirty": git_source_dirty(project_root),
            "command": [
                sys.executable,
                "-m",
                "benchmarks.ablation_study.train",
                *sys.argv[1:],
            ],
            "loss": config.loss_name,
            "dataset": str(datafolder.resolve()),
            "device": str(device),
            "epochs": args.epochs,
            "ground_truth_sha256": sha256_file(Path(target_csv)),
            "learning_rate": args.learning_rate,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "model": model_name,
            "model_kwargs": config.model_kwargs,
            "model_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "num_points": config.num_points,
            "processed_rebuilt": config.rebuild_processed,
            "optimizer": "Adam",
            "train_sampler": "shuffle",
            "scheduler": "ExponentialLR(gamma=0.97)",
            "seed": RANDOM_SEED,
            "splits_sha256": sha256_file(Path(splits_csv)),
            "status": "completed",
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "validation_prediction_std": val_prediction_std,
            "validation_samples_evaluated": len(val_loader_dataset),
            "versions": runtime_versions(),
            "weight_decay": 0.0001,
            "weight_sha256": sha256_file(checkpoint_path),
        },
    )


if __name__ == "__main__":
    main()
