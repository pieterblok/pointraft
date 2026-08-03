"""Train and evaluate the isolated PTv3 scalar-regression adapter."""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import flash_attn
import numpy as np
import spconv
import torch
import torch_scatter
from smoke import prepare_upstream
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

RANDOM_SEED = 133
GRID_SIZE = 0.002


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_state(project_root: Path) -> tuple[str, bool | None]:
    """Return the repository commit and source-dirty status."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)data/3DPotatoTwin",
                ":(exclude)weights",
                ":(exclude)results",
                ":(exclude)benchmarks/model_comparison/models/ptv3/.data",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def seed_everything() -> None:
    """Seed Python, NumPy, and Torch without forcing unsupported sparse kernels."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False


def augment(coordinates: torch.Tensor) -> torch.Tensor:
    """Apply the paper protocol's jitter, rotations, flips, and shear."""
    coordinates = coordinates.clone()
    coordinates += torch.empty_like(coordinates).uniform_(-0.0005, 0.0005)
    for axis in range(3):
        angle = math.pi * random.uniform(-2.0, 2.0) / 180.0
        sine, cosine = math.sin(angle), math.cos(angle)
        if axis == 0:
            matrix = [[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]]
        elif axis == 1:
            matrix = [[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]]
        else:
            matrix = [[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]]
        coordinates = coordinates @ coordinates.new_tensor(matrix).t()
    for axis in (0, 1):
        if random.random() < 0.5:
            coordinates[:, axis] *= -1
    shear = coordinates.new_empty(3, 3).uniform_(-0.2, 0.2)
    shear[torch.arange(3), torch.arange(3)] = 1
    return coordinates @ shear.t()


class ExportedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """Read a non-pickle NPZ split exported by the main environment."""

    def __init__(self, path: Path, *, apply_augmentation: bool = False) -> None:
        arrays = np.load(path)
        self.coordinates = arrays["coordinates"]
        self.offsets = arrays["offsets"]
        self.targets = arrays["targets"]
        self.classes = arrays["classes"]
        self.file_names = arrays["file_names"]
        self.unique_ids = arrays["unique_ids"]
        self.apply_augmentation = apply_augmentation

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        start, stop = self.offsets[index : index + 2]
        coordinates = torch.from_numpy(self.coordinates[start:stop])
        if self.apply_augmentation:
            coordinates = augment(coordinates)
        return (
            coordinates,
            torch.tensor(self.targets[index], dtype=torch.float32),
            str(self.file_names[index]),
            str(self.unique_ids[index]),
        )


def collate_clouds(
    samples: list[tuple[torch.Tensor, torch.Tensor, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Concatenate variable-length point clouds and retain sample metadata."""
    coordinates, targets, file_names, unique_ids = zip(*samples, strict=True)
    batch = torch.cat(
        [
            torch.full((len(points),), index, dtype=torch.long)
            for index, points in enumerate(coordinates)
        ]
    )
    return (
        torch.cat(coordinates),
        batch,
        torch.stack(targets),
        list(file_names),
        list(unique_ids),
    )


def evenly_spaced_indices(length: int, maximum: int | None) -> list[int]:
    """Return full or evenly bounded indices."""
    if maximum is None or length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    return [round(i * (length - 1) / (maximum - 1)) for i in range(maximum)]


def runtime_versions() -> dict[str, str]:
    """Return isolated runtime versions relevant to reproduction."""
    return {
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "flash_attn": flash_attn.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "spconv": spconv.__version__,
        "torch": torch.__version__,
        "torch_scatter": torch_scatter.__version__,
    }


def load_context(
    model_variant: str,
) -> tuple[Path, Path, dict[str, Any], type[nn.Module]]:
    """Prepare pinned PTv3 source and load the export manifest and adapter."""
    project_dir = Path(__file__).resolve().parent
    project_root = project_dir.parents[3]
    upstream_root = prepare_upstream(project_dir)
    sys.path.insert(0, str(upstream_root))
    from adapter import OfficialPTv3ThreeStageRegressor, PTv3Regressor

    model_classes = {
        "compact": PTv3Regressor,
        "official_3stage": OfficialPTv3ThreeStageRegressor,
    }

    manifest_path = project_dir / ".data" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "PTv3 export is missing; run export_data.py in the main environment"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        split_path = project_dir / ".data" / f"{split}.npz"
        if sha256_file(split_path) != manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"PTv3 export hash mismatch: {split_path}")
    return project_dir, project_root, manifest, model_classes[model_variant]


def train(args: argparse.Namespace) -> None:
    """Train a selected PTv3 adapter under the paper optimizer protocol."""
    project_dir, project_root, manifest, model_class = load_context(
        args.model_variant
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_dir = project_dir / ".data"
    train_dataset = ExportedDataset(data_dir / "train.npz", apply_augmentation=True)
    val_dataset = ExportedDataset(data_dir / "val.npz")

    class_counts = np.bincount(train_dataset.classes)
    sample_weights = 1.0 / class_counts[train_dataset.classes]
    sampler_generator = torch.Generator().manual_seed(RANDOM_SEED)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_clouds,
        num_workers=0,
    )
    val_count = (
        None if args.max_val_batches is None else args.max_val_batches * args.batch_size
    )
    val_subset = Subset(
        val_dataset,
        evenly_spaced_indices(len(val_dataset), val_count),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_clouds,
        num_workers=0,
    )

    model = model_class().to(device)
    use_amp = bool(getattr(model, "uses_flash_attention", False))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = nn.SmoothL1Loss(beta=20)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    checkpoint_path = project_root / "weights" / f"{args.run_name}.pth"
    if checkpoint_path.exists() and not args.overwrite:
        raise RuntimeError(f"{checkpoint_path} exists; pass --overwrite to replace it")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_root = project_root / "results" / "model_comparison"
    log_path = result_root / "logs" / f"{args.run_name}.train.csv"
    provenance_path = (
        result_root / "provenance" / f"{args.run_name}.train.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "epoch,train_loss,val_loss,val_prediction_std,learning_rate\n",
        encoding="utf-8",
    )

    best_loss = float("inf")
    best_epoch = 0
    val_prediction_std = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_index, (coordinates, batch, targets, _, _) in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            coordinates = coordinates.to(device)
            batch = batch.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(coordinates, batch, args.grid_size)
            loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        if not train_losses:
            raise RuntimeError("No PTv3 training batches were evaluated")

        model.eval()
        val_losses: list[float] = []
        val_predictions: list[torch.Tensor] = []
        with torch.inference_mode():
            for coordinates, batch, targets, _, _ in val_loader:
                coordinates = coordinates.to(device)
                batch = batch.to(device)
                targets = targets.to(device)
                predictions = model(coordinates, batch, args.grid_size)
                val_losses.append(criterion(predictions, targets).item())
                val_predictions.append(predictions.cpu())
        if not val_losses:
            raise RuntimeError("No PTv3 validation batches were evaluated")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_prediction_std = torch.cat(val_predictions).std(unbiased=False).item()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.5f}, "
            f"Val Loss: {val_loss:.5f}, Val Prediction SD: {val_prediction_std:.5f}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{epoch},{train_loss:.8f},{val_loss:.8f},"
                f"{val_prediction_std:.8f},{learning_rate:.12g}\n"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "grid_size": args.grid_size,
                    "model_state_dict": model.state_dict(),
                    "variant": model.variant,
                },
                checkpoint_path,
            )
            print("Saved best PTv3 model", flush=True)
        scheduler.step()

    peak_allocated_bytes = None
    peak_reserved_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        print(
            "Peak CUDA memory: "
            f"allocated={peak_allocated_bytes / 1024**2:.1f} MiB, "
            f"reserved={peak_reserved_bytes / 1024**2:.1f} MiB",
            flush=True,
        )

    commit, dirty = git_state(project_root)
    write_json(
        provenance_path,
        {
            "artifact": args.run_name,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "code_commit": commit,
            "code_dirty": dirty,
            "command": [sys.executable, *sys.argv],
            "dataset": manifest["dataset"],
            "device": str(device),
            "epochs": args.epochs,
            "export_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "grid_size": args.grid_size,
            "ground_truth_sha256": manifest["ground_truth_sha256"],
            "learning_rate": args.learning_rate,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "model": "ptv3",
            "model_selection": args.model_variant,
            "model_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "model_variant": model.variant,
            "optimizer": "Adam",
            "precision": "fp16-flash-fp32-rest" if use_amp else "float32",
            "peak_cuda_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_reserved_bytes": peak_reserved_bytes,
            "scheduler": "ExponentialLR(gamma=0.97)",
            "seed": RANDOM_SEED,
            "splits_sha256": manifest["splits_sha256"],
            "status": "completed",
            "train_samples": len(train_dataset),
            "validation_prediction_std": val_prediction_std,
            "validation_samples": len(val_dataset),
            "validation_samples_evaluated": len(val_subset),
            "versions": runtime_versions(),
            "weight_decay": 0.0001,
            "weight_sha256": sha256_file(checkpoint_path),
        },
    )


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a PTv3 checkpoint and write repository-standard artifacts."""
    project_dir, project_root, manifest, model_class = load_context(
        args.model_variant
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_dir = project_dir / ".data"
    test_dataset = ExportedDataset(data_dir / "test.npz")
    indices = evenly_spaced_indices(len(test_dataset), args.max_test_files)
    test_loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_clouds,
        num_workers=0,
    )
    checkpoint_path = project_root / "weights" / f"{args.run_name}.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint["variant"] != model_class.variant:
        raise RuntimeError(
            "Checkpoint variant does not match requested model: "
            f"{checkpoint['variant']} != {model_class.variant}"
        )
    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    use_amp = bool(getattr(model, "uses_flash_attention", False))
    grid_size = float(checkpoint["grid_size"])

    warmup = collate_clouds([test_dataset[indices[0]]])
    with torch.inference_mode():
        model(warmup[0].to(device), warmup[1].to(device), grid_size)
        if device.type == "cuda":
            torch.cuda.synchronize()

    predictions: list[float] = []
    targets: list[float] = []
    file_names: list[str] = []
    unique_ids: list[str] = []
    execution_times: list[float] = []
    with torch.inference_mode():
        for coordinates, batch, target, files, ids in test_loader:
            coordinates = coordinates.to(device)
            batch = batch.to(device)
            start = time.perf_counter()
            prediction = model(coordinates, batch, grid_size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            execution_times.append((time.perf_counter() - start) * 1000)
            predictions.append(float(prediction.item()))
            targets.append(float(target.item()))
            file_names.extend(files)
            unique_ids.extend(ids)

    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    residuals = target_array - prediction_array
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    target_variance = float(np.sum(np.square(target_array - target_array.mean())))
    r2 = float(1 - np.sum(np.square(residuals)) / target_variance)
    prediction_std = float(np.std(prediction_array))
    average_execution_ms = float(np.mean(execution_times))

    result_root = project_root / "results" / "model_comparison"
    result_path = result_root / "predictions" / f"{args.run_name}.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file_name", "unique_id", "gt", "pred", "diff"))
        writer.writerows(
            zip(
                file_names,
                unique_ids,
                targets,
                predictions,
                np.abs(residuals),
                strict=True,
            )
        )
    log_path = result_root / "logs" / f"{args.run_name}.test.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            (
                f"tubers={len(set(unique_ids))}",
                f"point_clouds={len(file_names)}",
                f"mae_g={mae:.8f}",
                f"rmse_g={rmse:.8f}",
                f"r2={r2:.8f}",
                f"prediction_std_g={prediction_std:.8f}",
                f"average_execution_ms={average_execution_ms:.8f}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"PTv3 test: point_clouds={len(file_names)}, MAE={mae:.3f} g, "
        f"RMSE={rmse:.3f} g, R2={r2:.4f}",
        flush=True,
    )

    commit, dirty = git_state(project_root)
    provenance_path = (
        result_root / "provenance" / f"{args.run_name}.test.json"
    )
    write_json(
        provenance_path,
        {
            "artifact": args.run_name,
            "average_execution_ms": average_execution_ms,
            "code_commit": commit,
            "code_dirty": dirty,
            "command": [sys.executable, *sys.argv],
            "dataset": manifest["dataset"],
            "device": str(device),
            "export_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "grid_size": grid_size,
            "ground_truth_sha256": manifest["ground_truth_sha256"],
            "mae_g": mae,
            "max_test_files": args.max_test_files,
            "model": "ptv3",
            "model_selection": args.model_variant,
            "model_variant": checkpoint["variant"],
            "point_clouds": len(file_names),
            "precision": "fp16-flash-fp32-rest" if use_amp else "float32",
            "prediction_sha256": sha256_file(result_path),
            "prediction_std_g": prediction_std,
            "r2": r2,
            "rmse_g": rmse,
            "splits_sha256": manifest["splits_sha256"],
            "status": "completed",
            "timing_scope": "isolated model forward only",
            "tubers": len(set(unique_ids)),
            "versions": runtime_versions(),
            "weight_sha256": sha256_file(checkpoint_path),
        },
    )


def main() -> None:
    """Parse isolated PTv3 train or test commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")
    for command_parser in (train_parser, test_parser):
        command_parser.add_argument("--run-name", required=True)
        command_parser.add_argument(
            "--model-variant",
            choices=("compact", "official_3stage"),
            default="compact",
        )
        command_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--grid-size", type=float, default=GRID_SIZE)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-val-batches", type=int)
    train_parser.add_argument("--overwrite", action="store_true")
    test_parser.add_argument("--max-test-files", type=int)
    args = parser.parse_args()

    for name in (
        "epochs",
        "batch_size",
        "max_train_batches",
        "max_val_batches",
        "max_test_files",
    ):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "learning_rate", 1.0) <= 0:
        parser.error("--learning-rate must be positive")
    if getattr(args, "grid_size", 1.0) <= 0:
        parser.error("--grid-size must be positive")
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a file-name-safe value")

    seed_everything()
    if args.action == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
