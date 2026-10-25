"""Run PointRAFT ablations with auditable canonical artifacts."""

import csv
import json
import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path

from utils.provenance import git_commit, git_source_dirty, write_json

from . import ABLATION_REGISTRY, ABLATIONS

SUMMARY_FIELDS = (
    "ablation",
    "description",
    "tubers",
    "point_clouds",
    "mae_g",
    "rmse_g",
    "r2",
    "prediction_std_g",
    "average_execution_ms",
    "model_parameters",
)


def artifact_paths(project_root: Path, ablation: str) -> dict[str, Path]:
    """Return canonical checkpoint and result paths for an ablation."""
    result_root = project_root / "results" / "ablation_study" / ablation
    return {
        "checkpoint": project_root / "weights" / "ablation_study" / f"{ablation}.pth",
        "train_log": result_root / "train_metrics.csv",
        "train_provenance": result_root / "train.provenance.json",
        "prediction": result_root / "predictions.csv",
        "test_log": result_root / "test_metrics.txt",
        "test_provenance": result_root / "test.provenance.json",
    }


def run_command(
    command: list[str],
    project_root: Path,
    ablation: str,
    action: str,
) -> bool:
    """Run one command, stream output, and retain failure evidence."""
    result_root = project_root / "results" / "ablation_study" / ablation
    log_path = result_root / f"{action}.runner.log"
    failure_path = result_root / f"{action}.failure.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recent_output: deque[str] = deque(maxlen=100)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Unable to capture ablation subprocess output")
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                recent_output.append(line.rstrip())
        return_code = process.wait()

    if return_code == 0:
        if failure_path.exists():
            failure_path.unlink()
        return True

    write_json(
        failure_path,
        {
            "ablation": ablation,
            "action": action,
            "code_commit": git_commit(project_root),
            "code_dirty": git_source_dirty(project_root),
            "command": command,
            "recent_output": list(recent_output),
            "return_code": return_code,
            "status": "failed",
        },
    )
    return False


def build_command(
    action: str,
    ablation: str,
    *,
    epochs: int = 50,
    learning_rate: float = 0.0005,
    batch_size: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_files: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Build a train or test command for one ablation."""
    command = [
        sys.executable,
        "-m",
        f"benchmarks.ablation_study.{action}",
        "--ablation",
        ablation,
    ]
    if action == "train":
        command.extend(
            [
                "--epochs",
                str(epochs),
                "--learning-rate",
                str(learning_rate),
                "--batch-size",
                str(batch_size or ABLATION_REGISTRY[ablation].batch_size),
            ]
        )
        if max_train_batches is not None:
            command.extend(["--max-train-batches", str(max_train_batches)])
        if max_val_batches is not None:
            command.extend(["--max-val-batches", str(max_val_batches)])
        if overwrite:
            command.append("--overwrite")
    elif max_test_files is not None:
        command.extend(["--max-test-files", str(max_test_files)])
    return command


def write_summary(project_root: Path) -> Path | None:
    """Collect every completed ablation test into one summary CSV."""
    rows = []
    for ablation in ABLATIONS:
        provenance_path = artifact_paths(project_root, ablation)["test_provenance"]
        if not provenance_path.exists():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        row = {field: provenance.get(field) for field in SUMMARY_FIELDS}
        row["ablation"] = ablation
        row["description"] = ABLATION_REGISTRY[ablation].description
        rows.append(row)
    if not rows:
        return None
    summary_path = project_root / "results" / "ablation_study" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def run_ablations(
    action: str,
    ablations: list[str],
    *,
    continue_on_error: bool = False,
    epochs: int = 50,
    learning_rate: float = 0.0005,
    batch_size: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_files: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run or print train/test commands for the selected ablations."""
    project_root = Path(__file__).resolve().parents[2]
    # Always run in registry order: the last ablations rebuild "processed".
    selected = [ablation for ablation in ABLATIONS if ablation in ablations]
    all_succeeded = True
    for ablation in selected:
        config = ABLATION_REGISTRY[ablation]
        command = build_command(
            action,
            ablation,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            max_test_files=max_test_files,
            overwrite=overwrite,
        )
        paths = artifact_paths(project_root, ablation)
        print(f"Ablation: {ablation} ({config.description})")
        print(f"Command: {shlex.join(command)}")
        print(f"Checkpoint: {paths['checkpoint']}")
        if action == "train":
            if config.rebuild_processed:
                print("Processed data: deleted and rebuilt before training")
            print(f"Log: {paths['train_log']}")
            print(f"Provenance: {paths['train_provenance']}")
        else:
            print(f"Predictions: {paths['prediction']}")
            print(f"Log: {paths['test_log']}")
            print(f"Provenance: {paths['test_provenance']}")
        if dry_run:
            continue
        succeeded = run_command(command, project_root, ablation, action)
        all_succeeded = all_succeeded and succeeded
        if not succeeded and not continue_on_error:
            break

    if action == "test" and not dry_run:
        summary_path = write_summary(project_root)
        if summary_path is not None:
            print(f"Summary: {summary_path}")
    return all_succeeded
