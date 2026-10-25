"""Run reviewer comparison models with auditable canonical artifacts."""

import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path

from utils.provenance import git_commit, git_source_dirty, write_json

PTV3_MODEL_VARIANTS = {
    "ptv3": "official_3stage",
    "ptv3_compact": "compact",
}


def artifact_paths(project_root: Path, model: str) -> dict[str, Path]:
    """Return canonical checkpoint and result paths for a comparison model."""
    result_root = project_root / "results" / "model_comparison"
    return {
        "checkpoint": project_root / "weights" / f"{model}.pth",
        "prediction": result_root / "predictions" / f"{model}.csv",
        "train_log": result_root / "logs" / f"{model}.train.csv",
        "test_log": result_root / "logs" / f"{model}.test.txt",
        "train_provenance": result_root / "provenance" / f"{model}.train.json",
        "test_provenance": result_root / "provenance" / f"{model}.test.json",
    }


def run_command(
    command: list[str],
    project_root: Path,
    model: str,
    action: str,
) -> bool:
    """Run one command, stream output, and retain failure evidence."""
    result_root = project_root / "results" / "model_comparison"
    log_path = result_root / "logs" / f"{model}.{action}.runner.log"
    failure_path = (
        result_root / "provenance" / f"{model}.{action}.failure.json"
    )
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
            raise RuntimeError("Unable to capture benchmark subprocess output")
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
            "action": action,
            "code_commit": git_commit(project_root),
            "code_dirty": git_source_dirty(project_root),
            "command": command,
            "recent_output": list(recent_output),
            "return_code": return_code,
            "model": model,
            "status": "failed",
        },
    )
    return False


def build_command(
    action: str,
    model: str,
    project_root: Path,
    *,
    epochs: int = 50,
    learning_rate: float = 0.0005,
    batch_size: int = 32,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_files: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Build a train or test command for one canonical model artifact."""
    if model in PTV3_MODEL_VARIANTS:
        project_dir = (
            project_root / "benchmarks" / "model_comparison" / "models" / "ptv3"
        )
        command = [
            "uv",
            "run",
            "--project",
            str(project_dir),
            "--locked",
            "python",
            str(project_dir / "regression.py"),
            action,
            "--run-name",
            model,
            "--model-variant",
            PTV3_MODEL_VARIANTS[model],
        ]
    else:
        command = [
            sys.executable,
            "-m",
            f"benchmarks.model_comparison.{action}",
            "--model",
            model,
        ]
    if action == "train":
        command.extend(
            [
                "--epochs",
                str(epochs),
                "--learning-rate",
                str(learning_rate),
                "--batch-size",
                str(batch_size),
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


def build_ptv3_export_command(
    project_root: Path,
    max_files_per_split: int | None = None,
) -> list[str]:
    """Build the main-environment command that exports data for PTv3."""
    script = (
        project_root
        / "benchmarks"
        / "model_comparison"
        / "models"
        / "ptv3"
        / "export_data.py"
    )
    command = [sys.executable, str(script)]
    if max_files_per_split is not None:
        command.extend(["--max-files-per-split", str(max_files_per_split)])
    return command


def run_models(
    action: str,
    models: list[str],
    *,
    continue_on_error: bool = False,
    epochs: int = 50,
    learning_rate: float = 0.0005,
    batch_size: int = 32,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_files: int | None = None,
    max_export_files_per_split: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run or print canonical train/test commands for selected models."""
    project_root = Path(__file__).resolve().parents[2]
    all_succeeded = True
    for model in models:
        if model in PTV3_MODEL_VARIANTS:
            export_command = build_ptv3_export_command(
                project_root, max_export_files_per_split
            )
            if dry_run:
                print(f"PTv3 export: {shlex.join(export_command)}")
            else:
                print(f"Preparing data for {model}", flush=True)
                prepared = run_command(export_command, project_root, model, "prepare")
                all_succeeded = all_succeeded and prepared
                if not prepared:
                    if continue_on_error:
                        continue
                    break

        command = build_command(
            action,
            model,
            project_root,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            max_test_files=max_test_files,
            overwrite=overwrite,
        )
        paths = artifact_paths(project_root, model)
        print(f"Model: {model}")
        print(f"Command: {shlex.join(command)}")
        print(f"Checkpoint: {paths['checkpoint']}")
        if action == "train":
            print(f"Log: {paths['train_log']}")
            print(f"Provenance: {paths['train_provenance']}")
        else:
            print(f"Predictions: {paths['prediction']}")
            print(f"Log: {paths['test_log']}")
            print(f"Provenance: {paths['test_provenance']}")
        if dry_run:
            continue
        succeeded = run_command(command, project_root, model, action)
        all_succeeded = all_succeeded and succeeded
        if not succeeded and not continue_on_error:
            break
    return all_succeeded
