"""Unified model-comparison and deployment command-line interface."""

import argparse
import shlex
import sys
from pathlib import Path

from .deployment import benchmark as deployment_benchmark
from .deployment import smoke_test as deployment_smoke
from .model_comparison import BENCHMARK_MODELS, DEFAULT_BENCHMARK_MODELS


def _add_model_comparison(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "model-comparison", help="Train or test reviewer comparison models."
    )
    action_parsers = parser.add_subparsers(dest="action", required=True)
    train_parser = action_parsers.add_parser("train", help="Train comparison models.")
    test_parser = action_parsers.add_parser("test", help="Test comparison models.")
    for command_parser in (train_parser, test_parser):
        command_parser.set_defaults(workflow="model-comparison")
        command_parser.add_argument(
            "--models",
            nargs="+",
            choices=BENCHMARK_MODELS,
            default=list(DEFAULT_BENCHMARK_MODELS),
        )
        command_parser.add_argument("--continue-on-error", action="store_true")
        command_parser.add_argument("--dry-run", action="store_true")
        command_parser.add_argument("--max-export-files-per-split", type=int)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--max-train-batches", type=int)
    train_parser.add_argument("--max-val-batches", type=int)
    train_parser.add_argument("--overwrite", action="store_true")
    test_parser.add_argument("--max-test-files", type=int)


def _add_deployment(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("deployment", help="Run deployment validation.")
    action_parsers = parser.add_subparsers(dest="action", required=True)
    smoke_parser = action_parsers.add_parser("smoke", help="Check one real sample.")
    deployment_smoke.configure_parser(smoke_parser)
    smoke_parser.set_defaults(workflow="deployment-smoke")
    run_parser = action_parsers.add_parser("run", help="Run deployment benchmark.")
    deployment_benchmark.configure_parser(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(workflow="deployment-run")


def main() -> None:
    """Parse and dispatch benchmark workflows."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_model_comparison(subparsers)
    _add_deployment(subparsers)
    args = parser.parse_args()

    if args.workflow == "model-comparison":
        from .model_comparison.runner import run_models

        succeeded = run_models(
            action=args.action,
            models=args.models,
            continue_on_error=args.continue_on_error,
            epochs=getattr(args, "epochs", 50),
            learning_rate=getattr(args, "learning_rate", 0.001),
            batch_size=getattr(args, "batch_size", 32),
            max_train_batches=getattr(args, "max_train_batches", None),
            max_val_batches=getattr(args, "max_val_batches", None),
            max_test_files=getattr(args, "max_test_files", None),
            max_export_files_per_split=args.max_export_files_per_split,
            overwrite=getattr(args, "overwrite", False),
            dry_run=args.dry_run,
        )
        if not succeeded:
            raise SystemExit(1)
        return

    if args.workflow == "deployment-smoke":
        deployment_smoke.run(args)
        return

    if args.dry_run:
        project_root = Path(__file__).resolve().parent.parent
        command = [sys.executable, "-m", "benchmarks.deployment.benchmark"]
        print(f"Command: {shlex.join(command)}")
        print(f"Checkpoint: {project_root / 'weights' / 'pointraft.pth'}")
        print(f"Results: {args.output_root}")
        return
    deployment_benchmark.run(args)
