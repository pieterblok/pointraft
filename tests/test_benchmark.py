import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.model_comparison import BENCHMARK_MODELS, DEFAULT_BENCHMARK_MODELS
from benchmarks.model_comparison.runner import (
    artifact_paths,
    build_command,
    build_ptv3_export_command,
    run_command,
    run_models,
)


class BenchmarkRunnerTest(unittest.TestCase):
    """Validate unified benchmark commands, logging, and failure evidence."""

    def test_public_models_remain_the_default(self) -> None:
        """Keep the no-argument public benchmark limited to verified baselines."""
        self.assertEqual(DEFAULT_BENCHMARK_MODELS, ("pointnet", "pointnet2"))

    def test_builds_bounded_paper_training_command(self) -> None:
        """Forward paper-model training controls through the shared runner."""
        project_root = Path("/project")
        command = build_command(
            "train",
            "pointmlp",
            project_root,
            epochs=3,
            learning_rate=0.003,
            batch_size=2,
            max_train_batches=5,
            max_val_batches=2,
            overwrite=True,
        )

        self.assertEqual(
            command[1:3], ["-m", "benchmarks.model_comparison.train"]
        )
        self.assertIn("pointmlp", command)
        self.assertIn("--max-train-batches", command)
        self.assertIn("--max-val-batches", command)
        self.assertIn("--overwrite", command)

    def test_builds_bounded_test_command(self) -> None:
        """Forward bounded test-file selection through the shared runner."""
        command = build_command(
            "test",
            "pointmlp",
            Path("/project"),
            max_test_files=64,
        )

        self.assertEqual(command[-2:], ["--max-test-files", "64"])

    def test_builds_isolated_ptv3_commands(self) -> None:
        """Route default PTv3 through the official-style isolated model."""
        project_root = Path("/project")
        export_command = build_ptv3_export_command(project_root, 8)
        train_command = build_command(
            "train",
            "ptv3",
            project_root,
            epochs=1,
            batch_size=2,
            max_train_batches=1,
            max_val_batches=1,
        )

        self.assertEqual(export_command[0], sys.executable)
        self.assertEqual(export_command[-2:], ["--max-files-per-split", "8"])
        self.assertEqual(train_command[:2], ["uv", "run"])
        self.assertIn(
            "/project/benchmarks/model_comparison/models/ptv3", train_command
        )
        self.assertIn(
            "/project/benchmarks/model_comparison/models/ptv3/regression.py",
            train_command,
        )
        self.assertIn("train", train_command)
        self.assertIn("--model-variant", train_command)
        self.assertIn("official_3stage", train_command)

    def test_builds_compact_ptv3_command(self) -> None:
        """Route the task-optimized PTv3 variant to separate artifacts."""
        command = build_command(
            "test",
            "ptv3_compact",
            Path("/project"),
            max_test_files=8,
        )

        self.assertIn("ptv3_compact", command)
        self.assertEqual(command[-3:], ["compact", "--max-test-files", "8"])

    def test_uses_flat_canonical_artifact_paths(self) -> None:
        """Keep every comparison model on the common artifact contract."""
        for model in BENCHMARK_MODELS:
            paths = artifact_paths(Path("/project"), model)
            self.assertEqual(
                paths["checkpoint"], Path(f"/project/weights/{model}.pth")
            )
            self.assertEqual(
                paths["prediction"],
                Path(f"/project/results/model_comparison/predictions/{model}.csv"),
            )
            self.assertEqual(
                paths["test_provenance"],
                Path(
                    f"/project/results/model_comparison/provenance/{model}.test.json"
                ),
            )

    def test_dry_run_does_not_launch_subprocesses(self) -> None:
        """Make virtual validation incapable of replacing artifacts."""
        with patch(
            "benchmarks.model_comparison.runner.run_command"
        ) as run_command_mock:
            succeeded = run_models(
                "train", ["pointnet", "ptv3"], overwrite=True, dry_run=True
            )

        self.assertTrue(succeeded)
        run_command_mock.assert_not_called()

    def test_failed_command_writes_evidence(self) -> None:
        """Retain the return code and recent output from a failed command."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            succeeded = run_command(
                [sys.executable, "-c", "print('bounded failure'); raise SystemExit(3)"],
                project_root,
                "example",
                "train",
            )

            failure_path = (
                project_root
                / "results"
                / "model_comparison"
                / "provenance"
                / "example.train.failure.json"
            )
            evidence = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertFalse(succeeded)
            self.assertEqual(evidence["return_code"], 3)
            self.assertIn("bounded failure", evidence["recent_output"])


if __name__ == "__main__":
    unittest.main()
