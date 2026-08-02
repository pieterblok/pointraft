import ast
import hashlib
import unittest

from benchmarks.deployment.pipeline import PROJECT_ROOT, checkpoint_sha256


class RepositoryContractTest(unittest.TestCase):
    """Validate canonical paths without launching training or deployment."""

    def _assigned_paths(self, relative_path: str) -> dict[str, str]:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        module = ast.parse(source)
        paths: dict[str, str] = {}
        for node in module.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "Path":
                    paths[target.id] = ast.literal_eval(node.value.args[0])
        return paths

    def test_root_entry_points_use_canonical_paths(self) -> None:
        """Keep root PointRAFT commands free of public run selectors."""
        train_paths = self._assigned_paths("train.py")
        test_paths = self._assigned_paths("test.py")

        self.assertEqual(train_paths["CHECKPOINT_PATH"], "weights/pointraft.pth")
        self.assertEqual(
            train_paths["TRAIN_METRICS_PATH"],
            "results/pointraft/train_metrics.csv",
        )
        self.assertEqual(
            test_paths["PREDICTIONS_PATH"], "results/pointraft/predictions.csv"
        )
        self.assertNotIn("--run-name", (PROJECT_ROOT / "train.py").read_text())
        self.assertNotIn("--run-name", (PROJECT_ROOT / "test.py").read_text())

    def test_deployment_hashes_the_selected_checkpoint(self) -> None:
        """Record checkpoint bytes instead of enforcing a repository-global hash."""
        checkpoint = PROJECT_ROOT / ".pytest-checkpoint"
        try:
            checkpoint.write_bytes(b"repository-specific checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(checkpoint_sha256(checkpoint), expected)
        finally:
            checkpoint.unlink(missing_ok=True)

        pipeline_source = (PROJECT_ROOT / "benchmarks/deployment/pipeline.py").read_text()
        self.assertNotIn("REFERENCE_CHECKPOINT_SHA256", pipeline_source)

    def test_results_is_not_a_python_package(self) -> None:
        """Keep generated result directories artifact-only."""
        self.assertFalse((PROJECT_ROOT / "results" / "__init__.py").exists())

    def test_legacy_source_directories_are_empty(self) -> None:
        """Prevent old import paths from becoming compatibility layers."""
        legacy_directories = (
            PROJECT_ROOT / "misc",
            PROJECT_ROOT / "benchmarks" / "models",
            PROJECT_ROOT / "benchmarks" / "inference",
        )
        for directory in legacy_directories:
            self.assertFalse(list(directory.glob("*.py")), directory)


if __name__ == "__main__":
    unittest.main()
