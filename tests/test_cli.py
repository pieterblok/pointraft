import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import cli
from benchmarks.deployment import benchmark


class BenchmarkCliTest(unittest.TestCase):
    """Validate top-level dispatch without launching benchmark work."""

    def test_deployment_rejects_negative_limits(self) -> None:
        """Apply deployment argument validation through the unified CLI."""
        argv = ["benchmarks", "deployment", "run", "--limit", "-1"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            cli.main()

    def test_deployment_import_does_not_load_model_runner(self) -> None:
        """Keep Python 3.8 deployment isolated from main-environment code."""
        source = Path(cli.__file__).read_text(encoding="utf-8")
        prefix = source.split("def main", maxsplit=1)[0]
        self.assertNotIn("model_comparison.runner", prefix)

    @patch.object(
        benchmark.psutil,
        "virtual_memory",
        return_value=SimpleNamespace(total=17_179_869_184),
    )
    def test_reads_total_memory_cross_platform(self, _virtual_memory) -> None:
        """Read physical memory without relying on POSIX-only APIs."""
        self.assertEqual(benchmark.total_memory_bytes(), 17_179_869_184)


if __name__ == "__main__":
    unittest.main()
