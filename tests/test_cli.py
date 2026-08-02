import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import cli


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


if __name__ == "__main__":
    unittest.main()
