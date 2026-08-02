"""Synthetic checks for the isolated PTv3 regression adapter."""

import sys
import unittest
from pathlib import Path

import torch
from smoke import prepare_upstream


class PTv3RegressorTest(unittest.TestCase):
    """Validate scalar output and gradient flow through the adapter."""

    @classmethod
    def setUpClass(cls) -> None:
        project_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(prepare_upstream(project_dir)))

    def test_forward_and_backward(self) -> None:
        """Produce one differentiable scalar for each synthetic cloud."""
        from adapter import PTv3Regressor

        device = torch.device("cuda")
        model = PTv3Regressor().to(device).train()
        coordinates = torch.rand(128, 3, device=device) * 0.08
        batch = torch.arange(2, device=device).repeat_interleave(64)

        prediction = model(coordinates, batch, 0.002)
        prediction.sum().backward()

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertIsNotNone(model.regression_head[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
