import unittest

import torch
from torch_geometric.data import Batch, Data

from benchmarks.model_comparison.models import PointMLP


class PointMLPTest(unittest.TestCase):
    """Validate the official-style PointMLP regression adapter."""

    @staticmethod
    def _small_model() -> PointMLP:
        """Build a reduced test model while preserving stage behavior."""
        return PointMLP(
            num_points=128,
            embed_dim=16,
            dim_expansion=(2, 2, 2, 2),
            pre_blocks=(1, 1, 1, 1),
            pos_blocks=(1, 1, 1, 1),
            k_neighbors=(8, 8, 8, 8),
            reducers=(2, 2, 2, 2),
        )

    def test_forward_and_backward(self) -> None:
        """Return one scalar per graph and propagate gradients."""
        model = self._small_model().train()
        batch = Batch.from_data_list(
            [Data(pos=torch.rand(128, 3)), Data(pos=torch.rand(128, 3))]
        )

        prediction = model(batch)
        loss = prediction.square().mean()
        loss.backward()

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertIsNotNone(model.regression_head[-1].weight.grad)

    def test_rejects_unexpected_point_count(self) -> None:
        """Reject batches that violate the fixed-point benchmark protocol."""
        model = self._small_model().eval()
        batch = Batch.from_data_list([Data(pos=torch.rand(127, 3))])

        with self.assertRaisesRegex(ValueError, "requires 128 points"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
