import unittest

import torch
from torch_geometric.data import Batch, Data

from benchmarks.model_comparison.models import DGCNN


class DGCNNTest(unittest.TestCase):
    """Validate the author-style DGCNN regression adapter."""

    def test_forward_and_backward(self) -> None:
        """Return one scalar per graph and propagate gradients."""
        model = DGCNN(k=8, num_points=64).train()
        batch = Batch.from_data_list(
            [Data(pos=torch.rand(64, 3)), Data(pos=torch.rand(64, 3))]
        )

        prediction = model(batch)
        loss = prediction.square().mean()
        loss.backward()

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertIsNotNone(model.regression_head[-1].weight.grad)

    def test_rejects_unexpected_point_count(self) -> None:
        """Reject batches that violate the fixed-point benchmark protocol."""
        model = DGCNN(k=8, num_points=64).eval()
        batch = Batch.from_data_list([Data(pos=torch.rand(63, 3))])

        with self.assertRaisesRegex(ValueError, "requires 64 points"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
