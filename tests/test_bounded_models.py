import unittest

import torch
from torch_geometric.data import Data

from benchmarks.model_comparison.models import PointTransformer, RandLANet


class BoundedModelCompatibilityTest(unittest.TestCase):
    """Validate the minimal PTv1 and RandLA-Net interface fixes."""

    def test_point_transformer_supports_cpu_without_features(self) -> None:
        """Create dummy features on the input tensor's device."""
        data = Data(
            pos=torch.rand(64, 3),
            batch=torch.zeros(64, dtype=torch.long),
        )
        model = PointTransformer(dim_model=[8, 16], k=4).eval()

        self.assertEqual(tuple(model(data).shape), (1,))

    def test_randlanet_derives_missing_batch_pointer(self) -> None:
        """Evaluate a manually constructed test sample without ``ptr``."""
        data = Data(
            pos=torch.rand(64, 3),
            batch=torch.zeros(64, dtype=torch.long),
        )
        model = RandLANet(num_neighbors=4).eval()

        self.assertEqual(tuple(model(data).shape), (1,))


if __name__ == "__main__":
    unittest.main()
