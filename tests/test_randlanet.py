import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn_graph

from benchmarks.model_comparison.models import BENCHMARK_MODEL_REGISTRY
from benchmarks.model_comparison.models.randlanet import (
    LocalFeatureAggregation,
    RandLANet,
)


class RandLANetTest(unittest.TestCase):
    """Validate the official-configuration RandLA-Net regression encoder."""

    @staticmethod
    def make_data(clouds: int, points: int) -> Data:
        """Create a deterministic concatenated synthetic point-cloud batch."""
        generator = torch.Generator().manual_seed(42)
        return Data(
            pos=torch.rand(clouds * points, 3, generator=generator),
            batch=torch.arange(clouds).repeat_interleave(points),
        )

    def test_official_configuration_and_hierarchy(self) -> None:
        """Retain all four official widths and 4x point reductions."""
        model = RandLANet().eval()
        data = self.make_data(clouds=2, points=1024)

        with torch.inference_mode():
            _, _, hierarchy = model.encode(data, return_hierarchy=True)

        self.assertEqual(model.d_out, (16, 64, 128, 256))
        self.assertEqual(model.block_output_channels, (32, 128, 256, 512))
        self.assertEqual(model.subsampling_ratios, (4, 4, 4, 4))
        self.assertEqual(model.num_neighbors, 16)
        self.assertEqual(
            [(level[0].size(0) // 2, level[0].size(1)) for level in hierarchy],
            [(256, 32), (64, 128), (16, 256), (4, 512)],
        )
        batch_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm1d)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(module.momentum == 0.01 for module in batch_norms))
        self.assertTrue(all(module.eps == 1e-6 for module in batch_norms))

    def test_canonical_registry_uses_official_configuration(self) -> None:
        """Make the official four-stage model the sole RandLA-Net entry."""
        self.assertIs(BENCHMARK_MODEL_REGISTRY["randlanet"], RandLANet)
        self.assertNotIn("randlanet_official", BENCHMARK_MODEL_REGISTRY)

    def test_local_feature_aggregation_shape_and_gradient(self) -> None:
        """Use ten spatial channels and two differentiable attentive pools."""
        positions = torch.rand(32, 3)
        features = torch.rand(32, 8, requires_grad=True)
        edge_index = knn_graph(positions, k=4, loop=True)
        aggregation = LocalFeatureAggregation(d_out=16).train()

        output = aggregation(features, positions, edge_index)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (32, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_relative_position_encoding_order(self) -> None:
        """Encode distance, center-minus-neighbor, center, then neighbor XYZ."""
        positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 3.0]])

        encoded = LocalFeatureAggregation.relative_position_encoding(
            positions,
            neighbor_index=torch.tensor([1]),
            center_index=torch.tensor([0]),
        )

        expected = torch.tensor([[5.0, -3.0, -4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 3.0]])
        self.assertTrue(torch.equal(encoded, expected))

    def test_scalar_output_finite_gradient_and_repeatability(self) -> None:
        """Produce repeatable scalar outputs with a functioning backward path."""
        data = self.make_data(clouds=2, points=64)
        model = RandLANet(num_neighbors=4).eval()

        first = model(data)
        second = model(data)
        first.sum().backward()

        self.assertEqual(tuple(first.shape), (2,))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))
        self.assertIsNotNone(model.regression_head[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
