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

    def test_three_stage_configuration_identity(self) -> None:
        """Retain the first three official encoder stages and settings."""
        from adapter import OfficialPTv3ThreeStageRegressor

        model = OfficialPTv3ThreeStageRegressor()

        self.assertEqual(model.variant, "ptv3-official-3stage-flash-regression")
        self.assertEqual(model.encoder_config["stride"], (2, 2))
        self.assertEqual(model.encoder_config["enc_depths"], (2, 2, 2))
        self.assertEqual(model.encoder_config["enc_channels"], (32, 64, 128))
        self.assertEqual(model.encoder_config["enc_num_head"], (2, 4, 8))
        self.assertEqual(model.encoder_config["enc_patch_size"], (1024,) * 3)
        self.assertEqual(len(model.backbone.enc), 3)
        self.assertEqual(model.flash_attention_module_count, 6)

    def test_three_stage_forward_and_backward(self) -> None:
        """Produce finite gradients from the official-style three-stage encoder."""
        from adapter import OfficialPTv3ThreeStageRegressor

        device = torch.device("cuda")
        model = OfficialPTv3ThreeStageRegressor().to(device).train()
        coordinates = torch.rand(128, 3, device=device) * 0.08
        batch = torch.arange(2, device=device).repeat_interleave(64)

        prediction = model(coordinates, batch, 0.002)
        prediction.sum().backward()

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(model.regression_head[-1].weight.grad)

if __name__ == "__main__":
    unittest.main()
