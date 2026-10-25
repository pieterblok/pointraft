"""Ablation 1: replace the Smooth L1 loss with the MSE loss."""

from torch import nn

from .config import AblationConfig

ABLATION = AblationConfig(
    name="mse",
    description="MSELoss instead of SmoothL1Loss(beta=20)",
    loss=nn.MSELoss,
    loss_name="MSELoss",
)
