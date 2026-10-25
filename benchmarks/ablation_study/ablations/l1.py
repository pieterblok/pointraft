"""Ablation 2: replace the Smooth L1 loss with the L1 loss."""

from torch import nn

from .config import AblationConfig

ABLATION = AblationConfig(
    name="l1",
    description="L1Loss instead of SmoothL1Loss(beta=20)",
    loss=nn.L1Loss,
    loss_name="L1Loss",
)
