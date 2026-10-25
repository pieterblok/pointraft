"""Shared configuration for PointRAFT ablations."""

from dataclasses import dataclass
from typing import Callable

from torch import nn


def smooth_l1_loss() -> nn.Module:
    """Return the default PointRAFT training loss."""
    return nn.SmoothL1Loss(beta=20)


@dataclass(frozen=True)
class AblationConfig:
    """One ablation: every field defaults to the PointRAFT training protocol."""

    name: str
    description: str
    loss: Callable[[], nn.Module] = smooth_l1_loss
    loss_name: str = "SmoothL1Loss(beta=20)"
    height_emb_dim: int = 8
    num_points: int = 1024
    batch_size: int = 32
    augmentation: bool = True
    # Delete the PointCloudDataset "processed" folder before training, so the
    # cached train/val tensors are rebuilt with this ablation's settings.
    rebuild_processed: bool = False

    @property
    def model_kwargs(self) -> dict:
        """Keyword arguments for PointRAFT(...)."""
        return {
            "height_emb_dim": self.height_emb_dim
        }
