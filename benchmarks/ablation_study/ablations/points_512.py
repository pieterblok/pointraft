"""Ablation 10: train and test with 512 input points instead of 1024."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="points_512",
    description="512 input points instead of 1024",
    num_points=512,
    rebuild_processed=True,
)
