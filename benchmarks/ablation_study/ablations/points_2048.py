"""Ablation 11: train and test with 2048 input points instead of 1024."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="points_2048",
    description="2048 input points instead of 1024",
    num_points=2048,
    # 2048 points at batch size 32 does not fit in 16 GB of GPU memory.
    batch_size=16,
    rebuild_processed=True,
)
