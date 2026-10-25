"""Ablation 8: wider height-embedding MLP [1, 64, 64] instead of [1, 8, 8]."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="height_emb_64",
    description="Height-embedding MLP [1, 64, 64] instead of [1, 8, 8]",
    height_emb_dim=64,
)
