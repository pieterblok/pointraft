"""Ablation 6: wider height-embedding MLP [1, 32, 32] instead of [1, 8, 8]."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="height_emb_32",
    description="Height-embedding MLP [1, 32, 32] instead of [1, 8, 8]",
    height_emb_dim=32,
)
