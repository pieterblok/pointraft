"""Ablation 7: wider height-embedding MLP [1, 48, 48] instead of [1, 8, 8]."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="height_emb_48",
    description="Height-embedding MLP [1, 48, 48] instead of [1, 8, 8]",
    height_emb_dim=48,
)
