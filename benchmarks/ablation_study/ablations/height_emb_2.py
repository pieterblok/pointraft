"""Ablation 3: narrower height-embedding MLP [1, 2, 2] instead of [1, 8, 8]."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="height_emb_2",
    description="Height-embedding MLP [1, 2, 2] instead of [1, 8, 8]",
    height_emb_dim=2,
)
