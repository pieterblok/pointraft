"""Ablation 4: narrower height-embedding MLP [1, 4, 4] instead of [1, 8, 8]."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="height_emb_4",
    description="Height-embedding MLP [1, 4, 4] instead of [1, 8, 8]",
    height_emb_dim=4,
)
