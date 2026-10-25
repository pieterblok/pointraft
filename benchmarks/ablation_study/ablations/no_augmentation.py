"""Ablation 9: train without data augmentation."""

from .config import AblationConfig

ABLATION = AblationConfig(
    name="no_augmentation",
    description="No jitter/rotate/flip/shear augmentation",
    augmentation=False,
    rebuild_processed=True,
)
