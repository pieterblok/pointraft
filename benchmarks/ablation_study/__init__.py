"""PointRAFT ablation-study training, evaluation, and orchestration."""

from .ablations import ABLATION_REGISTRY

# Registry order is the required run order (see ablations/__init__.py).
ABLATIONS = tuple(ABLATION_REGISTRY)

__all__ = [
    "ABLATIONS",
    "ABLATION_REGISTRY",
]
