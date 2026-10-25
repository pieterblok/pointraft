"""Model-comparison training, evaluation, and orchestration."""

# Models reported in the paper's comparison table; the default for train and test.
DEFAULT_BENCHMARK_MODELS = (
    "pointnet",
    "pointnet2",
    "dgcnn",
    "pointmlp",
    "randlanet",
    "ptv3_compact",
)
# Further comparison models that run only when selected with --models.
EXTRA_BENCHMARK_MODELS = (
    "pointtransformer",
    "ptv3",
)
BENCHMARK_MODELS = DEFAULT_BENCHMARK_MODELS + EXTRA_BENCHMARK_MODELS

__all__ = [
    "BENCHMARK_MODELS",
    "DEFAULT_BENCHMARK_MODELS",
    "EXTRA_BENCHMARK_MODELS",
]
