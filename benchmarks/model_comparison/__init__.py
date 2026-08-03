"""Model-comparison training, evaluation, and orchestration."""

DEFAULT_BENCHMARK_MODELS = ("pointnet", "pointnet2")
PAPER_BENCHMARK_MODELS = (
    "dgcnn",
    "pointmlp",
    "pointtransformer",
    "randlanet",
)
ISOLATED_BENCHMARK_MODELS = (
    "ptv3",
    "ptv3_official_3stage",
)
BENCHMARK_MODELS = (
    DEFAULT_BENCHMARK_MODELS + PAPER_BENCHMARK_MODELS + ISOLATED_BENCHMARK_MODELS
)

__all__ = [
    "BENCHMARK_MODELS",
    "DEFAULT_BENCHMARK_MODELS",
    "ISOLATED_BENCHMARK_MODELS",
    "PAPER_BENCHMARK_MODELS",
]
