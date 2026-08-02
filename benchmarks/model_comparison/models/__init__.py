"""Scalar-regression models used only by the benchmark suite."""

from .dgcnn import DGCNN
from .pointmlp import PointMLP
from .pointnet import PointNet
from .pointnet2 import PointNet2
from .pointtransformer import PointTransformer
from .randlanet import RandLANet

BENCHMARK_MODEL_REGISTRY = {
    "dgcnn": DGCNN,
    "pointmlp": PointMLP,
    "pointnet": PointNet,
    "pointnet2": PointNet2,
    "pointtransformer": PointTransformer,
    "randlanet": RandLANet,
}

__all__ = [
    "BENCHMARK_MODEL_REGISTRY",
    "DGCNN",
    "PointMLP",
    "PointNet",
    "PointNet2",
    "PointTransformer",
    "RandLANet",
]
