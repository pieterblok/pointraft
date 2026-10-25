import os
import random

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed supported random generators and request deterministic kernels.

    Parameters
    ----------
    seed
        Seed shared by Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a DataLoader worker.

    Parameters
    ----------
    worker_id
        Worker identifier supplied by PyTorch. The worker-specific PyTorch seed
        is used indirectly through :func:`torch.initial_seed`.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int = RANDOM_SEED) -> torch.Generator:
    """Create a deterministically seeded PyTorch generator.

    Parameters
    ----------
    seed
        Seed assigned to the generator.

    Returns
    -------
    torch.Generator
        Seeded CPU generator for samplers or data loaders.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
