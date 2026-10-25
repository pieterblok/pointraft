"""Registry of PointRAFT ablations, in the order they must be run.

The first eight ablations reuse the existing PointCloudDataset "processed"
folder. The last three change the cached data, so each one deletes that folder
before training and must therefore run after the others.
"""

from .config import AblationConfig
from .height_emb_2 import ABLATION as HEIGHT_EMB_2
from .height_emb_4 import ABLATION as HEIGHT_EMB_4
from .height_emb_16 import ABLATION as HEIGHT_EMB_16
from .height_emb_32 import ABLATION as HEIGHT_EMB_32
from .height_emb_48 import ABLATION as HEIGHT_EMB_48
from .height_emb_64 import ABLATION as HEIGHT_EMB_64
from .l1 import ABLATION as L1
from .mse import ABLATION as MSE
from .no_augmentation import ABLATION as NO_AUGMENTATION
from .points_512 import ABLATION as POINTS_512
from .points_2048 import ABLATION as POINTS_2048

ABLATION_REGISTRY: dict[str, AblationConfig] = {
    ablation.name: ablation
    for ablation in (
        MSE,
        L1,
        HEIGHT_EMB_2,
        HEIGHT_EMB_4,
        HEIGHT_EMB_16,
        HEIGHT_EMB_32,
        HEIGHT_EMB_48,
        HEIGHT_EMB_64,
        NO_AUGMENTATION,
        POINTS_512,
        POINTS_2048,
    )
}

__all__ = ["ABLATION_REGISTRY", "AblationConfig"]
