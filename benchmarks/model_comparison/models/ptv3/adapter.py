"""Point Transformer V3 adapter for point-cloud scalar regression."""

from functools import wraps
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_mean


def enable_fp16_flash_attention(backbone: nn.Module) -> int:
    """Run only serialized attention kernels under CUDA FP16 autocast.

    Parameters
    ----------
    backbone
        Upstream PTv3 backbone whose serialized attention modules are wrapped.

    Returns
    -------
    int
        Number of wrapped attention modules.

    Notes
    -----
    The pinned spconv build does not provide suitable FP16 kernels for every
    sparse CPE shape, so whole-model autocast is not viable. FlashAttention
    itself requires FP16 or BF16 inputs.
    """
    from pointtransformerv3.model import SerializedAttention

    wrapped_modules = 0
    for module in backbone.modules():
        if not isinstance(module, SerializedAttention):
            continue
        original_forward = module.forward

        @wraps(original_forward)
        def forward_with_autocast(*args: Any, _forward=original_forward, **kwargs: Any):
            point = args[0]
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=point.feat.is_cuda,
            ):
                return _forward(*args, **kwargs)

        module.forward = forward_with_autocast
        wrapped_modules += 1
    return wrapped_modules


class PTv3Regressor(nn.Module):
    """Pool a compact PTv3 encoder into one scalar per point cloud.

    Notes
    -----
    The pinned upstream package must be prepared and added to ``sys.path``
    before this class is instantiated.
    """

    variant = "ptv3-compact-3stage"
    output_channels = 128

    def __init__(self) -> None:
        super().__init__()
        from pointtransformerv3.model import PointTransformerV3

        self.backbone = PointTransformerV3(
            in_channels=3,
            order=("z",),
            stride=(2, 2),
            enc_depths=(2, 2, 2),
            enc_channels=(32, 64, self.output_channels),
            enc_num_head=(2, 4, 8),
            enc_patch_size=(64, 64, 64),
            cls_mode=True,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            shuffle_orders=False,
            drop_path=0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.output_channels, 64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        batch: torch.Tensor,
        grid_size: float,
    ) -> torch.Tensor:
        """Predict one scalar for each cloud in a concatenated point batch.

        Parameters
        ----------
        coordinates
            Centered XYZ coordinates with shape ``[points, 3]``.
        batch
            Cloud index for every point with shape ``[points]``.
        grid_size
            Voxel size used by PTv3 serialization.

        Returns
        -------
        torch.Tensor
            Scalar predictions with shape ``[clouds]``.
        """
        point: Any = self.backbone(
            {
                "feat": coordinates,
                "coord": coordinates,
                "grid_size": grid_size,
                "batch": batch,
            }
        )
        pooled = scatter_mean(point.feat, point.batch, dim=0)
        return self.regression_head(pooled).squeeze(-1)


class OfficialPTv3ThreeStageRegressor(nn.Module):
    """Adapt the first three official PTv3 stages to object regression.

    Notes
    -----
    This object-scale variant retains the official serialization, FlashAttention,
    patch, shuffle, and drop-path settings. It removes stages four and five,
    which over-compress a 1,024-point tuber cloud, and uses a scalar task head.
    """

    variant = "ptv3-official-3stage-flash-regression"
    uses_flash_attention = True
    output_channels = 128
    encoder_config = {
        "in_channels": 3,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2),
        "enc_depths": (2, 2, 2),
        "enc_channels": (32, 64, 128),
        "enc_num_head": (2, 4, 8),
        "enc_patch_size": (1024, 1024, 1024),
        "drop_path": 0.3,
        "shuffle_orders": True,
        "cls_mode": True,
        "enable_flash": True,
        "upcast_attention": False,
        "upcast_softmax": False,
    }

    def __init__(self) -> None:
        super().__init__()
        from pointtransformerv3.model import PointTransformerV3

        self.backbone = PointTransformerV3(**self.encoder_config)
        self.flash_attention_module_count = enable_fp16_flash_attention(
            self.backbone
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.output_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        batch: torch.Tensor,
        grid_size: float,
    ) -> torch.Tensor:
        """Predict one scalar per cloud from centered XYZ coordinates.

        Parameters
        ----------
        coordinates
            Centered XYZ coordinates with shape ``[points, 3]``.
        batch
            Cloud index for every point with shape ``[points]``.
        grid_size
            Voxel size used by PTv3 serialization.

        Returns
        -------
        torch.Tensor
            Scalar predictions with shape ``[clouds]``.
        """
        point: Any = self.backbone(
            {
                "feat": coordinates,
                "coord": coordinates,
                "grid_size": grid_size,
                "batch": batch,
            }
        )
        pooled = scatter_mean(point.feat, point.batch, dim=0)
        return self.regression_head(pooled).squeeze(-1)
