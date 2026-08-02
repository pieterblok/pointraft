"""Point Transformer V3 adapter for point-cloud scalar regression."""

from typing import Any

import torch
from torch import nn
from torch_scatter import scatter_mean


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
