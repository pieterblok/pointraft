"""Official-style PointMLP adapted to point-cloud scalar regression.

The architecture follows the full PointMLP ModelNet40 implementation:
https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ModelNet40/models/pointmlp.py
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch_geometric.nn import fps, knn


class ConvBNReLU1D(nn.Sequential):
    """Shared pointwise convolution used by PointMLP.

    Parameters
    ----------
    in_channels
        Input feature channels.
    out_channels
        Output feature channels.
    bias
        Whether the convolution includes a bias term.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvBNReLURes1D(nn.Module):
    """Two-layer residual MLP block from the official PointMLP.

    Parameters
    ----------
    channels
        Input and output feature channels.
    expansion
        Hidden-channel expansion ratio.
    bias
        Whether convolutions include bias terms.
    """

    def __init__(self, channels: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the residual pointwise MLP.

        Parameters
        ----------
        features
            Features with shape ``[batch, channels, points]``.

        Returns
        -------
        torch.Tensor
            Features with the same shape as the input.
        """
        return self.activation(self.net(features) + features)


class LocalGrouper(nn.Module):
    """Group local neighborhoods and apply geometric affine normalization.

    Parameters
    ----------
    channels
        Input feature channels.
    groups
        Number of farthest-point anchors.
    neighbors
        Number of nearest neighbors per anchor.
    """

    def __init__(self, channels: int, groups: int, neighbors: int) -> None:
        super().__init__()
        self.groups = groups
        self.neighbors = neighbors
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample anchors and group normalized neighbor features.

        Parameters
        ----------
        xyz
            Coordinates with shape ``[batch, points, 3]``.
        point_features
            Features with shape ``[batch, points, channels]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Anchor coordinates ``[batch, groups, 3]`` and grouped features
            ``[batch, groups, neighbors, 2 * channels]``.
        """
        batch_size, num_points, _ = xyz.shape
        flat_xyz = xyz.reshape(-1, 3)
        flat_features = point_features.reshape(-1, point_features.size(-1))
        point_batch = torch.arange(batch_size, device=xyz.device).repeat_interleave(
            num_points
        )

        anchor_indices = fps(
            flat_xyz,
            batch=point_batch,
            ratio=self.groups / num_points,
            random_start=False,
        )
        expected_anchors = batch_size * self.groups
        if anchor_indices.numel() != expected_anchors:
            raise RuntimeError(
                f"PointMLP expected {expected_anchors} FPS anchors; "
                f"received {anchor_indices.numel()}"
            )

        anchor_xyz = flat_xyz[anchor_indices].view(batch_size, self.groups, 3)
        anchor_features = flat_features[anchor_indices].view(
            batch_size, self.groups, -1
        )
        anchor_batch = torch.arange(batch_size, device=xyz.device).repeat_interleave(
            self.groups
        )
        edge_index = knn(
            flat_xyz,
            anchor_xyz.reshape(-1, 3),
            k=self.neighbors,
            batch_x=point_batch,
            batch_y=anchor_batch,
        )
        order = torch.argsort(edge_index[0], stable=True)
        neighbor_indices = edge_index[1][order].view(
            batch_size, self.groups, self.neighbors
        )
        grouped_features = flat_features[neighbor_indices]

        anchor_mean = anchor_features.unsqueeze(2)
        scale = torch.std(
            (grouped_features - anchor_mean).reshape(batch_size, -1),
            dim=-1,
            keepdim=True,
        ).view(batch_size, 1, 1, 1)
        grouped_features = (grouped_features - anchor_mean) / (scale + 1e-5)
        grouped_features = self.affine_alpha * grouped_features + self.affine_beta
        repeated_anchors = anchor_mean.expand(-1, -1, self.neighbors, -1)
        return anchor_xyz, torch.cat((grouped_features, repeated_anchors), dim=-1)


class PreExtraction(nn.Module):
    """Extract and pool features inside each PointMLP neighborhood.

    Parameters
    ----------
    in_channels
        Channels before the transfer projection.
    out_channels
        Channels after the transfer projection.
    blocks
        Number of residual MLP blocks.
    expansion
        Hidden-channel expansion in each residual block.
    bias
        Whether convolutions include bias terms.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        expansion: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias)
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(out_channels, expansion=expansion, bias=bias)
                for _ in range(blocks)
            ]
        )

    def forward(self, grouped_features: torch.Tensor) -> torch.Tensor:
        """Project, refine, and max-pool each local neighborhood.

        Parameters
        ----------
        grouped_features
            Features with shape ``[batch, groups, neighbors, channels]``.

        Returns
        -------
        torch.Tensor
            Pooled features with shape ``[batch, channels, groups]``.
        """
        batch_size, groups, neighbors, channels = grouped_features.shape
        features = grouped_features.permute(0, 1, 3, 2).reshape(
            batch_size * groups, channels, neighbors
        )
        features = self.operation(self.transfer(features))
        features = features.max(dim=-1).values
        return features.view(batch_size, groups, -1).permute(0, 2, 1)


class PosExtraction(nn.Sequential):
    """Refine pooled anchor features with residual pointwise MLP blocks.

    Parameters
    ----------
    channels
        Input and output feature channels.
    blocks
        Number of residual MLP blocks.
    expansion
        Hidden-channel expansion in each residual block.
    bias
        Whether convolutions include bias terms.
    """

    def __init__(
        self,
        channels: int,
        blocks: int,
        expansion: float,
        bias: bool,
    ) -> None:
        super().__init__(
            *[
                ConvBNReLURes1D(channels, expansion=expansion, bias=bias)
                for _ in range(blocks)
            ]
        )


class PointMLP(nn.Module):
    """Full official PointMLP backbone with a scalar regression head.

    Parameters
    ----------
    num_points
        Required number of points per graph.
    embed_dim
        Width of the initial point embedding.
    dim_expansion
        Per-stage channel multipliers.
    pre_blocks
        Residual block counts before neighborhood pooling.
    pos_blocks
        Residual block counts after neighborhood pooling.
    k_neighbors
        Per-stage kNN neighborhood sizes.
    reducers
        Per-stage point-count reduction factors.
    residual_expansion
        Hidden-channel expansion in residual blocks.
    bias
        Whether PointMLP convolutions include bias terms.
    """

    def __init__(
        self,
        num_points: int = 1024,
        embed_dim: int = 64,
        dim_expansion: Sequence[int] = (2, 2, 2, 2),
        pre_blocks: Sequence[int] = (2, 2, 2, 2),
        pos_blocks: Sequence[int] = (2, 2, 2, 2),
        k_neighbors: Sequence[int] = (24, 24, 24, 24),
        reducers: Sequence[int] = (2, 2, 2, 2),
        residual_expansion: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        stage_counts = {
            len(dim_expansion),
            len(pre_blocks),
            len(pos_blocks),
            len(k_neighbors),
            len(reducers),
        }
        if len(stage_counts) != 1:
            raise ValueError("PointMLP stage configurations must have equal lengths")

        self.num_points = num_points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias)
        self.local_groupers = nn.ModuleList()
        self.pre_extraction = nn.ModuleList()
        self.pos_extraction = nn.ModuleList()

        anchor_points = num_points
        channels = embed_dim
        for expansion, pre_count, pos_count, neighbors, reducer in zip(
            dim_expansion,
            pre_blocks,
            pos_blocks,
            k_neighbors,
            reducers,
            strict=True,
        ):
            anchor_points //= reducer
            out_channels = channels * expansion
            self.local_groupers.append(
                LocalGrouper(channels, groups=anchor_points, neighbors=neighbors)
            )
            self.pre_extraction.append(
                PreExtraction(
                    channels * 2,
                    out_channels,
                    blocks=pre_count,
                    expansion=residual_expansion,
                    bias=bias,
                )
            )
            self.pos_extraction.append(
                PosExtraction(
                    out_channels,
                    blocks=pos_count,
                    expansion=residual_expansion,
                    bias=bias,
                )
            )
            channels = out_channels

        self.regression_head = nn.Sequential(
            nn.Linear(channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Predict one scalar for each point cloud in a PyG batch.

        Parameters
        ----------
        data
            PyG data object containing ``pos`` and ``batch`` tensors.

        Returns
        -------
        torch.Tensor
            Scalar predictions with shape ``[batch]``.

        Raises
        ------
        ValueError
            If a graph does not contain exactly ``num_points`` points.
        """
        batch_size = int(data.batch.max().item()) + 1
        point_counts = torch.bincount(data.batch, minlength=batch_size)
        if not torch.all(point_counts == self.num_points):
            raise ValueError(
                f"PointMLP requires {self.num_points} points per graph; "
                f"received {point_counts.tolist()}"
            )

        xyz = data.pos.view(batch_size, self.num_points, 3)
        features = self.embedding(xyz.transpose(2, 1))
        for grouper, pre_extraction, pos_extraction in zip(
            self.local_groupers,
            self.pre_extraction,
            self.pos_extraction,
            strict=True,
        ):
            xyz, grouped_features = grouper(xyz, features.transpose(2, 1))
            features = pos_extraction(pre_extraction(grouped_features))

        global_features = features.max(dim=-1).values
        return self.regression_head(global_features).squeeze(-1)
