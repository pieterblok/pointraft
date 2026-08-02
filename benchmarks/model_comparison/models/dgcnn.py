"""Author-style DGCNN adapted from classification to scalar regression.

The architecture follows Wang et al.'s official ModelNet40 implementation:
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

import torch
from torch import nn


def _knn(features: torch.Tensor, k: int) -> torch.Tensor:
    """Find feature-space nearest neighbors using squared distances.

    Parameters
    ----------
    features
        Point features with shape ``[batch, channels, points]``.
    k
        Number of neighbors, including the query point itself.

    Returns
    -------
    torch.Tensor
        Neighbor indices with shape ``[batch, points, k]``.
    """
    inner = -2 * torch.matmul(features.transpose(2, 1), features)
    squared_norm = torch.sum(features**2, dim=1, keepdim=True)
    pairwise_distance = -squared_norm - inner - squared_norm.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1).indices


def _graph_features(features: torch.Tensor, k: int) -> torch.Tensor:
    """Construct EdgeConv features from a dynamically recomputed graph.

    Parameters
    ----------
    features
        Point features with shape ``[batch, channels, points]``.
    k
        Number of feature-space neighbors.

    Returns
    -------
    torch.Tensor
        Concatenated neighbor offsets and center features with shape
        ``[batch, 2 * channels, points, k]``.
    """
    batch_size, channels, num_points = features.shape
    neighbor_indices = _knn(features, k=k)
    batch_offsets = (
        torch.arange(batch_size, device=features.device).view(-1, 1, 1) * num_points
    )
    neighbor_indices = (neighbor_indices + batch_offsets).reshape(-1)

    point_features = features.transpose(2, 1).contiguous()
    neighbors = point_features.reshape(batch_size * num_points, channels)[
        neighbor_indices
    ]
    neighbors = neighbors.view(batch_size, num_points, k, channels)
    centers = point_features.view(batch_size, num_points, 1, channels).expand(
        -1, -1, k, -1
    )
    return torch.cat((neighbors - centers, centers), dim=3).permute(0, 3, 1, 2)


class DGCNN(nn.Module):
    """Four-stage dynamic graph CNN for point-cloud scalar regression.

    Parameters
    ----------
    k
        Number of nearest neighbors used by every EdgeConv stage.
    embedding_channels
        Width of the global point-feature embedding.
    dropout
        Dropout probability in the regression head.
    num_points
        Required number of points per graph.
    """

    def __init__(
        self,
        k: int = 20,
        embedding_channels: int = 1024,
        dropout: float = 0.5,
        num_points: int = 1024,
    ) -> None:
        super().__init__()
        self.k = k
        self.num_points = num_points

        self.edge_conv1 = self._edge_conv(6, 64)
        self.edge_conv2 = self._edge_conv(128, 64)
        self.edge_conv3 = self._edge_conv(128, 128)
        self.edge_conv4 = self._edge_conv(256, 256)
        self.embedding = nn.Sequential(
            nn.Conv1d(512, embedding_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_channels * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _edge_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build one official-style EdgeConv projection block.

        Parameters
        ----------
        in_channels
            Number of concatenated edge-feature channels.
        out_channels
            Number of projected point-feature channels.

        Returns
        -------
        torch.nn.Sequential
            Shared 1x1 convolution, batch normalization, and LeakyReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
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
                f"DGCNN requires {self.num_points} points per graph; "
                f"received {point_counts.tolist()}"
            )

        features = data.pos.view(batch_size, self.num_points, 3).transpose(2, 1)

        edge_features = self.edge_conv1(_graph_features(features, self.k))
        x1 = edge_features.max(dim=-1).values

        edge_features = self.edge_conv2(_graph_features(x1, self.k))
        x2 = edge_features.max(dim=-1).values

        edge_features = self.edge_conv3(_graph_features(x2, self.k))
        x3 = edge_features.max(dim=-1).values

        edge_features = self.edge_conv4(_graph_features(x3, self.k))
        x4 = edge_features.max(dim=-1).values

        embedded = self.embedding(torch.cat((x1, x2, x3, x4), dim=1))
        max_features = embedded.max(dim=-1).values
        mean_features = embedded.mean(dim=-1)
        prediction = self.regression_head(
            torch.cat((max_features, mean_features), dim=1)
        )
        return prediction.squeeze(-1)
