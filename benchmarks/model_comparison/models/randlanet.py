"""Official-configuration RandLA-Net encoder adapted to scalar regression.

This is an independent PyTorch/PyG implementation based on the RandLA-Net
paper and its published SemanticKITTI configuration. It does not copy the
official TensorFlow implementation.
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.pool import knn, knn_graph
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import softmax

if not WITH_TORCH_CLUSTER:
    raise ImportError("RandLA-Net requires 'torch-cluster'")


class SharedMLP(nn.Module):
    """Apply the RandLA-Net linear, batch-normalization, and activation order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels, momentum=0.01, eps=1e-6)
        self.activation = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, features: Tensor) -> Tensor:
        """Transform a point or edge feature matrix.

        Parameters
        ----------
        features
            Feature matrix with shape ``[items, channels]``.

        Returns
        -------
        Tensor
            Transformed feature matrix.
        """
        return self.activation(self.batch_norm(self.linear(features)))


class AttentivePooling(nn.Module):
    """Aggregate neighboring features with channel-wise learned attention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = nn.Linear(in_channels, in_channels, bias=False)
        self.output_mlp = SharedMLP(in_channels, out_channels)

    def forward(
        self,
        neighbor_features: Tensor,
        center_index: Tensor,
        num_centers: int,
    ) -> Tensor:
        """Pool edge features onto their center points.

        Parameters
        ----------
        neighbor_features
            Edge feature matrix with shape ``[edges, channels]``.
        center_index
            Target point index for every edge.
        num_centers
            Number of target points.

        Returns
        -------
        Tensor
            Pooled center features with shape ``[num_centers, out_channels]``.
        """
        scores = softmax(
            self.attention(neighbor_features),
            index=center_index,
            num_nodes=num_centers,
        )
        pooled = neighbor_features.new_zeros(num_centers, neighbor_features.size(1))
        pooled.index_add_(0, center_index, scores * neighbor_features)
        return self.output_mlp(pooled)


class LocalFeatureAggregation(nn.Module):
    """Apply two official RandLA-Net local spatial encodings and pools."""

    relative_position_channels = 10

    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.d_out = d_out
        self.position_mlp1 = SharedMLP(self.relative_position_channels, d_out // 2)
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.position_mlp2 = SharedMLP(d_out // 2, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

    @staticmethod
    def relative_position_encoding(
        positions: Tensor,
        neighbor_index: Tensor,
        center_index: Tensor,
    ) -> Tensor:
        """Build distance, relative XYZ, center XYZ, and neighbor XYZ features."""
        center = positions[center_index]
        neighbor = positions[neighbor_index]
        relative = center - neighbor
        distance = torch.linalg.vector_norm(relative, dim=1, keepdim=True)
        return torch.cat((distance, relative, center, neighbor), dim=1)

    def forward(self, features: Tensor, positions: Tensor, edge_index: Tensor) -> Tensor:
        """Aggregate local features around every center point.

        Parameters
        ----------
        features
            Reduced point features with shape ``[points, d_out // 2]``.
        positions
            XYZ coordinates with shape ``[points, 3]``.
        edge_index
            Source-to-target KNN edges with shape ``[2, edges]``.

        Returns
        -------
        Tensor
            Aggregated point features with shape ``[points, d_out]``.
        """
        neighbor_index, center_index = edge_index
        relative = self.relative_position_encoding(
            positions, neighbor_index, center_index
        )
        spatial_features = self.position_mlp1(relative)
        first_neighbors = torch.cat(
            (features[neighbor_index], spatial_features), dim=1
        )
        first_pooled = self.pool1(first_neighbors, center_index, features.size(0))

        spatial_features = self.position_mlp2(spatial_features)
        second_neighbors = torch.cat(
            (first_pooled[neighbor_index], spatial_features), dim=1
        )
        return self.pool2(second_neighbors, center_index, features.size(0))


class DilatedResidualBlock(nn.Module):
    """Official RandLA-Net bottleneck, two-pool LFA, and residual projection."""

    def __init__(self, d_in: int, d_out: int, num_neighbors: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.output_channels = 2 * d_out
        self.num_neighbors = num_neighbors
        self.input_mlp = SharedMLP(d_in, d_out // 2)
        self.local_feature_aggregation = LocalFeatureAggregation(d_out)
        self.output_mlp = SharedMLP(d_out, self.output_channels, activation=False)
        self.shortcut = SharedMLP(d_in, self.output_channels, activation=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, features: Tensor, positions: Tensor, batch: Tensor) -> Tensor:
        """Encode one point hierarchy level.

        Parameters
        ----------
        features
            Input features with shape ``[points, d_in]``.
        positions
            XYZ coordinates with shape ``[points, 3]``.
        batch
            Cloud index for each point.

        Returns
        -------
        Tensor
            Encoded features with shape ``[points, 2 * d_out]``.
        """
        points_per_cloud = torch.bincount(batch)
        neighbors = min(self.num_neighbors, int(points_per_cloud.min().item()))
        edge_index = knn_graph(
            positions,
            k=neighbors,
            batch=batch,
            loop=True,
        )
        encoded = self.input_mlp(features)
        encoded = self.local_feature_aggregation(encoded, positions, edge_index)
        return self.activation(self.output_mlp(encoded) + self.shortcut(features))


class SeededRandomSubsampling(nn.Module):
    """Randomly select centers and max-pool their source neighborhoods."""

    def __init__(self, ratio: int, num_neighbors: int, seed: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.num_neighbors = num_neighbors
        self.seed = seed
        self.max_aggregation = MaxAggregation()

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Subsample every cloud and apply neighborhood max pooling.

        Parameters
        ----------
        features
            Source point features.
        positions
            Source XYZ coordinates.
        batch
            Source cloud indices.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Pooled features, selected coordinates, and selected cloud indices.
        """
        selected_indices = []
        for cloud_id in torch.unique(batch, sorted=True):
            cloud_indices = torch.nonzero(batch == cloud_id, as_tuple=False).flatten()
            output_count = max(cloud_indices.numel() // self.ratio, 1)
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + int(cloud_id.item())
            )
            permutation = torch.randperm(
                cloud_indices.numel(), generator=generator, device="cpu"
            )[:output_count].to(cloud_indices.device)
            selected_indices.append(cloud_indices[permutation])

        selected = torch.cat(selected_indices)
        sampled_positions = positions[selected]
        sampled_batch = batch[selected]
        points_per_cloud = torch.bincount(batch)
        neighbors = min(self.num_neighbors, int(points_per_cloud.min().item()))
        center_index, neighbor_index = knn(
            positions,
            sampled_positions,
            k=neighbors,
            batch_x=batch,
            batch_y=sampled_batch,
        )
        pooled = self.max_aggregation(
            features[neighbor_index],
            center_index,
            dim_size=sampled_positions.size(0),
        )
        return pooled, sampled_positions, sampled_batch


class RandLANet(nn.Module):
    """Four-stage SemanticKITTI-configuration encoder for object regression."""

    variant = "randlanet-official-config-4stage-regression"
    d_out = (16, 64, 128, 256)
    block_output_channels = (32, 128, 256, 512)
    subsampling_ratios = (4, 4, 4, 4)
    num_neighbors = 16
    sampling_seed = 42

    def __init__(
        self,
        *,
        num_neighbors: int = num_neighbors,
        sampling_seed: int = sampling_seed,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.sampling_seed = sampling_seed
        self.initial_mlp = SharedMLP(3, 8)

        input_channels = (8, *self.block_output_channels[:-1])
        self.encoder_blocks = nn.ModuleList(
            DilatedResidualBlock(d_in, d_out, num_neighbors)
            for d_in, d_out in zip(input_channels, self.d_out, strict=True)
        )
        self.subsamplers = nn.ModuleList(
            SeededRandomSubsampling(
                ratio,
                num_neighbors,
                sampling_seed + level * 10_000,
            )
            for level, ratio in enumerate(self.subsampling_ratios)
        )
        self.regression_head = nn.Sequential(
            SharedMLP(self.block_output_channels[-1], 64),
            SharedMLP(64, 32),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
        )

    def encode(
        self,
        data: Any,
        *,
        return_hierarchy: bool = False,
    ) -> tuple[Any, ...]:
        """Encode XYZ input through all four randomly sampled levels.

        Parameters
        ----------
        data
            PyG data object containing ``pos`` and optionally ``batch``.
        return_hierarchy
            Return intermediate feature/position/batch tensors for tests.

        Returns
        -------
        tuple
            Deep features and batch indices, optionally followed by all levels.
        """
        positions = data.pos
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = positions.new_zeros(positions.size(0), dtype=torch.long)
        features = self.initial_mlp(positions)
        hierarchy = []

        for block, subsampler in zip(
            self.encoder_blocks, self.subsamplers, strict=True
        ):
            features = block(features, positions, batch)
            features, positions, batch = subsampler(features, positions, batch)
            if return_hierarchy:
                hierarchy.append((features, positions, batch))

        if return_hierarchy:
            return features, batch, hierarchy
        return features, batch

    def forward(self, data: Any) -> Tensor:
        """Predict one scalar for each point cloud in a PyG batch."""
        features, batch = self.encode(data)
        pooled = global_max_pool(features, batch)
        return self.regression_head(pooled).squeeze(-1)
