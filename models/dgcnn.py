import torch
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.nn.models.mlp import MLP
from torch.nn import Linear

class DGCNN(torch.nn.Module):
    def __init__(self, k=20, aggr='max'):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)
        self.mlp = MLP([1024, 512, 256, 1], dropout=0.5, norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out