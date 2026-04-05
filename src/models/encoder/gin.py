import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_add_pool

from src.registry.registry import ENCODERS


class GINLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop=0.2):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        mlp = Sequential(
            Linear(dim, hidden_dim),
            BN(hidden_dim),
            ReLU(),
            nn.Dropout(drop),
            Linear(hidden_dim, dim),
        )

        self.norm = BN(dim)
        self.conv = GINConv(nn=mlp)
        self.activation = ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = self.activation(h)
        h = self.drop(h)
        return x + h


@ENCODERS.register("gin")
class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, mlp_ratio=2, drop=0.2, pool=True):
        super().__init__()
        self.pool = pool

        self.lin0 = Linear(in_channels, hidden_dim)

        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch=None):
        x = self.lin0(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        if self.pool:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_add_pool(x, batch)

        return x
