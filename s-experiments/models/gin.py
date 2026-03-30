import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_add_pool


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


class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers, mlp_ratio=2, drop=0.2):
        super().__init__()
        self.drop = drop

        self.lin0 = Linear(in_channels, hidden_dim)

        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(num_layers)
        ])

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_channels)
        self.activation = ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.lin0(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        if batch is not None:
            x = global_add_pool(x, batch)

        else:
            x = x.sum(dim=0, keepdim=True)

        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin2(x)

        return x