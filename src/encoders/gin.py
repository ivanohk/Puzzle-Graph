"""GIN encoder: stack of GINConv layers with residuals, pre-norm, and final projection."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_add_pool

from src.registry import ENCODERS


class GINLayer(nn.Module):
    """GINConv with pre-norm BatchNorm, ReLU, dropout, and residual connection."""

    def __init__(self, dim: int, mlp_ratio: int = 2, drop: float = 0.2):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # No BN inside the MLP (matches gml-test)
        mlp = Sequential(
            Linear(dim, hidden_dim),
            ReLU(),
            nn.Dropout(drop),
            Linear(hidden_dim, dim),
        )

        self.norm = BN(dim)
        self.conv = GINConv(nn=mlp)
        self.activation = ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index):
        h = self.norm(x)  # pre-norm before conv
        h = self.conv(h, edge_index)
        h = self.activation(h)
        h = self.drop(h)
        return x + h

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        self.norm.reset_parameters()


@ENCODERS.register("gin")
class GINEncoder(nn.Module):
    """GIN backbone: embedding → GINLayer stack → linear projection head.

    Architecture:
        lin0(in_channels → hidden_dim)
        → num_layers × GINLayer(hidden_dim) with pre-norm and residuals
        → lin1(hidden_dim → hidden_dim) → ReLU → Dropout
        → lin2(hidden_dim → out_dim)

    Args:
        in_channels: Input node feature dimension.
        hidden_dim: Hidden dimension used throughout the GINLayer stack.
        out_dim: Final output dimension (defaults to hidden_dim).
        num_layers: Number of GINLayer blocks.
        mlp_ratio: Width multiplier for the inner MLP of each GINConv.
        drop: Dropout rate used in GINLayer and the final projection.
        pool: If True, apply global_add_pool to produce graph-level embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int | None = None,
        num_layers: int = 3,
        mlp_ratio: int = 2,
        drop: float = 0.2,
        pool: bool = False,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        self.pool = pool
        self._drop = drop

        self.lin0 = Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(num_layers)
        ])
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.activation = ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.lin0(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.activation(self.lin1(x))
        x = F.dropout(x, p=self._drop, training=self.training)
        x = self.lin2(x)

        return x

    def reset_parameters(self) -> None:
        self.lin0.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
