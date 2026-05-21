"""GCN encoder: 2-layer GCNConv with optional BN/LN, PReLU, weight standardization."""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool

from src.registry import ENCODERS
from src.nn.norm import apply_weight_standardization


@ENCODERS.register("gcn")
class GCNEncoder(nn.Module):
    """2-layer GCNConv backbone.

    Architecture per layer: GCNConv → (BatchNorm | LayerNorm) → PReLU.
    BatchNorm and LayerNorm are mutually exclusive.

    Args:
        in_channels: Input node feature dimension.
        hidden_dim: Hidden (and output) dimension for layer 1.
        out_dim: Output dimension for layer 2 (defaults to hidden_dim).
        batchnorm: Use BatchNorm1d after each conv.
        layernorm: Use LayerNorm after each conv (mutually exclusive with batchnorm).
        weight_standardization: Standardize GCNConv weights before layer 2 forward.
        batchnorm_mm: BatchNorm momentum as the EMA coefficient of running stats.
                      Note: PyTorch momentum = 1 - batchnorm_mm.
        pool: If True, apply global_mean_pool to produce graph-level embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int | None = None,
        batchnorm: bool = True,
        layernorm: bool = False,
        weight_standardization: bool = False,
        batchnorm_mm: float = 0.99,
        pool: bool = False,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim

        assert not (batchnorm and layernorm), (
            "batchnorm and layernorm are mutually exclusive"
        )
        self.weight_standardization = weight_standardization
        self.pool = pool

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        momentum = 1.0 - batchnorm_mm
        if batchnorm:
            self.norm1: nn.Module = nn.BatchNorm1d(hidden_dim, momentum=momentum)
            self.norm2: nn.Module = nn.BatchNorm1d(out_dim, momentum=momentum)
        elif layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x, edge_index)))

        if self.weight_standardization:
            apply_weight_standardization(self.conv2)
        x = self.act2(self.norm2(self.conv2(x, edge_index)))

        return x

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for norm in (self.norm1, self.norm2):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        self.act1 = nn.PReLU().to(next(self.parameters()).device)
        self.act2 = nn.PReLU().to(next(self.parameters()).device)
