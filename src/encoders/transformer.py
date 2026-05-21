"""Graph Transformer encoder: stack of TransformerConv blocks with pre-norm."""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv

from src.registry import ENCODERS


def _make_norm(name: str, dim: int) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm layer: {name!r}. Choose 'batchnorm' or 'layernorm'.")


def _make_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name!r}. Choose 'gelu' or 'relu'.")


class TransformerBlock(nn.Module):
    """Pre-norm TransformerConv + FFN block with residual connections."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        mlp_ratio: float,
        concat: bool,
        norm_layer_name: str,
        act_layer_name: str,
    ):
        super().__init__()
        # concat=True: TransformerConv outputs heads * out_channels.
        # We keep the total dim constant, so out_channels = dim // heads.
        attn_out_channels = dim // heads if concat else dim

        self.norm1 = _make_norm(norm_layer_name, dim)
        self.attn = TransformerConv(
            dim, attn_out_channels, heads=heads, dropout=attn_dropout, concat=concat
        )

        ffn_hidden = int(dim * mlp_ratio)
        self.norm2 = _make_norm(norm_layer_name, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            _make_act(act_layer_name),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), edge_index)
        x = x + self.ffn(self.norm2(x))
        return x

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        for m in self.ffn.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for norm in (self.norm1, self.norm2):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()


@ENCODERS.register("transformer")
class TransformerEncoder(nn.Module):
    """Stack of pre-norm TransformerConv blocks.

    Args:
        in_channels: Input node feature size.
        hidden_dim: Uniform hidden/output dimension throughout the stack.
        num_layers: Number of TransformerBlock layers.
        heads: Number of attention heads (hidden_dim must be divisible by heads).
        dropout: Dropout in the FFN.
        attn_dropout: Dropout inside TransformerConv attention.
        mlp_ratio: FFN hidden dim = hidden_dim * mlp_ratio.
        concat: If True, TransformerConv concatenates heads (see TransformerBlock).
        norm_layer_name: 'batchnorm' or 'layernorm'.
        act_layer_name: 'gelu' or 'relu'.
        pool: Apply global_mean_pool to produce graph-level embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        concat: bool = True,
        norm_layer_name: str = "batchnorm",
        act_layer_name: str = "gelu",
        pool: bool = True,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.pool = pool

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    mlp_ratio=mlp_ratio,
                    concat=concat,
                    norm_layer_name=norm_layer_name,
                    act_layer_name=act_layer_name,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = _make_norm(norm_layer_name, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.norm(x)
        return x

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
