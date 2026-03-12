import torch
from torch import nn
from torch_geometric.nn import GINConv

from graph.registry.registry import ENCODERS
from graph.models.encoders.base import NodeEncoder

def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "silu": return nn.SiLU()
    if name == "elu":  return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")

def _make_norm(name: str, dim: int) -> nn.Module:
    name = name.lower()
    if name in ("none", "null"): return nn.Identity()
    if name in ("batch", "batchnorm", "bn"): return nn.BatchNorm1d(dim)
    if name in ("layer", "layernorm", "ln"): return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm: {name}")

class GINEncoder(NodeEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        act: str = "relu", # Activation function to use in the MLPs of the GINConv layers. Default is "relu". Supported activations are "relu", "gelu", "silu", and "elu".
        norm: str = "none", # Normalization to apply after each GINConv layer. Default is "none". Supported norms are "none", "batch", and "layer".
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.convs = nn.ModuleList() # List to hold the GINConv layers since the number of layers is variable (determined by num_layers from yaml)
        self.norms = nn.ModuleList() # List to hold the normalization layers corresponding to each GINConv layer
        act_layer = _make_activation(act)

        dim = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                act_layer,
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(_make_norm(norm, hidden_dim))
            dim = hidden_dim # Update dim for the next layer to be hidden_dim since GINConv will output hidden_dim features. This ensures that all layers are compatible in terms of input and output dimensions.

        self.out_dim = hidden_dim # store the output dimension so the next block knows the dim of the embeddings

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training) # Apply dropout after each layer except the last one. This is a common practice to prevent overfitting. The dropout is applied during training and not during evaluation (inference).
        return x

@ENCODERS.register("gin")
def build_gin(*, in_dim: int, params: dict):
    return GINEncoder(in_dim=in_dim, **params)