from __future__ import annotations
from torch import nn
from graphssl.models.heads.base import Head
from graphssl.registry.registry import HEADS


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPHead(Head):
    def __init__( # in_dim is the out_dim of the encoder, use_bn is whether to use batch normalization
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        use_bn: bool = False,
        act: str = "relu",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act_layer = _make_activation(act)

        layers = []
        dim = in_dim

        if num_layers == 1: #if num_layers is 1, the MLP is just a linear layer (can be used for BYOL)
            layers.append(nn.Linear(dim, out_dim))
        else:
            for _ in range(num_layers - 1): #standard MLP structure leaving the last layer (linear) for output
                layers.append(nn.Linear(dim, hidden_dim))
                if use_bn: #if use_bn is True, add batch normalization after each linear layer
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_layer)
                dim = hidden_dim
            layers.append(nn.Linear(dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


@HEADS.register("mlp") #register the MLPHead in the HEADS registry with the name "mlp"
def build_mlp(*, in_dim: int, params: dict):
    return MLPHead(in_dim=in_dim, **params)
 