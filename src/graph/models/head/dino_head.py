from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from graph.models.heads.base import Head
from graph.registry.registry import HEADS


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class DINOHead(Head):
    """
    MLP projector -> bottleneck -> (L2 norm) -> weight-norm linear -> logits.
    forward_features() returns bottleneck features (normalized) before last layer.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        num_layers: int = 3,
        use_bn: bool = False,
        act: str = "gelu",
        norm_last_layer: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act_layer = _make_activation(act)

        layers = []
        dim = in_dim

        if num_layers == 1:
            layers.append(nn.Linear(dim, bottleneck_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_layer)
                dim = hidden_dim
            layers.append(nn.Linear(dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) # weight normalization for last layer
        self.last_layer.weight_g.data.fill_(1.0)

        if norm_last_layer: # freeze the last layer scaling parameter (weight_g) to prevent it from growing during training, which can lead to instability in DINO loss
            self.last_layer.weight_g.requires_grad = False

        self.out_dim = out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: # bottleneck features (normalized) before last layer
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)   # L2 normalize before last layer (features now have norm 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor: # logits for DINO loss after last layer weight normalization
        feats = self.forward_features(x)
        return self.last_layer(feats)


@HEADS.register("dino_head")
def build_dino_head(*, in_dim: int, params: dict):
    return DINOHead(in_dim=in_dim, **params)