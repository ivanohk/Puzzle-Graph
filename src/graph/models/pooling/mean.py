from __future__ import annotations
from torch_geometric.nn import global_mean_pool
from graph.models.pooling.base import Pooler
from graph.registry.registry import POOLERS

class MeanPool(Pooler):
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

@POOLERS.register("mean")
def build_mean_pooler(*, in_dim: int, params: dict):
    return MeanPool(in_dim=in_dim)