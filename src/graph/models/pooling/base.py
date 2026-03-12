# src/graphssl/models/pooling/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from torch import nn

class Pooler(nn.Module, ABC):
    """
    Pooler interface: maps node embeddings -> graph embeddings.
    """
    out_dim: int

    @abstractmethod
    def forward(self, x, batch):
        """
        x: [num_nodes, d]
        batch: [num_nodes] mapping each node to a graph id
        returns: [num_graphs, out_dim]
        """
        raise NotImplementedError