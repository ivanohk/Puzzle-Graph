from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import nn


class NodeEncoder(nn.Module, ABC):
    """
    Base interface for node-level encoders.

    Contract:
      - must expose `out_dim` (int)
      - forward(x, edge_index) -> node embeddings [num_nodes, out_dim]
    """
    out_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError