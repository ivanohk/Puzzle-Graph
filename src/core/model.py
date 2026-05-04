from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple
import torch
from torch import nn
from torch_geometric.data import Data

class BaseModel(nn.Module, ABC):
    """Base interface for all self-supervised graph SSL models."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embeddings, loss) for a batch of graphs."""

    @abstractmethod
    def student_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters to optimize; passed to the optimizer and gradient clipper."""

    @abstractmethod
    def post_step(self) -> None:
        """Per-batch housekeeping after optimizer.step(): EMA, centering, momentum updates."""
