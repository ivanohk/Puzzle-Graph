from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
from torch import nn
from torch_geometric.data import Data

class BaseModel(nn.Module, ABC):
    " Base interface for all self-supervised models. --> to make the training and models not dependent on each other. "
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        pass