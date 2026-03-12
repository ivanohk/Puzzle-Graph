from __future__ import annotations
from abc import ABC, abstractmethod
from torch import nn
import torch


class Head(nn.Module, ABC):
    """
    Common interface for projection/prediction heads.
    """

    out_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns head output (e.g., projected embedding or logits).
        """
        raise NotImplementedError

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional: returns intermediate features before the last linear/logit layer. --> for DINO
        Default: same as forward.
        """
        return self.forward(x)