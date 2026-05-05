from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


class BaseModel(nn.Module, ABC):
    """Base for all models. forward() returns embeddings only."""

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Extract node/graph embeddings from a batch."""


class BaseSSLModel(BaseModel):
    """Extension for self-supervised models.

    The Trainer loop is:
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        model.post_step()           # EMA, center updates, etc.

    For evaluation:
        embeddings = model(batch)   # calls forward(), no grad needed
    """

    @abstractmethod
    def compute_loss(self, data: Data) -> Tensor:
        """Compute the SSL training loss. Augmentation is handled internally."""

    @abstractmethod
    def student_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters to optimize. Excludes frozen teacher/target parameters."""

    def post_step(self) -> None:
        """Post-optimizer housekeeping called by the Trainer after optimizer.step().

        Override to implement EMA teacher updates, DINO center updates, etc.
        Default is a no-op for models that don't need it.
        """
