"""DINOTrainer: orchestrates one epoch of SSL self-distillation."""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.models.model_types.base import BaseModel


class DINOTrainer:
    """Stateless trainer — owns no model or optimizer state.

    Works with any BaseModel subclass: calls model.student_parameters() for
    gradient clipping and model.post_step() for per-batch housekeeping (EMA,
    centering, etc.), so no DINO-specific attributes are accessed directly.

    Args:
        grad_clip_norm: Clip student gradient norm to this value.
                        ~3.0 helps stabilize early DINO training.
    """

    def __init__(self, grad_clip_norm: Optional[float] = 3.0):
        self.grad_clip_norm = grad_clip_norm

    def train_epoch(
        self,
        model: BaseModel,
        loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
    ) -> float:
        """Run a full training epoch and return the average loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            embeddings, loss = model(batch)

            optimizer.zero_grad()
            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.student_parameters()),
                    self.grad_clip_norm,
                )

            optimizer.step()
            model.post_step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
