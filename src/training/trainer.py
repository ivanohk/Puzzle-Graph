"""DINOTrainer: orchestrates one epoch of DINO self-distillation."""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.utils.ema import update_ema_params


class DINOTrainer:
    """Stateless trainer — owns no model or optimizer state.

    Args:
        grad_clip_norm: Clip student gradient norm to this value.
                        ~3.0 helps stabilize early DINO training.
    """

    def __init__(self, grad_clip_norm: Optional[float] = 3.0):
        self.grad_clip_norm = grad_clip_norm

    def train_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        ema_tau: float,
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
                    list(model.student_enc.parameters())
                    + list(model.student_head.parameters()),
                    self.grad_clip_norm,
                )

            optimizer.step()

            update_ema_params(model.student_enc, model.teacher_enc, ema_tau)
            update_ema_params(model.student_head, model.teacher_head, ema_tau)

            if model.last_teacher_out is not None:
                model.student_head.update_center(model.last_teacher_out)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
