import numpy as np
import torch
from torch import nn


@torch.no_grad()
def update_ema_params(student: nn.Module, teacher: nn.Module, tau: float):
    """EMA update of teacher parameters from the student.

    Buffers (e.g. BatchNorm running stats) are copied directly, not averaged.
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


class EMA:
    """Scalar EMA with cosine-annealed alpha."""
    def __init__(self, alpha, epochs) -> None:
        super().__init__()
        self.alpha = alpha
        self.steps = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        
        alpha = 1 - (1 - self.alpha) * (np.cos(np.pi * self.steps / self.total_steps) + 1) / 2.0

        self.steps += 1

        ema = alpha * old + (1 - alpha) * new

        return ema