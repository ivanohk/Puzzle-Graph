import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN


from src.registry import HEADS

@HEADS.register("dino")
class DINOHead(nn.Module):
    def __init__(self, hidden_dim, proj_hidden, bottleneck_dim, 
                 n_prototypes, student_temp=0.1, 
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        self.projector = Sequential(
            Linear(hidden_dim, proj_hidden),
            BN(proj_hidden),
            ReLU(),
            Linear(proj_hidden, bottleneck_dim),
        )

        # weight_norm keeps prototype rows unit-norm; using parametrizations API
        # since the old weight_norm is deprecated and breaks deepcopy.
        proto_linear = Linear(bottleneck_dim, n_prototypes, bias=False)
        self.proto = nn.utils.parametrizations.weight_norm(proto_linear, dim=0)

        self.register_buffer("center", torch.zeros(1, n_prototypes))

    def forward(self, x, use_teacher_temp=False):
        h = self.projector(x)
        h = F.normalize(h, dim=-1)
        logits = self.proto(h)

        if use_teacher_temp:
            return F.softmax((logits - self.center) / self.teacher_temp, dim=-1)
        else:
            return F.log_softmax(logits / self.student_temp, dim=-1)
        
    @torch.no_grad()
    def update_center(self, teacher_out):
        self.center = (
            self.center * self.center_momentum
            + teacher_out.mean(0, keepdim=True) * (1 - self.center_momentum)
        )