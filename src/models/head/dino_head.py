import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d as BN


from src.registry.registry import HEADS

@HEADS.register("dino")
class DINOHead(nn.Module):
    def __init__(self, hidden_dim, proj_hidden, bottleneck_dim, 
                 n_prototypes, student_temp=0.1, 
                 teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Projector
        self.projector = Sequential(
            Linear(hidden_dim, proj_hidden),
            BN(proj_hidden),
            ReLU(),
            Linear(proj_hidden, bottleneck_dim),
        )

        # Prototype scorer
        self.proto = nn.utils.weight_norm(
            Linear(bottleneck_dim, n_prototypes, bias=False)
        )
        self.proto.weight_g.data.fill_(1)
        self.proto.weight_g.requires_grad = False

        # Centering buffer
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