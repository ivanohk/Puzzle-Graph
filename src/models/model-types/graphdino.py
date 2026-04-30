"""GraphDINO: DINO-style self-distillation for graph-level SSL."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from .base import BaseModel
from src.registry.registry import ENCODERS, HEADS
from src.data.dino_augmentations import compose


class GraphDINO(BaseModel):
    """
    Args:
        config: Config dict (see graphdino.yaml).
        in_channels: Node feature size, resolved from the dataset at runtime.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__(config)

        self.student_enc = self._build_encoder(config["encoder"], in_channels)
        self.student_head = self._build_head(config["head"], config["encoder"]["hidden_dim"])

        # Build fresh instances rather than deepcopy: weight_norm creates non-leaf
        # tensors that break copy.deepcopy.
        self.teacher_enc = self._build_encoder(config["encoder"], in_channels)
        self.teacher_head = self._build_head(config["head"], config["encoder"]["hidden_dim"])
        self.teacher_enc.load_state_dict(self.student_enc.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_enc.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.aug_list: List[Tuple[str, dict]] = []
        for aug_cfg in config.get("augment", []):
            aug_cfg = dict(aug_cfg)
            name = aug_cfg.pop("name")
            self.aug_list.append((name, aug_cfg))

        # Stored during forward so the trainer can call update_center after each step.
        self.last_teacher_out: Tensor | None = None

    @staticmethod
    def _build_encoder(enc_cfg: Dict, in_channels: int) -> nn.Module:
        cfg = dict(enc_cfg)
        name = cfg.pop("name")
        return ENCODERS.build(name, in_channels=in_channels, **cfg)

    @staticmethod
    def _build_head(head_cfg: Dict, hidden_dim: int) -> nn.Module:
        cfg = dict(head_cfg)
        name = cfg.pop("name")
        return HEADS.build(name, hidden_dim=hidden_dim, **cfg)

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            embeddings: Student encoding of the original graph, detached.
            loss: Symmetric DINO cross-entropy loss.
        """
        view1 = compose(data, self.aug_list)
        view2 = compose(data, self.aug_list)

        s_emb1 = self.student_enc(view1.x, view1.edge_index, view1.batch)
        s_emb2 = self.student_enc(view2.x, view2.edge_index, view2.batch)
        s_logits1 = self.student_head(s_emb1)
        s_logits2 = self.student_head(s_emb2)

        with torch.no_grad():
            t_emb1 = self.teacher_enc(view1.x, view1.edge_index, view1.batch)
            t_emb2 = self.teacher_enc(view2.x, view2.edge_index, view2.batch)
            t_targets1 = self.teacher_head(t_emb1, use_teacher_temp=True)
            t_targets2 = self.teacher_head(t_emb2, use_teacher_temp=True)

        self.last_teacher_out = torch.cat([t_targets1, t_targets2], dim=0)

        # L = -0.5 * (t2 * log(s1) + t1 * log(s2))
        loss = -0.5 * (
            (t_targets2 * s_logits1).sum(dim=-1).mean()
            + (t_targets1 * s_logits2).sum(dim=-1).mean()
        )

        with torch.no_grad():
            embeddings = self.student_enc(
                data.x, data.edge_index, data.batch
            ).detach()

        return embeddings, loss
