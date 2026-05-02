"""GraphDINO: DINO-style self-distillation for graph-level SSL."""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Data

from .base import BaseModel
from src.registry import ENCODERS, HEADS
from src.augmentation import compose
from src.config.schema import EncoderConfig, GraphDINOConfig, HeadConfig
from src.utils import update_ema_params


class GraphDINO(BaseModel):
    """
    Args:
        config: Config dict (see graphdino.yaml); validated against GraphDINOConfig.
        in_channels: Node feature size, resolved from the dataset at runtime.
    """

    def __init__(self, config: Dict, in_channels: int):
        super().__init__(config)
        cfg = GraphDINOConfig.from_dict(config)  # validates all fields at load time

        self.student_enc = self._build_encoder(cfg.encoder, in_channels)
        self.student_head = self._build_head(cfg.head, cfg.encoder.hidden_dim)

        # Build fresh instances rather than deepcopy: weight_norm creates non-leaf
        # tensors that break copy.deepcopy.
        self.teacher_enc = self._build_encoder(cfg.encoder, in_channels)
        self.teacher_head = self._build_head(cfg.head, cfg.encoder.hidden_dim)
        self.teacher_enc.load_state_dict(self.student_enc.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_enc.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.aug_list: List[Tuple[str, dict]] = [
            (a.name, a.kwargs) for a in cfg.augment
        ]
        self._ema_tau: float = cfg.ema_tau

        # Stored during forward so post_step can call update_center.
        self.last_teacher_out: Tensor | None = None

    @staticmethod
    def _build_encoder(enc_cfg: EncoderConfig, in_channels: int) -> nn.Module:
        kwargs = {k: v for k, v in vars(enc_cfg).items() if k != "name"}
        return ENCODERS.build(enc_cfg.name, in_channels=in_channels, **kwargs)

    @staticmethod
    def _build_head(head_cfg: HeadConfig, hidden_dim: int) -> nn.Module:
        kwargs = {k: v for k, v in vars(head_cfg).items() if k != "name"}
        return HEADS.build(head_cfg.name, hidden_dim=hidden_dim, **kwargs)

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

    def student_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters to optimize; passed to the optimizer and gradient clipper."""
        yield from self.student_enc.parameters()
        yield from self.student_head.parameters()

    def post_step(self) -> None:
        """EMA teacher update and prototype center update after each optimizer step."""
        update_ema_params(self.student_enc, self.teacher_enc, self._ema_tau)
        update_ema_params(self.student_head, self.teacher_head, self._ema_tau)
        if self.last_teacher_out is not None:
            self.student_head.update_center(self.last_teacher_out)
