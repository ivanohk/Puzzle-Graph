"""Validated configuration dataclasses for all model components.

Each dataclass mirrors the corresponding YAML block and raises ValueError
on construction if a value is out of range, so misconfigured runs fail at
load time rather than mid-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EncoderConfig:
    """Validated configuration for graph encoders (currently: GIN)."""
    name: str
    hidden_dim: int
    num_layers: int
    mlp_ratio: float = 2.0
    drop: float = 0.2
    pool: bool = True

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if not (0.0 <= self.drop < 1.0):
            raise ValueError(f"drop must be in [0, 1), got {self.drop}")


@dataclass
class HeadConfig:
    """Validated configuration for projection heads (currently: DINO head)."""
    name: str
    proj_hidden: int
    bottleneck_dim: int
    n_prototypes: int
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 0

    def __post_init__(self):
        for fname, val in [
            ("proj_hidden", self.proj_hidden),
            ("bottleneck_dim", self.bottleneck_dim),
            ("n_prototypes", self.n_prototypes),
        ]:
            if val <= 0:
                raise ValueError(f"{fname} must be > 0, got {val}")
        if self.student_temp <= 0 or self.teacher_temp <= 0:
            raise ValueError("student_temp and teacher_temp must be > 0")
        if self.warmup_teacher_temp <= 0:
            raise ValueError("warmup_teacher_temp must be > 0")
        if self.warmup_teacher_temp > self.teacher_temp:
            raise ValueError(
                f"warmup_teacher_temp ({self.warmup_teacher_temp}) must be "
                f"<= teacher_temp ({self.teacher_temp})"
            )
        if self.warmup_teacher_temp_epochs < 0:
            raise ValueError(
                f"warmup_teacher_temp_epochs must be >= 0, got {self.warmup_teacher_temp_epochs}"
            )
        if not (0.0 <= self.center_momentum < 1.0):
            raise ValueError(
                f"center_momentum must be in [0, 1), got {self.center_momentum}"
            )


@dataclass
class AugmentConfig:
    """Name and keyword arguments for a single augmentation step."""
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Augmentation name cannot be empty")

    @classmethod
    def from_dict(cls, d: dict) -> AugmentConfig:
        d = dict(d)
        name = d.pop("name")
        return cls(name=name, kwargs=d)


@dataclass
class GraphDINOConfig:
    """Validated top-level configuration for GraphDINO."""
    encoder: EncoderConfig
    head: HeadConfig
    augment: List[AugmentConfig] = field(default_factory=list)
    ema_tau: float = 0.996
    ema_tau_base: float = 0.996
    total_steps: int = 0
    freeze_last_layer_epochs: int = 1

    def __post_init__(self):
        if not (0.0 < self.ema_tau <= 1.0):
            raise ValueError(f"ema_tau must be in (0, 1], got {self.ema_tau}")
        if not (0.0 < self.ema_tau_base <= 1.0):
            raise ValueError(f"ema_tau_base must be in (0, 1], got {self.ema_tau_base}")
        if self.ema_tau_base > self.ema_tau:
            raise ValueError(
                f"ema_tau_base ({self.ema_tau_base}) must be <= ema_tau ({self.ema_tau})"
            )
        if self.total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {self.total_steps}")
        if self.freeze_last_layer_epochs < 0:
            raise ValueError(
                f"freeze_last_layer_epochs must be >= 0, got {self.freeze_last_layer_epochs}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> GraphDINOConfig:
        encoder = EncoderConfig(**d["encoder"])
        head = HeadConfig(**d["head"])
        augment = [AugmentConfig.from_dict(a) for a in d.get("augment", [])]
        return cls(
            encoder=encoder,
            head=head,
            augment=augment,
            ema_tau=d.get("ema_tau", 0.996),
            ema_tau_base=d.get("ema_tau_base", d.get("ema_tau", 0.996)),
            total_steps=d.get("total_steps", 0),
            freeze_last_layer_epochs=d.get("freeze_last_layer_epochs", 1),
        )
