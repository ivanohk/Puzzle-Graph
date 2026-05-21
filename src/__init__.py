"""Puzzle-Graph: a modular Graph Machine Learning library in pure PyTorch."""

# ── Core abstractions ─────────────────────────────────────────────────────────
from src.core.model import BaseModel, BaseSSLModel
from src.core.encoder import BaseEncoder
from src.core.augmentation import BaseAugmentation
from src.core.callback import Callback
from src.core.registry import Registry

# ── Encoders ──────────────────────────────────────────────────────────────────
from src.encoders import GINEncoder, GCNEncoder, TransformerEncoder

# ── Models ────────────────────────────────────────────────────────────────────
from src.models import (
    GraphDINO,
    Supervised,
    DGI,
    GraphCL,
    BGRL,
    AFGRL,
    VICReg,
    BarlowTwins,
)

# ── Losses ────────────────────────────────────────────────────────────────────
from src.losses import (
    DINOLoss,
    NTXentLoss,
    VICRegLoss,
    BarlowTwinsLoss,
    CosineRegressionLoss,
)

# ── Neural network building blocks ────────────────────────────────────────────
from src.nn import MLP, Projector, Predictor, DINOHead
from src.nn import pool_graph_embeddings, loss_inputs_from_embeddings

# ── Augmentation ──────────────────────────────────────────────────────────────
from src.augmentation import compose, MultiView
from src.augmentation.transforms import (
    EdgeDrop,
    EdgeAdd,
    Subgraph,
    FeatMask,
    FeatNoise,
    FeatShuffle,
)

# ── Utilities ─────────────────────────────────────────────────────────────────
from src.utils import update_ema_params, EMA, CosineDecayScheduler, CosineEMAScheduler

# ── Data ──────────────────────────────────────────────────────────────────────
from src.data import DataModule

# ── Evaluation ────────────────────────────────────────────────────────────────
from src.evaluation import LogRegEvaluator, KNNEvaluator, extract_embeddings

# ── Training ──────────────────────────────────────────────────────────────────
from src.training import (
    DINOTrainer,
    EmbeddingLoggerCallback,
    LinearEvalCallback,
    VisualizationCallback,
)
