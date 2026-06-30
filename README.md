# Puzzle-Graph

A modular Python library for **Self-Supervised Learning on graphs**, built on PyTorch and PyTorch Geometric. No Lightning, no Hydra — just clean, readable training loops you can step through with a debugger.

> **Status:** Alpha. All models train end-to-end and pass tests, but large-scale benchmarks are still in progress.

## Supported Methods

| Method | Family | Paper |
|---|---|---|
| **DGI** | Mutual Information | Veličković et al., ICLR 2019 |
| **GraphCL** | Contrastive (NT-Xent) | You et al., NeurIPS 2020 |
| **BGRL** | Teacher-Student + EMA | Thakoor et al., ICLR 2022 |
| **AFGRL** | Augmentation-Free Mining | Lee et al., AAAI 2022 |
| **VICReg** | Variance-Invariance-Covariance | Bardes et al., ICLR 2022 |
| **Barlow Twins** | Cross-Correlation | Zbontar et al., ICML 2021 |
| **GraphDINO** | Self-Distillation | Adapted from Caron et al., ICCV 2021 |

Plus a **Supervised** baseline (encoder + linear head) for comparison.

Encoder backbones are swappable: **GCN**, **GIN**, and **Graph Transformer** are included and registered via a simple decorator.

## Quick Start

Every model follows the same API:

```python
from src.models import BGRL
from src.training import DINOTrainer

config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 3, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}],
    "pred_hidden": 512,
}

model = BGRL(config, in_channels=dataset.num_features)
trainer = DINOTrainer(device="cuda")
losses = trainer.train(model, loader, optimizer, num_epochs=100)
```

Models expose `compute_loss(batch)`, `post_backward()`, and `post_step()` hooks, so you can also write your own training loop if you prefer.

## Installation

```bash
pip install torch torch_geometric
# optional
pip install faiss-cpu      # needed by AFGRL's PositiveMiner
pip install umap-learn matplotlib seaborn  # for visualization callbacks
```

Then clone this repo and run from the root:

```bash
pytest tests/ -v
```

All 8 model test suites should pass. AFGRL tests are auto-skipped if `faiss-cpu` is missing.

## Project Structure

```
src/
├── core/           # BaseSSLModel, Callback, Registry, Protocol interfaces
├── config/         # Dataclass schemas + YAML loading
├── encoders/       # GCN, GIN, Transformer (registry-based)
├── models/         # All SSL models + supervised baseline
├── losses/         # NT-Xent, DINO, VICReg, Barlow Twins, cosine regression
├── augmentation/   # 7 transforms, compose(), MultiView
├── nn/             # MLP, Projector, DINOHead, pooling utilities
├── evaluation/     # Linear probing (LogRegEvaluator), KNN, UMAP
├── training/       # DINOTrainer + callbacks (LinearEval, Visualization, EmbeddingLogger)
├── data/           # DataModule (wraps PyG datasets, provides loaders)
└── utils/          # EMA, cosine schedulers, PositiveMiner
```

## Key Design Choices

- **Hook-driven training.** The trainer calls `post_backward()` and `post_step()` at fixed points in the loop. Models use these to do things like freeze prototype gradients (GraphDINO) or update the EMA teacher (BGRL, AFGRL).
- **Protected-node augmentations.** When using `NeighborLoader` for mini-batch training, seed nodes are protected from `node_drop` so the loss stays valid.
- **Registry pattern.** Encoders and augmentations are registered by name (`@ENCODERS.register("gin")`), so swapping them is a one-line config change.

## About `canonic-pyg/`

Legacy reference implementations of BGRL and GraphDINO in vanilla PyG style. Not part of the library — kept around for potential upstream contributions.

## Acknowledgments

Developed at [NECSTLab](https://necst.it), Politecnico di Milano.
