# Puzzle-Graph

Libreria Python per Graph Machine Learning (GML) con PyTorch puro.
Implementa i principali algoritmi SSL su grafi con un'architettura modulare e zero dipendenze da Lightning o Hydra.

## Modelli implementati

| Modello | Tipo | File |
|---|---|---|
| DGI | Mutual information maximization | `src/models/dgi.py` |
| GraphCL | Contrastive (NT-Xent) | `src/models/graphcl.py` |
| BGRL | Teacher-student + EMA | `src/models/bgrl.py` |
| AFGRL | Augmentation-free + PositiveMiner | `src/models/afgrl.py` |
| VICReg | Variance-Invariance-Covariance | `src/models/vicreg.py` |
| Barlow Twins | Cross-correlation | `src/models/barlow_twins.py` |
| GraphDINO | DINO adattato ai grafi | `src/models/graphdino.py` |
| Supervised | Encoder + linear head (baseline) | `src/models/supervised.py` |

## API uniforme

Tutti i modelli usano la stessa firma:

```python
model = ModelClass(config: dict, in_channels: int)
# Supervised aggiunge num_classes
model = Supervised(config, in_channels, num_classes)
```

```python
from src.models import BGRL

config = {
    "encoder": {"name": "gin", "hidden_dim": 256, "num_layers": 3, "pool": False},
    "augment": [{"name": "edge_drop", "p": 0.2}, {"name": "feat_mask", "p": 0.1}],
    "pred_hidden": 512,
}

model = BGRL(config, in_channels=dataset.num_features)
loss = model.compute_loss(batch)
loss.backward()
optimizer.step()
model.post_step()  # EMA update del teacher
```

## Dipendenze

- Python 3.10+, PyTorch, PyTorch Geometric
- `faiss-cpu` (opzionale, richiesto solo da AFGRL)
- `umap-learn`, `matplotlib`, `seaborn` (opzionali, per visualization)

## Test

```bash
pytest tests/ -v
```

82 test, tutti i modelli coperti. AFGRL viene saltato automaticamente se `faiss-cpu`
non è installato.

## Struttura

```
src/
├── core/          # ABC: BaseModel, BaseSSLModel, Callback, Registry
├── encoders/      # GCN, GIN, Transformer
├── models/        # DGI, GraphCL, BGRL, AFGRL, GraphDINO, VICReg, BarlowTwins, Supervised
├── nn/            # MLP, DINOHead, norm, pooling
├── augmentation/  # functional, transforms, compose
├── losses/        # nt_xent, dino, vicreg, barlow, regression
├── evaluation/    # linear_probe, knn, visualization
├── training/      # DINOTrainer, callbacks
├── data/          # DataModule
├── utils/         # ema, schedulers, positive_miner
└── config/        # schema.py (dataclass validation), load.py
```

## Note su canonic-pyg/

Contiene implementazioni di riferimento BGRL e GraphDINO nello stile nativo PyG
(pensate per una futura PR upstream). Non è parte della libreria principale.
