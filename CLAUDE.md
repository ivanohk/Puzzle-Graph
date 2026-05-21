Sto sviluppando **Puzzle-Graph**, una libreria Python per Graph Machine Learning (GML).
L'obiettivo è implementare i principali algoritmi SSL su grafi in PyTorch puro,
con un'architettura modulare e riutilizzabile.

---

## STACK TECNOLOGICO

- Python 3.10+
- PyTorch + PyTorch Geometric (PyG)
- SOLO PyTorch puro nel core: zero dipendenze da PyTorch Lightning
- FAISS-cpu per il positive mining (AFGRL)
- UMAP + scikit-learn per evaluation e visualization (dipendenze opzionali)
- Nessun Hydra, nessun OmegaConf

---

## PUBLIC API — COSTRUZIONE UNIFORME

**Tutti i modelli usano la stessa firma pubblica:**
```python
model = ModelClass(config: Dict, in_channels: int)
# Supervised aggiunge: num_classes: int
model = Supervised(config, in_channels, num_classes)
```

Il `config` dict viene validato da un dataclass dedicato in `src/config/schema.py`.
`EncoderConfig.build(in_channels)` istanzia l'encoder tramite il registry `ENCODERS`.

**Dataclass per modello:**
| Modello      | Config dataclass    | File                          |
|--------------|---------------------|-------------------------------|
| DGI          | `DGIConfig`         | `src/config/schema.py`        |
| GraphCL      | `GraphCLConfig`     | `src/config/schema.py`        |
| VICReg       | `VICRegConfig`      | `src/config/schema.py`        |
| BarlowTwins  | `BarlowTwinsConfig` | `src/config/schema.py`        |
| BGRL         | `BGRLConfig`        | `src/config/schema.py`        |
| AFGRL        | `AFGRLConfig`       | `src/config/schema.py`        |
| Supervised   | `SupervisedConfig`  | `src/config/schema.py`        |
| GraphDINO    | `GraphDINOConfig`   | `src/config/schema.py`        |

---

## ALGORITMI IMPLEMENTATI

Tutti i modelli sono `nn.Module` puri con solo `forward()` e `compute_loss()`.
Nessun accesso a trainer, logger, o datamodule dall'interno del modello.

### Supervised
- Config: `SupervisedConfig` — encoder + num_classes passato al costruttore
- Encoder + linear head `nn.Linear(hidden_dim, num_classes)`, CrossEntropyLoss
- Supporto full-batch e mini-batch (crop a `[:batch_size]` per seed nodes)
- `graph_level` derivato da `cfg.encoder.pool`

### DGI (Deep Graph Infomax)
- Config: `DGIConfig` — encoder + corruption type
- Discrimina embedding reale vs corrotto tramite un discriminatore con matrice W learnable
  (`nn.Parameter`, inizializzata con `xavier_uniform_`)
- Corruzione: shuffle dei nodi (permutazione di x) o shuffle delle edge destinations
- Summary globale `s = sigmoid(mean(h_pos))` per node-level, `global_mean_pool` per graph-level
- Discriminator: `pos_logits = (h_pos * W@s).sum(-1)`
- Loss: BCEWithLogitsLoss su [pos_logits, neg_logits] con label [1,1,...,0,0,...]
- Metrica aggiuntiva: discriminator accuracy

### GraphCL (Graph Contrastive Learning)
- Config: `GraphCLConfig` — encoder + augment list + proj_dim + tau
- NT-Xent loss su 2 augmented view
- Encoder + projector 2-layer `Projector(hidden_dim, hidden_dim, proj_dim)`
- `protected_nodes` passato a `compose()` per i seed nodes in mini-batch node training
- Iperparametri: `proj_dim=128, tau=0.5`

### BGRL (Bootstrapped Graph Representation Learning)
- Config: `BGRLConfig` — encoder + augment list + pred_hidden + ema params + total_steps
- Teacher-student con EMA update sul target encoder
- Online encoder (student) + predictor MLP `Predictor(hidden_dim, pred_hidden, hidden_dim)`; target encoder (teacher, no grad)
- Loss: `2 - cos(pred1, target2) - cos(pred2, target1)` — `CosineRegressionLoss(symmetric=True)`
  - Chiamata corretta: `loss_fn(p1, t1, p2, t2)` → la loss internamente usa t2 con p1 e t1 con p2
- EMA update del target encoder in `post_step()` via `update_ema_params()`
- Momentum τ cosine-annealed da `ema_tau` a `ema_tau_end` su `total_steps` — `CosineEMAScheduler`.
  Se `total_steps=0`, τ fisso.
- LR scheduling delegato all'optimizer/trainer esterno
- Target encoder: `copy.deepcopy(encoder)` + `reset_parameters()` — pesi intenzionalmente diversi
  dall'online encoder (cruciale per la convergenza, App. B del paper BGRL)
- `protected_nodes` passato a `compose()` per i seed nodes in mini-batch node training
- `graph_level` derivato da `cfg.encoder.pool`
- Iperparametri: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, pred_hidden=512`

### AFGRL (Augmentation-Free Graph Representation Learning)
- Config: `AFGRLConfig` — encoder + pred_hidden + ema params + topk + miner params (no augment list)
- Come BGRL ma senza augmentation strutturale: nessuna vista augmentata, positive pairs minati
- Online encoder (student) + predictor MLP `Predictor(hidden_dim, pred_hidden, hidden_dim)`; target encoder (teacher, EMA, no grad)
- **Differenza da BGRL**: target encoder inizia con stessi pesi dell'online (`deepcopy` senza reset)
- **PositiveMiner** (`utils/positive_miner.py`): unione di due tipi di positivi per ogni nodo:
  - Local: top-k vicini per cosine similarity CHE SONO ANCHE adiacenti nel grafo (kNN ∩ adj)
  - Global: top-k vicini che condividono un cluster k-means in ALMENO UNA delle `num_kmeans` run
  - Restituisce coppie `(src, dst)` su cui calcolare la loss
- **Loss node-level**: simmetrica sulle coppie positive minate:
  `(2−2·cos(pred[src], target[dst]) + 2−2·cos(pred[dst], target[src])).mean()`
- **Loss graph-level**: nessun miner (non ha senso tra grafi diversi); teacher-student semplice
  su embeddings poolati — `CosineRegressionLoss(symmetric=False)`
- EMA update del target encoder in `post_step()` via `update_ema_params()`
- Momentum τ cosine-annealed da `ema_tau` a `ema_tau_end` su `total_steps` — `CosineEMAScheduler`.
  Se `total_steps=0`, τ fisso.
- `graph_level` derivato da `cfg.encoder.pool`
- Iperparametri: `ema_tau=0.99, ema_tau_end=1.0, total_steps=0, topk=5,
  num_centroids=50, num_kmeans=4, clus_num_iters=20, pred_hidden=512`

### GraphDINO
- Adattamento di DINO (ViT) ai grafi
- Student encoder + student head; teacher encoder + teacher head (EMA, no grad)
- `n_views` viste totali generate da `compose()`; le prime `n_global_views` vanno al teacher,
  tutte le `n_views` allo student (default: `n_global_views=2, n_views=2`)
- **DINOLoss**: cross-entropy su tutti i pari (teacher_i, student_j) con i ≠ j, mediata
  per numero di termini. Con n=2 coincide con `−0.5*(t2·s1 + t1·s2)`.
  - Student output: `log_softmax(logits / student_temp)` — calcolato in `DINOHead`
  - Teacher output: `softmax((logits − center) / teacher_temp_eff)` — calcolato in `DINOHead`
- **Teacher temperature warmup**: `teacher_temp_eff` parte da `warmup_teacher_temp`
  e cresce linearmente fino a `teacher_temp` nell'arco di `warmup_teacher_temp_epochs` epoche.
  Dopo, rimane fisso a `teacher_temp`. `DINOHead.set_epoch(epoch)` aggiorna il valore corrente.
- **Center update** (EMA): `center = c_mom * center + (1 − c_mom) * mean(teacher_out)`
  Gestito da `DINOHead.update_center()`, chiamato da `post_step()` del modello.
- **EMA teacher**: momentum crescente cosine da `ema_tau_base` a `ema_tau` su `total_steps`.
  Se `total_steps=0` il momentum è fisso a `ema_tau_base`.
- **freeze_last_layer_epochs**: nelle prime N epoche i gradienti del layer prototipo
  (`student_head.proto`) vengono azzerati dopo il backward. Implementato in `post_backward()`.
- Il trainer chiama `on_epoch_start(epoch)` (aggiorna teacher temp) e `on_epoch_end(epoch)`
  (avanza il contatore epoca per il freeze) prima e dopo ogni epoca.
- **Mini-batch node training**: `protected_nodes = torch.arange(batch.batch_size)` viene
  passato a `compose()` e propagato agli augment. Solo l'embedding dei seed nodes
  (`h[:batch_size]`) viene usato per la loss.
- Iperparametri: `student_temp=0.1, teacher_temp=0.07, warmup_teacher_temp=0.04,
  warmup_teacher_temp_epochs=30, ema_tau=0.996, freeze_last_layer_epochs=1,
  n_views=2, n_global_views=2`

### VICReg
- Config: `VICRegConfig` — encoder + augment list + proj_dim + invariance/variance/covariance coeffs
- Encoder + projector 3-layer `Projector(hidden_dim, hidden_dim*2, proj_dim)` (bottleneck espanso)
- Loss tripla: invariance (MSE), variance (hinge su std), covariance (off-diagonal) — `VICRegLoss`
- `protected_nodes` passato a `compose()` per i seed nodes in mini-batch node training
- `graph_level` derivato da `cfg.encoder.pool`
- Iperparametri: `proj_dim=256, invariance=25.0, variance=25.0, covariance=1.0`

### Barlow Twins
- Config: `BarlowTwinsConfig` — encoder + augment list + proj_dim + lambda_param (opzionale)
- Encoder + projector 3-layer `Projector(hidden_dim, hidden_dim*2, proj_dim)` (identico a VICReg)
- Cross-correlation matrix `C = (z1_norm.T @ z2_norm) / N`
- Loss: `sum((1−C_ii)²) + lambda * sum_{i≠j}(C_ij²)` — `BarlowTwinsLoss`
- `lambda_param=None` → default `1/proj_dim` calcolato dentro `BarlowTwinsLoss`
- `protected_nodes` passato a `compose()` per i seed nodes in mini-batch node training
- `graph_level` derivato da `cfg.encoder.pool`
- Iperparametri: `proj_dim=256, lambda_param=None`

---

## ENCODER (GNN BACKBONES)

Tutti espongono `forward(x, edge_index, batch) → Tensor[N, out_dim]`.

### GCN
- 2 layer GCNConv con PyG Sequential
- Ogni layer: GCNConv → (BatchNorm | LayerNorm) → PReLU
- Weight standardization opzionale: per ogni GCNConv (tranne il primo),
  normalizza i pesi per-riga: `weight = (weight − mean_i) / sqrt(var_i + 1e-5)`
  (normalizzazione applicata su `dim=1`, non globale)
- `reset_parameters()` esposto (usato da BGRL per target encoder)

### GIN (Graph Isomorphism Network)
- Stack di `GINLayer` con residual connections
- Ogni `GINLayer`: `GINConv(mlp) → norm → activation → dropout`
- `mlp_ratio`: moltiplica la dimensione hidden per il MLP interno di GINConv

### Transformer
- Stack di `TransformerBlock` (TransformerConv da PyG), pre-norm con residual
- Parametri: `in_dim, hidden_dim, num_layers=4, heads=4, dropout=0.1,
  attn_dropout=0.1, mlp_ratio=4.0`

---

## SISTEMA DI AUGMENTATION

Sistema componibile stile torchvision in `augmentation/`:
- `functional.py`: funzioni pure `(data, *, protected_nodes=None, **kwargs) → Data`
- `transforms.py`: classi callable che wrappano le funzioni
- `compose.py`: `compose(data, aug_list, protected_nodes=None)` applica la lista in sequenza,
  propagando `protected_nodes` a ogni augment

Operazioni disponibili e parametri:

1. `edge_drop(data, p)` — rimuove edge con probabilità p
2. `edge_add(data, p)` — aggiunge edge casuali con probabilità p
3. `feat_mask(data, p)` — maschera features dei nodi con probabilità p
4. `feat_noise(data, std)` — aggiunge rumore gaussiano alle features
5. `feat_shuffle(data, p)` — scambia features tra nodi casuali con probabilità p
6. `subgraph(data, num_hops)` — estrae un sottografo k-hop da un seed casuale
7. `node_drop(data, p, protected_nodes)` — rimuove nodi con rimappatura edge_index;
   i nodi in `protected_nodes` non vengono mai rimossi (usato per i seed nodes in mini-batch)

`MultiView(transforms, n_views)` genera n viste indipendenti e accetta `protected_nodes`.

---

## UTILITY CHIAVE

### extract_embeddings (`evaluation/visualization.py`)
Estrae embedding da un modello dato un datamodule. Gestisce:
1. Graph-level: DataLoader + `global_mean_pool`
2. Node full-batch: forward sull'intero grafo, slicing per split
3. Node mini-batch: NeighborLoader con `global_to_local` mapping per ricostruire l'ordine

### LogRegEvaluator (`evaluation/linear_probe.py`)
Evaluator lineare PyTorch puro. Per multilabel (es. ogbg-molpcba) usa BCE + average precision.

### EncoderConfig.build() (`config/schema.py`)
`cfg.encoder.build(in_channels)` — istanzia l'encoder via `ENCODERS.build(name, ...)`.
Tutti i campi di `EncoderConfig` tranne `name` vengono passati come kwargs.
È il punto unico in cui encoder viene creato — nessun modello costruisce l'encoder inline.

### EMA (`utils/ema.py`)
`update_ema_params(student, teacher, tau)` — aggiorna parametri e buffer (es. BatchNorm running stats).

### pool_graph_embeddings / loss_inputs_from_embeddings (`nn/pooling.py`)
- `pool_graph_embeddings`: `global_mean_pool` con fallback se `batch=None`
- `loss_inputs_from_embeddings`: taglia `z[:batch.batch_size]` per mini-batch node-level

### Schedulers (`utils/schedulers.py`)
- `CosineDecayScheduler(max_val, min_val, total_steps, warmup_steps)` — LR cosine decay con warmup
- `CosineEMAScheduler(ema_base, ema_end, total_steps)` — EMA momentum crescente cosine

---

## COME ADATTARE UN MODELLO (istruzioni per Claude)

Quando porti un modello da gml-test a Puzzle-Graph, segui questo processo:

### 1. Leggi prima, scrivi poi
Leggi **entrambe** le versioni (gml-test e Puzzle-Graph) e lista le differenze prima di
modificare qualsiasi file. Identifica cosa manca, cosa è sbagliato, cosa è migliorato.

### 2. Usa la firma pubblica uniforme
**Tutti i modelli usano `__init__(config: Dict, in_channels: int)`, senza eccezioni.**
(Supervised aggiunge `num_classes: int` come terzo argomento.)

Per ogni nuovo modello:
1. Crea un dataclass `ModelNameConfig` in `src/config/schema.py` con `__post_init__` validation
   e `from_dict(cls, d: dict)` classmethod. Segui la struttura degli altri config dataclass.
2. Esponi il config in `src/config/__init__.py`.
3. Il costruttore del modello fa solo: `cfg = ModelNameConfig.from_dict(config)` e usa `cfg.*`.
4. Usa `cfg.encoder.build(in_channels)` per istanziare l'encoder — non costruirlo inline.
5. `graph_level` si deriva sempre da `cfg.encoder.pool`, mai come parametro separato.

### 3. Non perdere nessuna feature di gml-test
Controlla punto per punto:
- **EMA scheduling**: gml-test usa `CosineDecayScheduler` per tau. Puzzle-Graph usa
  `CosineEMAScheduler`. Assicurati che sia presente e collegato in `post_step()`.
- **graph_level vs node-level**: gml-test usa `is_graph_level(self)` (trainer-dipendente).
  Puzzle-Graph usa `self.graph_level = cfg.encoder.pool` — derivato dal config, non passato esternamente.
- **Mini-batch: crop degli embedding** a `[:batch_size]` quando `not graph_level`.
- **Protected nodes**: passati a `compose()` per non corrompere i seed nodes in mini-batch.
- **Ordine argomenti delle loss**: verificare sempre contro la firma della loss class.
  Es: `CosineRegressionLoss(p1, t1, p2, t2)` = `2 - cos(p1,t2) - cos(p2,t1)`.

### 4. Aggiungi miglioramenti fondati
Aggiungi solo miglioramenti chiari e giustificabili:
- Se gml-test aveva un'asimmetria errata (es. EMA tau contato in batches invece di epoche),
  correggila e documentala nel CLAUDE.md.
- Se una feature mancava in gml-test ma è logicamente necessaria (es. protected_nodes in
  BGRL), aggiungila — ma documenta che è un'aggiunta rispetto a gml-test.

### 5. Dipendenze: controlla sempre tutti i moduli collegati
Per ogni modello modificato, verifica e aggiorna se necessario:
- `losses/` — la loss è corretta e testabile standalone?
- `utils/ema.py`, `utils/schedulers.py` — EMA e scheduler sono usati correttamente?
- `augmentation/` — `compose()` passa `protected_nodes`? La funzione augment accetta il param?
- `nn/` — MLP, head, pooling usati con le dimensioni giuste?
- `config/schema.py` — aggiungere config dataclass per ogni nuovo modello (tutti sono config-driven).
- `augmentation/__init__.py`, `config/__init__.py` — export aggiornati?

### 6. Aggiorna il CLAUDE.md per ogni modello toccato
Descrivi con precisione:
- Il dataclass config usato e i suoi campi principali
- La formula esatta della loss
- I dettagli dell'EMA (fisso vs schedulato, da dove a dove)
- Differenze intenzionali rispetto al paper o a gml-test (con motivazione)
- Iperparametri di default

---

## TEST

```bash
pytest tests/ -v
```

Un file per modello, tutti in `tests/`:

| File | Modello | Note |
|---|---|---|
| `test_bgrl.py` | BGRL | teacher frozen, reset→pesi diversi, EMA scheduler, step counter |
| `test_dgi.py` | DGI | W learnable, shuffle_nodes/shuffle_edges, ratio parziale |
| `test_graphcl.py` | GraphCL | projector dim, NT-Xent ≥ 0 |
| `test_vicreg.py` | VICReg | projector 3-layer, loss ≥ 0 |
| `test_barlow_twins.py` | BarlowTwins | lambda default=1/proj_dim, 3-layer projector |
| `test_afgrl.py` | AFGRL | **skip automatico se faiss non installato**; target=online all'init |
| `test_supervised.py` | Supervised | head dim, mini-batch crop |
| `test_graphdino.py` | GraphDINO | freeze last layer, teacher temp warmup, DINOTrainer hooks |

Tutti i test usano l'encoder **GIN**. `EncoderConfig.build()` usa `inspect.signature`
per filtrare i kwargs per ogni encoder, quindi GCN e Transformer sono pienamente
compatibili anche se non accettano tutti i campi di `EncoderConfig` (es. `num_layers`).

---

## PRINCIPI ARCHITETTURALI

1. **Separazione a strati**: `core → encoders/models/nn/augmentation/losses → evaluation → training → data`

2. **Config-driven uniform API**: tutti i modelli `__init__(config: Dict, in_channels: int)`.
   Il config viene validato da un dataclass dedicato in `schema.py`. L'encoder è sempre
   costruito da `cfg.encoder.build(in_channels)` via il registry — nessuna dipendenza inline.

3. **Loss come `nn.Module` standalone** in `losses/`, testabili indipendentemente.

4. **Registry pattern**: `@ENCODERS.register("gcn")` per componenti custom.

5. **Callback al posto di `on_train_epoch_end`**: `EmbeddingLoggerCallback`, `LinearEvalCallback`,
   `VisualizationCallback` invocati dal Trainer.

6. Il Trainer (`training/trainer.py`) è semplice: loop, device, checkpoint, callback.
   Loop per batch:
   ```
   compute_loss → backward → clip_grad → post_backward → optimizer.step → post_step
   ```
   Loop per epoca:
   ```
   on_epoch_start → batches → on_epoch_end
   ```

---

## STRUTTURA

```
src/
├── core/          ← ABC/Protocol: BaseModel, BaseSSLModel, Callback, Registry
├── encoders/      ← GCN, GIN, Transformer (auto-registrati)
├── models/        ← DGI, GraphCL, BGRL, AFGRL, GraphDINO, VICReg, BarlowTwins, Supervised
├── nn/            ← MLP, DINOHead, norm (weight_standardize), pooling
├── augmentation/  ← functional.py, transforms.py, compose.py
├── losses/        ← nt_xent, dino, vicreg, barlow, regression
├── evaluation/    ← linear_probe, knn, visualization (extract_embeddings)
├── training/      ← trainer.py (DINOTrainer), callbacks.py
├── data/          ← DataModule puro Python
├── utils/         ← ema.py, schedulers.py, positive_miner.py
└── config/        ← schema.py (dataclass validate), load.py
```

---

## NOTE IMPLEMENTATIVE

### graph-level vs node-level senza trainer
`self.graph_level` è derivato da `cfg.encoder.pool` nel costruttore — mai passato come parametro
separato. Nessun modello lo inferisce runtime dal batch.

### Mini-batch: protected nodes
In mini-batch node training, `batch.batch_size` indica quanti nodi sono i seed nodes.
La loss va calcolata solo su `z[:batch.batch_size]`.
`compose()` accetta `protected_nodes=torch.arange(batch.batch_size)` e lo propaga a ogni
augment; `node_drop` usa questo parametro per non rimuovere mai i seed nodes.

### BGRL vs AFGRL: inizializzazione del target encoder
- **BGRL**: `deepcopy(encoder)` + `reset_parameters()` — pesi DIVERSI da online (crucial, App. B paper)
- **AFGRL**: `deepcopy(encoder)` senza reset — pesi IDENTICI a online all'inizio
Questa differenza è intenzionale: BGRL ha bisogno dell'asimmetria per non collassare;
AFGRL parte già allineato perché la diversity viene dal miner, non dall'asimmetria dei pesi.

### GraphDINO: ordine delle operazioni nel trainer
```
forward → loss → backward → clip_grad → post_backward (freeze last layer)
→ optimizer.step → post_step (EMA teacher + center update)
```
L'EMA va aggiornata DOPO `optimizer.step`, non prima.

### LogRegEvaluator: multilabel
Per ogbg-molpcba le label sono multilabel float `[N, 128]` con NaN.
Usa average precision score invece di accuracy.

### extract_embeddings: global_to_local mapping
In mini-batch, i batch del NeighborLoader arrivano in ordine diverso dai node_ids originali.
Si usa una mappa `global_to_local[node_id] = posizione in z_cpu` per ricostruire l'ordine.

### pyproject.toml optional dependencies
```toml
[project.optional-dependencies]
viz  = ["umap-learn", "matplotlib", "seaborn"]
full = ["puzzle-graph[viz]", "faiss-cpu"]
```
