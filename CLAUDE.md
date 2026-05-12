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

## ALGORITMI IMPLEMENTATI

Tutti i modelli sono `nn.Module` puri con solo `forward()` e `compute_loss()`.
Nessun accesso a trainer, logger, o datamodule dall'interno del modello.

### Supervised
- Encoder + linear head, CrossEntropyLoss
- Supporto full-batch e mini-batch

### DGI (Deep Graph Infomax)
- Discrimina embedding reale vs corrotto tramite un discriminatore con matrice W learnable
- Corruzione: shuffle dei nodi (permutazione di x) o shuffle delle edge destinations
- Summary globale `s = sigmoid(mean(h_pos))` per node-level, `global_mean_pool` per graph-level
- Discriminator: `pos_logits = (h_pos * W@s).sum(-1)`
- Loss: BCEWithLogitsLoss su [pos_logits, neg_logits] con label [1,1,...,0,0,...]
- Metrica aggiuntiva: discriminator accuracy

### GraphCL (Graph Contrastive Learning)
- NT-Xent loss su 2 augmented view
- Encoder + projection head (MLP)
- Iperparametri chiave: `tau=0.5`

### BGRL (Bootstrapped Graph Representation Learning)
- Teacher-student con EMA update sul target encoder
- Online encoder (student) + predictor head; target encoder (teacher, no grad)
- Loss: `2 - cosine_similarity(online_pred1, target_emb2) - cosine_similarity(online_pred2, target_emb1)`
- LR e momentum τ seguono `CosineDecayScheduler`
- EMA update: `param_k = mm * param_k + (1-mm) * param_q`
- Target encoder inizializzato con `reset_parameters()` (non copia identica)
- Iperparametri: `lr=0.0005, tau_base=0.99, warmup_steps=30, max_steps=300`

### AFGRL (Augmentation-Free Graph Representation Learning)
- Come BGRL ma senza augmentation strutturale
- I positive pair vengono trovati tramite FAISS k-means + top-k similarity
- `PositiveMiner`: clustering k-means → per ogni nodo trova i top-k vicini nello
  spazio embedding del teacher che appartengono allo stesso cluster
- Loss: regression cosine similarity sulle coppie positive minate
- Iperparametri: `lr=0.01, topk=5, num_centroids=50, num_kmeans=4, clus_num_iters=20`

### GraphDINO
- Adattamento di DINO (ViT) ai grafi
- Student encoder + student head; teacher encoder + teacher head (EMA, no grad)
- 2 view per batch (stessa aug list); cross-entropy simmetrica tra teacher e student
- **DINOLoss**: `−0.5 * ((t2 * s1).sum(−1).mean() + (t1 * s2).sum(−1).mean())`
  - Student output: `log_softmax(logits / student_temp)`
  - Teacher output: `softmax((logits − center) / teacher_temp_eff)`
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
- Iperparametri: `student_temp=0.1, teacher_temp=0.07, warmup_teacher_temp=0.04,
  warmup_teacher_temp_epochs=30, ema_tau=0.996, freeze_last_layer_epochs=1`

### VICReg
- Loss tripla: invariance (MSE), variance (hinge su std), covariance (off-diagonal)
- Coefficienti: `invariance=25.0, variance=25.0, covariance=1.0`

### Barlow Twins
- Cross-correlation matrix `C = (z1_norm.T @ z2_norm) / N`
- Loss: `sum((1−C_ii)²) + lambda * sum_{i≠j}(C_ij²)`
- `lambda_param = 1/out_dim`

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
- `functional.py`: funzioni pure `(data, **kwargs) → Data`
- `transforms.py`: classi callable che wrappano le funzioni
- `compose.py`: `compose(data, aug_list)` applica la lista in sequenza

Operazioni disponibili e parametri (tutte le probabilità sono randomizzate `uniform(0, p)`):

1. `edge_drop(data, p)` — rimuove edge con probabilità p
2. `feat_mask(data, p)` — maschera features dei nodi (setta a −1) con probabilità p
3. `feat_noise(data, std)` — aggiunge rumore gaussiano alle features
4. `subgraph(data, p)` — estrae un sottografo tramite k-hop
5. `node_drop(data, p)` — rimuove nodi con rimappatura edge_index

Il concetto di **protected_nodes** (seed nodes nei mini-batch) non è ancora implementato
nel sistema componibile. Vedere NOTE IMPLEMENTATIVE.

---

## UTILITY CHIAVE

### extract_embeddings (`evaluation/visualization.py`)
Estrae embedding da un modello dato un datamodule. Gestisce:
1. Graph-level: DataLoader + `global_mean_pool`
2. Node full-batch: forward sull'intero grafo, slicing per split
3. Node mini-batch: NeighborLoader con `global_to_local` mapping per ricostruire l'ordine

### LogRegEvaluator (`evaluation/linear_probe.py`)
Evaluator lineare PyTorch puro. Per multilabel (es. ogbg-molpcba) usa BCE + average precision.

### EMA (`utils/ema.py`)
`update_ema_params(student, teacher, tau)` — aggiorna parametri e buffer (es. BatchNorm running stats).

### pool_graph_embeddings / loss_inputs_from_embeddings (`nn/pooling.py`)
- `pool_graph_embeddings`: `global_mean_pool` con fallback se `batch=None`
- `loss_inputs_from_embeddings`: taglia `z[:batch.batch_size]` per mini-batch node-level

### Schedulers (`utils/schedulers.py`)
- `CosineDecayScheduler(max_val, min_val, total_steps, warmup_steps)` — LR cosine decay con warmup
- `CosineEMAScheduler(ema_base, ema_end, total_steps)` — EMA momentum crescente cosine

---

## PRINCIPI ARCHITETTURALI

1. **Separazione a strati**: `core → encoders/models/nn/augmentation/losses → evaluation → training → data`

2. **Dependency injection**: i modelli ricevono encoder, augmenter, loss come parametri.

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
I modelli ricevono `graph_level` come parametro del costruttore o lo inferiscono
dal batch (se `batch.batch` ha più grafi → graph-level).

### Mini-batch: protected nodes
In mini-batch node training, `batch.batch_size` indica quanti nodi sono i seed nodes.
La loss va calcolata solo su `z[:batch.batch_size]`.
L'augmenter dovrebbe proteggere `torch.arange(batch.batch_size)` dalla corruzione —
questo non è ancora implementato nel sistema componibile.

### BGRL: target encoder inizializzato diversamente
Il target encoder NON è una copia dell'online encoder: viene deepcopy-ato e poi viene
chiamato `reset_parameters()`. Questo è cruciale per la convergenza.

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
