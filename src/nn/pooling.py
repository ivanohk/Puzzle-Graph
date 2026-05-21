"""Graph-level pooling utilities."""

from __future__ import annotations
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.nn import global_mean_pool


def pool_graph_embeddings(node_embeddings: Tensor, batch: Optional[Tensor]) -> Tensor:
    """Pool node embeddings to one embedding per graph.

    If batch is None (single graph), returns the mean of all nodes.
    """
    if batch is None:
        return node_embeddings.mean(dim=0, keepdim=True)
    return global_mean_pool(node_embeddings, batch)


def loss_inputs_from_embeddings(
    z1: Tensor, z2: Tensor, batch
) -> Tuple[Tensor, Tensor]:
    """Crop embeddings to seed nodes for mini-batch node training.

    In a NeighborLoader batch, batch.batch_size is the number of seed (target)
    nodes — the loss is computed only on those. For graph-level or full-batch,
    returns z1 and z2 unchanged.
    """
    if batch is not None and hasattr(batch, "batch_size"):
        bs = batch.batch_size
        return z1[:bs], z2[:bs]
    return z1, z2
