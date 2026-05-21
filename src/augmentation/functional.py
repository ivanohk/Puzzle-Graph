from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.registry import AUGMENTS


@AUGMENTS.register("edge_drop")
def edge_drop(data: Data, p=0.2, protected_nodes: Optional[Tensor] = None):
    assert data.edge_index is not None
    mask = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > p
    return Data(x=data.x, edge_index=data.edge_index[:, mask], batch=data.batch)


@AUGMENTS.register("edge_add")
def edge_add(data: Data, p=0.1, protected_nodes: Optional[Tensor] = None):
    assert data.edge_index is not None
    assert data.num_nodes is not None
    n = data.num_nodes
    n_add = max(1, int(data.edge_index.size(1) * p))
    src = torch.randint(0, n, (n_add,), device=data.edge_index.device)
    dst = torch.randint(0, n, (n_add,), device=data.edge_index.device)
    new_edges = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return Data(x=data.x, edge_index=edge_index, batch=data.batch)


@AUGMENTS.register("subgraph")
def subgraph(data: Data, num_hops=2, protected_nodes: Optional[Tensor] = None):
    assert data.edge_index is not None
    assert data.num_nodes is not None
    n = data.num_nodes
    seed = int(torch.randint(0, n, (1,), device=data.edge_index.device).item())
    node_idx, edge_index_sub, _, _ = k_hop_subgraph(
        node_idx=seed,
        num_hops=num_hops,
        edge_index=data.edge_index,
        num_nodes=n,
        relabel_nodes=True,
    )
    x_sub = data.x[node_idx] if data.x is not None else None
    batch_sub = data.batch[node_idx] if data.batch is not None else None
    return Data(x=x_sub, edge_index=edge_index_sub, batch=batch_sub)


@AUGMENTS.register("feat_mask")
def feat_mask(data: Data, p=0.2, protected_nodes: Optional[Tensor] = None):
    assert data.x is not None
    mask = (torch.rand(data.x.size(1), device=data.x.device) > p).float()
    return Data(x=data.x * mask, edge_index=data.edge_index, batch=data.batch)


@AUGMENTS.register("feat_noise")
def feat_noise(data: Data, std=0.1, protected_nodes: Optional[Tensor] = None):
    assert data.x is not None
    x = data.x + torch.randn_like(data.x) * std
    return Data(x=x, edge_index=data.edge_index, batch=data.batch)


@AUGMENTS.register("feat_shuffle")
def feat_shuffle(data: Data, p=0.1, protected_nodes: Optional[Tensor] = None):
    assert data.x is not None
    n = data.x.size(0)
    perm = torch.randperm(n, device=data.x.device)
    node_mask = torch.rand(n, device=data.x.device) < p
    x = data.x.clone()
    x[node_mask] = data.x[perm[node_mask]]
    return Data(x=x, edge_index=data.edge_index, batch=data.batch)


@AUGMENTS.register("node_drop")
def node_drop(data: Data, p=0.1, protected_nodes: Optional[Tensor] = None):
    """Drop nodes with probability p, remapping edge_index.

    Nodes in protected_nodes are never dropped (used to preserve mini-batch
    seed nodes in NeighborLoader training).
    """
    assert data.x is not None
    assert data.edge_index is not None
    n = data.num_nodes
    assert n is not None

    keep_mask = torch.rand(n, device=data.x.device) > p
    if protected_nodes is not None:
        keep_mask[protected_nodes] = True

    keep_idx = keep_mask.nonzero(as_tuple=True)[0]

    # Build a remapping table: old node id → new node id (-1 = dropped)
    new_idx = torch.full((n,), -1, dtype=torch.long, device=data.x.device)
    new_idx[keep_idx] = torch.arange(keep_idx.size(0), device=data.x.device)

    # Keep only edges where both endpoints survive
    src, dst = data.edge_index
    edge_mask = keep_mask[src] & keep_mask[dst]
    new_edge_index = new_idx[data.edge_index[:, edge_mask]]

    x_new = data.x[keep_idx]
    batch_new = data.batch[keep_idx] if data.batch is not None else None
    return Data(x=x_new, edge_index=new_edge_index, batch=batch_new)
