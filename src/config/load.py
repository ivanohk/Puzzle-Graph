from torch_geometric.loader import DataLoader, NeighborLoader
from registry.registry import Registry


LOADERS = Registry()

@LOADERS.register("graph")
def build_graph_loader(*, dataset, params):
    # Standard graph-level loader, batches whole graphs together.
    return DataLoader(
        dataset,
        batch_size=params.get("batch_size", 32),
        shuffle=params.get("shuffle", True),
        num_workers=params.get("num_workers", 0),
    )

@LOADERS.register("neighbor")
def build_neighbor_loader(*, dataset, params):
    # Samples neighbor subgraphs per node; useful for large single-graph datasets.
    return NeighborLoader(
        dataset,
        num_neighbors=params.get("num_neighbors", [10, 10]),
        batch_size=params.get("batch_size", 1024),
        shuffle=True,
    )
