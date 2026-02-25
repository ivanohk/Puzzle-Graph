from torch_geometric.loader import DataLoader, NeighborLoader
from registry.registry import Registry


LOADERS = Registry() #creao registry for data loaders

#gestiamo il loader "graph level"
@LOADERS.register("graph")
def build_graph_loader(*, dataset, params):
    return DataLoader(
        dataset,
        batch_size=params.get("batch_size", 32),
        shuffle=params.get("shuffle", True),
        num_workers=params.get("num_workers", 0),
    )
#DataLoader classico che ritorna batch (con concatenazione di nodi e edge) 

#gestiamo loader per sottografi (grafo singolo con molti nodi)
@LOADERS.register("neighbor")
def build_neighbor_loader(*, dataset, params):
    return NeighborLoader(
        dataset,
        num_neighbors=params.get("num_neighbors", [10, 10]),
        batch_size=params.get("batch_size", 1024),
        shuffle=True,
    )

#questo si comporta in modo diverso: campiona batch_size nodi e per ogni nodo prende 10 neighbor, poi 10 neighbor dei neighbor
#restituisce un subrafo

