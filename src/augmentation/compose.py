from torch_geometric.data import Data

from src.registry import AUGMENTS


def compose(data: Data, augments: list):
    for name, kwargs in augments:
        data = AUGMENTS.build(name, data=data, **kwargs)
    return data