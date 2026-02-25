from __future__ import annotations
from typing import Callable, Dict, Any

class Registry:
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            if name in self._builders:
                raise KeyError(f"Duplicate registration: {name}")
            self._builders[name] = fn
            return fn
        return deco

    def build(self, name: str, **kwargs):
        if name not in self._builders:
            raise KeyError(f"Unknown component '{name}'. Available: {list(self._builders)}")
        return self._builders[name](**kwargs)

ENCODERS = Registry()
HEADS = Registry()
AUGMENTS = Registry()
OBJECTIVES = Registry()
DATASETS = Registry()