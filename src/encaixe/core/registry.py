from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


class UnknownImplementationError(KeyError):
    def __init__(self, kind: str, name: str, available: list[str]):
        self.kind, self.name, self.available = kind, name, available
        super().__init__(
            f"Implementação desconhecida '{name}' para '{kind}'. "
            f"Disponíveis: {', '.join(available) or '(nenhuma)'}"
        )

    def __str__(self) -> str:
        return self.args[0]


def register(kind: str, name: str):
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY.setdefault(kind, {})[name] = factory
        return factory

    return deco


def create(kind: str, name: str, **kwargs: Any) -> Any:
    impls = _REGISTRY.get(kind, {})
    if name not in impls:
        raise UnknownImplementationError(kind, name, sorted(impls))
    return impls[name](**kwargs)
