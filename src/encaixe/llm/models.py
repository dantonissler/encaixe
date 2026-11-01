from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.ollama import OllamaProvider

from encaixe.settings import Settings


def build_model(spec: str, settings: Settings) -> Model:
    """`ollama:<nome>` -> OllamaModel; `test` -> TestModel (apenas testes)."""
    if spec == "test":
        return TestModel()
    prefix, sep, name = spec.partition(":")
    if prefix == "ollama" and sep and name:
        provider = OllamaProvider(base_url=f"{settings.ollama_base_url}/v1")
        return OllamaModel(name, provider=provider)
    raise ValueError(f"Especificação de modelo desconhecida: {spec!r}")
