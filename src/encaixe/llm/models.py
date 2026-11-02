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
        # Sem novas tentativas no cliente OpenAI (padrão: 2). Cada tentativa espera o
        # timeout por requisição inteiro (ver agent.factory.model_request_timeout): um
        # principal travado comeria o prazo do reserva. Falha é tratada pela cadeia
        # (FallbackModel + breaker).
        provider.client.max_retries = 0
        return OllamaModel(name, provider=provider)
    raise ValueError(f"Especificação de modelo desconhecida: {spec!r}")
