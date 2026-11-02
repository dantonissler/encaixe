from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel

from encaixe.llm.breaker import BreakerRegistry, CircuitBreaker
from encaixe.llm.models import build_model
from encaixe.settings import Settings


class BreakerModel(WrapperModel):
    """Delega request/request_stream e registra sucesso/falha no breaker."""

    def __init__(self, wrapped: Model, breaker: CircuitBreaker) -> None:
        super().__init__(wrapped)
        self.breaker = breaker

    async def request(self, messages, model_settings, model_request_parameters):
        self.breaker.try_acquire()
        try:
            response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        except Exception:
            self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return response

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        self.breaker.try_acquire()
        entered = False
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                entered = True
                try:
                    yield stream
                except (ModelAPIError, httpx.TransportError):
                    self.breaker.record_failure()
                    raise
        except Exception:
            if not entered:
                self.breaker.record_failure()
            raise
        self.breaker.record_success()


def wrap_with_breaker(model: Model, spec: str, breakers: BreakerRegistry) -> Model:
    return BreakerModel(model, breakers.get(spec))


def build_chain(
    primary: str, fallbacks: list[str], settings: Settings, breakers: BreakerRegistry
) -> tuple[Model, list[str]]:
    specs = [primary, *fallbacks]
    effective = [s for s in specs if breakers.get(s).available()]
    if not effective:
        effective = [specs[-1]]  # nunca devolve cadeia vazia
    models = [wrap_with_breaker(build_model(s, settings), s, breakers) for s in effective]
    if len(models) == 1:
        return models[0], effective
    chain = FallbackModel(*models, fallback_on=(ModelAPIError, httpx.TransportError))
    return chain, effective
