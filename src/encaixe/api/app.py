"""Fábrica da aplicação: `uvicorn encaixe.api.app:create_app --factory --port 8100`."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from encaixe.api.errors import install_error_handlers
from encaixe.api.routes import chat, conversations, documents, health, models
from encaixe.observability.logging import setup_logging
from encaixe.observability.tracing import setup_tracing
from encaixe.pipeline.stages import current_trace_id
from encaixe.settings import Settings, get_settings
from encaixe.warmup import warm_up
from encaixe.wiring import Deps, build_deps, close_deps, set_runtime_env

log = structlog.get_logger(__name__)

# A instrumentação OTel do FastAPI (contrib) já cria o span HTTP; a nativa fica desligada
# para não duplicar spans nem configurar exportadores pelo ambiente.
_NATIVE_TELEMETRY_OFF: dict[str, Any] = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
    "auto_configure": False,
}


class TraceIdMiddleware:
    """`X-Trace-Id` em toda resposta (spec §6.8), se a rota não tiver posto um."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if "x-trace-id" not in headers:
                    headers["X-Trace-Id"] = current_trace_id() or uuid.uuid4().hex
            await send(message)

        await self.app(scope, receive, send_with_trace)


def create_app(
    deps: Deps | None = None,
    *,
    settings: Settings | None = None,
    tracer_provider: Any = None,
) -> FastAPI:
    """Sem `deps` (produção), o lifespan liga tracing e logs, monta as dependências reais e as
    fecha no desligamento. Com `deps` prontas (testes), usa-as como estão e não as fecha."""
    set_runtime_env()  # antes de qualquer biblioteca da Hugging Face ser carregada
    cfg = settings or (deps.settings if deps is not None else get_settings())

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        if deps is not None:
            yield
            return
        setup_logging(cfg)
        provider = setup_tracing(cfg)
        built = await build_deps(cfg)  # o mesmo Settings: guards compartilhados com o pipeline
        api.state.deps = built
        if cfg.warmup:  # modelos carregados antes da 1ª requisição; Ollama fora não impede
            await warm_up(built)
        log.info("api_started", env=cfg.env, apps=sorted(built.apps))
        try:
            yield
        finally:
            await close_deps(built)
            provider.force_flush(timeout_millis=2000)
            log.info("api_stopped")

    api = FastAPI(title="encaixe", lifespan=lifespan, telemetry=_NATIVE_TELEMETRY_OFF)
    if deps is not None:
        api.state.deps = deps
    api.state.started_at = int(time.time())
    install_error_handlers(api)
    api.add_middleware(TraceIdMiddleware)
    for module in (chat, models, documents, conversations, health):
        api.include_router(module.router)
    # None → provider global (o lifespan o define via setup_tracing; o tracer é resolvido
    # no primeiro span). Sem `send`/`receive`: um span por chunk SSE seria só ruído.
    FastAPIInstrumentor.instrument_app(
        api, tracer_provider=tracer_provider, exclude_spans=["receive", "send"]
    )
    return api
