"""Aquecimento na partida da API: carrega os modelos antes da primeira requisição.

Sem isso a 1ª pergunta depois de subir a API e o Ollama levava ~25 s (reranker 11,5 s, Qwen
7,4 s, Llama Guard 3,5 s, bge-m3 1,8 s; D4 da Task 15). Carrega no processo o reranker (e o
classificador de injeção, se o backend `model` estiver em uso) e manda uma requisição mínima
a cada modelo do Ollama que alguma app usa: o principal, o guard de conteúdo e o embedder.
O reserva fica de fora (custa ~2,5 GiB de RAM). Nunca derruba a partida: falha vira log.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from encaixe.core.config import LoadedApp
from encaixe.settings import Settings

if TYPE_CHECKING:
    from encaixe.wiring import Deps

log = structlog.get_logger(__name__)

# A carga a frio do Qwen na GPU com a RAM no limite passou de 7 s; com swap, mais.
WARMUP_TIMEOUT_S = 120.0
PING = "oi"


@dataclass
class WarmupPlan:
    chat_models: list[str] = field(default_factory=list)
    embed_models: list[str] = field(default_factory=list)
    reranker: bool = False
    injection_model: bool = False


def _ollama_name(spec: str) -> str | None:
    provider, sep, name = spec.partition(":")
    return name if provider == "ollama" and sep and name else None


def plan_warmup(settings: Settings, apps: Mapping[str, LoadedApp]) -> WarmupPlan:
    """O que aquecer, a partir do que as apps usam de fato."""
    plan = WarmupPlan()
    chat: list[str] = []
    for app in apps.values():
        cfg = app.config
        if name := _ollama_name(cfg.models.primary):
            chat.append(name)
        guards = {*cfg.guardrails.input, *cfg.guardrails.output}
        if "content_safety" in guards:
            chat.append(settings.safety_model)
        if cfg.rag.enabled:
            plan.reranker = True
            plan.embed_models = [settings.embedding_model]
        if settings.injection_backend == "model" and "prompt_injection" in cfg.guardrails.input:
            plan.injection_model = True
    plan.chat_models = list(dict.fromkeys(chat))
    return plan


async def _chat(client: httpx.AsyncClient, base: str, model: str) -> None:
    resp = await client.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PING}],
            "stream": False,
            "options": {"num_predict": 1},
        },
    )
    resp.raise_for_status()


async def _embed(client: httpx.AsyncClient, base: str, model: str) -> None:
    resp = await client.post(f"{base}/api/embed", json={"model": model, "input": PING})
    resp.raise_for_status()


async def warm_up(deps: Deps, *, timeout_s: float = WARMUP_TIMEOUT_S) -> dict[str, str]:
    """Aquece o que as apps usam. Devolve `{componente: "ok" | "falhou: Erro"}`; nunca levanta."""
    plan = plan_warmup(deps.settings, deps.apps)
    results: dict[str, str] = {}

    async def step(component: str, work: Awaitable[Any]) -> None:
        started = time.perf_counter()
        try:
            await work
        except Exception as exc:  # noqa: BLE001 - partida segue; a 1ª requisição carrega
            results[component] = f"falhou: {type(exc).__name__}"
            log.warning("warmup_failed", component=component, error=type(exc).__name__)
            return
        results[component] = "ok"
        log.info(
            "warmup_done", component=component, seconds=round(time.perf_counter() - started, 2)
        )

    async def in_process() -> None:
        if plan.reranker and getattr(deps, "reranker", None) is not None:
            await step("reranker", deps.reranker.warm_up())
        if plan.injection_model and getattr(deps, "injection", None) is not None:
            await step("injection", deps.injection.warm_up())

    async def ollama() -> None:
        # Um modelo por vez: o Ollama roda com NUM_PARALLEL=1 e a RAM é curta.
        base = deps.settings.ollama_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for model in plan.chat_models:
                await step(f"ollama:{model}", _chat(client, base, model))
            for model in plan.embed_models:
                await step(f"ollama:{model}", _embed(client, base, model))

    await asyncio.gather(in_process(), ollama())
    return results
