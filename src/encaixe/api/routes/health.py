"""`GET /health` (processo vivo) e `GET /ready` (Postgres, Redis e modelos no Ollama)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Annotated, Any

import httpx
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from encaixe.api.auth import get_deps
from encaixe.wiring import Deps

log = structlog.get_logger(__name__)
router = APIRouter()

CHECK_TIMEOUT_S = 3.0


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def required_models(deps: Deps) -> list[str]:
    """Modelos Ollama da config: os das apps (principal e reservas), os guards de conteúdo
    (principal e secundário, se ligado) e o embedder."""
    s = deps.settings
    names = [s.safety_model, s.safety_secondary_model, s.embedding_model]
    for app in deps.apps.values():
        for spec in (app.config.models.primary, *app.config.models.fallbacks):
            provider, sep, name = spec.partition(":")
            if provider == "ollama" and sep and name:
                names.append(name)
    return list(dict.fromkeys(n for n in names if n))


def _canonical(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


async def _postgres_ok(deps: Deps) -> bool:
    if deps.pool is None:
        return False
    async with asyncio.timeout(CHECK_TIMEOUT_S):
        return await deps.pool.fetchval("SELECT 1") == 1


async def _redis_ok(deps: Deps) -> bool:
    if deps.redis is None:
        return False
    async with asyncio.timeout(CHECK_TIMEOUT_S):
        return bool(await deps.redis.ping())


async def _ollama_models(deps: Deps) -> set[str]:
    url = f"{deps.settings.ollama_base_url}/api/tags"
    async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_S) as client:
        response = await client.get(url)
        response.raise_for_status()
    present: set[str] = set()
    for model in response.json().get("models", []):
        present.update(_canonical(model[k]) for k in ("name", "model") if model.get(k))
    return present


async def _probe(name: str, check: Awaitable[bool]) -> bool:
    try:
        return bool(await check)
    except Exception as exc:  # noqa: BLE001 - dependência fora do ar = não pronto
        log.warning("ready_check_failed", dependency=name, error=type(exc).__name__)
        return False


@router.get("/ready")
async def ready(deps: Annotated[Deps, Depends(get_deps)]) -> JSONResponse:
    postgres, redis, models = await asyncio.gather(
        _probe("postgres", _postgres_ok(deps)),
        _probe("redis", _redis_ok(deps)),
        _models_or_none(deps),
    )
    missing = [name for name, ok in (("postgres", postgres), ("redis", redis)) if not ok]
    if models is None:
        missing.append("ollama")
    else:
        missing += [f"ollama:{m}" for m in required_models(deps) if _canonical(m) not in models]
    body: dict[str, Any] = {"status": "unavailable" if missing else "ready", "missing": missing}
    return JSONResponse(body, status_code=503 if missing else 200)


async def _models_or_none(deps: Deps) -> set[str] | None:
    try:
        return await _ollama_models(deps)
    except Exception as exc:  # noqa: BLE001
        log.warning("ready_check_failed", dependency="ollama", error=type(exc).__name__)
        return None
