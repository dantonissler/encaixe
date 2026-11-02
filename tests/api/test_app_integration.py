"""`create_app()` de produção: o lifespan monta as dependências reais e as fecha no fim."""

from __future__ import annotations

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider

from encaixe.api import app as app_module
from encaixe.db.repos.keys import create_key

pytestmark = pytest.mark.integration


async def test_lifespan_builds_and_closes_real_deps(settings, pg_pool, monkeypatch):
    calls: list[str] = []
    # Tracing/logging globais ficam de fora (não exporta spans de teste para o Phoenix).
    monkeypatch.setattr(
        app_module, "setup_tracing", lambda s: calls.append("tracing") or TracerProvider()
    )
    monkeypatch.setattr(app_module, "setup_logging", lambda s: calls.append("logging"))

    async def warm_up(deps):  # sem carregar modelos no teste (tests/api/test_lifespan_warmup.py)
        calls.append("warm_up")
        return {}

    monkeypatch.setattr(app_module, "warm_up", warm_up)
    key = await create_key(pg_pool, "exemplo")
    api = app_module.create_app(settings=settings)
    async with api.router.lifespan_context(api):
        deps = api.state.deps
        assert "exemplo" in deps.apps and deps.gate is not None
        transport = httpx.ASGITransport(app=api)
        headers = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", headers=headers
        ) as c:
            models = await c.get("/v1/models")
            invalid = await c.get("/v1/models", headers={"Authorization": "Bearer enc_x"})
    assert sorted(calls) == ["logging", "tracing", "warm_up"]
    assert [m["id"] for m in models.json()["data"]] == ["exemplo"]
    assert invalid.status_code == 401
    assert deps.pool.is_closing()  # close_deps rodou no desligamento
