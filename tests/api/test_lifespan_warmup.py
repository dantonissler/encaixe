"""O lifespan de produção aquece os modelos antes de aceitar requisições (D4 da Task 15)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider

from encaixe.api import app as app_module
from encaixe.settings import Settings


@pytest.fixture
def fake_runtime(monkeypatch):
    events: list[str] = []
    built = SimpleNamespace(apps={"exemplo": object()})

    async def build_deps(settings):
        events.append("build")
        return built

    async def close_deps(deps):
        events.append("close")

    async def warm_up(deps):
        assert deps is built
        events.append("warm_up")
        return {"ollama:qwen3.5:4b": "ok"}

    monkeypatch.setattr(app_module, "build_deps", build_deps)
    monkeypatch.setattr(app_module, "close_deps", close_deps)
    monkeypatch.setattr(app_module, "warm_up", warm_up)
    monkeypatch.setattr(app_module, "setup_tracing", lambda s: TracerProvider())
    monkeypatch.setattr(app_module, "setup_logging", lambda s: None)
    return events


async def test_lifespan_warms_up_before_serving(fake_runtime):
    api = app_module.create_app(settings=Settings())
    async with api.router.lifespan_context(api):
        assert fake_runtime == ["build", "warm_up"]
    assert fake_runtime == ["build", "warm_up", "close"]


async def test_warm_up_can_be_turned_off(fake_runtime):
    api = app_module.create_app(settings=Settings(warmup=False))
    async with api.router.lifespan_context(api):
        pass
    assert fake_runtime == ["build", "close"]
