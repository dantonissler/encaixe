from __future__ import annotations

import os
from pathlib import Path

import pytest

from encaixe import wiring
from encaixe.core.config import load_apps

WORKER_KEYS = {
    "pool",
    "embedder",
    "store",
    "injection",
    "chunker",
    "settings",
    "history_store",
    "apps",
}


def test_runtime_env_defaults(monkeypatch):
    for name in ("HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "PYDANTIC_AI_NO_BANNER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DO_NOT_TRACK", "0")  # setdefault: não sobrescreve escolha explícita
    wiring.set_runtime_env()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["PYDANTIC_AI_NO_BANNER"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "0"


def test_canaries_generated_once_per_app():
    apps = load_apps(Path("apps"))
    canaries = wiring.make_canaries(apps)
    assert set(canaries) == set(apps)
    assert all(c.startswith("cnr-") for c in canaries.values())


@pytest.mark.integration
async def test_build_and_close_deps(settings):
    deps = await wiring.build_deps(settings)
    try:
        assert "exemplo" in deps.apps and set(deps.canaries) == set(deps.apps)
        assert deps.gate.available == settings.max_concurrent_generations
        assert await deps.redis.ping()
        assert await deps.corpus_version("exemplo") >= 0
        _model, effective = deps.chain_factory("ollama:qwen3.5:4b", ["ollama:llama3.2-3b-cpu"])
        assert effective == ["ollama:qwen3.5:4b", "ollama:llama3.2-3b-cpu"]
        assert deps.approvals._injection is deps.injection
        assert deps.retriever._reranker is deps.reranker  # o aquecido é o usado nas buscas
    finally:
        await wiring.close_deps(deps)


@pytest.mark.integration
async def test_build_worker_deps(settings):
    ctx = await wiring.build_worker_deps(settings)
    try:
        assert WORKER_KEYS <= set(ctx)
        assert "redis" not in ctx  # o arq guarda o próprio pool em ctx["redis"]
        assert "exemplo" in ctx["apps"]
    finally:
        await wiring.close_worker_deps(ctx)


def test_worker_chunker_uses_docling_settings(tmp_path):
    from encaixe.settings import Settings

    chunker = wiring.make_chunker(Settings(docling_artifacts_path=tmp_path, docling_ocr=True))
    assert chunker.artifacts_path == tmp_path and chunker.do_ocr is True
