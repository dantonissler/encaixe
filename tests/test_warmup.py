"""Aquecimento na partida da API (D4 da Task 15)."""

from __future__ import annotations

import json

import httpx
import jinja2
import pytest
import respx
from structlog.testing import capture_logs

from encaixe.core.config import (
    AppConfig,
    GuardrailsConfig,
    LoadedApp,
    ModelsConfig,
    RagConfig,
)
from encaixe.guardrails.injection import InjectionGuard
from encaixe.output.schemas import RespostaBase
from encaixe.rag.rerank import CrossEncoderReranker
from encaixe.settings import Settings
from encaixe.warmup import plan_warmup, warm_up

BASE = "http://ollama.test"


def make_app(
    app_id: str = "a",
    *,
    primary: str = "ollama:qwen3.5:4b",
    fallbacks: list[str] | None = None,
    rag: bool = True,
    input_guards: list[str] | None = None,
    output_guards: list[str] | None = None,
) -> LoadedApp:
    cfg = AppConfig(
        id=app_id,
        models=ModelsConfig(primary=primary, fallbacks=fallbacks or []),
        rag=RagConfig(enabled=rag),
        guardrails=GuardrailsConfig(
            input=input_guards if input_guards is not None else ["prompt_injection"],
            output=output_guards if output_guards is not None else ["content_safety"],
        ),
    )
    return LoadedApp(cfg, None, jinja2.Template("s"), "v", RespostaBase)  # type: ignore[arg-type]


def settings(**kw) -> Settings:
    return Settings(ollama_base_url=BASE, **kw)


def test_plan_covers_primary_guard_embedder_and_reranker_not_fallbacks():
    app = make_app(fallbacks=["ollama:llama3.2-3b-cpu"])
    plan = plan_warmup(settings(), {"a": app})
    # o reserva fica de fora: carregá-lo custa ~2,5 GiB de RAM (D5)
    assert plan.chat_models == ["qwen3.5:4b", "llama-guard3-1b-cpu"]
    assert plan.embed_models == ["bge-m3-cpu"]
    assert plan.reranker is True
    assert plan.injection_model is False  # backend heurístico (padrão)


def test_plan_skips_what_no_app_uses():
    app = make_app(primary="test:x", rag=False, input_guards=[], output_guards=["pii_leak"])
    plan = plan_warmup(settings(injection_backend="model"), {"a": app})
    assert plan.chat_models == [] and plan.embed_models == []
    assert plan.reranker is False and plan.injection_model is False


def test_plan_loads_injection_model_only_with_model_backend_in_use():
    app = make_app(input_guards=["prompt_injection"])
    assert plan_warmup(settings(injection_backend="model"), {"a": app}).injection_model is True


def test_plan_dedupes_models_across_apps():
    apps = {"a": make_app("a"), "b": make_app("b", input_guards=["content_safety"])}
    plan = plan_warmup(settings(), apps)
    assert plan.chat_models == ["qwen3.5:4b", "llama-guard3-1b-cpu"]


class FakeDeps:
    def __init__(self, apps, cfg, reranker=None, injection=None):
        self.settings = cfg
        self.apps = apps
        self.reranker = reranker
        self.injection = injection


class FakeWarm:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    async def warm_up(self) -> None:
        self.calls += 1
        if self.fail:
            raise OSError("pesos ausentes")


@respx.mock
async def test_warm_up_sends_one_tiny_request_per_model_and_loads_reranker():
    chat = respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "ok"}})
    )
    embed = respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.0]]})
    )
    reranker = FakeWarm()
    deps = FakeDeps({"a": make_app()}, settings(), reranker=reranker, injection=FakeWarm())
    results = await warm_up(deps)  # type: ignore[arg-type]
    assert results == {
        "reranker": "ok",
        "ollama:qwen3.5:4b": "ok",
        "ollama:llama-guard3-1b-cpu": "ok",
        "ollama:bge-m3-cpu": "ok",
    }
    assert reranker.calls == 1
    sent = [json.loads(c.request.content) for c in chat.calls]
    assert [b["model"] for b in sent] == ["qwen3.5:4b", "llama-guard3-1b-cpu"]
    assert all(b["options"]["num_predict"] == 1 and b["stream"] is False for b in sent)
    assert json.loads(embed.calls[0].request.content)["model"] == "bge-m3-cpu"


@respx.mock
async def test_ollama_down_does_not_fail_startup_and_is_logged():
    respx.post(f"{BASE}/api/chat").mock(side_effect=httpx.ConnectError("recusada"))
    respx.post(f"{BASE}/api/embed").mock(side_effect=httpx.ConnectError("recusada"))
    reranker = FakeWarm()
    deps = FakeDeps({"a": make_app()}, settings(), reranker=reranker)
    with capture_logs() as logs:
        results = await warm_up(deps)  # type: ignore[arg-type]
    assert results["reranker"] == "ok"
    assert results["ollama:qwen3.5:4b"] == "falhou: ConnectError"
    assert results["ollama:bge-m3-cpu"] == "falhou: ConnectError"
    failed = [e for e in logs if e["event"] == "warmup_failed"]
    assert {e["component"] for e in failed} == {
        "ollama:qwen3.5:4b",
        "ollama:llama-guard3-1b-cpu",
        "ollama:bge-m3-cpu",
    }
    assert all(e["log_level"] == "warning" for e in failed)


@respx.mock
async def test_reranker_failure_does_not_stop_ollama_warm_up():
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, json={}))
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(200, json={}))
    deps = FakeDeps({"a": make_app()}, settings(), reranker=FakeWarm(fail=True))
    results = await warm_up(deps)  # type: ignore[arg-type]
    assert results["reranker"] == "falhou: OSError"
    assert results["ollama:qwen3.5:4b"] == "ok"


@respx.mock
async def test_injection_model_warmed_when_planned():
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, json={}))
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(200, json={}))
    injection = FakeWarm()
    cfg = settings(injection_backend="model")
    deps = FakeDeps({"a": make_app()}, cfg, reranker=FakeWarm(), injection=injection)
    results = await warm_up(deps)  # type: ignore[arg-type]
    assert results["injection"] == "ok" and injection.calls == 1


async def test_cross_encoder_warm_up_loads_once_and_predicts():
    loads: list[str] = []
    predicted: list = []

    class Model:
        def predict(self, pairs):
            predicted.append(pairs)
            return [0.5 for _ in pairs]

    def factory(model_id):
        loads.append(model_id)
        return Model()

    reranker = CrossEncoderReranker("m", model_factory=factory)
    await reranker.warm_up()
    await reranker.warm_up()
    assert loads == ["m"]
    assert len(predicted) == 2 and len(predicted[0]) == 1


@pytest.mark.parametrize("backend,loads", [("heuristic", 0), ("model", 1)])
async def test_injection_warm_up_only_loads_model_backend(backend, loads):
    created: list[str] = []

    def factory(model_id):
        created.append(model_id)
        return lambda texts, top_k=None: [[{"label": "SAFE", "score": 1.0}] for _ in texts]

    guard = InjectionGuard("m", 0.9, factory, backend=backend)
    await guard.warm_up()
    assert len(created) == loads
