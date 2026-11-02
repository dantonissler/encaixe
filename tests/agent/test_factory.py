from __future__ import annotations

from dataclasses import dataclass

import jinja2
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.toolsets import FunctionToolset

from encaixe.agent.factory import (
    build_agent,
    model_request_timeout,
    run_outcome_on_limit,
    usage_limits_for,
)
from encaixe.agent.loop_guard import LoopDetected
from encaixe.core.config import AgentConfig, AppConfig, ContextConfig, LoadedApp, ModelsConfig
from encaixe.messages import LIMIT_REACHED
from encaixe.output.schemas import RespostaBase


@dataclass
class StubDeps:
    pass


def make_app(fallbacks: list[str] | None = None, **agent) -> LoadedApp:
    cfg = AppConfig(
        id="a",
        models=ModelsConfig(primary="test", fallbacks=fallbacks or []),
        context=ContextConfig(num_ctx=4096),
        agent=AgentConfig(**agent),
    )
    return LoadedApp(
        config=cfg,
        root=None,  # type: ignore[arg-type]
        system_prompt=jinja2.Template("x"),
        prompt_version="v",
        output_model=RespostaBase,
    )


async def test_structured_output_validated():
    app = make_app()
    # NativeOutput: TestModel só aceita custom_output_args em modo tool; em nativo, usa texto JSON.
    model = TestModel(
        custom_output_text='{"resposta": "ok", "citacoes": []}',
        profile=ModelProfile(supports_json_schema_output=True),
    )
    agent = build_agent(app, model, [])
    result = await agent.run("oi", deps=StubDeps(), instructions="sys")
    assert result.output.resposta == "ok"
    assert isinstance(result.output, app.output_model)


def test_settings_and_limits():
    app = make_app(max_steps=2, max_tool_calls=5, timeout_s=33)
    agent = build_agent(app, TestModel(), [])
    ms = agent.model_settings
    # reasoning_effort "none": o Qwen3.5 pensa por padrão e, no /v1, gasta o orçamento em `reasoning`
    assert ms["extra_body"] == {"options": {"num_ctx": 4096}, "reasoning_effort": "none"}
    assert ms["timeout"] == 33
    lim = usage_limits_for(app)
    assert lim.request_limit == 2 and lim.tool_calls_limit == 5


async def test_usage_limit_produces_honest_message():
    app = make_app(max_steps=2)
    ts = FunctionToolset()

    @ts.tool_plain
    def buscar(q: str) -> str:
        return "r"

    n = {"i": 0}

    def fn(messages, info):
        n["i"] += 1
        return ModelResponse(parts=[ToolCallPart("buscar", {"q": str(n["i"])})])

    agent = build_agent(app, FunctionModel(fn), [ts])
    with pytest.raises(UsageLimitExceeded) as ei:
        await agent.run("oi", deps=StubDeps(), usage_limits=usage_limits_for(app))
    out = run_outcome_on_limit(ei.value)
    assert out.resposta.startswith(LIMIT_REACHED)
    assert out.citacoes == []


def test_outcome_lists_tools_and_handles_loop():
    out = run_outcome_on_limit(LoopDetected("buscar", 3), ["buscar", "ler"])
    assert out.resposta == f"{LIMIT_REACHED} Ferramentas usadas: buscar, ler."
    assert run_outcome_on_limit(LoopDetected("b", 3)).resposta == LIMIT_REACHED


@pytest.mark.parametrize(
    "timeout_s,fallbacks,expected",
    [
        (60, [], 60.0),  # sem reserva: o prazo inteiro
        (60, ["ollama:reserva"], 30.0),  # principal travado ainda deixa 30 s ao reserva
        (90, ["ollama:r1", "ollama:r2"], 30.0),
        (20, ["ollama:r1", "ollama:r2"], 10.0),  # piso de 10 s
    ],
)
def test_model_request_timeout_leaves_room_for_fallbacks(timeout_s, fallbacks, expected):
    app = make_app(fallbacks=fallbacks, timeout_s=timeout_s)
    assert model_request_timeout(app) == expected
    assert build_agent(app, TestModel(), []).model_settings["timeout"] == expected
