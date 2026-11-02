from __future__ import annotations

from dataclasses import dataclass

import jinja2
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
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
from encaixe.output.schemas import OUTPUT_TOOL, TEXT_ANSWER_RETRY, RespostaBase


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
    # Saída por ferramenta (ToolOutput): o modelo entrega a resposta chamando `responder`.
    model = TestModel(custom_output_args={"resposta": "ok", "citacoes": []})
    agent = build_agent(app, model, [])
    result = await agent.run("oi", deps=StubDeps(), instructions="sys")
    assert result.output.resposta == "ok"
    assert isinstance(result.output, app.output_model)


async def test_output_is_a_tool_so_the_model_can_still_call_tools():
    """Com saída nativa (gramática do Ollama) o modelo nunca chamava ferramentas (D1 da
    Task 15). A saída precisa ser uma ferramenta, oferecida junto com as da app."""
    app = make_app()
    ts = FunctionToolset()

    @ts.tool_plain
    def listar_tarefas() -> str:
        return "tarefa a"

    seen: list[dict] = []

    def fn(messages, info):
        params = info.model_request_parameters
        seen.append(
            {
                "tools": sorted(t.name for t in info.function_tools),
                "outputs": [t.name for t in info.output_tools],
                # nada de schema nativo (`response_format`): no Ollama ele vira gramática
                "native_schema": params.output_object is not None,
                "mode": params.output_mode,
            }
        )
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart("listar_tarefas", {})])
        args = {"resposta": "Você tem: tarefa a", "citacoes": []}
        return ModelResponse(parts=[ToolCallPart(OUTPUT_TOOL, args)])

    agent = build_agent(app, FunctionModel(fn), [ts])
    result = await agent.run("liste minhas tarefas", deps=StubDeps())
    assert result.output.resposta == "Você tem: tarefa a"
    assert seen[0] == {
        "tools": ["listar_tarefas"],
        "outputs": [OUTPUT_TOOL],
        "native_schema": False,
        "mode": "tool",
    }
    assert len(seen) == 2


async def test_text_answer_wrapped_without_citations_when_allowed():
    """Depois de uma ferramenta MCP (ex.: aprovação pendente) o Qwen responde em texto puro em
    ~7 de 9 vezes; forçado a repetir pela ferramenta de saída, às vezes inventava "Tarefa
    criada com sucesso". Quando o turno não precisa de citação, o texto vale como resposta."""
    app = make_app()

    def fn(messages, info):
        return ModelResponse(parts=[TextPart("Confirma a criação? Responda sim ou não.")])

    agent = build_agent(app, FunctionModel(fn), [], accept_text=lambda: True)
    result = await agent.run("crie a tarefa", deps=StubDeps())
    assert isinstance(result.output, app.output_model)
    assert result.output.resposta == "Confirma a criação? Responda sim ou não."
    assert result.output.citacoes == []


async def test_text_answer_retried_in_portuguese_when_citations_may_be_needed():
    app = make_app()
    calls: list[list] = []

    def fn(messages, info):
        calls.append(list(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart("São 30 dias.")])
        args = {"resposta": "São 30 dias.", "citacoes": ["T1"]}
        return ModelResponse(parts=[ToolCallPart(OUTPUT_TOOL, args)])

    agent = build_agent(app, FunctionModel(fn), [])  # padrão: texto puro não vale
    result = await agent.run("férias?", deps=StubDeps())
    assert result.output.citacoes == ["T1"]
    assert len(calls) == 2
    feedback = calls[1][-1].parts[-1]  # retorno de validação que o modelo recebe
    assert TEXT_ANSWER_RETRY in str(feedback.content)


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


@pytest.mark.parametrize(
    "leaked",
    [
        'Você tem 2 tarefas.\n<tool_use>\n<parameter=tool_name>\nresponder\n</parameter>',
        'Pronto.\n<tool_call>\n{"name": "responder", "arguments": {}}\n</tool_call>',
        "Ok. <function=listar_tarefas></function>",
    ],
)
async def test_tool_call_markup_in_text_is_retried_even_when_text_allowed(leaked):
    """Visto ao vivo: o Qwen escreveu a chamada de `responder` como texto (formato que o Ollama
    não reconheceu). Isso não é resposta: volta ao modelo como nova tentativa."""
    app = make_app()
    calls: list[list] = []

    def fn(messages, info):
        calls.append(list(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart(leaked)])
        return ModelResponse(parts=[ToolCallPart(OUTPUT_TOOL, {"resposta": "Ok.", "citacoes": []})])

    agent = build_agent(app, FunctionModel(fn), [], accept_text=lambda: True)
    result = await agent.run("x", deps=StubDeps())
    assert len(calls) == 2
    assert result.output.resposta == "Ok."
