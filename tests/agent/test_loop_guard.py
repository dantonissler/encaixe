from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from encaixe.agent.loop_guard import LoopDetected, LoopGuard, wrap_toolsets


def make_toolset(calls: list | None = None) -> FunctionToolset:
    ts = FunctionToolset()

    @ts.tool_plain
    def buscar(q: str) -> str:
        if calls is not None:
            calls.append(q)
        return f"res {q}"

    return ts


def caller(args_seq):
    state = {"i": 0}

    def fn(messages, info):
        i = state["i"]
        state["i"] += 1
        if i >= len(args_seq):
            return ModelResponse(parts=[TextPart("pronto")])
        return ModelResponse(parts=[ToolCallPart("buscar", args_seq[i])])

    return FunctionModel(fn)


async def test_detects_repeated_identical_calls():
    calls: list = []
    agent = Agent(caller([{"q": "a"}] * 5), toolsets=wrap_toolsets([make_toolset(calls)], max_repeats=3))
    with pytest.raises(LoopDetected) as ei:
        await agent.run("oi")
    assert ei.value.tool_name == "buscar"
    assert len(calls) == 2  # a 3a chamada idêntica é barrada antes de executar


async def test_counts_do_not_leak_across_runs():
    calls: list = []
    ts = make_toolset(calls)
    for _ in range(2):  # guards novos por execução: 2 chamadas idênticas por run, sem loop
        agent = Agent(caller([{"q": "a"}] * 2), toolsets=wrap_toolsets([ts], max_repeats=3))
        result = await agent.run("oi")
        assert result.output
    assert len(calls) == 4


async def test_different_args_not_a_loop():
    agent = Agent(
        caller([{"q": "a"}, {"q": "b"}, {"q": "c"}, {"q": "d"}]),
        toolsets=[LoopGuard(make_toolset(), max_repeats=3)],
    )
    result = await agent.run("oi")
    assert result.output


async def test_arg_key_order_is_canonical_and_counts_are_per_instance():
    g1 = LoopGuard(make_toolset(), max_repeats=2)
    g2 = LoopGuard(make_toolset(), max_repeats=2)
    assert g1._key("t", {"a": 1, "b": 2}) == g1._key("t", {"b": 2, "a": 1})
    assert g1 is not g2 and g1._counts is not g2._counts
