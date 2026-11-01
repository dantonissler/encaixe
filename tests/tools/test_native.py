from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from encaixe.core.registry import UnknownImplementationError
from encaixe.core.types import Chunk, ScoredChunk
from encaixe.tools.native import AgentDeps, build_native_toolset

from .helpers import make_ctx


class FakeRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, app_id, principals, query, top_k, rerank_top_n):
        self.calls.append((app_id, principals, query))
        return [
            ScoredChunk(Chunk("c1", "app1", "d", "Texto um", ("H",), ("app:*",), {"title": "Doc"}), 0.9),
            ScoredChunk(Chunk("c2", "app1", "d", "Texto dois", (), ("app:*",), {}), 0.8),
        ]


async def test_buscar_documentos_wraps_and_records_retrieved():
    ctx = make_ctx()
    retr = FakeRetriever()
    deps = AgentDeps(ctx=ctx, retriever=retr, injection=None, approvals=None, boundary="B1")  # type: ignore[arg-type]
    state = {"n": 0}

    def fn(messages, info):
        state["n"] += 1
        if state["n"] <= 2:  # duas buscas: não duplica ids
            return ModelResponse(parts=[ToolCallPart("buscar_documentos", {"consulta": f"q{state['n']}"})])
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(fn), toolsets=[build_native_toolset(["buscar_documentos"])], deps_type=AgentDeps)
    res = await agent.run("x", deps=deps)
    assert res.output == "ok"
    assert [s.chunk.id for s in ctx.retrieved] == ["c1", "c2"]
    assert retr.calls[0][0] == "app1" and retr.calls[0][1] is ctx.principals
    tool_return = next(
        p.content for m in res.all_messages() for p in m.parts if p.part_kind == "tool-return"
    )
    assert tool_return.startswith("<<B1 tipo=trechos>>") and tool_return.endswith("<<FIM B1>>")
    assert "Texto um" in tool_return


def test_unknown_native_tool():
    with pytest.raises(UnknownImplementationError):
        build_native_toolset(["nao_existe"])
