from __future__ import annotations

from fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from encaixe.core.types import GuardResult
from encaixe.tools.approvals import ApprovalService
from encaixe.tools.mcp import build_mcp_toolsets
from encaixe.tools.native import AgentDeps

from .helpers import FakeApprovalRepo, make_ctx


def make_server(calls: list):
    s = FastMCP("t")

    @s.tool
    def listar_tarefas() -> str:
        """Lista tarefas."""
        calls.append("listar")
        return "tarefa a"

    @s.tool
    def criar_tarefa(titulo: str) -> str:
        """Cria tarefa."""
        calls.append(("criar", titulo))
        return "criada"

    @s.tool
    def apagar_tudo() -> str:
        """Perigosa."""
        calls.append("apagar")
        return "apagado"

    return s


class FakeInjection:
    def __init__(self, block_on: str | None = None):
        self.block_on = block_on

    async def check_text(self, text):
        if self.block_on and self.block_on in text:
            return GuardResult("injection", "block", "x")
        return GuardResult("injection", "pass")


def build(ctx, server, *, allow_write=True):
    return build_mcp_toolsets(ctx.app, ctx, allow_write=allow_write, source=lambda cfg, headers: server)


async def tool_names(toolsets, deps):
    agent = Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])), toolsets=toolsets, deps_type=AgentDeps)
    seen = {}

    def fn(messages, info):
        seen["names"] = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(fn), toolsets=toolsets, deps_type=AgentDeps)
    await agent.run("x", deps=deps)
    return seen["names"]


def mk_deps(ctx, repo=None, injection=None):
    svc = ApprovalService(repo or FakeApprovalRepo(), None)  # type: ignore[arg-type]
    return AgentDeps(ctx=ctx, retriever=None, injection=injection or FakeInjection(), approvals=svc, boundary="B")  # type: ignore[arg-type]


def calling_model(name, args):
    state = {"n": 0}

    def fn(messages, info):
        state["n"] += 1
        if state["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("fim")])

    return FunctionModel(fn)


def returns(res):
    return [p.content for m in res.all_messages() for p in m.parts if p.part_kind == "tool-return"]


async def test_allowlist_filters_tools():
    ctx = make_ctx(allow=["listar_tarefas", "criar_tarefa"], require_approval=[])
    names = await tool_names(build(ctx, make_server([])), mk_deps(ctx))
    assert names == ["criar_tarefa", "listar_tarefas"]


async def test_write_tool_hidden_without_identity():
    ctx = make_ctx()
    names = await tool_names(build(ctx, make_server([]), allow_write=False), mk_deps(ctx))
    assert names == ["listar_tarefas"]


async def test_require_approval_must_be_in_allow():
    ctx = make_ctx(allow=["listar_tarefas"], require_approval=["apagar_tudo"])
    names = await tool_names(build(ctx, make_server([])), mk_deps(ctx))
    assert names == ["listar_tarefas"]


async def test_write_tool_replaced_by_approval_request():
    calls: list = []
    ctx = make_ctx()
    repo = FakeApprovalRepo()
    deps = mk_deps(ctx, repo)
    agent = Agent(calling_model("criar_tarefa", {"titulo": "X"}), toolsets=build(ctx, make_server(calls)), deps_type=AgentDeps)
    res = await agent.run("x", deps=deps)
    assert calls == []
    assert len(repo.rows) == 1
    assert repo.rows[0]["args"] == {"titulo": "X"} and repo.rows[0]["tool"] == "criar_tarefa"
    assert repo.rows[0]["server"] == "tarefas" and repo.rows[0]["user_ref"] == "u1"
    out = returns(res)[0]
    assert "confirmação" in out and "criar_tarefa(titulo=X)" in out


async def test_read_tool_wrapped_and_recorded():
    calls: list = []
    ctx = make_ctx()
    agent = Agent(calling_model("listar_tarefas", {}), toolsets=build(ctx, make_server(calls)), deps_type=AgentDeps)
    res = await agent.run("x", deps=mk_deps(ctx))
    out = returns(res)[0]
    assert out.startswith("<<B tipo=ferramenta>>") and "tarefa a" in out and "listar_tarefas" in out
    assert ctx.tools_called == ["listar_tarefas"]


async def test_tool_result_injection_blocked():
    ctx = make_ctx()
    agent = Agent(calling_model("listar_tarefas", {}), toolsets=build(ctx, make_server([])), deps_type=AgentDeps)
    res = await agent.run("x", deps=mk_deps(ctx, injection=FakeInjection(block_on="tarefa a")))
    assert returns(res) == ["[resultado bloqueado por segurança]"]
