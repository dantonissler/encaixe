from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from encaixe.core.config import LoadedApp, McpServerConfig
from encaixe.core.types import RequestContext
from encaixe.messages import APPROVAL_PENDING_RESULT
from encaixe.observability.spans import traced
from encaixe.output.schemas import OUTPUT_TOOL
from encaixe.tools.approvals import identity_headers
from encaixe.tools.native import AgentDeps
from encaixe.tools.results import UNAVAILABLE, sanitize_tool_result

log = structlog.get_logger(__name__)

def _describe(tool: str, args: dict) -> str:
    def short(v: Any) -> str:
        r = repr(v)
        return r if len(r) <= 80 else r[:77] + "..."

    return f"{tool}({', '.join(f'{k}={short(v)}' for k, v in args.items())})"


class McpGuardToolset(WrapperToolset[AgentDeps]):
    """Intercepta chamadas MCP: pede confirmação nas de escrita e trata o resultado como não confiável."""

    def __init__(self, wrapped: AbstractToolset[AgentDeps], server: str, require_approval: set[str]) -> None:
        super().__init__(wrapped)
        self.server = server
        self.require_approval = require_approval

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDeps], tool: Any) -> Any:
        deps = ctx.deps
        if name in self.require_approval:
            with traced("tool.mcp.approval_request", tool=name):
                await deps.approvals.request(deps.ctx, self.server, name, tool_args)
            deps.ctx.tools_called.append(f"{name}:pendente")
            return APPROVAL_PENDING_RESULT.format(
                acao=_describe(name, tool_args), ferramenta=name, saida=OUTPUT_TOOL
            )
        with traced("tool.mcp.call", tool=name, server=self.server) as span:
            try:
                result = await super().call_tool(name, tool_args, ctx, tool)
            except ModelRetry as exc:
                # Erro da ferramenta (ex.: argumento inválido): o modelo corrige e tenta de novo.
                # O texto vem do adaptador: é dado não confiável como qualquer resultado (§6.6).
                # Sem orçamento, o Pydantic AI encerraria a execução; aí vira "indisponível".
                if ctx.retry < tool.max_retries:
                    safe = await sanitize_tool_result(deps.injection, name, exc.message, deps.boundary)
                    raise ModelRetry(safe) from exc
                log.warning("mcp_tool_retries_exhausted", tool=name, server=self.server, error=exc.message)
                if span is not None:
                    span.set_attribute("tool.retries_exhausted", True)
                return UNAVAILABLE
            except Exception as exc:  # noqa: BLE001 (CancelledError não é Exception)
                log.warning("mcp_tool_unavailable", tool=name, server=self.server, error=type(exc).__name__)
                if span is not None:
                    span.set_attribute("tool.unavailable", True)
                return UNAVAILABLE
        deps.ctx.tools_called.append(name)
        return await sanitize_tool_result(deps.injection, name, result, deps.boundary)


def build_mcp_toolsets(
    app: LoadedApp,
    deps_ctx: RequestContext,
    *,
    allow_write: bool,
    source: Callable[[McpServerConfig, dict[str, str]], Any] | None = None,
) -> list[AbstractToolset[Any]]:
    """`source` permite injetar o servidor (testes); por padrão conecta por Streamable HTTP na URL."""
    headers = identity_headers(deps_ctx)
    toolsets: list[AbstractToolset[Any]] = []
    for server in app.config.tools.mcp_servers:
        target = source(server, headers) if source else server.url
        mcp = MCPToolset(target, headers=headers) if source is None else MCPToolset(target)
        allowed = set(server.allow)
        approval = set(server.require_approval) & allowed

        def keep(_ctx: Any, td: ToolDefinition, allowed=allowed, approval=approval) -> bool:
            if td.name not in allowed:
                return False
            return allow_write or td.name not in approval

        toolsets.append(McpGuardToolset(mcp.filtered(keep), server.name, approval))
    return toolsets
