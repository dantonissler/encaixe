from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from encaixe.context.untrusted import wrap_untrusted
from encaixe.core.config import LoadedApp, McpServerConfig
from encaixe.core.types import RequestContext
from encaixe.messages import APPROVAL_PROMPT
from encaixe.observability.spans import traced
from encaixe.tools.approvals import identity_headers
from encaixe.tools.native import AgentDeps

BLOCKED = "[resultado bloqueado por segurança]"


def _describe(tool: str, args: dict) -> str:
    return f"{tool}({', '.join(f'{k}={v}' for k, v in args.items())})"


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


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
            deps.ctx.tools_called.append(name)
            return APPROVAL_PROMPT.format(acao=_describe(name, tool_args))
        with traced("tool.mcp.call", tool=name, server=self.server):
            result = _stringify(await super().call_tool(name, tool_args, ctx, tool))
        deps.ctx.tools_called.append(name)
        if (await deps.injection.check_text(result)).verdict == "block":
            return BLOCKED
        return wrap_untrusted("ferramenta", {"ferramenta": name, "resultado": result}, deps.boundary)


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
