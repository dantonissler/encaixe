from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from encaixe.core.types import RequestContext
from encaixe.messages import APPROVAL_CANCELLED
from encaixe.observability.spans import traced
from encaixe.tools.results import sanitize_tool_result

MCP_TIMEOUT_S = 15.0
McpCaller = Callable[[str, str, dict, dict[str, str]], Awaitable[str]]

_NO = re.compile(r"\b(n[aã]o|cancel\w*|desist\w*|pare|nunca)\b", re.IGNORECASE)
_YES = re.compile(
    r"^(sim|s|confirmo|confirmado|pode|pode sim|ok|okay|isso|claro|manda ver|pode criar"
    r"|sim,? pode( criar)?)$"
)
_TRAILING = re.compile(r"[\W_]+$")


def _is_affirmative(message: str) -> bool:
    return bool(_YES.match(_TRAILING.sub("", message.strip().casefold())))


def identity_headers(ctx: RequestContext) -> dict[str, str]:
    return {
        "X-User-Ref": ctx.principals.user_ref or "",
        "X-User-Groups": ",".join(ctx.principals.groups),
        "X-App-Id": ctx.app.config.id,
    }


async def default_mcp_caller(url: str, tool: str, args: dict, headers: dict[str, str]) -> str:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    async def _call():
        async with Client(StreamableHttpTransport(url, headers=headers)) as client:
            return await client.call_tool(tool, args)

    result = await asyncio.wait_for(_call(), timeout=MCP_TIMEOUT_S)
    text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))
    return text or str(result.data)


class ApprovalService:
    def __init__(self, repo: Any, mcp_caller: McpCaller | None = None, injection: Any = None) -> None:
        self._repo = repo
        self._injection = injection
        self._call = mcp_caller or default_mcp_caller

    async def request(self, ctx: RequestContext, server: str, tool: str, args: dict) -> str:
        if not ctx.conversation_id:
            raise ValueError("aprovação exige conversation_id")
        with traced("tool.approval.request", server=server, tool=tool):
            return await self._repo.create_pending(
                ctx.app.config.id, ctx.conversation_id, ctx.principals.user_ref, server, tool, args
            )

    async def resolve_turn(self, ctx: RequestContext, boundary: str) -> str | None:
        if not (ctx.has_conversation_identity and ctx.conversation_id):
            return None
        app_id = ctx.app.config.id
        pending = await self._repo.get_pending(app_id, ctx.conversation_id, ctx.principals.user_ref)
        if pending is None:
            return None
        with traced("tool.approval.resolve", server=pending.server, tool=pending.tool):
            if _NO.search(ctx.user_message):
                await self._repo.resolve(app_id, pending.id, "cancelled")
                return APPROVAL_CANCELLED
            if _is_affirmative(ctx.user_message):
                # Marca antes de executar: uma queda não permite execução dupla.
                if not await self._repo.resolve(app_id, pending.id, "approved"):
                    return None
                return await self._execute(ctx, pending, boundary)
            await self._repo.resolve(app_id, pending.id, "cancelled")
            return None

    async def _execute(self, ctx: RequestContext, pending: Any, boundary: str) -> str:
        url = next(
            (s.url for s in ctx.app.config.tools.mcp_servers if s.name == pending.server), None
        )
        if url is None:
            return f"A ação {pending.tool} falhou: servidor '{pending.server}' não configurado."
        try:
            result = await self._call(url, pending.tool, pending.args, identity_headers(ctx))
        except Exception as exc:  # noqa: BLE001
            return f"A ação {pending.tool} falhou: {type(exc).__name__}."
        ctx.tools_called.append(pending.tool)
        body = await sanitize_tool_result(self._injection, pending.tool, result, boundary)
        return f"Ação confirmada e executada: {pending.tool}.\n{body}"
