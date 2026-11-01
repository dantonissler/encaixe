from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from encaixe.core.types import RequestContext
from encaixe.messages import APPROVAL_CANCELLED
from encaixe.observability.spans import traced

McpCaller = Callable[[str, str, dict, dict[str, str]], Awaitable[str]]

_YES = re.compile(r"^\s*(sim|s|confirmo|pode|ok)\b", re.IGNORECASE)
_NO = re.compile(r"^\s*(n[aã]o|cancela)", re.IGNORECASE)


def identity_headers(ctx: RequestContext) -> dict[str, str]:
    return {
        "X-User-Ref": ctx.principals.user_ref or "",
        "X-User-Groups": ",".join(ctx.principals.groups),
        "X-App-Id": ctx.app.config.id,
    }


async def default_mcp_caller(url: str, tool: str, args: dict, headers: dict[str, str]) -> str:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    async with Client(StreamableHttpTransport(url, headers=headers)) as client:
        result = await client.call_tool(tool, args)
    text = "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))
    return text or str(result.data)


class ApprovalService:
    def __init__(self, repo: Any, mcp_caller: McpCaller | None = None) -> None:
        self._repo = repo
        self._call = mcp_caller or default_mcp_caller

    async def request(self, ctx: RequestContext, server: str, tool: str, args: dict) -> str:
        if not ctx.conversation_id:
            raise ValueError("aprovação exige conversation_id")
        with traced("tool.approval.request", server=server, tool=tool):
            return await self._repo.create_pending(
                ctx.app.config.id, ctx.conversation_id, ctx.principals.user_ref, server, tool, args
            )

    async def resolve_turn(self, ctx: RequestContext) -> str | None:
        if not (ctx.has_conversation_identity and ctx.conversation_id):
            return None
        app_id = ctx.app.config.id
        pending = await self._repo.get_pending(app_id, ctx.conversation_id, ctx.principals.user_ref)
        if pending is None:
            return None
        with traced("tool.approval.resolve", server=pending.server, tool=pending.tool):
            if _YES.match(ctx.user_message):
                # Marca antes de executar: uma queda não permite execução dupla.
                if not await self._repo.resolve(app_id, pending.id, "approved"):
                    return None
                return await self._execute(ctx, pending)
            if _NO.match(ctx.user_message):
                await self._repo.resolve(app_id, pending.id, "cancelled")
                return APPROVAL_CANCELLED
            await self._repo.resolve(app_id, pending.id, "cancelled")
            return None

    async def _execute(self, ctx: RequestContext, pending: Any) -> str:
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
        return f"Ação confirmada e executada: {pending.tool}. Resultado: {result}"
