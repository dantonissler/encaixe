"""`GET /v1/conversations/{id}`: histórico para a app exibir; só o dono (`user_ref`) lê."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from encaixe.api.auth import get_deps, require_app
from encaixe.api.errors import service_unavailable
from encaixe.api.identity import resolve_identity, verify_conversation_owner
from encaixe.wiring import Deps

router = APIRouter()


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    key_app: Annotated[str, Depends(require_app)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> dict[str, Any]:
    principals, _, _ = resolve_identity(request, None)
    cid = await verify_conversation_owner(deps.history_store, key_app, conversation_id, principals)
    async with service_unavailable("postgres"):
        messages = await deps.history_store.list_messages(key_app, cid)
    return {
        "id": cid,
        "object": "conversation",
        "app_id": key_app,
        "messages": [
            {"seq": m.seq, "role": m.role, "content": m.content, "tool_calls": m.tool_calls}
            for m in messages
        ],
    }
