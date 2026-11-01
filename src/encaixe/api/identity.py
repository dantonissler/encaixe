"""Identidade declarada pelo backend da app (spec §6.4a) e dono da conversa (spec §7)."""

from __future__ import annotations

import uuid
from typing import Any

from starlette.requests import Request

from encaixe.api.errors import ApiError, service_unavailable
from encaixe.core.protocols import HistoryStore
from encaixe.core.types import Principals
from encaixe.db.repos.conversations import ConversationNotFound

CONVERSATION_NOT_FOUND = "Conversa não encontrada."


def _bad(message: str) -> ApiError:
    return ApiError(400, message, code="invalid_identity")


def _text(value: Any, field: str) -> str | None:
    """Texto aparado; vazio vira ausente (passa para a próxima fonte)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad(f"'{field}' deve ser texto.")
    return value.strip() or None


def _groups(value: Any, field: str) -> tuple[str, ...]:
    """Lista (metadata) ou CSV (header/metadata); sem vazios e sem repetidos, na ordem."""
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        items = value  # vírgula dentro de um item é recusada pelos Principals (400)
    else:
        raise _bad(f"'{field}' deve ser uma lista de textos ou um CSV.")
    return tuple(dict.fromkeys(g.strip() for g in items if g.strip()))


def resolve_identity(
    request: Request, body: Any | None = None
) -> tuple[Principals, str | None, str | None]:
    """(principais, conversation_id, model_override) da requisição.

    `user_ref`: `body.user` → `body.metadata.user_ref` → header `X-User-Ref`.
    Grupos: `body.metadata.groups` (lista) → header `X-User-Groups` (CSV).
    `conversation_id`: header `X-Conversation-Id` → `body.metadata.conversation_id`.
    `model_override`: header `X-Encaixe-Model` (o pipeline só o aceita em dev).
    Valor com vírgula (injetaria principais extras) → 400.
    """
    metadata = getattr(body, "metadata", None) or {}
    headers = request.headers
    user_ref = (
        _text(getattr(body, "user", None), "user")
        or _text(metadata.get("user_ref"), "metadata.user_ref")
        or _text(headers.get("x-user-ref"), "X-User-Ref")
    )
    if metadata.get("groups") is not None:
        groups = _groups(metadata["groups"], "metadata.groups")
    else:
        groups = _groups(headers.get("x-user-groups"), "X-User-Groups")
    conversation_id = _text(headers.get("x-conversation-id"), "X-Conversation-Id") or _text(
        metadata.get("conversation_id"), "metadata.conversation_id"
    )
    model_override = _text(headers.get("x-encaixe-model"), "X-Encaixe-Model")
    try:
        principals = Principals(user_ref, groups)
    except ValueError:
        raise _bad("user_ref e grupos não podem conter vírgula.") from None
    return principals, conversation_id, model_override


async def verify_conversation_owner(
    store: HistoryStore, app_id: str, conversation_id: str, principals: Principals
) -> str:
    """Id canônico da conversa se ela é desta app e deste `user_ref`; senão 404 (sem dizer
    se existe)."""
    not_found = ApiError(404, CONVERSATION_NOT_FOUND, code="conversation_not_found")
    try:
        cid = str(uuid.UUID(conversation_id))
    except ValueError:
        raise not_found from None
    async with service_unavailable("postgres"):
        try:
            owner = await store.get_owner(app_id, cid)
        except ConversationNotFound:
            raise not_found from None
    if owner != principals.user_ref:
        raise not_found
    return cid
