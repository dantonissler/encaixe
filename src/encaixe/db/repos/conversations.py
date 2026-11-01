from __future__ import annotations

import asyncpg

from encaixe.core.types import ChatMessage
from encaixe.db.pool import app_scope
from encaixe.observability.spans import traced


class ConversationNotFound(Exception):
    pass


class PgHistoryStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_conversation(self, app_id: str, user_ref: str | None) -> str:
        with traced("db.history.create_conversation"):
            async with app_scope(self._pool, app_id, None) as conn:
                cid = await conn.fetchval(
                    "INSERT INTO conversations (app_id, user_ref) VALUES ($1, $2) RETURNING id",
                    app_id,
                    user_ref,
                )
            return str(cid)

    async def get_owner(self, app_id: str, conversation_id: str) -> str | None:
        with traced("db.history.get_owner"):
            async with app_scope(self._pool, app_id, None) as conn:
                row = await conn.fetchrow(
                    "SELECT user_ref FROM conversations WHERE id = $1", conversation_id
                )
            if row is None:
                raise ConversationNotFound(conversation_id)
            return row["user_ref"]

    async def append(
        self,
        app_id: str,
        conversation_id: str,
        msg: ChatMessage,
        *,
        trace_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
    ) -> int:
        with traced("db.history.append"):
            async with app_scope(self._pool, app_id, None) as conn:
                # Trava a conversa para serializar a atribuição de seq.
                locked = await conn.fetchval(
                    "SELECT 1 FROM conversations WHERE id = $1 FOR UPDATE", conversation_id
                )
                if locked is None:
                    raise ConversationNotFound(conversation_id)
                seq = await conn.fetchval(
                    "INSERT INTO messages (app_id, conversation_id, seq, role, content, "
                    "tool_calls, tokens, trace_id, prompt_version, model) "
                    "SELECT $1, $2, COALESCE(MAX(seq), 0) + 1, $3, $4, $5, $6, $7, $8, $9 "
                    "FROM messages WHERE conversation_id = $2 RETURNING seq",
                    app_id,
                    conversation_id,
                    msg.role,
                    msg.content,
                    msg.tool_calls,
                    msg.tokens,
                    trace_id,
                    prompt_version,
                    model,
                )
            return seq

    async def list_messages(
        self, app_id: str, conversation_id: str, after_seq: int = 0
    ) -> list[ChatMessage]:
        with traced("db.history.list_messages"):
            async with app_scope(self._pool, app_id, None) as conn:
                rows = await conn.fetch(
                    "SELECT seq, role, content, tool_calls, tokens FROM messages "
                    "WHERE conversation_id = $1 AND seq > $2 ORDER BY seq",
                    conversation_id,
                    after_seq,
                )
            return [
                ChatMessage(
                    role=r["role"],
                    content=r["content"],
                    seq=r["seq"],
                    tool_calls=r["tool_calls"],
                    tokens=r["tokens"],
                )
                for r in rows
            ]

    async def get_summary(self, app_id: str, conversation_id: str) -> tuple[str, int] | None:
        with traced("db.history.get_summary"):
            async with app_scope(self._pool, app_id, None) as conn:
                row = await conn.fetchrow(
                    "SELECT summary, covers_until_seq FROM conversation_summaries "
                    "WHERE conversation_id = $1",
                    conversation_id,
                )
            return None if row is None else (row["summary"], row["covers_until_seq"])

    async def save_summary(
        self, app_id: str, conversation_id: str, summary: str, covers_until_seq: int
    ) -> None:
        with traced("db.history.save_summary"):
            async with app_scope(self._pool, app_id, None) as conn:
                row = await conn.fetchval(
                    "INSERT INTO conversation_summaries "
                    "(conversation_id, app_id, summary, covers_until_seq) "
                    "SELECT id, app_id, $2, $3 FROM conversations WHERE id = $1 "
                    "ON CONFLICT (conversation_id) DO UPDATE "
                    "SET summary = EXCLUDED.summary, covers_until_seq = EXCLUDED.covers_until_seq "
                    "RETURNING 1",
                    conversation_id,
                    summary,
                    covers_until_seq,
                )
                if row is None:
                    raise ConversationNotFound(conversation_id)
