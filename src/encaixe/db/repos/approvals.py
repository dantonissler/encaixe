from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import asyncpg

from encaixe.db.pool import app_scope
from encaixe.observability.spans import traced


@dataclass
class Approval:
    id: str
    app_id: str
    conversation_id: str
    user_ref: str | None
    server: str
    tool: str
    args: dict
    status: str
    expires_at: datetime


class PgApprovalStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_pending(
        self,
        app_id: str,
        conversation_id: str,
        user_ref: str | None,
        server: str,
        tool: str,
        args: dict,
        ttl_s: float = 600,
    ) -> str:
        with traced("db.approvals.create_pending"):
            async with app_scope(self._pool, app_id, None) as conn:
                aid = await conn.fetchval(
                    "INSERT INTO tool_approvals (app_id, conversation_id, user_ref, server, "
                    "tool, args, expires_at) VALUES ($1, $2, $3, $4, $5, $6, "
                    "now() + $7 * interval '1 second') RETURNING id",
                    app_id,
                    conversation_id,
                    user_ref,
                    server,
                    tool,
                    args,
                    float(ttl_s),
                )
            return str(aid)

    async def get_pending(
        self, app_id: str, conversation_id: str, user_ref: str | None
    ) -> Approval | None:
        with traced("db.approvals.get_pending"):
            async with app_scope(self._pool, app_id, None) as conn:
                r = await conn.fetchrow(
                    "SELECT * FROM tool_approvals WHERE conversation_id = $1 "
                    "AND user_ref IS NOT DISTINCT FROM $2 AND status = 'pending' "
                    "AND expires_at > now() ORDER BY created_at DESC LIMIT 1",
                    conversation_id,
                    user_ref,
                )
            if r is None:
                return None
            return Approval(
                id=str(r["id"]),
                app_id=r["app_id"],
                conversation_id=str(r["conversation_id"]),
                user_ref=r["user_ref"],
                server=r["server"],
                tool=r["tool"],
                args=r["args"],
                status=r["status"],
                expires_at=r["expires_at"],
            )

    async def resolve(
        self, app_id: str, approval_id: str, status: Literal["approved", "cancelled"]
    ) -> bool:
        """Resolve só aprovações pendentes e não expiradas; True se alguma linha mudou."""
        with traced("db.approvals.resolve"):
            async with app_scope(self._pool, app_id, None) as conn:
                row = await conn.fetchval(
                    "UPDATE tool_approvals SET status = $2 WHERE id = $1 "
                    "AND status = 'pending' AND expires_at > now() RETURNING 1",
                    approval_id,
                    status,
                )
            return row is not None
