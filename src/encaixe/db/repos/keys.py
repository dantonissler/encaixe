from __future__ import annotations

import hashlib
import secrets

import asyncpg

from encaixe.observability.spans import traced


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


async def create_key(pool: asyncpg.Pool, app_id: str) -> str:
    """Cria uma chave; o segredo só é devolvido aqui (guarda-se apenas o sha256)."""
    secret = "enc_" + secrets.token_urlsafe(32)
    with traced("db.keys.create"):
        await pool.execute(
            "INSERT INTO api_keys (app_id, key_hash) VALUES ($1, $2)", app_id, _hash(secret)
        )
    return secret


async def resolve_key(pool: asyncpg.Pool, secret: str) -> str | None:
    """Única consulta fora de app_scope (api_keys não tem RLS)."""
    with traced("db.keys.resolve"):
        return await pool.fetchval(
            "SELECT app_id FROM api_keys WHERE key_hash = $1 AND revoked_at IS NULL",
            _hash(secret),
        )
