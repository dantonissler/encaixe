from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from pgvector.asyncpg import register_vector

from encaixe.core.types import Principals


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, init=_init_connection)


@asynccontextmanager
async def app_scope(
    pool: asyncpg.Pool, app_id: str, principals: Principals | None
) -> AsyncIterator[asyncpg.Connection]:
    """Transação com `encaixe.app_id`/`encaixe.principals` locais (valem para o RLS)."""
    # Premissa: principais nunca contêm vírgula (a lista é separada por vírgulas).
    principals_csv = ",".join(principals.as_list()) if principals is not None else ""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('encaixe.app_id', $1, true), "
            "set_config('encaixe.principals', $2, true)",
            app_id,
            principals_csv,
        )
        yield conn
