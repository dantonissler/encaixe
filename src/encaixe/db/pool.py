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
    pool: asyncpg.Pool,
    app_id: str,
    principals: Principals | None,
    *,
    system: bool = False,
) -> AsyncIterator[asyncpg.Connection]:
    """Transação com `encaixe.app_id`/`encaixe.principals` locais (valem para o RLS).

    `system=True` ignora a ACL de chunks (o app_id continua valendo). Use apenas em
    caminhos de ingestão/remoção/manutenção, nunca em requisições de usuário.
    """
    if not app_id:
        raise ValueError("app_id vazio")
    items = principals.as_list() if principals is not None else []
    # Premissa: principais nunca contêm vírgula (a lista é separada por vírgulas).
    if any("," in p for p in items):
        raise ValueError("principal com vírgula")
    principals_csv = ",".join(items)
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('encaixe.app_id', $1, true), "
            "set_config('encaixe.principals', $2, true), "
            "set_config('encaixe.acl_bypass', $3, true)",
            app_id,
            principals_csv,
            "on" if system else "off",
        )
        yield conn
