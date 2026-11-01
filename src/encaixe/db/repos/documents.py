from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from encaixe.db.pool import app_scope
from encaixe.observability.spans import traced


@dataclass
class Document:
    id: str
    app_id: str
    title: str
    source_path: str | None
    acl: list[str]
    status: str
    quarantined: bool
    error: str | None


async def create_document(
    pool: asyncpg.Pool,
    app_id: str,
    title: str,
    source_path: str | None,
    content_hash: str,
    acl: list[str],
) -> str | None:
    """Cria o documento; devolve None se o mesmo hash já existe na app (dedup)."""
    with traced("db.documents.create"):
        async with app_scope(pool, app_id, None, system=True) as conn:
            row = await conn.fetchval(
                "INSERT INTO documents (app_id, title, source_path, content_hash, acl) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (app_id, content_hash) WHERE content_hash IS NOT NULL "
                "DO NOTHING RETURNING id::text",
                app_id,
                title,
                source_path,
                content_hash,
                acl,
            )
    return row


async def set_status(
    pool: asyncpg.Pool,
    app_id: str,
    document_id: str,
    status: str,
    *,
    quarantined: bool = False,
    error: str | None = None,
) -> None:
    with traced("db.documents.set_status", status=status):
        async with app_scope(pool, app_id, None, system=True) as conn:
            await conn.execute(
                "UPDATE documents SET status = $3, quarantined = $4, error = $5 "
                "WHERE app_id = $1 AND id = $2::uuid",
                app_id,
                document_id,
                status,
                quarantined,
                error,
            )


_DOCUMENT_COLUMNS = "id::text, app_id, title, source_path, acl, status, quarantined, error"


def _document(r: asyncpg.Record | None) -> Document | None:
    if r is None:
        return None
    return Document(
        r["id"], r["app_id"], r["title"], r["source_path"], list(r["acl"]),
        r["status"], r["quarantined"], r["error"],
    )  # fmt: skip


async def get_document(pool: asyncpg.Pool, app_id: str, document_id: str) -> Document | None:
    with traced("db.documents.get"):
        async with app_scope(pool, app_id, None, system=True) as conn:
            r = await conn.fetchrow(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE app_id = $1 AND id = $2::uuid",
                app_id,
                document_id,
            )
    return _document(r)


async def get_document_by_hash(
    pool: asyncpg.Pool, app_id: str, content_hash: str
) -> Document | None:
    """O documento da app com este hash (o que barrou um upload duplicado)."""
    with traced("db.documents.get_by_hash"):
        async with app_scope(pool, app_id, None, system=True) as conn:
            r = await conn.fetchrow(
                f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE app_id = $1 AND content_hash = $2",
                app_id,
                content_hash,
            )
    return _document(r)


async def delete_document(pool: asyncpg.Pool, app_id: str, document_id: str) -> None:
    """Remove o documento; os chunks saem por ON DELETE CASCADE."""
    with traced("db.documents.delete"):
        async with app_scope(pool, app_id, None, system=True) as conn:
            await conn.execute(
                "DELETE FROM documents WHERE app_id = $1 AND id = $2::uuid", app_id, document_id
            )


async def bump_corpus_version(pool: asyncpg.Pool, app_id: str) -> int:
    with traced("db.documents.bump_corpus_version"):
        return await pool.fetchval(
            "INSERT INTO app_runtime (app_id, corpus_version) VALUES ($1, 1) "
            "ON CONFLICT (app_id) DO UPDATE "
            "SET corpus_version = app_runtime.corpus_version + 1 RETURNING corpus_version",
            app_id,
        )


async def get_corpus_version(pool: asyncpg.Pool, app_id: str) -> int:
    with traced("db.documents.get_corpus_version"):
        v = await pool.fetchval("SELECT corpus_version FROM app_runtime WHERE app_id = $1", app_id)
    return v or 0
