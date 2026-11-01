from __future__ import annotations

import asyncpg

from encaixe.core.types import Chunk, Principals, ScoredChunk
from encaixe.db.pool import app_scope
from encaixe.observability.spans import traced


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = soma de 1/(k+rank), rank a partir de 1."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # dict preserva a ordem de inserção e sorted é estável: empates mantêm a 1ª aparição.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


EMBEDDING_DIMS = 1024

_VECTOR_SQL = """
SELECT id::text FROM chunks
WHERE app_id = $1 AND acl_principals && $2::text[] AND embedding IS NOT NULL
ORDER BY embedding <=> $3::vector
LIMIT $4
"""

_FTS_SQL = """
SELECT id::text FROM chunks
WHERE app_id = $1 AND acl_principals && $2::text[]
  AND numnode(websearch_to_tsquery('pt_unaccent', $3)) > 0
  AND tsv @@ websearch_to_tsquery('pt_unaccent', $3)
ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('pt_unaccent', $3)) DESC, id
LIMIT $4
"""

_ROWS_SQL = """
SELECT id::text, app_id, document_id::text, text, heading_path, acl_principals, metadata
FROM chunks
WHERE app_id = $1 AND acl_principals && $2::text[] AND id = ANY($3::uuid[])
"""

_UPSERT_SQL = """
INSERT INTO chunks (id, app_id, document_id, text, heading_path, acl_principals,
                    metadata, embedding)
VALUES ($1::uuid, $2, $3::uuid, $4, $5, $6, $7, $8::vector)
ON CONFLICT (id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    text = EXCLUDED.text,
    heading_path = EXCLUDED.heading_path,
    acl_principals = EXCLUDED.acl_principals,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding
"""


def _to_chunk(r: asyncpg.Record) -> Chunk:
    return Chunk(
        id=r["id"],
        app_id=r["app_id"],
        document_id=r["document_id"],
        text=r["text"],
        heading_path=tuple(r["heading_path"]),
        acl_principals=tuple(r["acl_principals"]),
        metadata=r["metadata"],
    )


class PgVectorStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        app_ids = {c.app_id for c in chunks}
        if len(app_ids) != 1:
            raise ValueError("upsert com chunks de apps diferentes")
        if any(c.embedding is None for c in chunks):
            raise ValueError("chunk sem embedding")
        if any(len(c.embedding or ()) != EMBEDDING_DIMS for c in chunks):
            raise ValueError(f"embedding deve ter {EMBEDDING_DIMS} dimensões")
        (app_id,) = app_ids
        rows = [
            (
                c.id,
                c.app_id,
                c.document_id,
                c.text,
                list(c.heading_path),
                list(c.acl_principals),
                c.metadata,
                c.embedding,
            )
            for c in chunks
        ]
        with traced("vectorstore.upsert", chunks=len(chunks)):
            async with app_scope(self._pool, app_id, None, system=True) as conn:
                await conn.executemany(_UPSERT_SQL, rows)

    async def search_hybrid(
        self,
        app_id: str,
        principals: Principals,
        text: str,
        embedding: list[float],
        k: int,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        acl = principals.as_list()
        with traced("vectorstore.search_hybrid", k=k):
            async with app_scope(self._pool, app_id, principals) as conn:
                await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                vec_ids = [
                    r["id"] for r in await conn.fetch(_VECTOR_SQL, app_id, acl, embedding, k)
                ]
                fts_ids: list[str] = []
                if text.strip():
                    fts_ids = [r["id"] for r in await conn.fetch(_FTS_SQL, app_id, acl, text, k)]
                fused = rrf_fuse([vec_ids, fts_ids])
                if not fused:
                    return []
                rows = await conn.fetch(_ROWS_SQL, app_id, acl, [i for i, _ in fused])
        by_id = {r["id"]: _to_chunk(r) for r in rows}
        return [ScoredChunk(by_id[i], s) for i, s in fused if i in by_id][:k]

    async def delete_document(self, app_id: str, document_id: str) -> None:
        with traced("vectorstore.delete_document"):
            async with app_scope(self._pool, app_id, None, system=True) as conn:
                await conn.execute(
                    "DELETE FROM chunks WHERE app_id = $1 AND document_id = $2::uuid",
                    app_id,
                    document_id,
                )
