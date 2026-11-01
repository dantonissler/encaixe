from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import structlog

from encaixe.core.types import Chunk
from encaixe.db.repos import documents as docs
from encaixe.observability.spans import traced

log = structlog.get_logger(__name__)


async def ingest_document(ctx: dict, app_id: str, document_id: str) -> str:
    """Job arq: parse -> guard -> embed -> (re)indexa. Idempotente. Devolve o status final."""
    pool = ctx["pool"]
    store = ctx["store"]
    with traced("ingest.document", app_id=app_id, document_id=document_id) as span:
        try:
            doc = await docs.get_document(pool, app_id, document_id)
            if doc is None:
                log.warning("ingest.missing", app_id=app_id, document_id=document_id)
                span.set_attribute("ingest.missing", True)
                return "missing"
            await docs.set_status(pool, app_id, document_id, "processing")

            with traced("ingest.parse"):
                drafts = await asyncio.to_thread(
                    ctx["chunker"].chunk, Path(doc.source_path or ""), doc.title
                )

            with traced("ingest.guard"):
                for d in drafts:
                    if (await ctx["injection"].check_text(d.text)).verdict == "block":
                        removed = await store.delete_chunks(app_id, document_id)
                        await docs.set_status(
                            pool, app_id, document_id, "quarantined", quarantined=True,
                            error="possível injeção de prompt no documento",
                        )  # fmt: skip
                        if removed:
                            await docs.bump_corpus_version(pool, app_id)
                        return "quarantined"

            with traced("ingest.embed", chunks=len(drafts)):
                vectors = await ctx["embedder"].embed([d.text for d in drafts]) if drafts else []

            acl = tuple(doc.acl) or ("app:*",)
            ns = uuid.UUID(document_id)
            chunks = [
                Chunk(
                    id=str(uuid.uuid5(ns, str(i))),
                    app_id=app_id,
                    document_id=document_id,
                    text=d.text,
                    heading_path=d.heading_path,
                    acl_principals=acl,
                    metadata={"title": doc.title, "page": d.page},
                    embedding=v,
                )
                for i, (d, v) in enumerate(zip(drafts, vectors, strict=True))
            ]
            with traced("ingest.upsert", chunks=len(chunks)):
                removed = await store.delete_chunks(app_id, document_id)
                await store.upsert(chunks)

            await docs.set_status(pool, app_id, document_id, "indexed")
            if removed or chunks:
                await docs.bump_corpus_version(pool, app_id)
            return "indexed"
        except Exception as exc:  # noqa: BLE001 - qualquer falha vira status failed
            log.error("ingest.failed", app_id=app_id, document_id=document_id, error=str(exc))
            span.set_attribute("ingest.error", str(exc))
            try:
                await store.delete_chunks(app_id, document_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                log.error("ingest.cleanup_failed", document_id=document_id, error=str(cleanup_exc))
                span.set_attribute("ingest.cleanup_error", str(cleanup_exc))
            try:
                await docs.set_status(pool, app_id, document_id, "failed", error=str(exc))
            except Exception as status_exc:  # noqa: BLE001
                log.error("ingest.status_failed", document_id=document_id, error=str(status_exc))
                span.set_attribute("ingest.status_error", str(status_exc))
            return "failed"
