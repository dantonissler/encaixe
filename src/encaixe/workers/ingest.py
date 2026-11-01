from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from encaixe.core.types import Chunk
from encaixe.db.repos import documents as docs
from encaixe.observability.spans import traced


async def ingest_document(ctx: dict, app_id: str, document_id: str) -> str:
    """Job arq: parse -> guard -> embed -> upsert. Devolve o status final."""
    pool = ctx["pool"]
    with traced("ingest.document", app_id=app_id, document_id=document_id):
        try:
            doc = await docs.get_document(pool, app_id, document_id)
            if doc is None:
                raise ValueError("documento não encontrado")
            await docs.set_status(pool, app_id, document_id, "processing")

            with traced("ingest.parse"):
                drafts = await asyncio.to_thread(
                    ctx["chunker"].chunk, Path(doc.source_path or ""), doc.title
                )

            with traced("ingest.guard"):
                for d in drafts:
                    if (await ctx["injection"].check_text(d.text)).verdict == "block":
                        await docs.set_status(
                            pool, app_id, document_id, "quarantined", quarantined=True,
                            error="possível injeção de prompt no documento",
                        )  # fmt: skip
                        return "quarantined"

            with traced("ingest.embed", chunks=len(drafts)):
                vectors = await ctx["embedder"].embed([d.text for d in drafts]) if drafts else []

            acl = tuple(doc.acl) or ("app:*",)
            chunks = [
                Chunk(
                    id=str(uuid.uuid4()),
                    app_id=app_id,
                    document_id=document_id,
                    text=d.text,
                    heading_path=d.heading_path,
                    acl_principals=acl,
                    metadata={"title": doc.title, "page": d.page},
                    embedding=v,
                )
                for d, v in zip(drafts, vectors, strict=True)
            ]
            with traced("ingest.upsert", chunks=len(chunks)):
                await ctx["store"].upsert(chunks)

            await docs.set_status(pool, app_id, document_id, "indexed")
            await docs.bump_corpus_version(pool, app_id)
            return "indexed"
        except Exception as exc:  # noqa: BLE001 - qualquer falha vira status failed
            await docs.set_status(pool, app_id, document_id, "failed", error=str(exc))
            return "failed"
