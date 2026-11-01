from __future__ import annotations

import structlog

from encaixe.core.protocols import Embedder, Reranker, VectorStore
from encaixe.core.types import Principals, ScoredChunk
from encaixe.observability.spans import traced

log = structlog.get_logger()


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, reranker: Reranker) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    async def retrieve(
        self,
        app_id: str,
        principals: Principals,
        query: str,
        top_k: int,
        rerank_top_n: int,
    ) -> list[ScoredChunk]:
        if top_k <= 0 or not query.strip():
            return []
        with traced("rag.retrieve") as span:
            with traced("rag.search"):
                (embedding,) = await self._embedder.embed([query])
                found = await self._store.search_hybrid(
                    app_id, principals, query, embedding, top_k
                )
            if not found:
                return []
            try:
                with traced("rag.rerank"):
                    return await self._reranker.rerank(query, found, rerank_top_n)
            except Exception as exc:  # noqa: BLE001 - degrada para a ordem da busca
                log.warning("rerank_failed", error_type=type(exc).__name__)
                span.set_attribute("rerank.degraded", True)
                return found[: max(rerank_top_n, 0)]
