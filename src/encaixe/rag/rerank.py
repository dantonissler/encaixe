from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from encaixe.core.types import ScoredChunk


class NoopReranker:
    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return chunks[: max(top_n, 0)]


def _default_factory(model_id: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, device="cpu")


class CrossEncoderReranker:
    def __init__(
        self, model_id: str, model_factory: Callable[[str], Any] = _default_factory
    ) -> None:
        self.model_id = model_id
        self._factory = model_factory
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._predict_lock = threading.Lock()

    async def _get_model(self) -> Any:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._factory, self.model_id)
        return self._model

    async def warm_up(self) -> None:
        """Carrega o modelo e faz uma predição mínima (aquecimento na partida da API)."""
        model = await self._get_model()
        await asyncio.to_thread(self._predict, model, [("aquecimento", "aquecimento")])

    def _predict(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        with self._predict_lock:  # modelo compartilhado entre requisições
            return [float(s) for s in model.predict(pairs)]

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not chunks or top_n <= 0:
            return []
        model = await self._get_model()
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = await asyncio.to_thread(self._predict, model, pairs)
        ranked = sorted(
            (ScoredChunk(c.chunk, s) for c, s in zip(chunks, scores, strict=True)),
            key=lambda x: x.score,
            reverse=True,
        )
        return ranked[:top_n]
