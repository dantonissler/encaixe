from __future__ import annotations

import httpx

from encaixe.cache.embeddings import EmbeddingCache
from encaixe.observability.spans import traced
from encaixe.settings import Settings

BATCH_SIZE = 32


class EmbeddingDimensionError(ValueError):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        settings: Settings,
        cache: EmbeddingCache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._client = client
        self.model_name = settings.embedding_model
        self.dims = settings.embedding_dims

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._http().post(
            f"{self._settings.ollama_base_url}/api/embed",
            json={"model": self.model_name, "input": batch},
        )
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]
        if len(vectors) != len(batch):
            raise EmbeddingDimensionError(
                f"esperados {len(batch)} vetores, recebidos {len(vectors)}"
            )
        for v in vectors:
            if len(v) != self.dims:
                raise EmbeddingDimensionError(f"dimensão {len(v)} != {self.dims}")
        return vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        with traced("embeddings.embed", model=self.model_name, count=len(texts)):
            result: list[list[float] | None] = [None] * len(texts)
            if self._cache is not None:
                result = await self._cache.get_many(self.model_name, texts)
            missing = [i for i, v in enumerate(result) if v is None]
            new_texts: list[str] = []
            new_vecs: list[list[float]] = []
            for start in range(0, len(missing), BATCH_SIZE):
                idx = missing[start : start + BATCH_SIZE]
                batch = [texts[i] for i in idx]
                vecs = await self._embed_batch(batch)
                for i, v in zip(idx, vecs, strict=True):
                    result[i] = v
                new_texts += batch
                new_vecs += vecs
            if self._cache is not None and new_texts:
                await self._cache.set_many(self.model_name, new_texts, new_vecs)
            return result  # type: ignore[return-value]
