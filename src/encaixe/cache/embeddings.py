from __future__ import annotations

import hashlib
import json

from redis.asyncio import Redis


class EmbeddingCache:
    """Cache de vetores no Redis. Formato: lista JSON de floats (compacta o bastante
    e independente de decode_responses). Chave: enc:emb:{model}:{sha256(text)}."""

    def __init__(self, redis: Redis, ttl_s: int = 7 * 24 * 3600) -> None:
        self._redis = redis
        self._ttl = ttl_s

    @staticmethod
    def _key(model: str, text: str) -> str:
        return f"enc:emb:{model}:{hashlib.sha256(text.encode()).hexdigest()}"

    async def get_many(self, model: str, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        raws = await self._redis.mget([self._key(model, t) for t in texts])
        return [json.loads(r) if r is not None else None for r in raws]

    async def set_many(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not texts:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for t, v in zip(texts, vectors, strict=True):
                pipe.set(self._key(model, t), json.dumps(v), ex=self._ttl)
            await pipe.execute()
