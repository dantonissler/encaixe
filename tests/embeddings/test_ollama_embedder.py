import httpx
import pytest
import respx

from encaixe.cache.embeddings import EmbeddingCache
from encaixe.embeddings.ollama import EmbeddingDimensionError, OllamaEmbedder


def _responder(dims=1024):
    def handler(request):
        import json

        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.5] * dims for _ in range(n)]})

    return handler


@respx.mock
async def test_batches_and_caches(settings, fake_redis):
    route = respx.post(f"{settings.ollama_base_url}/api/embed").mock(side_effect=_responder())
    emb = OllamaEmbedder(settings, EmbeddingCache(fake_redis))
    texts = [f"t{i}" for i in range(40)]
    out = await emb.embed(texts)
    assert len(out) == 40 and len(out[0]) == 1024
    assert route.call_count == 2
    out2 = await emb.embed(texts)
    assert route.call_count == 2
    assert out2 == out
    await emb.aclose()


@respx.mock
async def test_wrong_dimension_raises(settings):
    respx.post(f"{settings.ollama_base_url}/api/embed").mock(side_effect=_responder(8))
    emb = OllamaEmbedder(settings, None)
    with pytest.raises(EmbeddingDimensionError):
        await emb.embed(["a"])
    await emb.aclose()
