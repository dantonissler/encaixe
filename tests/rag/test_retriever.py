import numpy as np

from encaixe.core.types import Chunk, Principals, ScoredChunk
from encaixe.rag.rerank import CrossEncoderReranker, NoopReranker
from encaixe.rag.retriever import HybridRetriever


def sc(i: int) -> ScoredChunk:
    return ScoredChunk(Chunk(str(i), "a", "d", f"t{i}", (), ("app:*",), {}), 1.0 / (i + 1))


class FakeEmbedder:
    model_name = "f"
    dims = 3

    async def embed(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]


class FakeStore:
    def __init__(self):
        self.calls = []

    async def search_hybrid(self, app_id, principals, text, embedding, k):
        self.calls.append(k)
        return [sc(i) for i in range(k)]


class BoomReranker:
    async def rerank(self, query, chunks, top_n):
        raise RuntimeError("boom")


async def test_reranker_failure_degrades():
    store = FakeStore()
    r = HybridRetriever(FakeEmbedder(), store, BoomReranker())
    out = await r.retrieve("a", Principals(None), "q", 10, 3)
    assert [s.chunk.id for s in out] == ["0", "1", "2"]


async def test_empty_or_zero_returns_empty():
    store = FakeStore()
    r = HybridRetriever(FakeEmbedder(), store, NoopReranker())
    assert await r.retrieve("a", Principals(None), "  ", 10, 3) == []
    assert await r.retrieve("a", Principals(None), "q", 0, 3) == []
    assert store.calls == []


async def test_noop_reranker():
    out = await NoopReranker().rerank("q", [sc(i) for i in range(5)], 2)
    assert [s.chunk.id for s in out] == ["0", "1"]


async def test_cross_encoder_sorts_with_fake_model():
    loads = []

    class M:
        def predict(self, pairs):
            return np.array([float(p[1][1:]) for p in pairs])

    def factory(model_id):
        loads.append(model_id)
        return M()

    rr = CrossEncoderReranker("m", model_factory=factory)
    out = await rr.rerank("q", [sc(1), sc(3), sc(2)], 2)
    assert [s.chunk.id for s in out] == ["3", "2"]
    assert out[0].score == 3.0
    await rr.rerank("q", [sc(1)], 1)
    assert loads == ["m"]
    assert await rr.rerank("q", [], 2) == []
