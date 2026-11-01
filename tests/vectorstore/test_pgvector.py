import random
import uuid

import pytest

from encaixe.core.types import Chunk, Principals
from encaixe.db.pool import app_scope
from encaixe.vectorstore.pgvector import PgVectorStore

pytestmark = pytest.mark.integration


def vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.random() - 0.5 for _ in range(1024)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


async def make_doc(pool, app_id: str) -> str:
    async with app_scope(pool, app_id, None, system=True) as conn:
        return str(
            await conn.fetchval(
                "INSERT INTO documents (app_id, title) VALUES ($1, 't') RETURNING id", app_id
            )
        )


def mk(app_id, doc, text, seed, acl=("app:*",)) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        app_id=app_id,
        document_id=doc,
        text=text,
        heading_path=("H",),
        acl_principals=acl,
        metadata={"k": 1},
        embedding=vec(seed),
    )


async def test_acl_filters_restricted_chunk(pg_pool):
    store = PgVectorStore(pg_pool)
    doc = await make_doc(pg_pool, "a")
    pub = mk("a", doc, "regras gerais", 1)
    rh = mk("a", doc, "salarios sigilosos", 2, ("group:rh",))
    await store.upsert([pub, rh])
    r1 = await store.search_hybrid("a", Principals("u1"), "salarios", vec(2), 10)
    assert {s.chunk.id for s in r1} == {pub.id}
    r2 = await store.search_hybrid("a", Principals("u2", ("rh",)), "salarios", vec(2), 10)
    assert {s.chunk.id for s in r2} == {pub.id, rh.id}
    assert r2[0].chunk.id == rh.id
    assert r2[0].chunk.metadata == {"k": 1}
    assert r2[0].chunk.heading_path == ("H",)


async def test_other_app_isolated(pg_pool):
    store = PgVectorStore(pg_pool)
    doc = await make_doc(pg_pool, "a")
    await store.upsert([mk("a", doc, "texto alfa", 1)])
    assert await store.search_hybrid("b", Principals("u"), "alfa", vec(1), 10) == []


async def test_fts_matches_without_accents(pg_pool):
    store = PgVectorStore(pg_pool)
    doc = await make_doc(pg_pool, "a")
    c = mk("a", doc, "Política de remuneração", 5)
    await store.upsert([c])
    res = await store.search_hybrid("a", Principals(None), "remuneracao", vec(99), 10)
    assert c.id in {s.chunk.id for s in res}


async def test_principals_none_sees_only_public(pg_pool):
    store = PgVectorStore(pg_pool)
    doc = await make_doc(pg_pool, "a")
    pub = mk("a", doc, "publico", 1)
    prv = mk("a", doc, "privado", 2, ("user:u1",))
    await store.upsert([pub, prv])
    res = await store.search_hybrid("a", Principals(None), "privado", vec(2), 10)
    assert {s.chunk.id for s in res} == {pub.id}


async def test_upsert_validation_and_delete(pg_pool):
    store = PgVectorStore(pg_pool)
    doc = await make_doc(pg_pool, "a")
    c = mk("a", doc, "um", 1)
    with pytest.raises(ValueError):
        await store.upsert([c, mk("b", doc, "dois", 2)])
    noemb = mk("a", doc, "tres", 3)
    noemb.embedding = None
    with pytest.raises(ValueError):
        await store.upsert([noemb])
    await store.upsert([c])
    c.text = "um editado"
    await store.upsert([c])
    res = await store.search_hybrid("a", Principals(None), "editado", vec(1), 10)
    assert [s.chunk.text for s in res] == ["um editado"]
    await store.delete_document("a", doc)
    assert await store.search_hybrid("a", Principals(None), "editado", vec(1), 10) == []
