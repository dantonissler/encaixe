"""Rotas auxiliares: modelos, histórico de conversa, saúde e prontidão."""

from __future__ import annotations

import httpx
import respx

from encaixe.core.types import ChatMessage
from encaixe.db.migrate import migration_versions
from encaixe.settings import Settings
from pipeline.helpers import make_app

from .helpers import KEY_B, ApiHistory, make_api

OLLAMA = "http://ollama.test"
TAGS = {
    "models": [
        {"name": "qwen3.5:4b", "model": "qwen3.5:4b"},
        {"name": "llama3.2-3b-cpu:latest", "model": "llama3.2-3b-cpu:latest"},
        {"name": "llama-guard3-1b-cpu:latest", "model": "llama-guard3-1b-cpu:latest"},
        {"name": "shieldgemma-2b-cpu:latest", "model": "shieldgemma-2b-cpu:latest"},
        {"name": "bge-m3-cpu:latest", "model": "bge-m3-cpu:latest"},
    ]
}


ALL_MIGRATIONS = migration_versions()


class FakePool:
    """`applied`: versões em schema_migrations (None = tabela inexistente, banco virgem)."""

    def __init__(self, fail: bool = False, applied: list[str] | None = ALL_MIGRATIONS):
        self.fail = fail
        self.applied = applied

    async def fetchval(self, query, *args):
        if self.fail:
            raise ConnectionRefusedError("postgres fora do ar")
        if "to_regclass" in query:
            return self.applied is not None
        return 1

    async def fetch(self, query, *args):
        if self.fail:
            raise ConnectionRefusedError("postgres fora do ar")
        assert self.applied is not None, "consultou schema_migrations sem a tabela existir"
        return [{"version": v} for v in self.applied]


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def ping(self):
        if self.fail:
            raise ConnectionError("redis fora do ar")
        return True


def ready_api(*, pool_fail=False, redis_fail=False, applied=ALL_MIGRATIONS):
    app = make_app()
    app.config.models.primary = "ollama:qwen3.5:4b"
    app.config.models.fallbacks = ["ollama:llama3.2-3b-cpu"]
    api = make_api(app, settings=Settings(ollama_base_url=OLLAMA))
    api.h.deps.pool = FakePool(pool_fail, applied)
    api.h.deps.redis = FakeRedis(redis_fail)
    return api


async def test_models_lists_only_key_app():
    api = make_api(extra_apps=(make_app(app_id="b"), make_app(app_id="c")))
    async with api.client() as c:
        r = await c.get("/v1/models")
        one = await c.get("/v1/models/a")
        other = await c.get("/v1/models/b")
    async with api.client(key=None) as c:
        anonymous = await c.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"
    assert [m["id"] for m in r.json()["data"]] == ["a"]
    assert r.json()["data"][0]["object"] == "model"
    assert one.json()["id"] == "a"
    assert other.status_code == 404 and anonymous.status_code == 401


async def test_get_conversation_only_owner():
    history = ApiHistory(
        messages=[ChatMessage("user", "oi", seq=1), ChatMessage("assistant", "olá", seq=2)]
    )
    cid = history.add("a", "ana")
    api = make_api(history=history, extra_apps=(make_app(app_id="b"),))
    async with api.client(**{"X-User-Ref": "ana"}) as c:
        mine = await c.get(f"/v1/conversations/{cid}")
    async with api.client(**{"X-User-Ref": "bruno"}) as c:
        other_user = await c.get(f"/v1/conversations/{cid}")
    async with api.client(key=KEY_B, **{"X-User-Ref": "ana"}) as c:
        other_app = await c.get(f"/v1/conversations/{cid}")
    assert mine.status_code == 200
    body = mine.json()
    assert body["id"] == cid and body["app_id"] == "a"
    assert [(m["role"], m["content"], m["seq"]) for m in body["messages"]] == [
        ("user", "oi", 1),
        ("assistant", "olá", 2),
    ]
    assert other_user.status_code == other_app.status_code == 404


async def test_health_is_always_ok():
    api = ready_api(pool_fail=True, redis_fail=True)
    async with api.client(key=None) as c:
        r = await c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert r.headers["x-trace-id"]


async def test_ready_when_everything_is_up():
    api = ready_api()
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").respond(json=TAGS)
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready" and r.json()["missing"] == []


async def test_ready_503_lists_what_is_missing():
    api = ready_api(redis_fail=True)
    tags = {"models": [m for m in TAGS["models"] if not m["name"].startswith("qwen")]}
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").respond(json=tags)
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 503
    assert r.json()["missing"] == ["redis", "ollama:qwen3.5:4b"]


async def test_ready_503_when_ollama_and_postgres_are_down():
    api = ready_api(pool_fail=True)
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").mock(side_effect=httpx.ConnectError("recusado"))
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 503
    assert r.json()["missing"] == ["postgres", "ollama"]


async def test_ready_requires_both_content_guards():
    api = ready_api()
    tags = {"models": [m for m in TAGS["models"] if not m["name"].startswith("shieldgemma")]}
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").respond(json=tags)
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["ollama:shieldgemma-2b-cpu"]


async def test_ready_503_when_database_was_never_migrated():
    """Revisão final I4: banco novo sem `make migrate` não está pronto."""
    api = ready_api(applied=None)
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").respond(json=TAGS)
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["postgres:schema"]


async def test_ready_503_when_a_migration_is_pending():
    assert len(ALL_MIGRATIONS) >= 2
    api = ready_api(applied=ALL_MIGRATIONS[:-1])
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").respond(json=TAGS)
        async with api.client(key=None) as c:
            r = await c.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["postgres:schema"]
