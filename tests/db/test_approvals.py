import asyncio

import pytest

from encaixe.db.pool import app_scope
from encaixe.db.repos.approvals import PgApprovalStore
from encaixe.db.repos.conversations import PgHistoryStore

pytestmark = pytest.mark.integration


async def test_expired_pending_not_returned(pg_pool):
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    await store.create_pending("a", cid, "u", "srv", "tool", {"x": 1}, ttl_s=-1)
    assert await store.get_pending("a", cid, "u") is None


async def test_pending_resolve_and_isolation(pg_pool):
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    aid = await store.create_pending("a", cid, "u", "srv", "tool", {"x": 1})
    got = await store.get_pending("a", cid, "u")
    assert got is not None and got.id == aid and got.args == {"x": 1} and got.tool == "tool"
    assert await store.get_pending("a", cid, "other") is None
    assert await store.get_pending("b", cid, "u") is None
    await store.resolve("a", aid, "approved")
    assert await store.get_pending("a", cid, "u") is None


async def test_resolve_expired_returns_false_and_unchanged(pg_pool):
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    aid = await store.create_pending("a", cid, "u", "s", "t", {}, ttl_s=-1)
    assert await store.resolve("a", aid, "approved") is False
    async with app_scope(pg_pool, "a", None) as conn:
        assert await conn.fetchval("SELECT status FROM tool_approvals WHERE id=$1", aid) == "pending"
    live = await store.create_pending("a", cid, "u", "s", "t", {})
    assert await store.resolve("a", live, "cancelled") is True
    assert await store.resolve("a", live, "approved") is False


async def test_same_pending_request_reuses_row(pg_pool):
    """Mesma ferramenta e mesmos args pedidos de novo na conversa (ex.: o modelo repete a
    chamada no mesmo turno) reaproveitam a aprovação pendente em vez de criar outra órfã."""
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    first = await store.create_pending("a", cid, "u", "srv", "criar", {"t": "X", "p": "1"})
    again = await store.create_pending("a", cid, "u", "srv", "criar", {"p": "1", "t": "X"})
    assert again == first
    other_args = await store.create_pending("a", cid, "u", "srv", "criar", {"t": "Y"})
    other_user = await store.create_pending("a", cid, "v", "srv", "criar", {"t": "X", "p": "1"})
    assert len({first, other_args, other_user}) == 3
    async with app_scope(pg_pool, "a", None) as conn:
        n = await conn.fetchval("SELECT count(*) FROM tool_approvals WHERE conversation_id=$1", cid)
    assert n == 3


async def test_concurrent_identical_requests_create_one_row(pg_pool):
    """Duas chamadas iguais na mesma resposta do modelo rodam em paralelo."""
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    ids = await asyncio.gather(
        *(store.create_pending("a", cid, "u", "srv", "criar", {"t": "X"}) for _ in range(4))
    )
    assert len(set(ids)) == 1


async def test_resolved_or_expired_request_is_not_reused(pg_pool):
    cid = await PgHistoryStore(pg_pool).create_conversation("a", "u")
    store = PgApprovalStore(pg_pool)
    expired = await store.create_pending("a", cid, "u", "srv", "criar", {"t": "X"}, ttl_s=-1)
    live = await store.create_pending("a", cid, "u", "srv", "criar", {"t": "X"})
    assert live != expired
    assert await store.resolve("a", live, "approved") is True
    new = await store.create_pending("a", cid, "u", "srv", "criar", {"t": "X"})
    assert new not in (expired, live)
    got = await store.get_pending("a", cid, "u")
    assert got is not None and got.id == new
