import pytest

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
