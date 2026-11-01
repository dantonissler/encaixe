import pytest

from encaixe.core.types import ChatMessage
from encaixe.db.repos.conversations import ConversationNotFound, PgHistoryStore

pytestmark = pytest.mark.integration


async def test_append_and_list_in_order(pg_pool):
    store = PgHistoryStore(pg_pool)
    cid = await store.create_conversation("a", "u1")
    assert await store.get_owner("a", cid) == "u1"
    s1 = await store.append("a", cid, ChatMessage(role="user", content="oi"))
    s2 = await store.append(
        "a",
        cid,
        ChatMessage(role="assistant", content="olá", tool_calls=[{"name": "x"}], tokens=3),
        trace_id="t",
        prompt_version="v1",
        model="m",
    )
    assert (s1, s2) == (1, 2)
    msgs = await store.list_messages("a", cid)
    assert [(m.seq, m.role, m.content) for m in msgs] == [(1, "user", "oi"), (2, "assistant", "olá")]
    assert msgs[1].tool_calls == [{"name": "x"}] and msgs[1].tokens == 3
    assert [m.seq for m in await store.list_messages("a", cid, after_seq=1)] == [2]


async def test_get_owner_unknown_or_other_app(pg_pool):
    store = PgHistoryStore(pg_pool)
    cid = await store.create_conversation("a", None)
    assert await store.get_owner("a", cid) is None
    with pytest.raises(ConversationNotFound):
        await store.get_owner("b", cid)


async def test_summary_roundtrip(pg_pool):
    store = PgHistoryStore(pg_pool)
    cid = await store.create_conversation("a", "u")
    assert await store.get_summary("a", cid) is None
    await store.save_summary("a", cid, "resumo 1", 4)
    await store.save_summary("a", cid, "resumo 2", 8)
    assert await store.get_summary("a", cid) == ("resumo 2", 8)
