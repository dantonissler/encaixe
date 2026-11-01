import pytest

from encaixe.core.types import Principals
from encaixe.db.pool import app_scope

pytestmark = pytest.mark.integration


async def test_other_app_cannot_read_conversation(pg_pool):
    async with app_scope(pg_pool, "a", None) as conn:
        cid = await conn.fetchval(
            "INSERT INTO conversations (app_id, user_ref) VALUES ('a', 'u') RETURNING id"
        )
    async with app_scope(pg_pool, "b", None) as conn:
        rows = await conn.fetch("SELECT id FROM conversations WHERE id = $1", cid)
    assert rows == []
    async with app_scope(pg_pool, "a", None) as conn:
        assert await conn.fetchval("SELECT count(*) FROM conversations") == 1


async def test_unset_scope_sees_nothing(pg_pool):
    async with app_scope(pg_pool, "a", None) as conn:
        await conn.execute("INSERT INTO conversations (app_id) VALUES ('a')")
    async with pg_pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM conversations") == 0


async def test_chunk_acl_policy_hides_restricted(pg_pool):
    async with app_scope(pg_pool, "a", Principals(user_ref=None)) as conn:
        doc = await conn.fetchval(
            "INSERT INTO documents (app_id, title) VALUES ('a', 't') RETURNING id"
        )
        await conn.execute(
            "INSERT INTO chunks (app_id, document_id, text, acl_principals) "
            "VALUES ('a', $1, 'Remuneração do cargo', ARRAY['group:rh'])",
            doc,
        )
    async with app_scope(pg_pool, "a", Principals(user_ref="x")) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0
    async with app_scope(pg_pool, "a", Principals(user_ref="x", groups=("rh",))) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 1
        # tsv generated with unaccent
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('pt_unaccent', 'remuneracao')"
            )
            == 1
        )
