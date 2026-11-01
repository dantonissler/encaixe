import asyncpg
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


async def _seed_restricted(pg_pool, app_id="a"):
    async with app_scope(pg_pool, app_id, None, system=True) as conn:
        doc = await conn.fetchval(
            "INSERT INTO documents (app_id, title) VALUES ($1, 't') RETURNING id", app_id
        )
        await conn.execute(
            "INSERT INTO chunks (app_id, document_id, text, acl_principals) "
            "VALUES ($1, $2, 'x', ARRAY['group:rh'])",
            app_id,
            doc,
        )
    return doc


async def test_system_scope_can_select_and_delete_restricted(pg_pool):
    doc = await _seed_restricted(pg_pool)
    async with app_scope(pg_pool, "a", None) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0
        assert await conn.fetchval("DELETE FROM chunks WHERE document_id=$1 RETURNING 1", doc) is None
    async with app_scope(pg_pool, "a", None, system=True) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 1
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc)
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0


async def test_system_scope_still_isolated_by_app(pg_pool):
    await _seed_restricted(pg_pool, "a")
    async with app_scope(pg_pool, "b", None, system=True) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0


async def test_no_principals_sees_only_public_chunks(pg_pool):
    async with app_scope(pg_pool, "a", None, system=True) as conn:
        doc = await conn.fetchval(
            "INSERT INTO documents (app_id, title) VALUES ('a', 't') RETURNING id"
        )
        for acl in (["app:*"], ["group:rh"]):
            await conn.execute(
                "INSERT INTO chunks (app_id, document_id, text, acl_principals) "
                "VALUES ('a', $1, 'x', $2)",
                doc,
                acl,
            )
    async with app_scope(pg_pool, "a", None) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 0
    async with app_scope(pg_pool, "a", Principals(user_ref=None)) as conn:
        assert await conn.fetchval("SELECT count(*) FROM chunks") == 1


async def test_with_check_rejects_mismatched_app_id(pg_pool):
    async with app_scope(pg_pool, "a", None) as conn:
        with pytest.raises(asyncpg.PostgresError):
            await conn.execute("INSERT INTO conversations (app_id) VALUES ('b')")


async def test_app_scope_rejects_bad_input(pg_pool):
    with pytest.raises(ValueError):
        async with app_scope(pg_pool, "", None):
            pass


async def test_app_role_cannot_write_schema_migrations(pg_pool):
    async with pg_pool.acquire() as conn:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("INSERT INTO schema_migrations (version) VALUES ('x')")


async def test_app_scope_rejects_comma_principal(pg_pool):
    p = Principals(user_ref="ok")
    object.__setattr__(p, "user_ref", "x,group:rh")  # contorna a validação do construtor
    with pytest.raises(ValueError):
        async with app_scope(pg_pool, "a", p):
            pass
