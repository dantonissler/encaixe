import pytest

from encaixe.db.repos.keys import create_key, resolve_key

pytestmark = pytest.mark.integration


async def test_create_and_resolve_and_revoked(pg_pool, settings):
    secret = await create_key(pg_pool, "exemplo")
    assert secret.startswith("enc_")
    assert await resolve_key(pg_pool, secret) == "exemplo"
    assert await resolve_key(pg_pool, "enc_nope") is None
    async with pg_pool.acquire() as conn:
        raw = await conn.fetchval("SELECT count(*) FROM api_keys WHERE key_hash = $1", secret)
        assert raw == 0  # only the hash is stored
        await conn.execute("UPDATE api_keys SET revoked_at = now()")
    assert await resolve_key(pg_pool, secret) is None
