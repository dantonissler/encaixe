import pytest
import pytest_asyncio
from fakeredis import aioredis as fakeredis_aio

from encaixe.settings import Settings

TEST_DB = "postgresql://encaixe_app:encaixe_app_dev@localhost:55432/encaixe_test"
TEST_ADMIN_DB = "postgresql://postgres:postgres_dev@localhost:55432/encaixe_test"


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url=TEST_DB, database_admin_url=TEST_ADMIN_DB)


@pytest_asyncio.fixture
async def fake_redis():
    client = fakeredis_aio.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def pg_pool(settings):
    # Importado aqui porque o pacote encaixe.db só existe a partir da Task 4.
    import asyncpg
    from encaixe.db import migrate
    from encaixe.db import pool as pool_mod

    await migrate.apply_migrations(settings.database_admin_url)
    pool = await pool_mod.create_pool(settings.database_url)
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' "
            "AND tablename <> 'schema_migrations'"
        )
    if tables:
        admin = await asyncpg.connect(settings.database_admin_url)
        try:
            names = ", ".join(f'"{t["tablename"]}"' for t in tables)
            await admin.execute(f"TRUNCATE {names} RESTART IDENTITY CASCADE")
        finally:
            await admin.close()
    yield pool
    await pool.close()
