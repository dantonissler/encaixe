import pytest

from encaixe.db.migrate import apply_migrations, migration_versions, pending_migrations
from encaixe.db.pool import create_pool

pytestmark = pytest.mark.integration


async def test_apply_is_idempotent(settings):
    await apply_migrations(settings.database_admin_url)
    assert await apply_migrations(settings.database_admin_url) == []


def test_migration_versions_are_the_sql_files_in_order():
    assert migration_versions()[:2] == ["0001_init", "0002_documents_hash"]


async def test_nothing_pending_after_apply(settings):
    await apply_migrations(settings.database_admin_url)
    pool = await create_pool(settings.database_url)  # papel da API (só lê schema_migrations)
    try:
        assert await pending_migrations(pool) == []
    finally:
        await pool.close()
