import pytest

from encaixe.db.migrate import apply_migrations

pytestmark = pytest.mark.integration


async def test_apply_is_idempotent(settings):
    await apply_migrations(settings.database_admin_url)
    assert await apply_migrations(settings.database_admin_url) == []
