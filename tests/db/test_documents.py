import pytest

from encaixe.db.repos import documents as docs

pytestmark = pytest.mark.integration


async def test_dedup_status_and_corpus_version(pg_pool):
    d1 = await docs.create_document(pg_pool, "appx", "T", "/x.md", "h1", ["app:*"])
    assert d1
    assert await docs.create_document(pg_pool, "appx", "T", "/x.md", "h1", ["app:*"]) is None
    assert await docs.create_document(pg_pool, "appy", "T", "/x.md", "h1", ["app:*"])
    await docs.set_status(pg_pool, "appx", d1, "quarantined", quarantined=True, error="e")
    d = await docs.get_document(pg_pool, "appx", d1)
    assert (d.status, d.quarantined, d.error, d.acl) == ("quarantined", True, "e", ["app:*"])
    assert await docs.get_corpus_version(pg_pool, "appx") == 0
    assert await docs.bump_corpus_version(pg_pool, "appx") == 1
    assert await docs.bump_corpus_version(pg_pool, "appx") == 2
    await docs.delete_document(pg_pool, "appx", d1)
    assert await docs.get_document(pg_pool, "appx", d1) is None


async def test_get_document_by_hash_is_scoped_by_app(pg_pool):
    d1 = await docs.create_document(pg_pool, "appx", "T", "/x.md", "h1", ["group:rh"])
    found = await docs.get_document_by_hash(pg_pool, "appx", "h1")
    assert (found.id, found.acl, found.status) == (d1, ["group:rh"], "pending")
    assert await docs.get_document_by_hash(pg_pool, "appy", "h1") is None
    assert await docs.get_document_by_hash(pg_pool, "appx", "h2") is None
