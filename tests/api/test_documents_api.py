from __future__ import annotations

import hashlib

import pytest

from encaixe.api.routes import documents as documents_route
from encaixe.settings import Settings

from .helpers import KEY_B, make_api

DATA = "# Política de férias\n\nSão 30 dias corridos.\n".encode()


async def upload(c, name="politica-ferias.md", data=DATA, acl=None, app="a"):
    form = {"acl": acl} if acl is not None else {}
    return await c.post(
        f"/v1/apps/{app}/documents", files={"file": (name, data, "text/markdown")}, data=form
    )


def stored_files(tmp_path) -> list:
    return sorted(p for p in tmp_path.rglob("*") if p.is_file())


@pytest.fixture
def api(tmp_path):
    return make_api(settings=Settings(upload_dir=tmp_path / "uploads"))


async def test_upload_enqueues_and_dedups(api, tmp_path):
    async with api.client() as c:
        first = await upload(c, acl="group:rh, user:ana")
        again = await upload(c, name="outra-copia.md", acl="group:rh,user:ana")
    assert first.status_code == 202
    doc_id = first.json()["document_id"]
    assert first.json()["status"] == "pending" and first.json()["duplicate"] is False
    assert api.h.enqueued == [("ingest_document", "a", doc_id)]

    assert again.status_code == 200
    assert again.json()["document_id"] == doc_id and again.json()["duplicate"] is True
    assert api.h.enqueued == [("ingest_document", "a", doc_id)]  # não enfileira de novo

    files = stored_files(tmp_path)
    assert len(files) == 1  # a cópia duplicada não fica no disco
    assert files[0].parent == (tmp_path / "uploads" / "a").resolve()
    assert files[0].name.endswith("-politica-ferias.md") and files[0].read_bytes() == DATA
    doc, content_hash = api.docs.rows[("a", doc_id)]
    assert content_hash == hashlib.sha256(DATA).hexdigest()
    assert doc.acl == ["group:rh", "user:ana"]
    assert doc.source_path == str(files[0]) and doc.title == "politica-ferias"
    assert "source_path" not in first.json()  # caminho do servidor não vaza


async def test_upload_without_acl_is_public(api):
    async with api.client() as c:
        r = await upload(c)
    assert r.status_code == 202
    assert api.docs.rows[("a", r.json()["document_id"])][0].acl == ["app:*"]


async def test_get_document_status(api):
    async with api.client() as c:
        doc_id = (await upload(c)).json()["document_id"]
        api.docs.rows[("a", doc_id)][0].status = "quarantined"
        api.docs.rows[("a", doc_id)][0].quarantined = True
        r = await c.get(f"/v1/apps/a/documents/{doc_id}")
        missing = await c.get("/v1/apps/a/documents/00000000-0000-0000-0000-000000000000")
        malformed = await c.get("/v1/apps/a/documents/nao-e-uuid")
    assert r.status_code == 200
    assert r.json()["status"] == "quarantined" and r.json()["quarantined"] is True
    assert missing.status_code == malformed.status_code == 404


async def test_delete_bumps_corpus(api, tmp_path):
    async with api.client() as c:
        doc_id = (await upload(c)).json()["document_id"]
        assert len(stored_files(tmp_path)) == 1
        r = await c.delete(f"/v1/apps/a/documents/{doc_id}")
        gone = await c.get(f"/v1/apps/a/documents/{doc_id}")
        again = await c.delete(f"/v1/apps/a/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json() == {"document_id": doc_id, "deleted": True, "corpus_version": 1}
    assert api.docs.corpus == {"a": 1}
    assert ("a", doc_id) not in api.docs.rows
    assert stored_files(tmp_path) == []
    assert gone.status_code == again.status_code == 404


async def test_delete_never_removes_files_outside_upload_dir(api, tmp_path):
    outside = tmp_path / "fora.md"
    outside.write_bytes(DATA)
    doc_id = await api.docs.create("a", "fora", str(outside), "h", ["app:*"])
    async with api.client() as c:
        r = await c.delete(f"/v1/apps/a/documents/{doc_id}")
    assert r.status_code == 200
    assert outside.exists()


async def test_documents_of_another_app_are_404(api):
    async with api.client() as c:
        r = await upload(c, app="b")
        doc_id = (await upload(c)).json()["document_id"]
    async with api.client(key=KEY_B) as c:
        get = await c.get(f"/v1/apps/a/documents/{doc_id}")
        delete = await c.delete(f"/v1/apps/a/documents/{doc_id}")
    assert r.status_code == get.status_code == delete.status_code == 404
    assert ("a", doc_id) in api.docs.rows


async def test_upload_filename_path_traversal_is_neutralized(api, tmp_path):
    names = [
        "../../../etc/passwd",
        "..\\..\\segredo.md",
        "..",
        "/abs/../.oculto.md",
        "rel*at?ório:1.md",
    ]
    async with api.client() as c:
        responses = [await upload(c, name=n, data=f"doc {i}".encode()) for i, n in enumerate(names)]
    assert [r.status_code for r in responses] == [202] * len(names)
    root = (tmp_path / "uploads" / "a").resolve()
    files = stored_files(tmp_path)
    assert len(files) == len(names)
    assert all(f.parent == root for f in files)
    suffixes = sorted(f.name.split("-", 1)[1] for f in files)
    assert suffixes == sorted(
        ["passwd", "segredo.md", "documento", "oculto.md", "rel_at_ório_1.md"]
    )


async def test_oversized_upload_is_413(api, monkeypatch, tmp_path):
    big = b"x" * (documents_route.MAX_UPLOAD_BYTES + 1)
    async with api.client() as c:
        r = await upload(c, data=big)
    assert r.status_code == 413
    # Content-Length já acima do limite: recusa antes de ler o formulário
    monkeypatch.setattr(documents_route, "MAX_UPLOAD_BYTES", 1024)
    async with api.client() as c:
        early = await upload(c, data=b"y" * (documents_route.FORM_OVERHEAD + 4096))
    assert early.status_code == 413
    assert api.h.enqueued == [] and api.docs.rows == {}
    assert stored_files(tmp_path) == []


async def test_invalid_upload_requests_are_400(api):
    async with api.client() as c:
        bad_acl = await upload(c, acl="rh")
        empty_ref = await upload(c, acl="user:")
        no_file = await c.post("/v1/apps/a/documents", data={"acl": "app:*"})
        empty = await upload(c, data=b"")
    assert [r.status_code for r in (bad_acl, empty_ref, no_file, empty)] == [400] * 4
    assert api.docs.rows == {}


async def test_enqueue_failure_rolls_back(api, tmp_path):
    async def broken(*args):
        raise ConnectionError("redis fora do ar")

    api.h.deps.enqueue = broken
    async with api.client() as c:
        r = await upload(c)
    assert r.status_code == 503
    assert api.docs.rows == {} and stored_files(tmp_path) == []
