from encaixe.seed import format_table, prepare_upload, split_front_matter, wait_all

DOC = "---\nacl:\n  - group:rh\n---\n# Título\n\ntexto\n"


def test_split_front_matter():
    assert split_front_matter(DOC) == (["group:rh"], "# Título\n\ntexto\n")
    assert split_front_matter("---\nacl: []\n---\n# T\n") == ([], "# T\n")
    assert split_front_matter("# sem front-matter\n") == ([], "# sem front-matter\n")


def test_prepare_upload_strips_and_overrides(tmp_path):
    f = tmp_path / "d.md"
    f.write_text(DOC, encoding="utf-8")
    assert prepare_upload(f) == (b"# T\xc3\xadtulo\n\ntexto\n", ["group:rh"])
    assert prepare_upload(f, ["user:ana"])[1] == ["user:ana"]
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF")
    assert prepare_upload(pdf, None) == (b"%PDF", [])


def test_wait_all_polls_until_terminal_or_timeout():
    class Resp:
        def __init__(self, status):
            self.status = status

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": self.status}

    calls = []

    class Client:
        def get(self, url):
            calls.append(url)
            return Resp("indexed" if len(calls) > 1 else "processing")

    now = [0.0]
    rows = wait_all(
        Client(), "a", [{"document_id": "1", "status": "pending", "title": "x"}],
        sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0],
    )  # fmt: skip
    assert rows[0]["status"] == "indexed" and len(calls) == 2

    class Stuck:
        def get(self, url):
            return Resp("pending")

    rows = wait_all(
        Stuck(), "a", [{"document_id": "1", "status": "pending", "title": "x"}],
        timeout_s=4, sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0],
    )  # fmt: skip
    assert rows[0]["status"] == "pending"
    assert "x" in format_table(rows)
