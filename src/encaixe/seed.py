"""Envio de documentos pela API (`encaixe seed` e `encaixe ingest`).

Os arquivos .md podem trazer um front-matter YAML com `acl:` (lista de principais); ele vira o
campo `acl` do upload e é removido do texto, para o YAML não ser indexado.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import yaml

TERMINAL = {"indexed", "quarantined", "failed"}


def split_front_matter(text: str) -> tuple[list[str], str]:
    """(acl do front-matter, texto sem o front-matter). Sem front-matter: ([], texto)."""
    if not text.startswith("---"):
        return [], text
    head, sep, rest = text[3:].partition("\n---")
    if not sep:
        return [], text
    meta = yaml.safe_load(head) or {}
    acl = meta.get("acl") if isinstance(meta, dict) else None
    items = [acl] if isinstance(acl, str) else list(acl or [])
    body = rest.split("\n", 1)[1] if "\n" in rest else ""
    return [str(i) for i in items], body.lstrip("\n")


def prepare_upload(path: Path, acl_override: list[str] | None = None) -> tuple[bytes, list[str]]:
    """(conteúdo a enviar, acl). `acl_override` vence o front-matter."""
    raw = path.read_bytes()
    if path.suffix.lower() != ".md":
        return raw, acl_override or []
    acl, body = split_front_matter(raw.decode("utf-8"))
    return body.encode("utf-8"), acl_override if acl_override is not None else acl


def api_url() -> str:
    return os.environ.get("ENCAIXE_API_URL", "http://localhost:8100").rstrip("/")


def upload(
    client: httpx.Client, app_id: str, path: Path, acl_override: list[str] | None = None
) -> dict:
    content, acl = prepare_upload(path, acl_override)
    resp = client.post(
        f"/v1/apps/{app_id}/documents",
        files={"file": (path.name, content, "text/markdown")},
        data={"acl": ",".join(acl)} if acl else {},
    )
    resp.raise_for_status()
    return resp.json()


def wait_all(
    client: httpx.Client,
    app_id: str,
    docs: list[dict],
    *,
    timeout_s: float = 300,
    interval_s: float = 2,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict]:
    """Consulta cada documento até um estado final (ou o prazo); devolve as linhas finais."""
    deadline = clock() + timeout_s
    current = {d["document_id"]: d for d in docs}
    while True:
        for doc_id, doc in current.items():
            if doc["status"] not in TERMINAL:
                resp = client.get(f"/v1/apps/{app_id}/documents/{doc_id}")
                resp.raise_for_status()
                current[doc_id] = {**doc, **resp.json()}
        if all(d["status"] in TERMINAL for d in current.values()) or clock() >= deadline:
            return list(current.values())
        sleep(interval_s)


def format_table(rows: list[dict]) -> str:
    lines = [f"{'documento':<28} {'status':<12} {'acl':<14} observação"]
    for r in rows:
        lines.append(
            f"{r['title']:<28} {r['status']:<12} {','.join(r.get('acl') or []):<14} "
            f"{r.get('error') or ''}"
        )
    return "\n".join(lines)
