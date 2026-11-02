"""Documentos da app: upload (→ job de ingestão), status/quarentena e remoção.

O arquivo vai para `upload_dir/<app>/<uuid>-<nome saneado>`; o mesmo conteúdo (sha256) na
mesma app não é reprocessado (200 com o documento existente). Remover apaga a linha (os
chunks saem em cascata), o arquivo e incrementa a versão do corpus (invalida o cache).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, Any, Protocol

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from encaixe.api.auth import get_deps, require_app
from encaixe.api.errors import RETRY_AFTER, UNAVAILABLE, ApiError, service_unavailable
from encaixe.db.repos import documents as docs
from encaixe.db.repos.documents import Document
from encaixe.wiring import Deps

log = structlog.get_logger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FORM_OVERHEAD = 64 * 1024  # cabeçalhos multipart e campos além do arquivo
# Principais aceitos na ACL (spec §6.4a); vírgula separa os itens e nunca faz parte de um.
# Espaço e caracteres de controle também não: "user: ana" nunca casaria com o principal.
ACL_ITEM = re.compile(r"app:\*|(user|group):[^,\s\x00-\x1f\x7f-\x9f]+")
_UNSAFE = re.compile(r"[^\w.\- ]")
# O arquivo é `<uuid hex>-<nome>`; nomes de arquivo têm no máximo 255 bytes (ext4 etc.).
MAX_FILENAME_BYTES = 255 - len(uuid.uuid4().hex) - 1
MAX_SUFFIX_BYTES = 16


class DocumentRepo(Protocol):
    async def create(
        self, app_id: str, title: str, source_path: str, content_hash: str, acl: list[str]
    ) -> str | None: ...

    async def get(self, app_id: str, document_id: str) -> Document | None: ...

    async def get_by_hash(self, app_id: str, content_hash: str) -> Document | None: ...

    async def delete(self, app_id: str, document_id: str) -> None: ...

    async def bump_corpus(self, app_id: str) -> int: ...


class PgDocumentRepo:
    """`DocumentRepo` sobre as funções de `encaixe.db.repos.documents` (escopo de sistema)."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def create(self, app_id, title, source_path, content_hash, acl):
        return await docs.create_document(self.pool, app_id, title, source_path, content_hash, acl)

    async def get(self, app_id, document_id):
        return await docs.get_document(self.pool, app_id, document_id)

    async def get_by_hash(self, app_id, content_hash):
        return await docs.get_document_by_hash(self.pool, app_id, content_hash)

    async def delete(self, app_id, document_id):
        await docs.delete_document(self.pool, app_id, document_id)

    async def bump_corpus(self, app_id):
        return await docs.bump_corpus_version(self.pool, app_id)


async def document_repo(deps: Annotated[Deps, Depends(get_deps)]) -> DocumentRepo:
    return PgDocumentRepo(deps.pool)


Repo = Annotated[DocumentRepo, Depends(document_repo)]
KeyApp = Annotated[str, Depends(require_app)]
AppDeps = Annotated[Deps, Depends(get_deps)]


def _fit_bytes(text: str, limit: int) -> str:
    """Prefixo de `text` com no máximo `limit` bytes em UTF-8, sem partir caractere."""
    return text.encode()[: max(limit, 0)].decode("utf-8", "ignore")


def safe_filename(name: str | None) -> str:
    """Só o nome-base, sem diretórios (`/` ou `\\`), sem ponto inicial e só com caracteres
    seguros; o que sobrar vazio vira `documento`. Cabe em `MAX_FILENAME_BYTES` bytes (UTF-8),
    cortando o nome e mantendo a extensão (o Docling escolhe o formato por ela)."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = _UNSAFE.sub("_", unicodedata.normalize("NFC", base)).strip(" .")
    if len(base.encode()) > MAX_FILENAME_BYTES:
        stem, dot, ext = base.rpartition(".")
        suffix = f".{ext}" if dot and stem and len(ext.encode()) < MAX_SUFFIX_BYTES else ""
        stem = base[: len(base) - len(suffix)]
        base = _fit_bytes(stem, MAX_FILENAME_BYTES - len(suffix.encode())).rstrip(" .") + suffix
    return base or "documento"


def parse_acl(raw: str | None) -> list[str]:
    """CSV de principais; vazio → `["app:*"]` (público dentro da app)."""
    items: list[str] = []
    for part in (raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        if not ACL_ITEM.fullmatch(item):
            raise ApiError(
                400,
                f"ACL inválida: '{item}'. Use app:*, user:<id> ou group:<nome>.",
                code="invalid_acl",
            )
        if item not in items:
            items.append(item)
    return items or ["app:*"]


def document_view(doc: Document) -> dict[str, Any]:
    # source_path fica de fora: é um caminho do servidor.
    return {
        "document_id": doc.id,
        "title": doc.title,
        "status": doc.status,
        "quarantined": doc.quarantined,
        "error": doc.error,
        "acl": doc.acl,
    }


def _same_app(key_app: str, app_id: str) -> None:
    if app_id != key_app:
        raise ApiError(404, f"App '{app_id}' não encontrada.", code="model_not_found")


def _document_id(raw: str) -> str:
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        raise _not_found() from None


def _not_found() -> ApiError:
    return ApiError(404, "Documento não encontrado.", code="document_not_found")


def _too_large() -> ApiError:
    limit = MAX_UPLOAD_BYTES // (1024 * 1024)
    return ApiError(413, f"Arquivo maior que o limite de {limit} MB.", code="file_too_large")


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


async def _read_upload(request: Request) -> tuple[bytes, str | None, list[str]]:
    """(conteúdo, nome original, acl) do multipart, sem ler além do limite."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + FORM_OVERHEAD:
        raise _too_large()  # recusa antes de receber o corpo
    form = await request.form(max_files=1, max_fields=4)
    try:
        upload, acl_raw = form.get("file"), form.get("acl")
        if not isinstance(upload, UploadFile):
            raise ApiError(400, "Envie o arquivo no campo 'file' (multipart/form-data).")
        if acl_raw is not None and not isinstance(acl_raw, str):
            raise ApiError(400, "O campo 'acl' deve ser texto (CSV).")
        acl = parse_acl(acl_raw)
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        filename = upload.filename
    finally:
        await form.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise _too_large()
    if not data:
        raise ApiError(400, "Arquivo vazio.")
    return data, filename, acl


def _unlink(path: Path) -> None:
    path.unlink(missing_ok=True)


_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "acl": {
                            "type": "string",
                            "description": "CSV: app:*, user:<id>, group:<nome>",
                        },
                    },
                }
            }
        },
    }
}


@router.post("/v1/apps/{app_id}/documents", status_code=202, openapi_extra=_UPLOAD_OPENAPI)
async def upload_document(
    app_id: str, request: Request, key_app: KeyApp, deps: AppDeps, repo: Repo
) -> JSONResponse:
    _same_app(key_app, app_id)
    data, filename, acl = await _read_upload(request)
    content_hash = hashlib.sha256(data).hexdigest()
    name = safe_filename(filename)
    root = (Path(deps.settings.upload_dir) / app_id).resolve()
    path = root / f"{uuid.uuid4().hex}-{name}"
    if not _inside(path, root):  # defesa em profundidade: o nome já foi saneado
        raise ApiError(400, "Nome de arquivo inválido.")

    async with service_unavailable("postgres"):
        document_id = await repo.create(
            app_id, Path(name).stem or name, str(path), content_hash, acl
        )
        existing = None if document_id else await repo.get_by_hash(app_id, content_hash)
    if document_id is None:
        if existing is None:  # o original foi removido entre o INSERT e a busca
            raise ApiError(409, "Documento em remoção; tente de novo.", code="conflict")
        return JSONResponse({**document_view(existing), "duplicate": True}, status_code=200)

    try:
        root.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        if deps.enqueue is None:
            raise RuntimeError("fila de ingestão não configurada")
        await deps.enqueue("ingest_document", app_id, document_id)
    except Exception as exc:
        # Sem arquivo ou sem job o documento ficaria 'pending' para sempre e barraria o
        # reenvio (dedup): desfaz tudo e pede para tentar de novo.
        log.error("document_upload_failed", app_id=app_id, error=type(exc).__name__)
        try:
            await repo.delete(app_id, document_id)
            await asyncio.to_thread(_unlink, path)
        except Exception as cleanup:  # noqa: BLE001
            log.error("document_upload_cleanup_failed", error=type(cleanup).__name__)
        raise ApiError(503, UNAVAILABLE, code="service_unavailable", headers=RETRY_AFTER) from exc

    log.info("document_uploaded", app_id=app_id, document_id=document_id, bytes=len(data))
    body = {
        "document_id": document_id,
        "title": Path(name).stem or name,
        "status": "pending",
        "quarantined": False,
        "error": None,
        "acl": acl,
        "duplicate": False,
    }
    return JSONResponse(body, status_code=202)


@router.get("/v1/apps/{app_id}/documents/{document_id}")
async def get_document(
    app_id: str, document_id: str, key_app: KeyApp, repo: Repo
) -> dict[str, Any]:
    _same_app(key_app, app_id)
    async with service_unavailable("postgres"):
        doc = await repo.get(app_id, _document_id(document_id))
    if doc is None:
        raise _not_found()
    return document_view(doc)


@router.delete("/v1/apps/{app_id}/documents/{document_id}")
async def delete_document(
    app_id: str, document_id: str, key_app: KeyApp, deps: AppDeps, repo: Repo
) -> dict[str, Any]:
    _same_app(key_app, app_id)
    doc_id = _document_id(document_id)
    async with service_unavailable("postgres"):
        doc = await repo.get(app_id, doc_id)
        if doc is None:
            raise _not_found()
        await repo.delete(app_id, doc_id)
        version = await repo.bump_corpus(app_id)
    if doc.source_path:
        source = Path(doc.source_path)
        if _inside(source, Path(deps.settings.upload_dir)):
            try:
                await asyncio.to_thread(_unlink, source)
            except OSError as exc:  # a linha já saiu; o arquivo órfão não afeta respostas
                log.warning("document_file_remove_failed", error=type(exc).__name__)
        else:
            log.warning("document_file_outside_upload_dir", app_id=app_id, document_id=doc_id)
    log.info("document_deleted", app_id=app_id, document_id=doc_id, corpus_version=version)
    return {"document_id": doc_id, "deleted": True, "corpus_version": version}
