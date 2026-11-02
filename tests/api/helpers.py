"""API real (FastAPI + httpx ASGITransport) sobre dependências falsas.

Reaproveita os fakes do pipeline (`tests/pipeline/helpers.py`) e acrescenta o que só a API usa:
criação/dono de conversas, chaves, repositório de documentos e um cliente ASGI mínimo que
consegue desconectar no meio da resposta (o ASGITransport do httpx só devolve a resposta
depois que ela termina, então não serve para testar desconexão).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic_ai.messages import ModelRequest, ModelResponse

from encaixe.api.app import create_app
from encaixe.api.auth import key_resolver
from encaixe.api.routes.documents import document_repo
from encaixe.core.config import LoadedApp
from encaixe.db.repos.conversations import ConversationNotFound
from encaixe.db.repos.documents import Document
from encaixe.pipeline.gate import GenerationGate
from pipeline.helpers import FakeHistory, Harness, Script, make_app, make_harness

CHAT = "/v1/chat/completions"
KEY = "enc_chave_a"
KEY_B = "enc_chave_b"
KEYS = {KEY: "a", KEY_B: "b"}
QUESTION = "Qual é a política de férias?"


class ApiHistory(FakeHistory):
    """Histórico falso que também cria conversas e informa o dono (trabalho da API)."""

    def __init__(self, messages=None, summary=None):
        super().__init__(messages, summary)
        self.owners: dict[tuple[str, str], str | None] = {}

    def add(self, app_id: str, user_ref: str | None) -> str:
        cid = str(uuid.uuid4())
        self.owners[(app_id, cid)] = user_ref
        return cid

    async def create_conversation(self, app_id, user_ref):
        return self.add(app_id, user_ref)

    async def get_owner(self, app_id, conversation_id):
        try:
            return self.owners[(app_id, conversation_id)]
        except KeyError:
            raise ConversationNotFound(conversation_id) from None


class FakeDocs:
    """Repositório de documentos em memória (mesma semântica de dedup do Postgres)."""

    def __init__(self):
        self.rows: dict[tuple[str, str], tuple[Document, str]] = {}
        self.corpus: dict[str, int] = {}

    async def create(self, app_id, title, source_path, content_hash, acl):
        if await self.get_by_hash(app_id, content_hash) is not None:
            return None
        doc_id = str(uuid.uuid4())
        doc = Document(doc_id, app_id, title, source_path, list(acl), "pending", False, None)
        self.rows[(app_id, doc_id)] = (doc, content_hash)
        return doc_id

    async def get(self, app_id, document_id):
        row = self.rows.get((app_id, document_id))
        return row[0] if row else None

    async def get_by_hash(self, app_id, content_hash):
        for (a, _), (doc, h) in self.rows.items():
            if a == app_id and h == content_hash:
                return doc
        return None

    async def delete(self, app_id, document_id):
        self.rows.pop((app_id, document_id), None)

    async def bump_corpus(self, app_id):
        self.corpus[app_id] = self.corpus.get(app_id, 0) + 1
        return self.corpus[app_id]


@dataclass
class Api:
    app: FastAPI
    h: Harness
    history: ApiHistory
    docs: FakeDocs
    gate: GenerationGate

    def client(self, key: str | None = KEY, **headers: str) -> httpx.AsyncClient:
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test", headers=headers
        )


def make_api(
    app: LoadedApp | None = None,
    script: Script | None = None,
    *,
    gate: GenerationGate | None = None,
    history: ApiHistory | None = None,
    extra_apps: tuple[LoadedApp, ...] = (),
    tracer_provider: Any = None,
    **harness: Any,
) -> Api:
    app = app or make_app()
    history = history if history is not None else ApiHistory()
    h = make_harness(app, script, history=history, **harness)
    for extra in extra_apps:
        h.deps.apps[extra.config.id] = extra
    h.deps.gate = gate or GenerationGate(1, 5, 5)
    docs = FakeDocs()
    api = create_app(h.deps, tracer_provider=tracer_provider)

    async def resolve(secret: str) -> str | None:
        return KEYS.get(secret)

    async def override_keys():
        return resolve

    async def override_docs():
        return docs

    api.dependency_overrides[key_resolver] = override_keys
    api.dependency_overrides[document_repo] = override_docs
    return Api(api, h, history, docs, h.deps.gate)


def chat_body(text: str = QUESTION, **extra: Any) -> dict:
    return {"model": "a", "messages": [{"role": "user", "content": text}], **extra}


def sse_payloads(text: str) -> list[str]:
    """Os `data:` de um corpo SSE, na ordem (cada evento precisa ser só `data: ...`)."""
    payloads = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: ") and "\n" not in block, block
        payloads.append(block[len("data: ") :])
    return payloads


def sse_chunks(text: str) -> list[dict]:
    payloads = sse_payloads(text)
    assert payloads[-1] == "[DONE]", payloads[-3:]
    assert "[DONE]" not in payloads[:-1]
    return [json.loads(p) for p in payloads[:-1]]


def content_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)


def extensions(chunks: list[dict]) -> list[dict]:
    """O campo `encaixe` dos chunks de evento extra (delta vazio, sem finish_reason)."""
    return [
        c["encaixe"] for c in chunks if "encaixe" in c and c["choices"][0]["finish_reason"] is None
    ]


def model_saw(script: Script, call: int = 0) -> str:
    """Todo o texto (sistema, histórico, usuário) que o modelo recebeu na chamada `call`."""
    texts = []
    for message in script.messages[call]:
        if isinstance(message, ModelRequest | ModelResponse):
            texts += [str(getattr(p, "content", "")) for p in message.parts]
    return "\n".join(texts)


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condição não satisfeita a tempo")
        await asyncio.sleep(0.01)


@dataclass
class AsgiCall:
    """Uma requisição em andamento contra o app ASGI, com desconexão controlada pelo teste."""

    sent: list[dict]
    disconnect: asyncio.Event
    task: asyncio.Task
    held: asyncio.Event  # `send` ficou preso (ver `hold_send` em `start_asgi`)

    def body(self) -> str:
        return b"".join(
            m.get("body", b"") for m in self.sent if m["type"] == "http.response.body"
        ).decode()


def start_asgi(
    app: FastAPI,
    path: str,
    payload: dict,
    headers: dict[str, str],
    *,
    hold_send: Callable[[dict], bool] | None = None,
) -> AsgiCall:
    """Chama o app como o uvicorn faz (ASGI spec 2.3): o `receive` entrega o corpo e depois
    só devolve `http.disconnect` quando o teste mandar. Com `hold_send`, o primeiro `send` cuja
    mensagem satisfaz o predicado fica preso para sempre (socket lento): o cancelamento pela
    desconexão chega com o app parado dentro do `send`."""
    body = json.dumps(payload).encode()
    disconnect = asyncio.Event()
    held = asyncio.Event()
    delivered = False
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if hold_send is not None and not held.is_set() and hold_send(message):
            held.set()
            await asyncio.Event().wait()

    raw_headers = [
        (b"host", b"test"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *[(k.lower().encode(), v.encode()) for k, v in headers.items()],
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    return AsgiCall(sent, disconnect, task, held)
