from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from encaixe.core.types import (
    ChatMessage,
    Chunk,
    ChunkDraft,
    GuardResult,
    Principals,
    RequestContext,
    ScoredChunk,
)

if TYPE_CHECKING:
    from encaixe.output.schemas import RespostaBase


class Embedder(Protocol):
    model_name: str
    dims: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def search_hybrid(
        self,
        app_id: str,
        principals: Principals,
        text: str,
        embedding: list[float],
        k: int,
    ) -> list[ScoredChunk]: ...

    async def delete_document(self, app_id: str, document_id: str) -> None: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class Retriever(Protocol):
    async def retrieve(
        self,
        app_id: str,
        principals: Principals,
        query: str,
        top_k: int,
        rerank_top_n: int,
    ) -> list[ScoredChunk]: ...


class Guard(Protocol):
    name: str

    async def check(self, text: str, ctx: RequestContext) -> GuardResult: ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class DocumentChunker(Protocol):
    def chunk(self, path: Path, title: str) -> list[ChunkDraft]: ...


class HistoryStore(Protocol):
    async def create_conversation(self, app_id: str, user_ref: str | None) -> str: ...

    async def get_owner(self, app_id: str, conversation_id: str) -> str | None: ...

    async def append(
        self,
        app_id: str,
        conversation_id: str,
        msg: ChatMessage,
        *,
        trace_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
    ) -> int: ...

    async def list_messages(
        self, app_id: str, conversation_id: str, after_seq: int = 0
    ) -> list[ChatMessage]: ...

    async def get_summary(
        self, app_id: str, conversation_id: str
    ) -> tuple[str, int] | None: ...

    async def save_summary(
        self, app_id: str, conversation_id: str, summary: str, covers_until_seq: int
    ) -> None: ...


class ResponseCache(Protocol):
    async def get(self, key: str) -> dict | None: ...

    async def set(self, key: str, output: RespostaBase, ttl_s: int) -> None: ...
