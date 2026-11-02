from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from encaixe.core.config import LoadedApp


@dataclass(frozen=True)
class Principals:
    user_ref: str | None
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Os principais viajam como lista separada por vírgulas (setting do Postgres):
        # vírgula ou valor vazio permitiriam injetar principais extras.
        values = [v for v in (self.user_ref, *self.groups) if v is not None]
        if self.user_ref == "" or any(v == "" or "," in v for v in values):
            raise ValueError("principals inválidos: vazios ou com vírgula")

    def as_list(self) -> list[str]:
        result = ["app:*"]
        if self.user_ref:
            result.append(f"user:{self.user_ref}")
        result.extend(f"group:{g}" for g in self.groups)
        return result

    def fingerprint(self) -> str:
        # JSON da lista ordenada: codificação sem ambiguidade (com "|" o grupo `rh|user:x`
        # colidia com o usuário x no grupo rh na chave do cache exato).
        encoded = json.dumps(sorted(self.as_list()), ensure_ascii=False)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    seq: int | None = None
    tool_calls: list[dict] | None = None
    tokens: int | None = None


@dataclass
class Chunk:
    id: str
    app_id: str
    document_id: str
    text: str
    heading_path: tuple[str, ...]
    acl_principals: tuple[str, ...]
    metadata: dict
    embedding: list[float] | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class ChunkDraft:
    text: str
    heading_path: tuple[str, ...]
    page: int | None


@dataclass
class GuardResult:
    guard: str
    verdict: Literal["pass", "flag", "block"]
    reason: str = ""
    redacted_text: str | None = None


@dataclass
class RequestContext:
    request_id: str
    app: LoadedApp
    principals: Principals
    conversation_id: str | None
    has_conversation_identity: bool
    user_message: str
    client_history: list[ChatMessage]
    canary: str
    model_override: str | None = None
    retrieved: list[ScoredChunk] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    guard_results: list[GuardResult] = field(default_factory=list)
    fallback_used: bool = False
    cache_layer_hit: str | None = None
    extra_context: list[str] = field(default_factory=list)
    # alias curto mostrado ao modelo (T1, T2…) → id real do chunk (ver context/aliases.py)
    citation_aliases: dict[str, str] = field(default_factory=dict)
