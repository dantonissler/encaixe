from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from encaixe.core.config import LoadedApp


@dataclass(frozen=True)
class Principals:
    user_ref: str | None
    groups: tuple[str, ...] = ()

    def as_list(self) -> list[str]:
        result = ["app:*"]
        if self.user_ref:
            result.append(f"user:{self.user_ref}")
        result.extend(f"group:{g}" for g in self.groups)
        return result

    def fingerprint(self) -> str:
        return hashlib.sha256("|".join(sorted(self.as_list())).encode()).hexdigest()[:16]


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
