"""Cache exato de respostas: chave escopada por app, permissões e versão do corpus."""

from __future__ import annotations

import hashlib
import json

from encaixe.core.types import ChatMessage, GuardResult, Principals, RequestContext
from encaixe.observability.spans import traced
from encaixe.output.schemas import RespostaBase
from encaixe.tools.native import SEARCH_TOOL


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def exact_key(
    *,
    app_id: str,
    prompt_version: str,
    model: str,
    principals: Principals,
    history: list[ChatMessage],
    user_message: str,
    corpus_version: int,
) -> str:
    payload = {
        "app": app_id,
        "prompt": prompt_version,
        "model": model,
        "principals": principals.fingerprint(),
        "history": [[m.role, m.content] for m in history],
        "message": _normalize(user_message),
        "corpus": corpus_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "enc:resp:" + hashlib.sha256(canonical.encode()).hexdigest()


class ExactCache:
    def __init__(self, redis) -> None:
        self._redis = redis

    async def get(self, key: str) -> dict | None:
        with traced("cache.exact.get") as span:
            raw = await self._redis.get(key)
            span.set_attribute("hit", raw is not None)
            if raw is None:
                return None
            return json.loads(raw)

    async def set(self, key: str, output: RespostaBase, ttl_s: int) -> None:
        with traced("cache.exact.set", hit=False):
            await self._redis.set(key, json.dumps(output.model_dump(mode="json")), ex=ttl_s)


def is_cacheable(ctx: RequestContext, output_verdicts: list[GuardResult]) -> bool:
    if not ctx.app.config.cache.exact.enabled:
        return False
    # Busca nos documentos não impede: a chave já leva a versão do corpus e os principais.
    # Ferramenta MCP (dado por usuário que muda) e ação aprovada impedem.
    if ctx.extra_context or any(t != SEARCH_TOOL for t in ctx.tools_called):
        return False
    for r in ctx.guard_results:
        if r.verdict == "block" or (r.guard == "pii" and r.verdict == "flag"):
            return False
    return not any(r.verdict == "block" for r in output_verdicts)
