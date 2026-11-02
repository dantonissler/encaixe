"""Montagem das dependências de processo: `build_deps` (API) e `build_worker_deps` (arq)."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from arq import create_pool as create_arq_pool
from arq.connections import ArqRedis, RedisSettings
from pydantic_ai.models import Model
from redis.asyncio import Redis

from encaixe.cache.embeddings import EmbeddingCache
from encaixe.cache.exact import ExactCache
from encaixe.context.tokens import HeuristicTokenCounter
from encaixe.core.config import LoadedApp, McpServerConfig, load_apps
from encaixe.core.protocols import HistoryStore, ResponseCache, Retriever, TokenCounter
from encaixe.db.pool import create_pool
from encaixe.db.repos.approvals import PgApprovalStore
from encaixe.db.repos.conversations import PgHistoryStore
from encaixe.db.repos.documents import get_corpus_version
from encaixe.embeddings.ollama import OllamaEmbedder
from encaixe.guardrails.canary import make_canary
from encaixe.guardrails.runner import get_guard
from encaixe.llm.breaker import BreakerRegistry
from encaixe.llm.fallback import build_chain
from encaixe.pipeline.gate import GenerationGate
from encaixe.rag.rerank import CrossEncoderReranker
from encaixe.rag.retriever import HybridRetriever
from encaixe.settings import Settings
from encaixe.tools.approvals import ApprovalService
from encaixe.vectorstore.pgvector import PgVectorStore

log = structlog.get_logger(__name__)

# (primary, fallbacks) -> (modelo pronto para o agente, specs efetivas na ordem)
ChainFactory = Callable[[str, list[str]], tuple[Model, list[str]]]
# Fonte injetável dos servidores MCP (testes): (config do servidor, headers) -> alvo do MCPToolset
McpSource = Callable[[McpServerConfig, dict[str, str]], Any]


def set_runtime_env() -> None:
    """Sem telemetria de terceiros e sem o banner do Pydantic AI (sem sobrescrever o ambiente)."""
    for name in ("HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "PYDANTIC_AI_NO_BANNER"):
        os.environ.setdefault(name, "1")


def make_canaries(apps: dict[str, LoadedApp]) -> dict[str, str]:
    """Um canário por app, gerado uma vez por processo (system prompt estável → prefix cache)."""
    return {app_id: make_canary() for app_id in apps}


@dataclass
class Deps:
    """Tudo que o pipeline (e a API) usam; os campos opcionais existem só em produção."""

    settings: Settings
    apps: dict[str, LoadedApp]
    history_store: HistoryStore
    retriever: Retriever
    injection: Any  # InjectionGuard (check_text), compartilhado com o guard de entrada
    approvals: Any  # ApprovalService
    chain_factory: ChainFactory
    corpus_version: Callable[[str], Awaitable[int]]
    cache: ResponseCache | None = None
    enqueue: Callable[..., Awaitable[Any]] | None = None  # arq: pool.enqueue_job
    canaries: dict[str, str] = field(default_factory=dict)
    counter: TokenCounter = field(default_factory=HeuristicTokenCounter)
    mcp_source: McpSource | None = None
    gate: GenerationGate | None = None
    breakers: BreakerRegistry | None = None
    pool: Any = None  # asyncpg.Pool
    redis: Redis | None = None
    arq_pool: ArqRedis | None = None
    embedder: OllamaEmbedder | None = None
    store: PgVectorStore | None = None


async def build_deps(settings: Settings) -> Deps:
    set_runtime_env()
    apps = load_apps(settings.apps_dir)
    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        arq_pool = await create_arq_pool(RedisSettings.from_dsn(settings.redis_url))
    except Exception:
        await redis.aclose()
        await pool.close()
        raise

    embedder = OllamaEmbedder(settings, cache=EmbeddingCache(redis))
    store = PgVectorStore(pool)
    injection = get_guard("prompt_injection", settings)  # mesma instância do guard de entrada
    breakers = BreakerRegistry()  # um por processo; a cadeia é montada por requisição

    def chain_factory(primary: str, fallbacks: list[str]) -> tuple[Model, list[str]]:
        return build_chain(primary, fallbacks, settings, breakers)

    async def corpus_version(app_id: str) -> int:
        return await get_corpus_version(pool, app_id)

    return Deps(
        settings=settings,
        apps=apps,
        history_store=PgHistoryStore(pool),
        retriever=HybridRetriever(
            embedder, store, CrossEncoderReranker(settings.reranker_model_id)
        ),
        injection=injection,
        approvals=ApprovalService(PgApprovalStore(pool), injection=injection),
        chain_factory=chain_factory,
        corpus_version=corpus_version,
        cache=ExactCache(redis),
        enqueue=arq_pool.enqueue_job,
        canaries=make_canaries(apps),
        gate=GenerationGate(
            settings.max_concurrent_generations, settings.max_queue, settings.queue_timeout_s
        ),
        breakers=breakers,
        pool=pool,
        redis=redis,
        arq_pool=arq_pool,
        embedder=embedder,
        store=store,
    )


async def _close_quietly(name: str, close: Callable[[], Awaitable[Any]] | None) -> None:
    if close is None:
        return
    try:
        await close()
    except Exception as exc:  # noqa: BLE001 - desligamento: fecha o resto mesmo assim
        log.warning("close_failed", resource=name, error=type(exc).__name__)


async def close_deps(deps: Deps) -> None:
    await _close_quietly("arq", deps.arq_pool.aclose if deps.arq_pool else None)
    await _close_quietly("embedder", deps.embedder.aclose if deps.embedder else None)
    await _close_quietly("redis", deps.redis.aclose if deps.redis else None)
    await _close_quietly("postgres", deps.pool.close if deps.pool else None)


def make_chunker(settings: Settings) -> Any:
    """Chunker do worker com os modelos locais do Docling (offline) e o OCR conforme settings."""
    from encaixe.rag.chunking import DoclingChunker  # import leve; o Docling carrega no 1º uso

    return DoclingChunker(
        artifacts_path=settings.docling_artifacts_path, do_ocr=settings.docling_ocr
    )


async def build_worker_deps(settings: Settings) -> dict:
    """Chaves do ctx do arq usadas pelos jobs (workers/ingest.py e workers/summarize.py).

    Não usa a chave `redis`: o arq guarda nela o próprio pool.
    """
    set_runtime_env()
    apps = load_apps(settings.apps_dir)
    pool = await create_pool(settings.database_url)
    embedding_redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return {
        "pool": pool,
        "embedder": OllamaEmbedder(settings, cache=EmbeddingCache(embedding_redis)),
        "embedding_redis": embedding_redis,
        "store": PgVectorStore(pool),
        "injection": get_guard("prompt_injection", settings),
        "chunker": make_chunker(settings),
        "settings": settings,
        "history_store": PgHistoryStore(pool),
        "apps": apps,
    }


async def close_worker_deps(ctx: dict) -> None:
    embedder, redis, pool = ctx.get("embedder"), ctx.get("embedding_redis"), ctx.get("pool")
    await _close_quietly("embedder", embedder.aclose if embedder else None)
    await _close_quietly("redis", redis.aclose if redis else None)
    await _close_quietly("postgres", pool.close if pool else None)
