from __future__ import annotations

from typing import ClassVar

from arq.connections import RedisSettings

from encaixe.settings import Settings
from encaixe.workers.ingest import ingest_document
from encaixe.workers.summarize import summarize_conversation

_settings = Settings()


async def on_startup(ctx: dict) -> None:
    from encaixe.wiring import build_worker_deps  # criado na Task 13

    ctx.update(await build_worker_deps(_settings))


async def on_shutdown(ctx: dict) -> None:
    from encaixe.wiring import close_worker_deps

    await close_worker_deps(ctx)  # pool, cliente HTTP do embedder e Redis do cache de embeddings


class WorkerSettings:
    functions: ClassVar[list] = [ingest_document, summarize_conversation]
    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 2
