"""`POST /v1/chat/completions` compatível com OpenAI (JSON ou SSE).

Identidade da conversa (spec §7): com `conversation_id`, o servidor é a fonte da verdade
(confere o dono, usa o histórico guardado e lê do cliente só a última mensagem do usuário);
sem ele, cria uma conversa nova, aceita o histórico do cliente como não confiável e desliga as
ferramentas de escrita. A geração passa pela vaga de geração (`GenerationGate`); se o cliente
desconectar, o pipeline é cancelado e a vaga volta na hora (Review Focus 5).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import AbstractAsyncContextManager, aclosing, nullcontext, suppress
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from encaixe.api.auth import get_deps, require_app, resolve_app
from encaixe.api.errors import RETRY_AFTER, ApiError, service_unavailable
from encaixe.api.identity import resolve_identity, verify_conversation_owner
from encaixe.api.sse import DONE, ChunkEncoder, OutputFormat, final_extension, render_output
from encaixe.core.config import LoadedApp
from encaixe.core.types import ChatMessage, RequestContext
from encaixe.messages import ALL_FAILED, QUEUE_WAIT
from encaixe.pipeline.chat import ChatPipeline
from encaixe.pipeline.events import Final, Status
from encaixe.pipeline.gate import GenerationGate, QueueFull, QueueTimeout
from encaixe.pipeline.stages import current_trace_id
from encaixe.wiring import Deps

log = structlog.get_logger(__name__)
router = APIRouter()

BUSY = "Muitas requisições na fila. Tente novamente em instantes."
CLIENT_CLOSED = 499  # convenção do nginx: o cliente desistiu antes da resposta


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        """Conteúdo textual (partes `text` de mensagens multimodais; o resto é ignorado)."""
        if isinstance(self.content, str):
            return self.content
        parts = self.content or []
        return "\n".join(
            p["text"] for p in parts if p.get("type") == "text" and isinstance(p.get("text"), str)
        )


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "text"


class ChatCompletionRequest(BaseModel):
    """Corpo da OpenAI; parâmetros de amostragem e `tools` do cliente são ignorados (a app
    define modelo, ferramentas e limites)."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[MessageIn] = Field(min_length=1)
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None
    response_format: ResponseFormat | str | None = None


_FORMATS: dict[str, OutputFormat] = {
    "text": "text",
    "json_object": "json",
    "json_schema": "json",  # o schema é sempre o da app (spec §7)
    "json": "json",
    "toon": "toon",
}


def output_format(requested: ResponseFormat | str | None, app: LoadedApp) -> OutputFormat:
    kind = requested if isinstance(requested, str) else (requested.type if requested else "text")
    fmt = _FORMATS.get(kind)
    if fmt is None:
        raise ApiError(400, f"response_format '{kind}' não suportado.", code="invalid_format")
    if fmt != "text" and fmt not in app.config.output.formats:
        raise ApiError(400, f"A app não oferece o formato '{fmt}'.", code="invalid_format")
    return fmt


def split_messages(messages: list[MessageIn]) -> tuple[list[ChatMessage], str]:
    """(histórico user/assistant antes da última mensagem do usuário, essa mensagem).

    Mensagens `system`/`developer`/`tool` do cliente nunca entram: o prompt é o da app.
    """
    turns = [ChatMessage(m.role, m.text()) for m in messages if m.role in ("user", "assistant")]
    last = next((i for i in range(len(turns) - 1, -1, -1) if turns[i].role == "user"), None)
    if last is None or not turns[last].content.strip():
        raise ApiError(
            400, "Envie ao menos uma mensagem do usuário com texto.", code="no_user_message"
        )
    history = [t for t in turns[:last] if t.content.strip()]  # ex.: chamadas de ferramenta
    return history, turns[last].content


def _slot(gate: GenerationGate | None) -> AbstractAsyncContextManager[Any]:
    return gate.slot() if gate is not None else nullcontext()


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    key_app: Annotated[str, Depends(require_app)],
    deps: Annotated[Deps, Depends(get_deps)],
) -> Response:
    app = resolve_app(deps, key_app, body.model or request.headers.get("x-app-id"))
    app_id = app.config.id
    fmt = output_format(body.response_format, app)
    principals, client_cid, model_override = resolve_identity(request, body)
    history, user_message = split_messages(body.messages)

    if client_cid is not None:
        # Servidor é a fonte da verdade: só a última mensagem do usuário vem do cliente.
        cid = await verify_conversation_owner(deps.history_store, app_id, client_cid, principals)
        history = []
    gate = deps.gate
    if gate is not None and gate.full:  # antes de criar conversa: nada órfão no 429
        raise ApiError(429, BUSY, code="queue_full", headers=RETRY_AFTER)
    if client_cid is None:
        async with service_unavailable("postgres"):
            cid = await deps.history_store.create_conversation(app_id, principals.user_ref)

    ctx = RequestContext(
        request_id=uuid.uuid4().hex,
        app=app,
        principals=principals,
        conversation_id=cid,
        has_conversation_identity=client_cid is not None,
        user_message=user_message,
        client_history=history,
        canary=deps.canaries.get(app_id, ""),
        model_override=model_override,
    )
    trace_id = current_trace_id() or ctx.request_id  # o mesmo que o pipeline usa
    headers = {"X-Conversation-Id": cid, "X-Trace-Id": trace_id}
    completion_id, created = f"chatcmpl-{ctx.request_id}", int(time.time())
    pipeline = ChatPipeline(deps)

    if body.stream:
        encoder = ChunkEncoder(completion_id, app_id, created, fmt)
        return StreamingResponse(
            _sse(pipeline, ctx, gate, encoder, trace_id),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started = time.monotonic()
    try:
        final = await until_disconnect(request, _complete(pipeline, ctx, gate))
    except QueueFull:
        raise ApiError(429, BUSY, code="queue_full", headers=RETRY_AFTER) from None
    except QueueTimeout:
        raise ApiError(503, BUSY, code="queue_timeout", headers=RETRY_AFTER) from None
    _log_finished(ctx, started, stream=False, completed=final is not None)
    if final is None:
        return Response(status_code=CLIENT_CLOSED)
    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": app_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": render_output(final.output, fmt)},
                "finish_reason": "stop",
            }
        ],
        "encaixe": final_extension(final),
    }
    return JSONResponse(payload, headers=headers)


async def _complete(
    pipeline: ChatPipeline, ctx: RequestContext, gate: GenerationGate | None
) -> Final:
    """Roda o pipeline inteiro e devolve o `Final` (que já traz as substituições)."""
    final: Final | None = None
    async with _slot(gate), aclosing(pipeline.stream(ctx)) as events:
        async for event in events:
            if isinstance(event, Final):
                final = event
    assert final is not None  # o pipeline sempre termina com Final
    return final


async def _sse(
    pipeline: ChatPipeline,
    ctx: RequestContext,
    gate: GenerationGate | None,
    encoder: ChunkEncoder,
    trace_id: str,
) -> AsyncIterator[bytes]:
    """Corpo SSE. Se o cliente desconectar, o Starlette cancela este gerador: o pipeline é
    fechado (cancelando o agente) e o `finally` do `slot()` devolve a vaga."""
    started, completed = time.monotonic(), False
    try:
        yield encoder.start()
        try:
            if gate is not None and gate.available == 0:
                for chunk in encoder.encode(Status("queued", {"message": QUEUE_WAIT})):
                    yield chunk
            async with _slot(gate), aclosing(pipeline.stream(ctx)) as events:
                async for event in events:
                    for chunk in encoder.encode(event):
                        yield chunk
        except (QueueFull, QueueTimeout) as exc:
            log.warning("chat_queue_rejected", request_id=ctx.request_id, error=type(exc).__name__)
            for chunk in encoder.failure(ALL_FAILED, ctx.conversation_id, trace_id):
                yield chunk
        except Exception:
            log.exception("chat_stream_failed", request_id=ctx.request_id)
            if not encoder.finished:
                for chunk in encoder.failure(ALL_FAILED, ctx.conversation_id, trace_id):
                    yield chunk
        yield DONE
        completed = True
    finally:
        _log_finished(ctx, started, stream=True, completed=completed)


async def until_disconnect[T](request: Request, work: Awaitable[T]) -> T | None:
    """Espera `work`; se o cliente desconectar antes, cancela-o (liberando a vaga) e devolve
    None. Usado na resposta sem stream, em que o Starlette não observa a desconexão."""
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_client_gone(request))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if not task.done():  # o cliente saiu primeiro
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            return None
        return await task
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()


async def _client_gone(request: Request) -> None:
    """Retorna quando chega `http.disconnect` (o corpo já foi lido pelo FastAPI). Se não der
    para observar a conexão, nunca retorna: a geração segue até o fim."""
    with suppress(Exception):
        while (await request.receive())["type"] != "http.disconnect":
            pass
        return
    await asyncio.Event().wait()


def _log_finished(ctx: RequestContext, started: float, *, stream: bool, completed: bool) -> None:
    log.info(
        "chat_completed" if completed else "chat_client_disconnected",
        request_id=ctx.request_id,
        app_id=ctx.app.config.id,
        conversation_id=ctx.conversation_id,
        identity=ctx.has_conversation_identity,
        stream=stream,
        cache=ctx.cache_layer_hit,
        fallback=ctx.fallback_used,
        tools=list(ctx.tools_called),
        duration_ms=round((time.monotonic() - started) * 1000),
    )
