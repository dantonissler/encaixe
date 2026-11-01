from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, TracerProvider

if TYPE_CHECKING:
    from encaixe.core.types import RequestContext

current_ctx: ContextVar[RequestContext | None] = ContextVar("encaixe_current_ctx", default=None)

_provider: TracerProvider | None = None


def use_tracer_provider(provider: TracerProvider | None) -> None:
    """Define o provider usado por `traced` (None volta ao global)."""
    global _provider
    _provider = provider


def set_current_ctx(ctx: RequestContext | None) -> Token:
    return current_ctx.set(ctx)


def _ctx_attributes() -> dict[str, Any]:
    ctx = current_ctx.get()
    if ctx is None:
        return {}
    attrs: dict[str, Any] = {
        "app_id": ctx.app.config.id,
        "prompt.version": ctx.app.prompt_version,
        "principals.fingerprint": ctx.principals.fingerprint(),
    }
    if ctx.conversation_id is not None:
        attrs["conversation_id"] = ctx.conversation_id
    return attrs


class traced:
    """Abre o span `encaixe.<name>`; context manager sync e decorator (sync/async)."""

    def __init__(self, name: str, **attrs: Any) -> None:
        self.name = name
        self.attrs = attrs

    @contextmanager
    def _span(self):
        tracer = trace.get_tracer("encaixe", tracer_provider=_provider)
        attributes = {**_ctx_attributes(), **self.attrs}
        with tracer.start_as_current_span(
            f"encaixe.{self.name}",
            attributes=attributes,
            record_exception=True,
            set_status_on_exception=True,
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def __enter__(self):
        self._cm = self._span()
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)

    def __call__(self, fn: Callable) -> Callable:
        import inspect

        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def agen_wrapper(*a, **kw):
                with traced(self.name, **self.attrs):
                    async for item in fn(*a, **kw):
                        yield item

            return agen_wrapper
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*a, **kw):
                with traced(self.name, **self.attrs):
                    return await fn(*a, **kw)

            return wrapper

        @functools.wraps(fn)
        def swrapper(*a, **kw):
            with traced(self.name, **self.attrs):
                return fn(*a, **kw)

        return swrapper
