import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from encaixe.core.types import Principals, RequestContext
from encaixe.observability import spans
from encaixe.observability.spans import current_ctx, set_current_ctx, traced


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    spans.use_tracer_provider(provider)
    yield exp
    spans.use_tracer_provider(None)


def _ctx() -> RequestContext:
    from types import SimpleNamespace

    app = SimpleNamespace(config=SimpleNamespace(id="exemplo"), prompt_version="v1")
    return RequestContext(
        request_id="r1",
        app=app,  # type: ignore[arg-type]
        principals=Principals(user_ref="u1"),
        conversation_id="c1",
        has_conversation_identity=True,
        user_message="oi",
        client_history=[],
        canary="x",
    )


def test_traced_adds_ctx_attributes(exporter):
    ctx = _ctx()
    token = set_current_ctx(ctx)
    try:
        with traced("rag.retrieve", k=3):
            pass
    finally:
        current_ctx.reset(token)
    (span,) = exporter.get_finished_spans()
    assert span.name == "encaixe.rag.retrieve"
    assert span.attributes["app_id"] == "exemplo"
    assert span.attributes["conversation_id"] == "c1"
    assert span.attributes["prompt.version"] == "v1"
    assert span.attributes["principals.fingerprint"] == ctx.principals.fingerprint()
    assert span.attributes["k"] == 3


def test_traced_without_ctx(exporter):
    with traced("x"):
        pass
    (span,) = exporter.get_finished_spans()
    assert "app_id" not in span.attributes


def test_traced_records_error_and_reraises(exporter):
    with pytest.raises(ValueError, match="boom"), traced("falha"):
        raise ValueError("boom")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


async def test_traced_as_decorator(exporter):
    @traced("deco", a=1)
    async def f(x):
        return x * 2

    assert await f(2) == 4
    (span,) = exporter.get_finished_spans()
    assert span.name == "encaixe.deco"
    assert span.attributes["a"] == 1


async def test_traced_decorator_error(exporter):
    @traced("deco")
    async def f():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await f()
    assert exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR


def test_setup_tracing_idempotent(settings):
    from encaixe.observability import tracing

    p1 = tracing.setup_tracing(settings)
    n = len(p1._active_span_processor._span_processors)
    p2 = tracing.setup_tracing(settings)
    assert p1 is p2
    assert len(p2._active_span_processor._span_processors) == n
    spans.use_tracer_provider(None)
