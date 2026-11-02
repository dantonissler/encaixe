from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from encaixe.core.pii import mask_pii
from encaixe.observability import spans
from encaixe.settings import Settings

_lock = threading.Lock()
_provider: TracerProvider | None = None


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, Mapping):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(_mask_value(v) for v in value)
    return value


def _mask_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: _mask_value(v) for k, v in (attributes or {}).items()}


def mask_span(span: ReadableSpan) -> ReadableSpan:
    """Cópia do span com dados pessoais mascarados nos atributos, nos atributos dos eventos
    (inclusive exceções) e na descrição do status. Contexto, tempos e nomes não mudam."""
    status = span.status
    if status.description:
        status = Status(status.status_code, mask_pii(status.description))
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_mask_attributes(span.attributes),
        events=[Event(e.name, _mask_attributes(e.attributes), e.timestamp) for e in span.events],
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class MaskingSpanExporter(SpanExporter):
    """Envolve o exporter: o Phoenix só recebe cópias mascaradas (spec §6.13). Sempre ligado,
    como nos logs; `log_full_text` não o desliga. O texto completo (já mascarado) continua no
    trace para depuração."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([mask_span(s) for s in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


def setup_tracing(settings: Settings, service_name: str = "encaixe") -> TracerProvider:
    """Configura OTel -> Phoenix (OTLP HTTP). Idempotente."""
    global _provider
    with _lock:
        if _provider is not None:
            return _provider
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(OpenInferenceSpanProcessor())
        provider.add_span_processor(
            BatchSpanProcessor(
                MaskingSpanExporter(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
            )
        )
        trace.set_tracer_provider(provider)
        spans.use_tracer_provider(provider)
        Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
        _provider = provider
        return provider
