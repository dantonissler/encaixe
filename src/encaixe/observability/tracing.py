from __future__ import annotations

import threading

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from encaixe.observability import spans
from encaixe.settings import Settings

_lock = threading.Lock()
_provider: TracerProvider | None = None


def setup_tracing(settings: Settings, service_name: str = "encaixe") -> TracerProvider:
    """Configura OTel -> Phoenix (OTLP HTTP). Idempotente."""
    global _provider
    with _lock:
        if _provider is not None:
            return _provider
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(OpenInferenceSpanProcessor())
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
        spans.use_tracer_provider(provider)
        Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
        _provider = provider
        return provider
