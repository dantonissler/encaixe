"""Dados pessoais mascarados antes de os spans saírem para o Phoenix (spec §3, §6.13, §6.14;
revisão final I3)."""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from encaixe.observability import spans
from encaixe.observability.tracing import MaskingSpanExporter

CPF = "529.982.247-25"
EMAIL = "ana.souza@aurora.com.br"
PHONE = "(11) 98765-4321"
UUID = "550e8400-e29b-41d4-a716-446655440000"


def _emit(provider: TracerProvider) -> None:
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("encaixe.pipeline.chat") as span:
        span.set_attribute("input.value", f'{{"msg": "meu CPF é {CPF} e o e-mail {EMAIL}"}}')
        span.set_attribute("llm.input_messages", [f"ligue {PHONE}", "sem dado"])
        span.set_attribute("conversation_id", UUID)
        span.set_attribute("rag.retrieved", 3)
        span.add_event("tool.result", {"resultado": f"cliente {CPF}", "ok": True})
        span.set_status(Status(StatusCode.ERROR, f"falhou para {EMAIL}"))


def _all_text(span) -> str:
    parts = [str(v) for v in span.attributes.values()]
    for event in span.events:
        parts += [str(v) for v in event.attributes.values()]
    parts.append(span.status.description or "")
    return " ".join(parts)


def test_masking_exporter_masks_attributes_events_and_status():
    inner = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(MaskingSpanExporter(inner)))
    _emit(provider)
    (span,) = inner.get_finished_spans()
    text = _all_text(span)
    for raw in (CPF, EMAIL, PHONE):
        assert raw not in text
    assert span.attributes["input.value"] == '{"msg": "meu CPF é [CPF] e o e-mail [EMAIL]"}'
    assert span.attributes["llm.input_messages"] == ("ligue [TELEFONE]", "sem dado")
    assert span.events[0].attributes == {"resultado": "cliente [CPF]", "ok": True}
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "falhou para [EMAIL]"
    # ids e números não mudam; o contexto (trace/span id) também não
    assert span.attributes["conversation_id"] == UUID and span.attributes["rag.retrieved"] == 3
    assert span.name == "encaixe.pipeline.chat" and span.context.trace_id != 0


def test_setup_tracing_exports_masked_spans(settings, monkeypatch):
    """O exporter OTLP real recebe só cópias mascaradas."""
    from encaixe.observability import tracing

    exported: list = []

    class FakeOtlp:
        def __init__(self, *a, **kw):
            pass

        def export(self, batch):
            exported.extend(batch)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=30000):
            return True

    monkeypatch.setattr(tracing, "OTLPSpanExporter", FakeOtlp)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", lambda p: None)
    monkeypatch.setattr(tracing.Agent, "instrument_all", staticmethod(lambda *a, **k: None))
    provider = tracing.setup_tracing(settings)
    try:
        _emit(provider)
        provider.force_flush()
        assert exported, "nenhum span exportado"
        text = " ".join(_all_text(s) for s in exported)
        for raw in (CPF, EMAIL, PHONE):
            assert raw not in text
        assert "[CPF]" in text
    finally:
        spans.use_tracer_provider(None)
        provider.shutdown()
