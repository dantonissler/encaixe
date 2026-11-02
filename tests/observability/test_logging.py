import json

import structlog
from opentelemetry.sdk.trace import TracerProvider

from encaixe.observability.logging import setup_logging
from encaixe.settings import Settings


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_log_masks_cpf_and_has_trace_id(capsys):
    setup_logging(Settings())
    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("s"):
        structlog.get_logger().info("cliente 529.982.247-25", extra={"k": ["529.982.247-25"]})
    out = capsys.readouterr().out
    assert "529.982.247-25" not in out
    data = json.loads(out.strip().splitlines()[-1])
    assert len(data["trace_id"]) == 32 and len(data["span_id"]) == 16
    assert "[CPF]" in data["event"]


def test_no_trace_id_outside_span(capsys):
    setup_logging(Settings())
    structlog.get_logger().info("x")
    assert "trace_id" not in _last(capsys)


def test_drops_prompt_and_completion_when_disabled(capsys):
    setup_logging(Settings(log_full_text=False))
    structlog.get_logger().info("x", prompt="p", completion="c", other="o")
    data = _last(capsys)
    assert "prompt" not in data and "completion" not in data and data["other"] == "o"


def test_keeps_prompt_when_enabled(capsys):
    setup_logging(Settings(log_full_text=True))
    structlog.get_logger().info("x", prompt="p")
    assert _last(capsys)["prompt"] == "p"


def test_trace_ids_not_corrupted_by_pii_mask(capsys):
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    class Digits(IdGenerator):
        def generate_span_id(self):
            return int("1234567890123456", 16)

        def generate_trace_id(self):
            return int("12345678901234567890123456789012", 16)

    setup_logging(Settings())
    tracer = TracerProvider(id_generator=Digits()).get_tracer("t")
    with tracer.start_as_current_span("s"):
        structlog.get_logger().info("x")
    data = _last(capsys)
    assert data["trace_id"] == "12345678901234567890123456789012"
    assert data["span_id"] == "1234567890123456"
