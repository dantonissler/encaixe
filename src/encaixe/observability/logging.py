from __future__ import annotations

import logging
from typing import Any

import structlog
from opentelemetry import trace

from encaixe.core.pii import mask_pii
from encaixe.settings import Settings


def add_trace_ids(_logger, _method, event_dict: dict) -> dict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_mask(v) for v in value)
    return value


def mask_pii_processor(_logger, _method, event_dict: dict) -> dict:
    return {k: _mask(v) for k, v in event_dict.items()}


def make_drop_full_text(enabled: bool):
    def drop_full_text(_logger, _method, event_dict: dict) -> dict:
        if not enabled:
            event_dict.pop("prompt", None)
            event_dict.pop("completion", None)
        return event_dict

    return drop_full_text


def setup_logging(settings: Settings) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            make_drop_full_text(settings.log_full_text),
            structlog.processors.format_exc_info,
            mask_pii_processor,
            add_trace_ids,  # depois do mascaramento: ids hex nunca sao alterados
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
