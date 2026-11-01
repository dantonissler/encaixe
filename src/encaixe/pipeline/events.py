"""Eventos que o pipeline emite, na ordem em que acontecem, para a camada HTTP traduzir."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from encaixe.output.schemas import RespostaBase


@dataclass(frozen=True)
class TextDelta:
    """Pedaço novo do texto da resposta (concatenar na ordem)."""

    text: str


@dataclass(frozen=True)
class Status:
    """Evento extra: `queued` (fila), `tool` (ferramenta chamada) e `fallback` (modelo reserva)."""

    kind: Literal["queued", "tool", "fallback"]
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Retraction:
    """Todo texto já emitido deixa de valer; a resposta passa a ser `replacement`."""

    replacement: str


@dataclass(frozen=True)
class Final:
    """Último evento de todo stream: a saída validada que vale (já com substituições)."""

    output: RespostaBase
    conversation_id: str | None
    trace_id: str


ChatEvent = TextDelta | Status | Retraction | Final
