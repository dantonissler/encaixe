"""Formato de fio da OpenAI: eventos do pipeline → chunks SSE `chat.completion.chunk`.

- `TextDelta` → `choices[0].delta.content`.
- `Status` (fila, ferramenta, fallback) → chunk de extensão: `delta == {}` e o campo de topo
  `"encaixe": {"status": <tipo>, ...}`; clientes que não o conhecem o ignoram.
- `Retraction(r)` → chunk de extensão `{"retraction": r}` (o texto já recebido deixa de valer;
  a resposta passa a ser `r`) seguido de um delta de conteúdo com `r`, para que clientes
  OpenAI comuns também mostrem o aviso.
- `Final` → último chunk, `finish_reason: "stop"` e `"encaixe": {"citacoes", "output",
  "trace_id", "conversation_id"}`; depois vem `data: [DONE]`.

Com `response_format` JSON ou TOON, o conteúdo não é a `resposta` em pedaços: a saída
validada inteira vai num único delta logo antes do chunk final.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from encaixe.output.schemas import RespostaBase
from encaixe.output.toon import to_toon
from encaixe.pipeline.events import ChatEvent, Final, Retraction, Status, TextDelta
from encaixe.pipeline.stages import static_output

OutputFormat = Literal["text", "json", "toon"]

DONE = b"data: [DONE]\n\n"


def sse_data(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        + b"\n\n"
    )


def render_output(output: RespostaBase, fmt: OutputFormat) -> str:
    """O `content` da mensagem: a `resposta`, o JSON validado inteiro ou o mesmo em TOON."""
    if fmt == "json":
        return output.model_dump_json()
    if fmt == "toon":
        return to_toon(output)
    return output.resposta


def final_extension(final: Final) -> dict[str, Any]:
    return {
        "citacoes": list(final.output.citacoes),
        "output": final.output.model_dump(mode="json"),
        "trace_id": final.trace_id,
        "conversation_id": final.conversation_id,
    }


class ChunkEncoder:
    """Traduz os eventos de uma resposta em chunks SSE (um `id` por resposta)."""

    def __init__(self, completion_id: str, model: str, created: int, fmt: OutputFormat) -> None:
        self.completion_id = completion_id
        self.model = model
        self.created = created
        self.fmt = fmt
        self.sent_text = False
        self.finished = False

    def _chunk(
        self,
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
        extension: dict[str, Any] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extension is not None:
            payload["encaixe"] = extension
        return sse_data(payload)

    def _content(self, text: str) -> bytes:
        self.sent_text = True
        return self._chunk({"content": text})

    def start(self) -> bytes:
        """Primeiro chunk, como na OpenAI: só o papel."""
        return self._chunk({"role": "assistant", "content": ""})

    def extension(self, data: dict[str, Any]) -> bytes:
        return self._chunk({}, extension=data)

    def encode(self, event: ChatEvent) -> list[bytes]:
        if isinstance(event, TextDelta):
            return [self._content(event.text)] if self.fmt == "text" and event.text else []
        if isinstance(event, Status):
            return [self.extension({**event.data, "status": event.kind})]
        if isinstance(event, Retraction):
            if self.fmt != "text":
                return []  # nada foi liberado: o conteúdo só sai no fim
            visible = f"\n\n{event.replacement}" if self.sent_text else event.replacement
            return [self.extension({"retraction": event.replacement}), self._content(visible)]
        if isinstance(event, Final):
            chunks = []
            if self.fmt != "text":
                chunks.append(self._content(render_output(event.output, self.fmt)))
            chunks.append(self._chunk({}, finish_reason="stop", extension=final_extension(event)))
            self.finished = True
            return chunks
        return []

    def failure(self, text: str, conversation_id: str | None, trace_id: str) -> list[bytes]:
        """Encerra bem formado quando a resposta não pôde ser gerada (fila, erro inesperado)."""
        first: ChatEvent = Retraction(text) if self.sent_text else TextDelta(text)
        final = Final(static_output(text), conversation_id, trace_id)
        return [*self.encode(first), *self.encode(final)]
