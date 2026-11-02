from __future__ import annotations

import json
from typing import Any

from encaixe.context.untrusted import wrap_untrusted

BLOCKED = "[resultado bloqueado por segurança]"
UNAVAILABLE = "[ferramenta indisponível no momento]"
TRUNCATED = "[resultado truncado]"
# Padrão de `Settings.tool_result_max_chars` (ENCAIXE_TOOL_RESULT_MAX_CHARS).
DEFAULT_MAX_CHARS = 4000


def stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def truncate_result(text: str, max_chars: int) -> str:
    """Corta em `max_chars` com marcador visível (o modelo sabe que faltou algo)."""
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars, 0)] + "\n" + TRUNCATED


async def sanitize_tool_result(
    injection: Any, name: str, result: Any, boundary: str, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Resultado de ferramenta é dado não confiável: limite de tamanho (spec §6.6), guard de
    injeção e delimitação."""
    text = truncate_result(stringify(result), max_chars)
    if (await injection.check_text(text)).verdict == "block":
        return BLOCKED
    return wrap_untrusted("ferramenta", {"ferramenta": name, "resultado": text}, boundary)
