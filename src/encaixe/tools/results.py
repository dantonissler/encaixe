from __future__ import annotations

import json
from typing import Any

from encaixe.context.untrusted import wrap_untrusted

BLOCKED = "[resultado bloqueado por segurança]"
UNAVAILABLE = "[ferramenta indisponível no momento]"


def stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


async def sanitize_tool_result(injection: Any, name: str, result: Any, boundary: str) -> str:
    """Resultado de ferramenta é dado não confiável: guard de injeção e delimitação."""
    text = stringify(result)
    if (await injection.check_text(text)).verdict == "block":
        return BLOCKED
    return wrap_untrusted("ferramenta", {"ferramenta": name, "resultado": text}, boundary)
