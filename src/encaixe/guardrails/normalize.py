from __future__ import annotations

import re
import unicodedata

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced

# Largura zero, bidi, seletores de variação, preenchimentos invisíveis, Unicode Tags e
# controles C0/C1 (preserva \t, \n e \r).
_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x180E, 0x180E),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFF9, 0xFFFB),
    (0xE0000, 0xE007F), (0xE0100, 0xE01EF),
    (0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), (0x7F, 0x9F),
)
_INVISIBLE = re.compile(
    "[" + "".join(f"\\U{lo:08x}-\\U{hi:08x}" for lo, hi in _RANGES) + "]"
)


def strip_invisible(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return unicodedata.normalize("NFKC", text)


class NormalizeGuard:
    name = "normalize"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.normalize") as span:
            normalized = strip_invisible(text)
            if len(text) > self.max_chars or len(normalized) > self.max_chars:
                result = GuardResult(self.name, "block", "too_long")
            else:
                result = GuardResult(self.name, "pass", redacted_text=normalized)
            span.set_attribute("verdict", result.verdict)
            return result


@register("guard", "normalize")
def _factory(settings, **_):
    return NormalizeGuard(settings.max_user_message_chars)
