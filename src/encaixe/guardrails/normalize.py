from __future__ import annotations

import re
import unicodedata

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced

# Largura zero, marcas bidi (inclui isolates) e o bloco Unicode Tags.
_INVISIBLE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u00ad\U000e0000-\U000e007f]"
)


def strip_invisible(text: str) -> str:
    return unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))


class NormalizeGuard:
    name = "normalize"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.normalize") as span:
            if len(text) > self.max_chars:
                result = GuardResult(self.name, "block", "too_long")
            else:
                result = GuardResult(self.name, "pass", redacted_text=strip_invisible(text))
            span.set_attribute("verdict", result.verdict)
            return result


@register("guard", "normalize")
def _factory(settings, **_):
    return NormalizeGuard(settings.max_user_message_chars)
