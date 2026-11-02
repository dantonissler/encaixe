from __future__ import annotations

from encaixe.core.pii import find_pii, mask_pii
from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced


class PiiGuard:
    name = "pii"

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.pii") as span:
            if find_pii(text):
                result = GuardResult(self.name, "flag", "pii", redacted_text=mask_pii(text))
            else:
                result = GuardResult(self.name, "pass")
            span.set_attribute("verdict", result.verdict)
            return result


def _ids(text: str) -> set[str]:
    return {
        "".join(c for c in text[m.start : m.end] if c.isdigit())
        for m in find_pii(text)
        if m.kind in ("cpf", "cnpj")
    }


class PiiLeakGuard:
    name = "pii_leak"

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.pii_leak") as span:
            found = _ids(text)
            if found:
                known = _ids(ctx.user_message)
                for c in ctx.retrieved:
                    known |= _ids(c.chunk.text)
                result = (
                    GuardResult(self.name, "block", "pii_leak") if found - known
                    else GuardResult(self.name, "pass")
                )
            else:
                result = GuardResult(self.name, "pass")
            span.set_attribute("verdict", result.verdict)
            return result


@register("guard", "pii")
def _pii(**_):
    return PiiGuard()


@register("guard", "pii_leak")
def _leak(**_):
    return PiiLeakGuard()
