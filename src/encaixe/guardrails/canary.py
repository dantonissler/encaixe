from __future__ import annotations

import re
import secrets

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.guardrails.normalize import strip_invisible
from encaixe.observability.spans import traced


def make_canary() -> str:
    return "cnr-" + secrets.token_hex(6)


class CanaryGuard:
    name = "canary"

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        with traced("guard.canary") as span:
            hit = False
            if ctx.canary:
                squashed = re.sub(r"\s+", "", strip_invisible(text).casefold())
                hit = ctx.canary in text or ctx.canary.casefold() in squashed
            result = GuardResult(self.name, "block", "canary_leak") if hit else GuardResult(self.name, "pass")
            span.set_attribute("verdict", result.verdict)
            return result


@register("guard", "canary")
def _factory(**_):
    return CanaryGuard()
