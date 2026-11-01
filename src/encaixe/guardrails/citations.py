from __future__ import annotations

from typing import TYPE_CHECKING

from encaixe.core.registry import register
from encaixe.core.types import GuardResult, RequestContext
from encaixe.observability.spans import traced

if TYPE_CHECKING:
    from encaixe.output.schemas import RespostaBase


class CitationsGuard:
    name = "citations"

    async def check(self, text: str, ctx: RequestContext) -> GuardResult:
        return GuardResult(self.name, "pass")

    async def check_output(self, output: RespostaBase, ctx: RequestContext) -> GuardResult:
        with traced("guard.citations") as span:
            if not ctx.app.config.rag.require_citations:
                result = GuardResult(self.name, "pass")
            else:
                valid = {c.chunk.id for c in ctx.retrieved}
                unknown = [c for c in output.citacoes if c not in valid]
                result = (
                    GuardResult(self.name, "block", "unknown_citation:" + ",".join(unknown))
                    if unknown
                    else GuardResult(self.name, "pass")
                )
            span.set_attribute("verdict", result.verdict)
            return result


@register("guard", "citations")
def _factory(**_):
    return CitationsGuard()
