from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from encaixe.context.aliases import alias_for
from encaixe.context.untrusted import wrap_untrusted
from encaixe.core.protocols import Retriever
from encaixe.core.registry import UnknownImplementationError
from encaixe.core.types import RequestContext
from encaixe.guardrails.injection import InjectionGuard
from encaixe.observability.spans import traced

if TYPE_CHECKING:
    from encaixe.tools.approvals import ApprovalService


@dataclass
class AgentDeps:
    ctx: RequestContext
    retriever: Retriever
    injection: InjectionGuard
    approvals: ApprovalService
    boundary: str


@traced("tool.buscar_documentos")
async def buscar_documentos(ctx: RunContext[AgentDeps], consulta: str) -> str:
    """Busca trechos relevantes na base de documentos do aplicativo."""
    deps = ctx.deps
    rag = deps.ctx.app.config.rag
    found = await deps.retriever.retrieve(
        deps.ctx.app.config.id, deps.ctx.principals, consulta, rag.top_k, rag.rerank_top_n
    )
    known = {s.chunk.id for s in deps.ctx.retrieved}
    for s in found:
        if s.chunk.id not in known:
            known.add(s.chunk.id)
            deps.ctx.retrieved.append(s)
    # ids curtos continuando a numeração da pré-busca (o mapa fica no RequestContext)
    payload = [
        {
            "id": alias_for(deps.ctx.citation_aliases, s.chunk.id),
            "titulo": s.chunk.metadata.get("title", ""),
            "secao": " > ".join(s.chunk.heading_path),
            "texto": s.chunk.text,
        }
        for s in found
    ]
    deps.ctx.tools_called.append("buscar_documentos")
    return wrap_untrusted("trechos", payload, deps.boundary)


# Só lê o corpus: o resultado depende da versão do corpus e dos principais, como a pré-busca.
SEARCH_TOOL = "buscar_documentos"
_NATIVE: dict[str, Callable[..., Any]] = {SEARCH_TOOL: buscar_documentos}


def build_native_toolset(names: list[str]) -> FunctionToolset[AgentDeps]:
    toolset: FunctionToolset[AgentDeps] = FunctionToolset()
    for name in names:
        if name not in _NATIVE:
            raise UnknownImplementationError("tool", name, sorted(_NATIVE))
        toolset.add_function(_NATIVE[name], name=name)
    return toolset
