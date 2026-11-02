"""Ids curtos de citação (`T1`, `T2`…) para os trechos mostrados ao modelo.

Modelos pequenos copiam mal UUIDs (`id: <uuid>`, UUIDs inventados). O modelo vê e cita ids
curtos; o pipeline traduz de volta para o id real do chunk e descarta o que não resolve.
O mapa alias → chunk_id vive no `RequestContext` (`citation_aliases`): o `ContextBuilder`
numera os trechos da pré-busca e a ferramenta `buscar_documentos` continua a numeração.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

ALIAS_PREFIX = "T"
_ALIAS = re.compile(r"T0*(\d+)", re.IGNORECASE)
# Tokens aproveitáveis dentro de uma citação mal formada ("id: T1", "T1, T3", "[<uuid>]").
_TOKEN = re.compile(
    r"\bT\d+\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
# Moldura comum em volta do id: espaços, aspas, colchetes, prefixo "id:".
_FRAME = re.compile(r"^[\s\[(\"'`]*(?:id\s*[:=]\s*)?[\s\[(\"'`]*|[\s\])\"'`.,;]*$", re.IGNORECASE)


def next_alias(aliases: dict[str, str]) -> str:
    return f"{ALIAS_PREFIX}{len(aliases) + 1}"


def alias_for(aliases: dict[str, str], chunk_id: str) -> str:
    """Alias já dado ao chunk ou o próximo da numeração (registrado em `aliases`)."""
    for alias, known in aliases.items():
        if known == chunk_id:
            return alias
    alias = next_alias(aliases)
    aliases[alias] = chunk_id
    return alias


def _lookup(token: str, aliases: dict[str, str], valid: dict[str, str]) -> str | None:
    if (exact := valid.get(token.lower())) is not None:
        return exact
    match = _ALIAS.fullmatch(token)
    if match is None:
        return None
    target = aliases.get(f"{ALIAS_PREFIX}{int(match.group(1))}")
    return target if target is not None and target.lower() in valid else None


def _resolve_one(raw: str, aliases: dict[str, str], valid: dict[str, str]) -> list[str]:
    text = raw.strip()
    for candidate in (text, _FRAME.sub("", text)):
        if candidate and (found := _lookup(candidate, aliases, valid)) is not None:
            return [found]
    found_all = (_lookup(m.group(0), aliases, valid) for m in _TOKEN.finditer(text))
    return [f for f in found_all if f is not None]


def resolve_citations(
    citations: Iterable[str], aliases: dict[str, str], retrieved_ids: Iterable[str]
) -> tuple[list[str], int]:
    """(ids reais citados, sem repetição e na ordem, quantas citações foram descartadas).

    Aceita o alias (`T1`, `t1`, `[T1]`, `id: T1`, `T01`) e o id exato de um trecho recuperado
    (UUID em qualquer caixa, com ou sem `id:`). O que não resolve para um trecho recuperado
    é descartado.
    """
    valid = {cid.lower(): cid for cid in retrieved_ids}
    cited: list[str] = []
    dropped = 0
    for raw in citations:
        found = _resolve_one(str(raw), aliases, valid)
        if not found:
            dropped += 1
        for cid in found:
            if cid not in cited:
                cited.append(cid)
    return cited, dropped
