from __future__ import annotations

import re
import unicodedata

from encaixe.guardrails.normalize import strip_invisible

_VERB_IGN = r"(?:ignor\w*|desconsider\w*|esquec\w*|disregard|forget|override)"
_NOUN = r"(?:instrucoes|instructions|prompts?|diretrizes|orientacoes|guidelines)"
_QUAL = r"(?:anteriores|anterior|acima|previous|prior|above|do sistema)"
_DET = r"(?:(?:todas|all|any|the|your|suas|as|essas|these|those)\s+)*"
_DIRECTIVE = (
    r"(?:revel\w*|mostr\w*|imprim\w*|exib\w*|repit\w*|copi\w*|vaz\w*|"
    r"reveal|print|show|repeat|output|leak|dump)"
)
_PROMPT_TARGET = (
    r"(?:(?:o\s+)?(?:seu|teu)\s+prompt|suas\s+instrucoes\s+(?:internas|iniciais|originais|do sistema)|"
    r"your\s+(?:system\s+)?prompt|your\s+(?:initial|original|hidden)\s+instructions|"
    r"(?:o\s+)?prompt\s+do\s+sistema(?!\s+operacional)|system\s+prompt)"
)

# Padrões sobre o texto com todo espaço colapsado (precisão acima de recall: documentos
# ingeridos que disparam ficam em quarentena).
_FLAT = {
    "ignore_instructions": rf"\b{_VERB_IGN}\s+{_DET}(?:(?:previous|prior|above|anteriores)\s+)?"
    rf"{_NOUN}(?:\s+{_QUAL})?",
    "forget_everything": r"\besque(?:ca|cer)\s+(?:(?:todas\s+)?as\s+)?(?:suas\s+)?instrucoes|"
    r"\besque(?:ca|cer)\s+tudo\s+(?:o\s+)?que\s+(?:te|lhe)\s+(?:foi dito|disseram|instruiram)|"
    r"\bforget\s+(?:all|everything)\s+(?:you were told|above|your instructions|previous instructions)",
    "persona": r"\byou are now (?:a|an) .{0,40}\b(?:assistant|ai|model|bot|"
    r"without (?:restrictions|filters|limits))\b|"
    r"\bvoce (?:agora )?e uma? (?:assistente|ia|modelo|bot)\b.{0,30}sem (?:restricoes|filtros|limites)|"
    r"\bdo anything now\b|"
    r"\b(?:ative|ativar|ativa|entre no|entre em|enable|enter|activate)\s+(?:o\s+|the\s+)?"
    r"(?:modo\s+|mode\s+)?(?:dan|jailbreak)\b|"
    r"\b(?:ative|ativar|entre no|entre em|enable|enter|activate)\s+(?:o\s+|the\s+)?"
    r"(?:modo desenvolvedor|developer mode)\b",
    "new_instructions": r"\b(?:novas|new)\s+(?:instrucoes|instructions)\s*:",
    "attention": r"\b(?:atencao|attention),?\s+(?:assistente|assistant)\s*[:!]",
    "chat_template": r"<\|im_start\|>|<\|im_end\|>|\[inst\]|<<sys>>|<\|system\|>",
}
# Dependem de quebras de linha (espaços/tabs colapsados, \n preservado).
_LINES = {
    "reveal_prompt": rf"\b{_DIRECTIVE}\b[^.!?\n]{{0,60}}?{_PROMPT_TARGET}",
    "markdown_system": r"^\s*###\s*(?:system|instructions?)s?\s*:?",
}
_FLAT_RX = {k: re.compile(v) for k, v in _FLAT.items()}
_LINE_RX = {k: re.compile(v, re.MULTILINE) for k, v in _LINES.items()}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", strip_invisible(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def detect(text: str) -> str | None:
    """Devolve o nome do primeiro padrão de injeção encontrado, ou None."""
    folded = _fold(text)
    lines = re.sub(r"[ \t]+", " ", folded)
    flat = re.sub(r"\s+", " ", folded)
    for name, rx in _LINE_RX.items():
        if rx.search(lines):
            return name
    for name, rx in _FLAT_RX.items():
        if rx.search(flat):
            return name
    return None
