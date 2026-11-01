from __future__ import annotations

import re
import unicodedata

from encaixe.guardrails.normalize import strip_invisible

_VERB_IGN = r"(?:ignor\w*|desconsider\w*|esquec\w*|disregard|forget|override)"
_NOUN = r"(?:instrucoes|regras|orientacoes|diretrizes|instructions|rules|prompts?|guidelines)"
_QUAL = r"(?:anteriores|anterior|acima|do sistema|previous|prior|above|system)"
_DET = r"(?:(?:todas|all|any|the|your|suas|as|essas|these|those|those)\s+)*"
_SHOW = r"(?:revel\w*|mostr\w*|imprim\w*|exib\w*|repit\w*|copi\w*)"
_POSS = r"(?:(?:todas|todos|as|os|o|a|seu|seus|sua|suas)\s+)*"

# Padrões sobre o texto com todo espaço colapsado.
_FLAT = {
    "ignore_instructions": rf"\b{_VERB_IGN}\s+{_DET}{_NOUN}\s+{_QUAL}|"
    rf"\b{_VERB_IGN}\s+(?:todas|all)\s+(?:as\s+|the\s+)?(?:{_QUAL}\s+)?{_NOUN}|"
    rf"\b{_VERB_IGN}\s+(?:the\s+|any\s+|as\s+)?{_QUAL}\s+{_NOUN}",
    "forget_everything": r"\b(?:forget|esqueca|esquecer)\s+(?:everything|all|tudo)\s+"
    r"(?:above|previous|prior|o que foi dito|acima|que (?:foi dito|eu disse))",
    "reveal_prompt_pt": rf"\b{_SHOW}\s+{_POSS}(?:prompt|instrucoes)\s+"
    r"(?:do sistema|internas|iniciais|originais)|"
    rf"\b{_SHOW}\s+{_POSS}prompt\b",
    "reveal_prompt_en": r"\b(?:reveal|print|show|repeat|output|display)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+)?(?:prompt|instructions)",
    "system_prompt": r"\bprompt do sistema\b|\bsystem prompt\b",
    "persona": r"\bvoce (?:agora )?e um assistente sem (?:restricoes|filtros|limites)|"
    r"\byou are now\b|\bdeveloper mode\b|\bmodo desenvolvedor\b|\bdan\b|\bjailbreak\b|"
    r"\bassistente sem (?:restricoes|filtros|limites)",
    "new_instructions": r"\b(?:novas|new)\s+(?:instrucoes|instructions)\s*:",
    "attention": r"\batencao,?\s+(?:assistente|ia|modelo)\b|\battention,?\s+(?:assistant|ai|model)\b",
}
# Marcadores de papel dependem de quebra de linha.
_LINES = {
    "role_marker": r"^\s*(?:system|sistema|assistant|assistente)\s*:",
    "chat_template": r"<\|im_start\|>|\[inst\]|###\s*(?:system|instruction)",
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
