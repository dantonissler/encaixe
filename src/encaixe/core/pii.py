from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["cpf", "cnpj", "email", "telefone"]


@dataclass(frozen=True)
class PiiMatch:
    kind: Kind
    start: int
    end: int


_CPF = re.compile(r"(?<!\d)(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})(?!\d)")
_CNPJ = re.compile(r"(?<!\d)(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(
    r"(?<![\d\w])(?:\+55[\s-]?)?(?:\(\d{2}\)|\d{2})[\s-]?(?:9\d{4}|\d{4})-?\d{4}(?!\d)"
)
_MASKS = {"cpf": "[CPF]", "cnpj": "[CNPJ]", "email": "[EMAIL]", "telefone": "[TELEFONE]"}


def _digits(s: str) -> list[int]:
    return [int(c) for c in s if c.isdigit()]


def _valid_cpf(s: str) -> bool:
    d = _digits(s)
    if len(d) != 11 or len(set(d)) == 1:
        return False
    for n in (9, 10):
        total = sum(d[i] * (n + 1 - i) for i in range(n))
        if (total * 10 % 11) % 10 != d[n]:
            return False
    return True


def _valid_cnpj(s: str) -> bool:
    d = _digits(s)
    if len(d) != 14 or len(set(d)) == 1:
        return False
    for n in (12, 13):
        weights = ([5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
        if n == 13:
            weights = [6] + weights
        total = sum(d[i] * weights[i] for i in range(n))
        r = total % 11
        if (0 if r < 2 else 11 - r) != d[n]:
            return False
    return True


def find_pii(text: str) -> list[PiiMatch]:
    found: list[PiiMatch] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < m.end and m.start < b for m in found)

    for kind, rx, check in (
        ("email", _EMAIL, None),
        ("cnpj", _CNPJ, _valid_cnpj),
        ("cpf", _CPF, _valid_cpf),
        ("telefone", _PHONE, None),
    ):
        for m in rx.finditer(text):
            if check and not check(m.group()):
                continue
            if overlaps(m.start(), m.end()):
                continue
            found.append(PiiMatch(kind, m.start(), m.end()))  # type: ignore[arg-type]
    return sorted(found, key=lambda m: m.start)


def mask_pii(text: str) -> str:
    out, pos = [], 0
    for m in find_pii(text):
        out.append(text[pos : m.start])
        out.append(_MASKS[m.kind])
        pos = m.end
    out.append(text[pos:])
    return "".join(out)
