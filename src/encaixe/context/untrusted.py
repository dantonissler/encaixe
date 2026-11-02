from __future__ import annotations

import secrets

from encaixe.output.toon import to_toon


def new_boundary() -> str:
    return "DADOS-" + secrets.token_hex(8)


def wrap_untrusted(kind: str, payload: list[dict] | dict, boundary: str) -> str:
    return f"<<{boundary} tipo={kind}>>\n{to_toon(payload)}\n<<FIM {boundary}>>"
