import re

from encaixe.context.untrusted import new_boundary, wrap_untrusted
from encaixe.output.toon import from_toon


def test_new_boundary_format():
    assert re.fullmatch(r"DADOS-[0-9a-f]{16}", new_boundary())


def test_wrap_has_boundary_and_toon():
    b = new_boundary()
    payload = [{"id": "1", "texto": "olá"}]
    out = wrap_untrusted("trechos", payload, b)
    assert out.count(b) == 2
    lines = out.split("\n")
    assert lines[0] == f"<<{b} tipo=trechos>>"
    assert lines[-1] == f"<<FIM {b}>>"
    assert from_toon("\n".join(lines[1:-1])) == payload
