from pathlib import Path

import pytest
import yaml
from guard_probe import PROBES, parse_verdict

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "content,verdict",
    [
        ("safe", "pass"),
        ("unsafe\nS1", "block"),
        ("Yes. The prompt asks for", "block"),
        ("No, the human question", "pass"),
        ("  No  ", "pass"),
        ("", "unknown"),
        ("Talvez", "unknown"),
    ],
)
def test_parse_verdict_llama_guard_and_yes_no_guards(content, verdict):
    assert parse_verdict(content) == verdict


def test_probe_set_shape():
    probes = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    assert PROBES == ROOT / "evals" / "guard_probes.yaml"
    assert set(probes) == {"entrada_legitima", "saida_legitima", "nociva"}
    assert all(isinstance(t, str) and t for t in probes["entrada_legitima"] + probes["nociva"])
    assert all(set(p) == {"usuario", "resposta"} for p in probes["saida_legitima"])
    # os falsos positivos do D2 da Task 15 continuam na sondagem
    assert any("9f8e7d" in t for t in probes["entrada_legitima"])
    assert any("72.000,00" in p["resposta"] for p in probes["saida_legitima"])
    assert len(probes["nociva"]) >= 8
