from collections import Counter
from pathlib import Path

import yaml
from retrieval import load_cases

from encaixe.messages import NOT_FOUND, REFUSAL_INPUT

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "apps" / "exemplo" / "evals" / "tests.yaml"


def test_categories_and_fixed_messages():
    tests = yaml.safe_load(TESTS.read_text(encoding="utf-8"))
    counts = Counter(t["metadata"]["category"] for t in tests)
    assert counts == {
        "rag": 15, "acl": 3, "injecao_direta": 3, "injecao_indireta": 2,
        "ferramenta_leitura": 3, "fora_escopo": 2,
    }  # fmt: skip
    equals = {a["value"] for t in tests for a in t["assert"] if a["type"] == "equals"}
    assert equals == {NOT_FOUND, REFUSAL_INPUT}


def test_retrieval_cases_and_corpus_titles():
    cases = load_cases(TESTS)
    assert len(cases) == 15
    titles = {p.stem for p in (ROOT / "apps" / "exemplo" / "docs").glob("*.md")}
    assert {d for c in cases for d in c["expected_docs"]} <= titles
    assert any(c["groups"] == ("rh",) for c in cases)
