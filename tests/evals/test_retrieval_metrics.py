import retrieval
from retrieval import (
    AclControlError,
    acl_control_problems,
    count_leaks,
    mrr,
    recall_at_k,
    summarize,
)


def test_recall_and_mrr():
    assert recall_at_k({"a", "b"}, ["x", "a", "y"], 2) == 0.5
    assert mrr({"a"}, ["x", "a"]) == 0.5


def test_edge_cases():
    assert recall_at_k(set(), ["a"], 5) == 0.0
    assert recall_at_k({"a"}, ["x", "y", "a"], 2) == 0.0
    assert mrr({"a"}, ["x", "y"]) == 0.0
    assert mrr({"a", "b"}, ["x", "b", "a"]) == 0.5  # primeiro acerto vale


def test_acl_leak_counting_and_summary():
    assert count_leaks({"salarios"}, ["a", "salarios"]) == 1
    assert count_leaks({"salarios"}, ["a", "b"]) == 0
    rows = [
        {"expected": ["a"], "recall": 1.0, "mrr": 1.0, "leaks": 0},
        {"expected": [], "forbidden": ["s"], "recall": 0.0, "mrr": 0.0, "leaks": 1},
        {"expected": [], "forbidden": ["s"], "recall": 0.0, "mrr": 0.0, "leaks": 0},
    ]
    s = summarize(rows, 5)
    assert s["acl_leaks"] == 1 and s["acl_cases"] == 2
    assert s["recall@5"] == 1.0 and s["cases"] == 1  # casos de ACL fora das médias


ACL_CASES = [
    {"expected_docs": [], "forbidden_docs": ["salarios-diretoria"]},
    {"expected_docs": ["politica-ferias"], "forbidden_docs": ["salarios-diretoria"]},
    {"expected_docs": ["beneficios"], "forbidden_docs": []},
]


def test_acl_control_passes_when_every_forbidden_doc_is_indexed():
    """Revisão final M-T14: controle positivo; sem ele, perder o documento restrito deixava
    acl_leaks=0 e a queda de recall/MRR ficava abaixo do max_drop."""
    statuses = {"salarios-diretoria": {"indexed"}, "politica-ferias": {"indexed"}}
    assert acl_control_problems(ACL_CASES, statuses) == []
    # reenvio: basta um documento com o título estar indexado
    statuses["salarios-diretoria"] = {"failed", "indexed"}
    assert acl_control_problems(ACL_CASES, statuses) == []


def test_acl_control_flags_missing_or_not_indexed_forbidden_docs():
    assert acl_control_problems(ACL_CASES, {}) == ["salarios-diretoria: não existe"]
    problems = acl_control_problems(ACL_CASES, {"salarios-diretoria": {"quarantined"}})
    assert problems == ["salarios-diretoria: status quarantined (esperado indexed)"]


def test_main_fails_the_run_when_acl_control_fails(monkeypatch, tmp_path, capsys):
    async def broken(app_id, k):
        raise AclControlError(["salarios-diretoria: não existe"])

    monkeypatch.setattr(retrieval, "_run", broken)
    monkeypatch.setattr(retrieval, "RESULTS_DIR", tmp_path)
    assert retrieval.main(["--app", "exemplo"]) == 1
    assert not (tmp_path / "exemplo-retrieval.json").exists()
    assert "salarios-diretoria: não existe" in capsys.readouterr().err
