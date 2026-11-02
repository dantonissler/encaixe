import json

import multiturn


def test_check_turns():
    ok = multiturn.check_turns(
        "Você confirma a criação da tarefa?",
        "Criei a tarefa.",
        '[{"titulo":"Revisar Contrato"}]',
        "revisar contrato",
    )
    assert ok == []
    bad = multiturn.check_turns("Feito.", "Ok.", "[]", "revisar contrato")
    assert len(bad) == 3


def test_merge_extra_preserves_promptfoo_results(tmp_path):
    path = tmp_path / "exemplo.json"
    path.write_text(json.dumps({"results": {"stats": {"successes": 3}}}))
    multiturn.merge_extra(path, multiturn.summarize([{"id": "a", "failures": []}]))
    data = json.loads(path.read_text())
    assert data["results"]["stats"]["successes"] == 3
    assert data["extra"]["multiturn"]["passed"] == 1
    multiturn.merge_extra(tmp_path / "novo.json", {"passed": 0, "total": 2})
    assert json.loads((tmp_path / "novo.json").read_text())["extra"]["multiturn"]["total"] == 2
