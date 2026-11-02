import json

import multiturn
import pytest


def test_check_turns_requires_task_to_appear_after_turn_2():
    ask, ans, needle = "Você confirma a criação?", "Criei a tarefa.", "revisar contrato ab12"
    assert multiturn.check_turns(ask, ans, "[]", f'[{{"titulo":"{needle}"}}]', needle) == []
    # a tarefa já estava lá antes: a checagem seria vazia
    both = f'[{{"titulo":"{needle}"}}]'
    assert multiturn.check_turns(ask, ans, both, both, needle)
    # nada criado
    assert multiturn.check_turns(ask, ans, "[]", "[]", needle)
    assert len(multiturn.check_turns("Feito.", "Ok.", "[]", "[]", needle)) == 3


def test_cases_are_unique_per_run():
    a, b = multiturn.make_cases("aaaaaa"), multiturn.make_cases("bbbbbb")
    assert {c["user"] for c in a}.isdisjoint({c["user"] for c in b})
    assert {c["needle"] for c in a}.isdisjoint({c["needle"] for c in b})
    assert all(c["needle"] in c["prompt"] for c in a)


def test_merge_extra_preserves_promptfoo_results(tmp_path):
    path = tmp_path / "exemplo.json"
    path.write_text(json.dumps({"results": {"stats": {"successes": 3}}}))
    multiturn.merge_extra(path, multiturn.summarize([{"id": "a", "failures": []}]))
    data = json.loads(path.read_text())
    assert data["results"]["stats"]["successes"] == 3
    assert data["extra"]["multiturn"]["passed"] == 1


def test_merge_extra_refuses_to_create_the_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        multiturn.merge_extra(tmp_path / "novo.json", {"passed": 2, "total": 2})
    assert not (tmp_path / "novo.json").exists()


def test_main_fails_without_promptfoo_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ENCAIXE_EVAL_API_KEY", "k")
    assert multiturn.main(["--results", str(tmp_path / "x.json")]) == 1


async def test_write_turns_carry_the_conversation_id(monkeypatch):
    """Sem X-Conversation-Id não há ferramenta de escrita (spec §7): o pedido de criação e o
    "sim" precisam ir numa conversa já aberta."""
    calls: list[tuple[str, str | None]] = []
    created: set[str] = set()

    async def fake_chat(client, user, text, conversation_id=None):
        calls.append((text, conversation_id))
        if text == "sim" and conversation_id:
            created.add(user)
            return "Criei a tarefa.", conversation_id
        return "Você confirma a criação?", conversation_id or "cid-1"

    async def fake_mcp(user, tool, args=None):
        needle = next(c["needle"] for c in multiturn.make_cases("abc123") if c["user"] == user)
        return json.dumps([{"titulo": needle}] if user in created else [])

    monkeypatch.setattr(multiturn, "_chat", fake_chat)
    monkeypatch.setattr(multiturn, "_mcp", fake_mcp)
    summary = await multiturn.run_cases("abc123")
    assert summary["passed"] == summary["total"] == 2
    prompts = {c["prompt"] for c in multiturn.make_cases("abc123")}
    assert all(cid == "cid-1" for text, cid in calls if text in prompts or text == "sim")
