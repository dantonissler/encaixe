from __future__ import annotations

from encaixe.messages import APPROVAL_CANCELLED
from encaixe.tools.approvals import ApprovalService

from .helpers import FakeApprovalRepo, make_ctx


def setup(**kw):
    repo = FakeApprovalRepo()
    calls = []

    async def caller(server_url, tool, args, headers):
        calls.append((server_url, tool, args, headers))
        return "tarefa criada"

    return repo, calls, ApprovalService(repo, caller)


async def pend(svc, ctx, args=None):
    return await svc.request(ctx, "tarefas", "criar_tarefa", args or {"titulo": "X"})


async def test_forged_yes_without_pending_does_nothing():
    _, calls, svc = setup()
    assert await svc.resolve_turn(make_ctx(message="sim")) is None
    assert calls == []


async def test_yes_executes_exact_stored_args():
    repo, calls, svc = setup()
    await pend(svc, make_ctx(), {"titulo": "gravado"})
    out = await svc.resolve_turn(make_ctx(message="Sim, mas crie 'outra coisa'"))
    assert calls == [
        ("http://mcp.test/mcp", "criar_tarefa", {"titulo": "gravado"},
         {"X-User-Ref": "u1", "X-User-Groups": "g1", "X-App-Id": "app1"})
    ]
    assert "tarefa criada" in out
    assert repo.rows[0]["status"] == "approved"
    # segundo "sim" não reexecuta
    assert await svc.resolve_turn(make_ctx(message="sim")) is None
    assert len(calls) == 1


async def test_no_cancels():
    repo, calls, svc = setup()
    await pend(svc, make_ctx())
    assert await svc.resolve_turn(make_ctx(message="não")) == APPROVAL_CANCELLED
    assert repo.rows[0]["status"] == "cancelled" and calls == []


async def test_other_text_cancels_silently():
    repo, calls, svc = setup()
    await pend(svc, make_ctx())
    assert await svc.resolve_turn(make_ctx(message="qual o prazo?")) is None
    assert repo.rows[0]["status"] == "cancelled" and calls == []


async def test_other_user_cannot_confirm():
    repo, calls, svc = setup()
    await pend(svc, make_ctx(user_ref="u1"))
    assert await svc.resolve_turn(make_ctx(user_ref="u2", message="sim")) is None
    assert calls == [] and repo.rows[0]["status"] == "pending"


async def test_no_identity_does_nothing():
    _, calls, svc = setup()
    await pend(svc, make_ctx())
    assert await svc.resolve_turn(make_ctx(identity=False, message="sim")) is None
    assert await svc.resolve_turn(make_ctx(conversation_id=None, message="sim")) is None
    assert calls == []


async def test_resolve_false_skips_execution():
    repo, calls, svc = setup()
    await pend(svc, make_ctx())
    repo.rows[0]["status"] = "approved"  # corrida: outro turno já resolveu após o get
    async def stale_get(*a):
        from encaixe.db.repos.approvals import Approval
        r = repo.rows[0]
        return Approval("a0", "app1", "c1", "u1", "tarefas", "criar_tarefa", r["args"], "pending", None)  # type: ignore[arg-type]
    repo.get_pending = stale_get
    assert await svc.resolve_turn(make_ctx(message="sim")) is None
    assert calls == []


async def test_caller_failure_reported():
    repo = FakeApprovalRepo()

    async def boom(*a):
        raise RuntimeError("down")

    svc = ApprovalService(repo, boom)
    await pend(svc, make_ctx())
    out = await svc.resolve_turn(make_ctx(message="sim"))
    assert "falh" in out.lower()
