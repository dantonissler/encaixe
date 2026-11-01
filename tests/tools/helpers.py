from __future__ import annotations

import jinja2

from encaixe.core.config import AppConfig, LoadedApp, McpServerConfig, ModelsConfig, ToolsConfig
from encaixe.core.types import Principals, RequestContext
from encaixe.output.schemas import RespostaBase


def make_ctx(
    *,
    user_ref: str | None = "u1",
    groups: tuple[str, ...] = ("g1",),
    conversation_id: str | None = "c1",
    identity: bool = True,
    message: str = "oi",
    allow: list[str] | None = None,
    require_approval: list[str] | None = None,
) -> RequestContext:
    cfg = AppConfig(
        id="app1",
        models=ModelsConfig(primary="test"),
        tools=ToolsConfig(
            mcp_servers=[
                McpServerConfig(
                    name="tarefas",
                    url="http://mcp.test/mcp",
                    allow=allow if allow is not None else ["listar_tarefas", "criar_tarefa"],
                    require_approval=require_approval if require_approval is not None else ["criar_tarefa"],
                )
            ]
        ),
    )
    app = LoadedApp(
        config=cfg,
        root=None,  # type: ignore[arg-type]
        system_prompt=jinja2.Template("s"),
        prompt_version="v",
        output_model=RespostaBase,
    )
    return RequestContext(
        request_id="r",
        app=app,
        principals=Principals(user_ref, groups),
        conversation_id=conversation_id,
        has_conversation_identity=identity,
        user_message=message,
        client_history=[],
        canary="C",
    )


class FakeApprovalRepo:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def create_pending(self, app_id, conversation_id, user_ref, server, tool, args, ttl_s=600):
        row = {
            "id": f"a{len(self.rows)}", "app_id": app_id, "conversation_id": conversation_id,
            "user_ref": user_ref, "server": server, "tool": tool, "args": args, "status": "pending",
        }
        self.rows.append(row)
        return row["id"]

    async def get_pending(self, app_id, conversation_id, user_ref):
        from encaixe.db.repos.approvals import Approval

        for r in reversed(self.rows):
            if (r["conversation_id"], r["user_ref"], r["status"]) == (conversation_id, user_ref, "pending"):
                return Approval(r["id"], r["app_id"], r["conversation_id"], r["user_ref"], r["server"],
                                r["tool"], r["args"], r["status"], None)  # type: ignore[arg-type]
        return None

    async def resolve(self, app_id, approval_id, status):
        for r in self.rows:
            if r["id"] == approval_id and r["status"] == "pending":
                r["status"] = status
                return True
        return False
