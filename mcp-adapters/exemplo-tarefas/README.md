# Adaptador MCP de exemplo: tarefas

Servidor FastMCP com tarefas em memória, separadas por usuário (cabeçalho `X-User-Ref`; sem ele, `anon`).

Ferramentas: `listar_tarefas()` e `criar_tarefa(titulo, prazo=None)`.

Endpoint: `http://localhost:8765/mcp` (Streamable HTTP).

```bash
docker compose --profile tools up mcp-tarefas
# ou, sem Docker:
uv run python mcp-adapters/exemplo-tarefas/server.py
```

Os dados somem ao reiniciar. Para usar num app, declare o servidor em `tools.mcp_servers` do `app.yaml`
(`allow` e, para escrita, `require_approval: [criar_tarefa]`).
