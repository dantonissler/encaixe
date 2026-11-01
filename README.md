# encaixe

Plataforma local de agentes de IA montada em 13 peças plugáveis. Cada peça resolve um problema que a anterior criou, e cada aplicação se conecta ao bot por configuração.

> **Status:** planejamento. Ainda não há código. O design está em [docs/superpowers/specs/2026-11-01-chatbot-13-pecas-design.md](docs/superpowers/specs/2026-11-01-chatbot-13-pecas-design.md).

## As 13 peças

| # | Peça | Tecnologia prevista |
|---|------|---------------------|
| 1 | Janela de contexto | Histórico no Postgres, orçamento de tokens, resumo acumulado |
| 2 | RAG | Docling, busca híbrida (vetorial + full-text), rerank |
| 3 | Embeddings | bge-m3 |
| 4 | Banco vetorial | PostgreSQL + pgvector |
| 5 | Tool calling | Pydantic AI + camada MCP (FastMCP) para sistemas terceiros |
| 6 | Agente | Loop do Pydantic AI com limites de passos e detecção de loop |
| 7 | Structured output | JSON Schema por gramática + TOON |
| 8 | Streaming | SSE compatível com a API da OpenAI |
| 9 | Guardrails | Prompt Guard 2, Llama Guard 3, Presidio |
| 10 | Prompt caching | Cache de prefixo do motor + Redis 8 (exato, semântico, embeddings) |
| 11 | Fallback | Cadeia de modelos, circuit breaker, degradação por peça |
| 12 | Evals | promptfoo + métricas de recuperação, rodando a cada mudança de prompt |
| 13 | Observabilidade | OpenTelemetry → Arize Phoenix |

Tudo roda localmente: Ollama (Qwen3.5, Gemma 4, Llama), Postgres, Redis e Phoenix em Docker Compose.
