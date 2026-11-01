# Chatbot plugável com as 13 peças de um agente de IA — Design

- **Data:** 2026-11-01
- **Status:** rascunho para aprovação
- **Escopo deste documento:** design (o quê e por quê, e as decisões de como). O plano de implementação passo a passo vem depois, a partir deste documento.

---

## 1. Intenção

Construir uma **plataforma de chatbot/agente 100% local** que transforma uma chamada de LLM em um agente de produção, montada a partir de 13 peças. Cada peça resolve um problema que a anterior criou.

As aplicações que vão usar o bot **ainda não são conhecidas**. Por isso o bot é uma plataforma genérica: uma aplicação nova se pluga **por configuração**, sem mexer no código do núcleo. Cada peça também é plugável: dá para ligar, desligar ou trocar a implementação por aplicação.

**Prioridade declarada:** colocar em **uso real o quanto antes**. Isso orienta a regra de construção: nenhuma peça é escrita do zero quando existe biblioteca madura que resolve. O código próprio é a "cola" — interfaces, pipeline e configuração por aplicação.

### 1.1 A cadeia das 13 peças

| # | Peça | Problema que resolve | Problema que cria (→ próxima peça) |
|---|------|----------------------|------------------------------------|
| 1 | Janela de contexto | O modelo não lembra da conversa | Só sabe o que cabe no contexto e o que aprendeu no treino |
| 2 | RAG | Não conhece os documentos da aplicação | Como achar o trecho certo pelo significado, não pela palavra |
| 3 | Embeddings | Busca por significado | Onde guardar e buscar milhares/milhões de vetores rápido |
| 4 | Banco vetorial | Armazenamento e busca vetorial | O modelo só responde; não consulta nem age em sistemas |
| 5 | Tool calling (+ camada MCP) | O modelo chama funções/sistemas terceiros | Uma chamada não basta para tarefas de vários passos |
| 6 | Agente | Loop até terminar a tarefa | Saída em texto livre que sistemas não conseguem consumir com segurança |
| 7 | Structured output | Contrato de resposta garantido | Loop + validação deixam a resposta lenta |
| 8 | Streaming | Latência percebida | O texto sai antes de ser checado; documentos e ferramentas podem injetar instruções |
| 9 | Guardrails | Segurança de entrada, saída e conteúdo não confiável | Mais chamadas de modelo por requisição: custo e latência |
| 10 | Prompt caching | Custo e latência | Dependência de um modelo/servidor que pode falhar |
| 11 | Fallback | Disponibilidade | Vários modelos e caminhos: a qualidade varia sem ninguém perceber |
| 12 | Evals | Regressão mensurável a cada mudança | O eval diz *que* piorou, não *onde* nem *por quê* em produção |
| 13 | Observabilidade | Rastreio de cada passo | Fecha o ciclo: traces de produção viram novos casos de eval |

---

## 2. Objetivos, não-objetivos e anti-objetivos

### Objetivos
1. As 13 peças funcionando, cada uma atrás de uma interface própria e ativável por aplicação.
2. **Aplicação como conceito de primeira classe:** cada `app_id` tem documentos, prompt, ferramentas, schema de saída, políticas de cache e guardrails próprios, definidos em arquivo versionado.
   - **Permissões por usuário e grupo dentro da aplicação, desde a v1** (decisão do usuário): documentos com lista de acesso; busca, cache e ferramentas respeitam quem está perguntando (§6.4a).
3. **API compatível com o formato OpenAI** (`/v1/chat/completions`, com streaming SSE). Qualquer aplicação se conecta com SDKs que já existem.
4. **Camada MCP** para conectar APIs de sistemas terceiros como ferramentas do agente.
5. **100% local:** nenhuma chamada a serviço externo em tempo de execução, nem telemetria de bibliotecas.
6. Primeira aplicação em uso real (piloto interno) ao fim da Fase 2 (§9).
7. **Alvo de operação da v1 (decisão do usuário):** rodar **neste notebook**, para **poucos usuários simultâneos (até ~5)**. Um servidor maior fica para depois da v1.

### Não-objetivos (v1)
- Fine-tuning de modelos.
- Frontend próprio. Para testes internos usa-se o **Open WebUI** apontado para a API; as aplicações têm suas próprias interfaces.
- Alta disponibilidade multi-nó, Kubernetes, autoescala.
- Voz, imagem e outras entradas multimodais.
- Gerar automaticamente servidores MCP para toda API existente.
- Expor o próprio bot como servidor MCP (fica para depois da v1; ver §9, Fase 6).
- Conectores de ingestão (SharePoint, pastas de rede, sincronização). Na v1 os documentos entram **só por upload via API**.
- Servidor dedicado, vLLM, alta concorrência. A v1 roda no notebook (objetivo 7).

### Anti-objetivos (é fracasso, mesmo que funcione)
- O bot **obedecer instruções** vindas de documentos, resultados de ferramentas ou mensagens antigas do histórico.
- **Vazamento entre aplicações:** app A enxergando documentos, cache, histórico ou credenciais do app B.
- **Vazamento entre usuários da mesma aplicação:** usuário recebendo trecho de documento, resposta em cache ou dado de ferramenta que sua permissão não alcança.
- **Cache semântico vazando dado de um usuário para outro.**
- Mudança de prompt, modelo ou schema indo para produção **sem rodar evals**.
- Comportamento que não dá para rastrear: um passo do agente que não aparece no trace.
- Ação com efeito colateral (gravar, enviar, pagar, abrir chamado) executada sem confirmação.

---

## 3. Restrições

- **Hardware de desenvolvimento (verificado):** RTX 3060 Laptop com **6 GB de VRAM**, 14 GB de RAM, 16 núcleos, Docker, Python com `uv` e Node 20. Isso limita os modelos (§5.1) e força a divisão de modelos entre GPU e CPU (§5.2).
- **Hardware de produção da v1:** o mesmo notebook, para até ~5 usuários simultâneos (decisão do usuário). Consequências:
  - Uma GPU só, e o Ollama atende na prática uma geração por vez. A API tem uma **fila com limite de concorrência**: requisições excedentes esperam com timeout e recebem um evento de "aguardando" no stream, ou `429` se a fila encher.
  - **Metas iniciais de latência** (calibradas na Fase 0): primeiro token em < 3 s numa conversa sem ferramentas e sem fila; resposta completa com RAG em < 20 s.
  - Se o notebook desligar ou hibernar, o bot fica fora do ar. É aceitável no piloto.
  - Acesso pela rede local através de um proxy reverso com TLS (Caddy). Só a API e o Open WebUI ficam expostos; o resto fica na rede interna do Docker.
  - Backup diário do Postgres (`pg_dump`) para outro disco.
  - O motor de inferência continua intercambiável (vLLM ou llama.cpp server num servidor futuro) sem mudar o resto.
- **Construção:** uma pessoa (o usuário com o Claude), sem prazo fixo. As fases seguem em sequência completa, sem trabalhos paralelos.
- **Idioma:** português, principalmente. Todos os modelos escolhidos são multilíngues e precisam ser validados em português pelos evals.
- **Dados pessoais (LGPD):** as aplicações podem trafegar CPF, nomes e outros dados. Logs e traces mascaram dados pessoais, e o cache semântico tem regras de escopo (§6.10).
- **Cadeia de suprimentos:** em março de 2026 o LiteLLM teve as versões 1.82.7 e 1.82.8 comprometidas no PyPI, e o pgvector teve CVEs corrigidas em 2026. Toda dependência fica com versão fixada e hash (`uv.lock`), imagens Docker são referenciadas por digest e as atualizações são deliberadas.

---

## 4. Abordagens consideradas

| Abordagem | Resumo | Trade-off | Decisão |
|-----------|--------|-----------|---------|
| **A. Núcleo próprio em Python sobre bibliotecas maduras** | FastAPI enxuto com interfaces por peça; cada peça implementada por uma ferramenta consolidada | Mais peças para orquestrar e o código de cola é do time; em troca, não há teto e todas as 13 peças ficam como pedido | **Escolhida pelo usuário** |
| B. Plataforma pronta self-hosted (ex.: Dify) | Apps, RAG, agente e ferramentas prontos; protótipo em dias | Evals por mudança de prompt, saída estruturada garantida, guardrails sob medida e fallback ficam fracos ou rígidos; pesada em RAM; licença restringe multi-tenant | Descartada |
| C. Igual à A, em TypeScript (Mastra / Vercel AI SDK) | Viável com Node 20 | Ecossistema local de embeddings, guardrails e avaliação de RAG bem mais rico em Python | Descartada |

**Mudança depois da escolha (decisão do Claude, para revisão):** a proposta original de A incluía o **LiteLLM** para roteamento, fallback e cache. A pesquisa mostrou que o **Pydantic AI v2** já faz fallback entre modelos (`FallbackModel`) e fala com o Ollama nativamente, e que o **RedisVL** faz o cache semântico no Redis. O LiteLLM sai do núcleo: uma dependência a menos, e menos superfície depois do incidente de março de 2026. Ele volta como opção se um dia for preciso um gateway central com chaves e orçamento por aplicação para muitos backends.

---

## 5. Stack escolhida

### 5.1 Modelos

Ninguém escolhe o modelo por opinião: a **Fase 0 roda uma disputa entre modelos nos evals** e os números decidem. Ponto de partida:

| Papel | Desenvolvimento (6 GB) | Produção com GPU maior (referência) | Licença |
|-------|------------------------|-------------------------------------|---------|
| LLM principal | **Qwen3.5 4B** (Q4_K_M, ~3,3 GB) | Qwen3.5 9B ou Gemma 4 26B (MoE) | Apache 2.0 |
| LLM de fallback (outra família) | **Gemma 4 E2B** (na CPU; ver §5.2) | Gemma 4 E4B, ou o principal em outra instância | Apache 2.0 |
| Candidato na disputa | Llama 3.2 3B, Gemma 4 E4B | — | Llama Community / Apache 2.0 |
| Embeddings | **bge-m3** (1024 dimensões, multilíngue, 8k tokens) | o mesmo | MIT |
| Embeddings (alternativa) | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B/8B | Apache 2.0 |
| Reranker | **bge-reranker-v2-m3** | o mesmo | Apache 2.0 |
| Guarda contra injeção | **Llama Prompt Guard 2 86M** (multilíngue) | o mesmo | Llama 4 Community (repositório restrito no HF) |
| Guarda de conteúdo | **Llama Guard 3 1B** (`llama-guard3:1b`) | Llama Guard 3 8B, ou Qwen3Guard | Llama 3.2 Community |

Notas:
- Relatos da comunidade indicam que o Qwen3.5 4B acerta mais em tool calling do que o Gemma 4 E4B nesse tamanho. Os relatos são informais; é exatamente o que a disputa da Fase 0 mede.
- Modelos de 3–4B erram mais no loop de agente. Os limites do agente (§6.6), o fallback (§6.11) e os evals (§6.12) existem também para compensar isso.

### 5.2 Orçamento de VRAM e RAM (desenvolvimento)

Os 6 GB de VRAM ficam **reservados para o LLM principal** e seu KV cache. Os modelos auxiliares rodam na CPU (16 núcleos):

| Componente | Onde | Memória aproximada |
|------------|------|--------------------|
| Qwen3.5 4B Q4_K_M + KV cache (`num_ctx` 8192, KV `q8_0`, flash attention) | GPU | ~4,5–5,5 GB VRAM |
| bge-m3 (embeddings de consulta; ingestão em lote pode usar a GPU com o chat ocioso) | CPU | ~1,2 GB RAM |
| bge-reranker-v2-m3 + Prompt Guard 2 (ONNX Runtime) | CPU | ~1,5 GB RAM |
| Llama Guard 3 1B | CPU | ~1,5 GB RAM |
| Gemma 4 E2B (fallback) | CPU, carregado sob demanda | ~3 GB RAM |
| Postgres, Redis, Phoenix, API, worker | CPU | ~2 GB RAM |

Rodar o fallback **em outro dispositivo e de outra família** também protege contra falhas correlacionadas, como estouro de memória na GPU ou problema de driver. A RAM fica apertada (14 GB, com ~7 GB livres no momento da medição). O builder mede na Fase 0 e, se precisar, descarrega o fallback (`keep_alive` curto).

### 5.3 Infraestrutura e bibliotecas

| Peça / função | Tecnologia | Por quê |
|---------------|-----------|---------|
| API | **FastAPI** + Uvicorn, SSE | Async, padrão do ecossistema, streaming simples |
| Inferência de LLM | **Ollama** (≥ 0.19) | Simples em desenvolvimento; tool calling, saída por JSON Schema, cache de prefixo automático. Em produção pode virar vLLM ou llama.cpp server |
| Agente, tool calling, saída estruturada, fallback | **Pydantic AI v2** | Loop de agente, `output_type` com Pydantic, `FallbackModel`, cliente MCP nativo, ferramentas com aprovação, instrumentação OpenTelemetry |
| Camada MCP (adaptadores) | **FastMCP 3.x** | Servidores MCP finos sobre APIs REST de terceiros |
| Banco relacional + vetorial | **PostgreSQL 17/18 + pgvector ≥ 0.8.2** (preferir a última 0.8.x) | Um banco só para histórico, documentos, vetores e busca textual; HNSW com *iterative scan* para filtros por app |
| Busca textual (híbrida) | Full-text search do Postgres (config `portuguese`) | Sem serviço extra; combinado com vetores via RRF |
| Parsing de documentos | **Docling 2.x** | Local; PDF (com OCR), DOCX, PPTX, XLSX, HTML; chunker que respeita títulos |
| Cache | **Redis 8** + **RedisVL** (`SemanticCache`) | Busca vetorial embutida no Redis 8; cache exato, semântico e de embeddings |
| Fila de ingestão | **arq** (asyncio sobre Redis) | Leve, reusa o Redis |
| Serialização para contexto | **TOON** (implementação oficial `toon-format/toon-python`) | 30–60% menos tokens que JSON para dados tabulares no contexto |
| Mascaramento de dados pessoais | **Microsoft Presidio** + reconhecedores para CPF, CNPJ, RG, telefone | Local, extensível |
| Observabilidade | **OpenTelemetry** (convenções OpenInference) → **Arize Phoenix** self-hosted, num database próprio no mesmo Postgres | Um processo só, leve para 14 GB de RAM. Langfuse v3 exige ClickHouse + Redis + S3 e ~16 GB de RAM. Por ser OTel, dá para trocar depois |
| Logs | **structlog** em JSON com `trace_id`/`span_id` | Correlação log ↔ trace |
| Evals | **promptfoo** (MIT) + testes de recuperação em pytest + datasets do Phoenix | Asserts determinísticos, LLM-juiz, red team de injeção, comparação com baseline |
| UI de teste | **Open WebUI** apontado para `/v1` | Zero frontend para testar |
| Empacotamento | `uv` + `uv.lock` com hashes; Docker Compose | Reprodutível e com versões fixadas |

**Licenças que pedem atenção:** Phoenix usa Elastic License 2.0 (uso interno OK; não pode ser oferecido como serviço gerenciado). Redis 8 é tri-licenciado (AGPLv3/RSAL/SSPL); uso interno sem distribuir modificações é OK. Os modelos Llama Guard e Prompt Guard usam licenças comunitárias da Meta. promptfoo foi adquirido pela OpenAI em março de 2026 e continua MIT. Não é bloqueio, mas fica registrado como risco: se mudar de rumo, o caminho é DeepEval.

---

## 6. Arquitetura

### 6.1 Visão geral

```
                ┌──────────── Aplicações (desconhecidas) / Open WebUI ────────────┐
                │      SDK OpenAI → POST /v1/chat/completions (stream=true)       │
                └──────────────────────────────┬──────────────────────────────────┘
                                               │  API key → app_id
┌──────────────────────────────────── API (FastAPI) ───────────────────────────────────────┐
│ Pipeline por requisição (cada estágio = 1 span OTel; ligável/desligável por app)          │
│                                                                                          │
│ 1 guard.input → 2 cache.lookup → 3 context.build ──→ 4 agent.run ──→ 5 guard.output       │
│  (injeção,      (exato,          (histórico,        (Pydantic AI:      (schema, conteúdo, │
│   conteúdo,      semântico)       resumo, RAG,       LLM ⇄ tools/MCP,   dados pessoais,   │
│   dados pessoais)                 orçamento tokens)  fallback, limites)  canário, citação)│
│                                                                     │                    │
│                         6 stream SSE ◀───────────────────────────────┘                   │
│                         7 cache.store → 8 persist (histórico) → 9 resumo assíncrono      │
└───────┬───────────────┬───────────────┬───────────────┬───────────────┬──────────────────┘
        │               │               │               │               │
   Ollama (GPU/CPU)  Postgres+pgvector  Redis 8     Serviço aux. (CPU)  Servidores MCP
   LLMs, bge-m3,     histórico, docs,   cache,      reranker,           (adaptadores FastMCP
   Llama Guard       chunks, FTS        fila arq    Prompt Guard 2      para APIs terceiras)
        │
   Worker de ingestão (arq): Docling → chunking → Prompt Guard → embeddings → pgvector

   Tudo emite OTel → Phoenix (traces + datasets de eval)   |   promptfoo roda sobre a mesma API
```

### 6.2 Contrato "plug-and-play"

Cada peça é uma **interface Python (`Protocol`)** com uma ou mais implementações, registradas por nome. A configuração da aplicação escolhe a implementação e os parâmetros. Exemplo:

```python
class VectorStore(Protocol):
    async def upsert(self, app_id: str, chunks: list[Chunk]) -> None: ...
    async def search(self, app_id: str, query: SearchQuery, k: int) -> list[ScoredChunk]: ...

class Guard(Protocol):
    name: str
    async def check(self, item: GuardInput, ctx: RequestContext) -> GuardResult: ...
```

Interfaces previstas: `ChatModel` (via Pydantic AI), `Embedder`, `VectorStore`, `Retriever`, `Reranker`, `DocumentParser`, `Chunker`, `ContextBuilder`, `HistoryStore`, `Summarizer`, `Guard`, `Cache`, `ToolProvider` (nativo ou MCP), `OutputFormatter` (JSON/TOON), `Tracer`.

O pipeline é uma lista ordenada de **estágios** que recebem e devolvem um `RequestContext`. Cada estágio é instrumentado automaticamente (um decorator cria o span). Para acrescentar uma implementação basta criar a classe e uma linha de configuração; o pipeline não muda.

### 6.3 Configuração por aplicação (GitOps)

Aplicações, prompts e schemas ficam **em arquivos versionados no git**, não no banco. Assim "mudou o prompt → roda o eval" vira um gatilho natural (§6.12).

```
apps/
  <app_id>/
    app.yaml          # liga/desliga peças, modelos, limites, ferramentas
    prompts/
      system.md       # prompt de sistema (Jinja2), versionado
    schemas.py        # modelos Pydantic da saída estruturada
    evals/            # dataset dourado + config promptfoo da app
```

```yaml
# apps/exemplo/app.yaml (ilustrativo)
id: exemplo
models:
  primary: ollama:qwen3.5:4b
  fallbacks: [ollama-cpu:gemma4:e2b]
context:
  num_ctx: 8192
  budget: {system: 1500, summary: 500, history: 2500, rag: 2500, user: 800, output: 1000}
rag:
  enabled: true
  top_k: 20          # candidatos da busca híbrida
  rerank_top_n: 5    # vão para o contexto
  require_citations: true
tools:
  native: [buscar_documentos]
  mcp_servers:
    - name: sistema-x
      url: http://mcp-sistema-x:8000/mcp
      allow: [consultar_pedido, abrir_chamado]
      require_approval: [abrir_chamado]
agent:
  max_steps: 8
  max_tool_calls: 12
  timeout_s: 60
output:
  schema: schemas.Resposta
  formats: [json, toon]       # o cliente escolhe
cache:
  exact: {enabled: true, ttl_s: 3600}
  semantic: {enabled: false}  # ver regras em §6.10
guardrails:
  input: [normalize, prompt_injection, content_safety, pii]
  output: [schema, content_safety, pii_leak, canary, citations]
  on_guard_unavailable: block # fail-closed
streaming:
  mode: buffered_sentences    # ou: strict (só envia depois da checagem completa)
```

### 6.4 Modelo de dados (Postgres)

- `apps_runtime` (estado em tempo de execução, como versão do corpus e status) e `api_keys` (hash, `app_id`, escopos). A definição da app vem do YAML.
- `conversations` (`id`, `app_id`, `user_ref` opaco fornecido pela aplicação, timestamps)
- `messages` (`conversation_id`, papel, conteúdo, tool calls em jsonb, tokens, `trace_id`, versão do prompt, modelo usado)
- `conversation_summaries` (resumo acumulado e até qual mensagem cobre)
- `documents` (`app_id`, origem, hash do conteúdo, versão, status de ingestão, flag de quarentena)
- `chunks` (`app_id`, `document_id`, texto, `tsv tsvector`, `embedding vector(1024)`, nome e versão do modelo de embedding, metadados jsonb com caminho de títulos e página)
- `ingestion_jobs`, `tool_approvals`
- **Row-Level Security por `app_id`** em tudo que é dado de app: defesa em profundidade contra vazamento entre aplicações.
- `chunks.acl_principals text[]` (copiado de `documents.acl`) com índice GIN, e uma política de RLS que também filtra pelos principais da requisição (§6.4a).
- `data_retention` por app e job de expurgo (§6.14).

### 6.4a Identidade e permissões dentro da aplicação

- **Quem é o usuário:** a aplicação se autentica com a API key e **declara** quem é o usuário final e seus grupos, em `metadata.user_ref` e `metadata.groups` (ou nos headers `X-User-Ref` e `X-User-Groups`). O bot confia nessa declaração porque a chamada vem do **backend da aplicação**. A API key nunca vai para o navegador; isso é requisito de integração.
- **Principais da requisição:** `user:<user_ref>` + `group:<g>` para cada grupo + `app:*` (acesso público dentro da app).
- **Documentos:** o upload recebe `acl` (lista de principais). Sem `acl`, o documento vale como `app:*`. Mudar a ACL de um documento atualiza os chunks e incrementa a versão do corpus.
- **Busca:** a busca vetorial e a textual filtram por `app_id` **e** por `acl_principals && principais_da_requisição`, com o *iterative scan* do pgvector para o filtro não esvaziar o top-k. A RLS repete o filtro no banco (defesa em profundidade).
- **Cache:** a chave do cache exato inclui o hash dos principais. O cache semântico só guarda respostas em que **todos** os chunks usados são `app:*`.
- **Ferramentas e MCP:** o bot repassa `user_ref` e os grupos aos adaptadores MCP (header na chamada). **O adaptador aplica as permissões do sistema terceiro.** Repassar o token delegado do usuário fica a cargo de cada adaptador, quando o sistema terceiro suportar.
- **Histórico:** uma conversa pertence a um `user_ref`, e só ele a lê.
- **Evals de vazamento:** casos em que o usuário A pergunta sobre um documento restrito ao B. A recuperação precisa voltar vazia e a resposta não pode citar o conteúdo.

### 6.5 Peças 1–4: janela de contexto, RAG, embeddings, banco vetorial

**Janela de contexto (1)**
- O histórico completo fica persistido. A cada turno, o `ContextBuilder` monta o prompt dentro de um **orçamento de tokens por seção** (configurável por app).
- Quando o histórico estoura o orçamento, os turnos antigos viram um **resumo acumulado**. O resumo é gerado **de forma assíncrona depois da resposta**, para não somar latência. Os turnos recentes entram literalmente.
- Resultados de ferramentas de turnos antigos são truncados ou resumidos.
- Contagem de tokens pelo tokenizer do modelo (HF `tokenizers`, offline), calibrada com o `prompt_eval_count` do Ollama.
- **A ordem do prompt é pensada para o cache de prefixo (§6.10):** `[sistema + definição de ferramentas]` (estático por app) → `[resumo]` → `[histórico]` → `[contexto RAG + mensagem atual]`. O RAG muda a cada turno, então fica no fim para não quebrar o prefixo.

**RAG (2): ingestão**
- Upload via API → job no `arq` → **Docling** extrai texto com estrutura → **chunking que respeita títulos**, com orçamento de tokens casado com o tokenizer do bge-m3 (ponto de partida: ~400–600 tokens, sobreposição pequena).
- Cada chunk recebe um **cabeçalho contextual** (título do documento + caminho de seções) antes de gerar o embedding.
- **Prompt Guard 2 roda sobre cada chunk na ingestão.** Documento com injeção detectada vai para quarentena (não é indexado) e gera alerta.
- Deduplicação por hash. A versão do corpus da app é incrementada (isso invalida o cache semântico).

**RAG (2): recuperação**
- **Busca híbrida:** pgvector (HNSW, cosseno, filtro `app_id` com *iterative scan*) + FTS `portuguese`, fundidos por **RRF** → **rerank** com bge-reranker-v2-m3 → top-N vai para o contexto.
- Os chunks entram no prompt **serializados em TOON**, com id de fonte, dentro do bloco de dados não confiáveis (§6.9).
- RAG também é exposto **como ferramenta** (`buscar_documentos`), para o agente poder fazer buscas adicionais quando a primeira não bastar (RAG agêntico).
- Com `require_citations`, a resposta precisa citar ids de chunks que foram de fato recuperados (verificado no guard de saída).

**Embeddings (3)**
- bge-m3 via Ollama, 1024 dimensões. **Cache de embeddings no Redis** por hash do texto.
- A tabela guarda o modelo e a versão do embedding. Trocar de modelo exige reindexação controlada: indexa a nova coluna/tabela, compara nos evals de recuperação e só então vira a chave.

**Banco vetorial (4)**
- pgvector no mesmo Postgres. Índice HNSW. Opção de `halfvec` se a memória apertar.
- Um banco só simplifica backup, transações e a RLS por app.

### 6.6 Peças 5–6: tool calling, camada MCP e agente

**Tool calling (5)**
- Dois tipos de provedor de ferramentas: **nativas** (Python, dentro da API, como `buscar_documentos`) e **MCP** (servidores externos via Streamable HTTP ou stdio, usando o cliente MCP do Pydantic AI).
- Por app: **allowlist** de ferramentas; validação de argumentos pelo JSON Schema da ferramenta; timeout e limite de tamanho do resultado.
- **Classificação de efeito colateral:** ferramentas de leitura rodam direto; ferramentas de escrita (`require_approval`) **pedem confirmação**. Na v1 a confirmação acontece **pela própria conversa**: o bot descreve a ação e pergunta, e a ação só roda quando o usuário confirma no turno seguinte. Isso funciona com qualquer cliente, inclusive o Open WebUI. Um endpoint dedicado de aprovação (`POST /v1/approvals/{id}`) fica para quando alguma aplicação precisar.
- **A pendência fica no servidor:** ao pedir confirmação, o bot grava em `tool_approvals` a ferramenta, os argumentos exatos, a conversa, o `user_ref` e a expiração. O "sim" só executa se houver pendência válida **para aquela conversa e aquele usuário**, e executa **exatamente** os argumentos gravados. Um "sim" forjado no histórico enviado pelo cliente não executa nada (§7, identidade da conversa).
- Todo resultado de ferramenta é **dado não confiável:** passa pelo Prompt Guard, é serializado em TOON e entra delimitado (§6.9).

**Camada MCP para sistemas terceiros**
- Cada sistema terceiro ganha um **servidor MCP adaptador** em `mcp-adapters/<sistema>/`, escrito com FastMCP sobre a API REST do sistema.
- `FastMCP.from_openapi` serve para **protótipo**. Para produção, as ferramentas são **curadas à mão**: poucas, bem nomeadas e bem descritas. A própria documentação do FastMCP recomenda isso, porque modelos pequenos se perdem com dezenas de endpoints gerados.
- **Credenciais por aplicação:** o adaptador recebe as credenciais do sistema terceiro por variável de ambiente ou segredo do container, nunca pelo prompt. Quando o mesmo sistema atende várias apps com credenciais diferentes, sobe uma instância do adaptador por app (decisão do builder se houver forma melhor).
- Os adaptadores rodam numa rede Docker separada, a única com saída para os sistemas terceiros. Os containers de modelo **não têm saída de rede**.

**Agente (6)**
- Loop do Pydantic AI: modelo → chamadas de ferramenta → resultados → … até produzir o `output_type` final.
- **Limites obrigatórios:** máximo de passos, de chamadas de ferramenta e de tokens por execução (`UsageLimits`), timeout de parede, e **detecção de loop** (mesma ferramenta com os mesmos argumentos repetida N vezes → interrompe).
- Ao estourar um limite, a resposta é honesta ("não consegui concluir; fiz X e Y"), nunca um erro mudo.

### 6.7 Peça 7: structured output (JSON Schema + TOON)

- A saída de cada app é um **modelo Pydantic**. O Pydantic AI converte para JSON Schema e o Ollama **força o formato por gramática** (decodificação restrita). Se a validação falhar mesmo assim, há uma nova tentativa com o erro de validação devolvido ao modelo, e depois fallback (§6.11).
- **Sobre TOON:** os motores locais só garantem formato por **JSON Schema**. Por isso a garantia vem do JSON Schema, e o **TOON entra em dois lugares:**
  1. **Na entrada:** dados enviados ao modelo (chunks de RAG, resultados de ferramentas, tabelas) são serializados em TOON para economizar tokens.
  2. **Na saída, se o cliente pedir:** o objeto já validado é serializado em TOON (`response_format: toon` ou parâmetro equivalente). O cliente recebe TOON com a garantia de schema do JSON.
- Apps de conversa livre podem usar um schema mínimo (`{resposta: str, citacoes: list[str]}`) para manter a validação e a checagem de citação.

### 6.8 Peça 8: streaming

- SSE no formato de *chunks* da OpenAI (`data: {...}` / `data: [DONE]`), para compatibilidade com SDKs e o Open WebUI.
- Eventos extras (início e fim de ferramenta, fallback acionado, retratação) seguem como campos de extensão ignorados por clientes que não os entendem.
- Saída estruturada em stream: o Pydantic AI valida parcialmente enquanto o JSON chega. O texto da resposta é emitido à medida que fica válido.
- **Convivência com guardrails:** veja os modos em §6.9.
- O header `X-Trace-Id` vai em toda resposta, para suporte.

### 6.9 Peça 9: guardrails

Princípio: **todo texto que não veio do prompt de sistema ou do usuário atual é dado, nunca instrução.** Isso vale para documentos, resultados de ferramentas, resumo e histórico antigo.

**Entrada (mensagem do usuário)**
1. **Normalização:** limites de tamanho; remoção de caracteres invisíveis e Unicode "tag" (usados para esconder instruções).
2. **Prompt Guard 2 86M** (injeção/jailbreak, multilíngue, ~90 ms na CPU, janela de 512 tokens; textos maiores são fatiados).
3. **Llama Guard 3 1B** (categorias de conteúdo configuráveis por app).
4. **Presidio:** detecta e mascara dados pessoais **antes** do cache e dos logs. Mascarar também para o modelo é opção por app, porque alguns fluxos precisam do CPF para consultar um sistema.

**Conteúdo não confiável (documentos, ferramentas, histórico)**
5. Prompt Guard na ingestão (§6.5) e em cada resultado de ferramenta em tempo de execução.
6. **Delimitação ("spotlighting"):** o conteúdo entra num bloco com marcador aleatório por requisição, em TOON. O prompt de sistema declara a hierarquia: o conteúdo do bloco é informação, nunca ordem.
7. **Limite de capacidade:** allowlist de ferramentas e confirmação para escrita. É a defesa que funciona mesmo quando a detecção falha.

**Saída**
8. Validação de schema (§6.7).
9. Llama Guard sobre a resposta.
10. **Vazamento:** checagem de dados pessoais que não estavam autorizados e **token canário** no prompt de sistema (se aparecer na saída, houve vazamento do prompt → bloqueia).
11. **Citações:** em apps com RAG, toda citação precisa apontar para um chunk recuperado. Resposta sem base vira "não encontrei nos documentos".

**Streaming × guardrails (configurável por app)**
- `buffered_sentences` (padrão): o texto é liberado por frase depois das checagens rápidas (regex de dados pessoais, canário). O Llama Guard roda em paralelo sobre a janela acumulada. Se reprovar no meio do stream, sai um evento de **retratação** e uma mensagem substituta.
- `strict`: nada é enviado antes da checagem completa (para apps sensíveis). O stream vira "uma frase só".

**Falha de guardrail:** se o serviço de guarda cair, o padrão é **bloquear** (fail-closed), configurável por app.

### 6.10 Peça 10: prompt caching (quatro camadas)

| Camada | Onde | O que guarda | Chave / invalidação |
|--------|------|--------------|---------------------|
| **Cache de prefixo (KV)** | Motor de inferência (Ollama automático; vLLM APC em produção) | Processamento do prefixo comum do prompt | Funciona pela **ordem estável do prompt** (§6.5) e por manter o modelo carregado (`keep_alive`) |
| **Cache exato de respostas** | Redis | Resposta final | hash(`app_id`, versão do prompt, modelo, mensagens normalizadas, versão do corpus); TTL por app |
| **Cache semântico** | Redis 8 + RedisVL | Resposta para perguntas *parecidas* | Por app, com limiar de distância calibrado nos evals; invalidado quando a versão do corpus muda |
| **Cache de embeddings** | Redis | Vetores de consultas e de chunks | hash(texto, modelo de embedding) |

**Regras do cache semântico (anti-objetivo de vazamento):**
- Só para perguntas **de turno único**, sem dependência do histórico.
- **Nunca** quando a execução chamou ferramenta (os dados podem ser do usuário) ou quando a resposta contém dado pessoal.
- Escopo sempre por `app_id`; nunca entre apps.
- Desligado por padrão. A app liga explicitamente (caso típico: FAQ sobre documentos).

### 6.11 Peça 11: fallback (em camadas)

1. **Retry** com backoff para erros transitórios (timeout, 5xx).
2. **Reparo de saída:** schema inválido → nova tentativa com o erro de validação.
3. **Cadeia de modelos** (`FallbackModel`): principal na GPU → modelo de outra família na CPU. Em produção, a cadeia pode incluir outra instância/servidor.
4. **Circuit breaker** por modelo (estado no Redis): depois de N falhas, pula direto para o próximo durante um tempo.
5. **Degradação de peças:** reranker fora → segue sem rerank; RAG fora → responde avisando que não pôde consultar os documentos (ou recusa, conforme a app); guardrail fora → bloqueia (fail-closed).
6. **Resposta final estática** e honesta quando tudo falha.

Todo fallback acionado é marcado no trace (`fallback.used`, `fallback.reason`) e conta como sinal para os evals.

### 6.12 Peça 12: evals

**Regra:** toda mudança em `apps/**` (prompt, schema, `app.yaml`), em modelo ou em parâmetro de RAG **roda a suíte de evals da app**, que é comparada a um **baseline versionado**. Se piorar além do limite, a mudança não entra.

Tipos de eval:
1. **Determinísticos (base):** schema válido; citou a fonte esperada; chamou a ferramenta certa com os argumentos certos; recusou quando devia; não vazou dado pessoal nem o canário; latência dentro do orçamento.
2. **Recuperação (sem LLM):** recall@k e MRR do RAG sobre um dataset dourado de perguntas → chunks esperados. Mede embeddings, chunking e rerank isoladamente.
3. **LLM-juiz (com cautela):** fidelidade e relevância. Um juiz de 4B é fraco, então o juiz roda **offline com o maior modelo que couber** (mesmo lento na CPU) e é **calibrado contra rótulos humanos** antes de valer como portão.
4. **Red team:** plugins do promptfoo para injeção direta e **indireta** (documento e resultado de ferramenta envenenados), jailbreak e extração de dados pessoais e do prompt de sistema.
5. **Disputa de modelos (Fase 0):** a mesma suíte contra os candidatos de §5.1.

Execução: `make eval APP=<id>` local e **hook de git `pre-push`** quando `apps/**` muda. O CI entra quando o repositório tiver um (provedor de CI não definido).

**De onde vêm os datasets:**
- **Requisito de entrada de toda aplicação:** quem pluga a app entrega **30–50 casos dourados** (pergunta → fatos esperados, fontes esperadas, ferramenta esperada ou recusa esperada) e uma amostra rotulada à mão para calibrar o juiz. Sem isso a app não passa da configuração para o uso.
- **Disputa da Fase 0:** como nenhuma aplicação está definida, ela usa um **conjunto genérico em português** montado pelo builder: um corpus público pequeno (legislação, manuais, FAQs públicas) com perguntas de RAG, tarefas de tool calling com ferramentas falsas e casos de saída estruturada e de injeção. A disputa **é refeita** com o dataset da primeira aplicação real.

**Ciclo de produção → eval:** traces marcados (feedback negativo, guardrail acionado, fallback usado, loop interrompido) são exportados do Phoenix como novos casos de dataset.

### 6.13 Peça 13: observabilidade

- **OpenTelemetry** em tudo, com as convenções **OpenInference** (entendidas pelo Phoenix). Instrumentação nativa do Pydantic AI + spans próprios do pipeline.
- **Árvore de spans por requisição:** `request` → `guard.input.*` → `cache.lookup` → `context.build` (`history`, `summary`, `rag.retrieve` → `embed`, `search`, `rerank`) → `agent.run` → `llm.call` (modelo, tokens de entrada/saída, tempo até o primeiro token, prefixo em cache) → `tool.call` (servidor MCP, ferramenta, status, latência) → `guard.output.*` → `stream` → `persist`.
- **Atributos comuns:** `app_id`, `conversation_id`, `prompt.version` (hash do arquivo), `model`, `fallback.used`, `cache.layer_hit`, `guard.verdicts`.
- **Logs** structlog em JSON com `trace_id`. **Dados pessoais são mascarados antes de ir para logs e traces.** Em produção, o registro do texto completo de prompts e respostas é configurável por app (LGPD).
- **Phoenix** self-hosted: traces, latência, custo em tokens e datasets de eval. Métricas Prometheus/Grafana ficam fora da v1.
- **Telemetria externa desligada** em todas as bibliotecas; Hugging Face em modo offline depois do download inicial dos modelos.

### 6.14 Dados pessoais e retenção (LGPD)

Valores padrão escolhidos pelo Claude. Quem responde por proteção de dados deve validá-los antes do uso real.

- **Retenção configurável por app** (`data_retention` no `app.yaml`). Padrões: conversas e mensagens por **90 dias**; traces no Phoenix por **30 dias**, já mascarados; cache no Redis pelo TTL da camada.
- **Job diário de expurgo** (arq) apaga o que passou do prazo no Postgres, no Phoenix e no Redis.
- **Apagamento por titular:** `DELETE /v1/apps/{app_id}/users/{user_ref}/data` remove conversas, mensagens, resumos e aprovações; invalida as chaves de cache derivadas do usuário; e apaga os traces com aquele `user_ref`.
- Dados pessoais são mascarados antes de logs e traces (§6.13). No banco, o texto fica como chegou, quando a app não mascara para o modelo (§6.9). Criptografia em repouso fica a cargo do disco (LUKS/BitLocker) na v1.

---

## 7. Interface com as aplicações

| Endpoint | Função |
|----------|--------|
| `POST /v1/chat/completions` | Compatível com OpenAI. `model` = `app_id` (ou header `X-App-Id`); `stream: true` para SSE; `response_format` aceita o schema da app ou `toon` |
| `GET /v1/models` | Lista as apps que a chave enxerga (o Open WebUI usa isso) |
| `POST /v1/apps/{app_id}/documents` | Upload → job de ingestão; `GET` consulta status e quarentena |
| `DELETE /v1/apps/{app_id}/documents/{id}` | Remove documento e chunks; incrementa a versão do corpus |
| `GET /v1/conversations/{id}` | Histórico (para a app exibir) |
| `GET /health`, `/ready` | Saúde de cada dependência |

| `DELETE /v1/apps/{app_id}/users/{user_ref}/data` | Apagamento por titular (§6.14) |

**Autenticação:** API key por app (armazenada como hash), enviada como `Authorization: Bearer`. A identidade do usuário final (`user_ref` e grupos) é declarada pelo backend da aplicação (§6.4a). O bot não gerencia usuários finais.

**Identidade da conversa (o servidor é a fonte da verdade):**
- A aplicação envia `metadata.conversation_id` (ou o header `X-Conversation-Id`). Na primeira mensagem sem id, o bot cria a conversa e devolve o id no header `X-Conversation-Id`.
- **Com id:** o bot usa **o histórico guardado no servidor** e lê do array `messages` só a **última mensagem do usuário**. Mensagens de assistente ou de ferramenta enviadas pelo cliente são ignoradas, o que impede histórico forjado.
- **Sem id** (clientes OpenAI genéricos): o bot aceita o histórico do cliente como contexto, mas trata tudo como não confiável e **desliga as ferramentas de escrita** naquela requisição.
- **Open WebUI:** o builder verifica se ele envia um id de chat utilizável (ex.: via `metadata`). Se não enviar, ele funciona no modo sem id, o que basta para testes internos.

**Plugar uma aplicação nova** = criar `apps/<id>/` (yaml, prompt, schema, **dataset dourado**) + gerar a API key + integrar pelo backend da app (enviando `user_ref`, grupos e `conversation_id`) + (se houver) subir o adaptador MCP do sistema dela + enviar documentos com ACL + rodar os evals.

---

## 8. Estrutura do repositório (proposta)

```
chat-bot/
├── docker-compose.yml        # postgres+pgvector, redis, ollama, aux-ml, phoenix, api, worker, open-webui
├── pyproject.toml / uv.lock
├── Makefile                  # up, down, ingest, eval, bakeoff
├── src/chatbot/
│   ├── api/                  # rotas FastAPI, SSE, autenticação
│   ├── core/                 # Protocols, registry, pipeline, RequestContext, AppConfig
│   ├── context/              # orçamento de tokens, histórico, resumo
│   ├── rag/                  # ingestão (docling, chunking), retrieval híbrido, rerank
│   ├── embeddings/
│   ├── vectorstore/          # pgvector
│   ├── llm/                  # providers (ollama, vllm), fallback, circuit breaker
│   ├── agent/                # agente Pydantic AI, limites, detecção de loop
│   ├── tools/                # registry, cliente MCP, confirmação
│   ├── output/               # schemas base, TOON
│   ├── guardrails/
│   ├── cache/
│   ├── observability/        # setup OTel, decorators de span, logs
│   └── workers/              # jobs arq
├── services/aux-ml/          # reranker + Prompt Guard (ONNX Runtime, CPU)
├── mcp-adapters/             # um servidor FastMCP por sistema terceiro
├── apps/<app_id>/            # config, prompts, schemas, evals por aplicação
├── evals/                    # configs compartilhadas, baselines, red team
├── migrations/               # Alembic
└── docs/
```

---

## 9. Roadmap por fases

Observabilidade e evals são **transversais**: começam na Fase 0 e crescem em toda fase. **Toda peça nova chega com seus spans e seus casos de eval.** (Na lista original as duas eram as últimas peças. Trazê-las para o início é decisão do Claude: são baratas no começo e caras de encaixar depois.)

Construção sequencial, por uma pessoa e sem prazo fixo. Cada fase termina em algo utilizável e só começa depois que a anterior passa nos seus evals.

| Fase | Peças | Entrega utilizável ao fim |
|------|-------|---------------------------|
| **0. Fundação** | 13 (base), 12 (base) | Docker Compose de pé (com Caddy e backup); esqueleto FastAPI; carregador de `apps/`; OTel → Phoenix; harness de eval com o **dataset genérico em português**; **disputa de modelos** com números; medição real de VRAM, RAM e latência |
| **1. Conversa** | 1, 8 | Chat com histórico no servidor (identidade da conversa, §7), resumo, streaming e fila de concorrência via `/v1/chat/completions`; usável no Open WebUI |
| **2. Conhecimento** | 2, 3, 4 + **permissões por usuário/grupo** (§6.4a) + guardrails mínimos (Prompt Guard na entrada e na ingestão, dados pessoais nos logs) | Perguntas e respostas sobre documentos com citação, respeitando a ACL. **Piloto interno com usuários de confiança** |
| **3. Ação** | 5, 6, 7 + camada MCP | Agente com ferramentas nativas e MCP (com identidade repassada), confirmação de escrita com pendência no servidor, saída estruturada JSON/TOON; primeiro adaptador MCP real |
| **4. Proteção e robustez** | 9 (completo), 10, 11 + retenção/LGPD (§6.14) | Guardrails completos (incluindo modos de streaming), 4 camadas de cache, cadeia de fallback e circuit breaker, expurgo e apagamento por titular. **Liberado para uso real no notebook, até ~5 usuários** |
| **5. Qualidade contínua** | 12 (completo), 13 (completo) | Portão de evals no `pre-push`, red team (incluindo vazamento por ACL), juiz calibrado, ciclo trace → dataset |
| 6. Pós-v1 (opcional) | — | Servidor com GPU maior e vLLM; conectores de ingestão (SharePoint etc.); bot como servidor MCP; endpoint de aprovação; métricas Prometheus |

---

## 10. Decisões tomadas pelo Claude (para revisão do usuário)

1. LiteLLM fora do núcleo (§4).
2. Modelos de partida e divisão GPU/CPU (§5.1–5.2), sujeitos à disputa da Fase 0.
3. Phoenix em vez de Langfuse para observabilidade, pelo peso em RAM (§5.3).
4. TOON na entrada do modelo e como formato de saída opcional; garantia de formato via JSON Schema (§6.7).
5. Configuração das apps em git, não em banco (§6.3).
6. Confirmação de ações de escrita pela conversa na v1 (§6.6).
7. Cache semântico desligado por padrão e com regras de escopo (§6.10).
8. Guardrails em modo fail-closed por padrão (§6.9).
9. Observabilidade e evals trazidas para a Fase 0 (§9).
10. Servidor como fonte da verdade do histórico; "sim" só executa com pendência gravada no servidor; cliente sem id de conversa não tem ferramentas de escrita (§7, §6.6).
11. A aplicação declara a identidade do usuário pelo backend; o bot confia nela; os adaptadores MCP aplicam as permissões do sistema terceiro (§6.4a).
12. Dataset dourado de 30–50 casos como requisito de entrada de cada app; disputa da Fase 0 com dataset genérico em português (§6.12).
13. Ingestão só por upload na v1 (§2).
14. Padrões de retenção: 90 dias para conversas, 30 dias para traces, apagamento por titular (§6.14).
15. Metas iniciais de latência, fila de concorrência, Caddy com TLS na rede local e backup diário (§3).

## 11. Deixado para o builder

- Tamanho exato de chunk e sobreposição (calibrar nos evals de recuperação).
- Limiares do Prompt Guard, do cache semântico e do RRF (calibrar nos evals).
- Escolha entre Ollama e ONNX/TEI para os modelos auxiliares na CPU, desde que respeite o orçamento de §5.2.
- Detalhes das migrações, nomes de tabelas e índices.
- Fixar as versões exatas na implementação. Versões encontradas na pesquisa de 01/11/2026: Pydantic AI v2.x; FastMCP 3.x (há relato de 4.x, não confirmado); pgvector ≥ 0.8.2 (relato de 0.8.7 em out/2026); Redis 8; Docling 2.x; Ollama ≥ 0.19.

## 12. Questões em aberto

Nenhuma bloqueia o início. Elas pesam a partir das fases indicadas.

1. **Qual é a primeira aplicação real?** Precisa estar definida antes do fim da Fase 2 (piloto), com seu dataset dourado.
2. **Validação dos padrões de LGPD** (§6.14) por quem responde por proteção de dados, antes da Fase 4.
3. **Servidor pós-v1:** hardware e quantidade de usuários, quando o notebook deixar de bastar.
