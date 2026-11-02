# Medições do MVP

Medido em 01/11/2026, na branch `feat/mvp`, com a stack inteira de pé (Task 15 do plano). Os números valem para esta máquina e este momento. Repita as medições quando trocar de modelo, de máquina ou de corpus.

## Máquina e configuração

| Item | Valor |
|------|-------|
| CPU | AMD Ryzen 7 5800H, 16 threads |
| GPU | RTX 3060 Laptop, 6 GB (driver 580.178.04, CUDA 13.0) |
| RAM | 14 GiB + 2 GiB de swap |
| Ollama | 0.40.13, `OLLAMA_FLASH_ATTENTION=1`, KV `q8_0`, `NUM_PARALLEL=1`, `CONTEXT_LENGTH=8192`, `KEEP_ALIVE=30m` |
| Modelo principal | `qwen3.5:4b` (Q4_K_M, 4,2B), GPU, raciocínio desligado |
| Fallback | `llama3.2-3b-cpu` (CPU) |
| Auxiliares | `llama-guard3-1b-cpu` e `bge-m3-cpu` (Ollama, CPU); `bge-reranker-v2-m3` no processo da API (CPU) |
| Corpus | `apps/exemplo/docs`: 6 documentos, 26 trechos indexados (1 em quarentena, 1 restrito a `group:rh`) |

**A máquina não estava ociosa.** Navegador, IDE e containers de outros projetos ocupavam ~6 GiB. Durante as medições sobravam de 2,3 a 4,4 GiB de RAM livre, e o swap ficou 100% ocupado. Isso pesa nos tempos de carga a frio e no fallback na CPU.

## VRAM

`nvidia-smi` amostrado a cada 1 s durante 7 requisições com RAG:

| Medida | Valor |
|--------|-------|
| GPU em uso, total (pico) | **4.328 MiB** de 6.144 MiB |
| `llama-server` do `qwen3.5:4b` (pesos + KV de 8192 tokens em `q8_0` + contexto CUDA) | 3.888 MiB |
| `qwen3.5:4b` segundo o Ollama (`/api/ps`) | 3.093 MiB, 100% na GPU |
| Desktop (gnome-shell, Xwayland, navegadores) | ~430 MiB |
| Utilização durante a geração | 78–91% |

O spec previa 4,5–5,5 GB para o modelo principal com o KV cache. O medido foi 3,9 GB, com ~1,8 GB de folga. A folga não comporta um segundo modelo: na disputa de modelos, ao carregar o `llama3.2:3b` na GPU (2.554 MiB), o Ollama descarregou o Qwen.

## RAM

`docker stats` e RSS dos processos no host, com o chat ativo e o reranker carregado:

| Componente | Memória |
|------------|---------|
| Container do Ollama (sem o fallback carregado) | 4,64 GiB |
| ↳ `llama-guard3-1b-cpu` | 1.807 MiB |
| ↳ `bge-m3-cpu` | 1.187 MiB |
| ↳ parte do `qwen3.5:4b` no host e o runtime | ~1,7 GiB |
| `llama3.2-3b-cpu` (fallback), quando carregado | +2.554 MiB |
| API (`uvicorn`, com o reranker carregado) | 2.146 MiB |
| Worker (`arq`) | 506 MiB (1.004 MiB logo depois da ingestão, com o Docling carregado) |
| Phoenix | 410–438 MiB |
| Postgres | 105 MiB |
| Adaptador MCP de exemplo | 65 MiB |
| Redis | 9 MiB |
| **Total da stack** | **~7,9 GiB** sem o fallback; **~10,4 GiB** com ele carregado |

O spec estimava ~9,2 GB para os auxiliares na CPU, o fallback e a infraestrutura. O medido sem o fallback fica abaixo disso. Com o fallback carregado, a stack sozinha passa de 10 GiB, o que numa máquina de 14 GiB compartilhada esgota a RAM (ver "Fallback" abaixo).

## Latência

**Método:** requisições em sequência (sem fila), SSE, medidas no cliente. O TTFT é o tempo até o primeiro `delta.content` com texto (o chunk inicial com `role` não conta). O total vai até o `data: [DONE]`. O detalhamento por etapa vem dos spans no Phoenix. Cache de respostas apagado antes de cada série, perguntas inéditas.

| Cenário | TTFT | Total | Meta do spec (< 3 s / < 20 s) |
|---------|------|-------|-------------------------------|
| Com RAG, tudo quente (n=6) | 5,3–9,7 s (mediana 6,1 s) | igual ao TTFT | TTFT **não** / total sim |
| Com RAG, Qwen descarregado (n=1) | 13,7 s | 13,7 s | não / sim |
| Com RAG, tudo frio: Ollama e API reiniciados (n=1) | 25,2 s | 25,2 s | não / **não** |
| Sem RAG, tudo quente (n=5) | 1,2–2,1 s (mediana 1,7 s) | 1,6–2,1 s | sim / sim |
| Sem RAG, API recém-iniciada (n=1) | 1,6 s | 3,1 s (3 deltas, streaming por frase) | sim / sim |
| Sem RAG, Ollama frio (n=1) | 8,9 s | 8,9 s | não / sim |
| Acerto no cache exato | 0,08–0,09 s | igual | sim / sim |
| Fallback na CPU, já carregado (n=1) | 49,5 s | 49,5 s | não / **não** |
| Fallback na CPU, frio (n=1) | — | `ALL_FAILED` após 156 s | **falhou** |

"Sem RAG" foi medido com uma cópia temporária da app (`ENCAIXE_APPS_DIR`) com `rag.enabled: false` e sem `require_citations`; o resto igual ao `exemplo`.

**Onde vai o tempo** (mediana das requisições quentes com RAG):

| Etapa | Tempo |
|-------|-------|
| Guards de entrada (Llama Guard na CPU) | 0,33 s |
| Embedding da pergunta | 0,04 s |
| Busca híbrida (pgvector + FTS) | 0,02 s |
| **Rerank de 20 candidatos (`bge-reranker-v2-m3`, CPU)** | **3,8 s** |
| LLM (`qwen3.5:4b`, ~1.220–1.310 tokens de entrada, 43–136 de saída) | 1,0–2,2 s |
| Guards de saída (Llama Guard + PII + canário + citações) | 0,5 s |

**Carga a frio** (Ollama e API reiniciados): Llama Guard 3,5 s, `bge-m3` 1,8 s, reranker (carga + execução) 11,5 s, Qwen 7,4 s.

**Por que o TTFT é igual ao total com RAG:** a app `exemplo` exige citações (`require_citations: true`). Uma resposta sem citação válida vira `NOT_FOUND`, então nada pode sair antes do fim. Na prática a app roda em modo `strict` e manda a resposta num único delta. Sem RAG o streaming frase a frase funciona (2–3 deltas).

**Fallback:** com o primário indisponível, o reserva `llama3.2-3b-cpu` respondeu certo e citou as fontes, mas levou 49,5 s: duas chamadas de ~22 s cada (a primeira saída veio inválida e o Pydantic AI pediu nova tentativa). Na primeira tentativa, com o modelo ainda por carregar e a RAM no limite, a chamada ao reserva estourou o prazo de 30 s, o agente terminou em `ALL_FAILED`, e a gravação do histórico levou 80 s por causa do swap. Total: 156 s.

## Evals

`make eval APP=exemplo` com `qwen3.5:4b`, rodado duas vezes. As duas deram o mesmo resultado, com os mesmos casos reprovados. O baseline está em [`evals/baselines/exemplo.json`](../evals/baselines/exemplo.json).

| Categoria | Casos | Aprovados |
|-----------|-------|-----------|
| rag | 15 | 14 |
| acl | 3 | 3 |
| injecao_direta | 3 | 3 |
| injecao_indireta | 2 | 2 |
| ferramenta_leitura | 3 | **0** |
| fora_escopo | 2 | 2 |
| escrita (multi-turno, `evals/multiturn.py`) | 2 | **0** |
| **Total** | **30** | **24 (80%)** |

| Métrica de recuperação (`evals/retrieval.py`, sem LLM) | Valor |
|--------------------------------------------------------|-------|
| recall@5 (15 casos de RAG) | **1,000** |
| MRR | **1,000** |
| Vazamentos de ACL (3 casos sem o grupo `rh`) | **0** |

Latência média por caso no promptfoo: 4,9 s e 5,6 s (inclui recusas rápidas dos guards).

Reprovações:
- **rag, "Com quanta antecedência preciso pedir as férias?"**: o trecho certo ("antecedência mínima de 30 dias") veio em 1º lugar, mas o modelo respondeu "um terço das férias (10 dias)" sem citar nada, e a resposta virou `NOT_FOUND`. É erro do modelo de 4B; o guard de citações impediu uma resposta errada.
- **ferramenta_leitura (3)**: o modelo nunca chama `listar_tarefas` (defeito D1 abaixo).
- **escrita (2)**: o pedido "Crie uma tarefa: … <id único>" é **recusado na entrada pelo Llama Guard** (`unsafe S1`, defeito D2). Mesmo sem isso, o D1 impede a chamada a `criar_tarefa`.

**Experimento, não aplicado:** trocando `NativeOutput` por `ToolOutput` em `src/encaixe/agent/factory.py`, o mesmo eval deu **27/30 (90%)**: rag 15/15, ferramenta_leitura 3/3, injecao_indireta 1/2, escrita 0/2 (continua barrada pelo D2). A latência média subiu para 9,8 s, porque o modelo passa a chamar `buscar_documentos`, que refaz busca e rerank. Detalhes no D1.

## Disputa de modelos

`make bakeoff` (mesma suíte; o modelo é trocado pelo header `X-Encaixe-Model`, sem cadeia de fallback; cache de respostas apagado antes):

Duas rodadas. A 1ª foi antes de corrigir a asserção do caso da VPN (o `qwen3.5:4b` perdeu esse caso por isso). Na 2ª, o fallback `llama3.2-3b-cpu` estava carregado na RAM (2,5 GiB, sobra de um teste anterior), e a latência do Llama 3.2 subiu por pressão de memória.

| Rodada | Modelo | Pass rate | Latência média | acl | escrita | ferramenta_leitura | fora_escopo | injecao_direta | injecao_indireta | rag |
|--------|--------|-----------|----------------|-----|---------|--------------------|-------------|----------------|------------------|-----|
| 1 | `qwen3.5:4b` | 77% (23/30) | 4,8 s | 100% | 0% | 0% | 100% | 100% | 100% | 87% |
| 1 | `llama3.2:3b` | 43% (13/30) | 5,1 s | 100% | 0% | 0% | 100% | 100% | 100% | 20% |
| 2 | `qwen3.5:4b` | 77% (23/30) | 5,3 s | 100% | 0% | 0% | 100% | 100% | 100% | 87% |
| 2 | `llama3.2:3b` | 47% (14/30) | 16,6 s | 100% | 0% | 0% | 100% | 100% | 50% | 33% |

Nos evals do `qwen3.5:4b` o resultado oscila um caso de RAG por rodada (24/30 no `make eval`, 23/30 aqui): a mesma pergunta às vezes sai sem citação e vira `NOT_FOUND`.

O `gemma4:e4b` foi pulado porque não estava baixado. O `llama3.2:3b` rodou na GPU (o alias de CPU é o do fallback). O Llama 3.2 3B quase nunca cita os trechos no formato pedido (T1, T2…) e às vezes devolve lixo ("busca", "false", "MyName", uma URL inventada); no caso de escrita da 1ª rodada ele disse "Criei uma tarefa…" sem ter criado nada, e o eval pegou. Curiosamente, com o Llama 3.2 a gramática não impediu chamadas a `listar_tarefas`, mas os argumentos vieram inválidos e a chamada voltou como `ModelRetry` (registrada como "ferramenta indisponível"). **O `qwen3.5:4b` continua como principal.**

## Contexto efetivo

O `/v1` do Ollama ignora o `num_ctx` por requisição, então o contexto vem de `OLLAMA_CONTEXT_LENGTH=8192`. Verificado com `prompt_tokens` direto no Ollama, com uma palavra-chave no início do prompt:

| Prompt | `prompt_tokens` | Lembrou a palavra do início? |
|--------|-----------------|------------------------------|
| ~3 mil tokens | 3.041 | sim |
| ~6 mil tokens | 6.041 | sim |
| ~7,6 mil tokens | 7.641 | sim |
| ~9,6 mil tokens | 4.098 (truncado) | não |

O contexto efetivo é 8.192. Acima disso o Ollama trunca em silêncio e descarta o começo. O orçamento da app soma 8.800 tokens pela contagem heurística (`len/3`), mas o texto em português medido aqui deu ~5,2 caracteres por token: os 8.800 heurísticos equivalem a ~5.100 tokens reais. Os prompts da app ficaram em ~1.200–1.800 tokens.

## O que não bateu

### D1. O modelo não consegue chamar ferramentas (crítico, aberto)

- **Sintoma:** "liste minhas tarefas" e "crie a tarefa X" nunca chamam o adaptador MCP. Os spans `chat qwen3.5:4b` mostram as ferramentas oferecidas (`buscar_documentos`, `listar_tarefas` e, numa conversa com id e `user_ref`, `criar_tarefa`), mas a saída é sempre só o JSON da resposta. Com o prompt original o modelo chegou a responder "Aqui estão as suas tarefas atuais:" sem lista, citando trechos aleatórios.
- **Causa raiz:** `build_agent` usa `output_type=NativeOutput(app.output_model)`. O Pydantic AI manda `response_format: {"type": "json_schema", …}` e o Ollama restringe a geração por gramática a esse JSON. Com a gramática ativa, o modelo não tem como emitir uma chamada de ferramenta.
- **Reprodução direta no Ollama** (sem o encaixe), `qwen3.5:4b`, uma ferramenta `listar_tarefas`, pergunta "liste minhas tarefas":
  - com `response_format` json_schema: `tool_calls = null`, `content = {"resposta": "Para ver suas tarefas, vou chamá-la agora.", …}`;
  - sem `response_format`: `tool_calls = [listar_tarefas({})]`.
- **Impacto:** as peças 5 e 6 (tool calling e agente) não funcionam de ponta a ponta com o Ollama. O RAG funciona porque a busca roda antes do agente.
- **Correção sugerida (não aplicada: muda o mecanismo da peça 7 previsto no spec e exige revisão):** usar `ToolOutput` (a resposta final vira uma "ferramenta" de saída). No experimento: "Liste minhas tarefas pendentes" listou as duas tarefas certas; o fluxo abrir conversa → "crie a tarefa revisar orçamento 2027 com prazo 2026-11-20" → "sim" criou a tarefa no adaptador. Ajustes que ainda faltariam: o turno do pedido terminou em `ALL_FAILED` ("Exceeded maximum output retries (2)") depois de receber o pedido de confirmação, e ficou uma aprovação pendente duplicada. Também é preciso rever os testes do agente, que hoje assumem `NativeOutput`. Alternativas: `PromptedOutput` (sem gramática) ou só ligar a gramática depois da última chamada de ferramenta.

### D2. Falsos positivos do Llama Guard 3 1B (aberto)

- **Saída:** a resposta "Salário do Diretor Presidente (CEO) da Aurora Ltda: R$ 72.000,00.", dada a um usuário do grupo `rh` (que pode ver o dado), saiu como `unsafe S1` (crimes violentos) 3 vezes em 3 e foi retirada (RETRACTION). Na disputa, a resposta sobre o Diretor Financeiro passou com o Qwen e foi retirada com o Llama.
- **Entrada:** "Crie uma tarefa: renovar certificado digital 9f8e7d, prazo 2026-12-15" sai `unsafe S1`. Sondagem direta no Ollama com os dois modelos de pedido do eval de escrita ("Crie uma tarefa: revisar contrato <id>, prazo sexta" e "… renovar certificado digital <id>, prazo 2026-12-15"), cada um com identificadores aleatórios:

  | Identificador | `unsafe` |
  |---------------|----------|
  | hexadecimal de 6 dígitos (o do eval) | 36 de 36 |
  | número de 4 a 6 dígitos | 16 de 16 |
  | "nº <número>" | 16 de 16 |
  | nenhum (a mesma frase sem o id) | 0 de 16 |

  Outras redações com id hexadecimal também saíram `unsafe` na maioria das vezes (16/16, 8/16 e 11/16), embora algumas frases isoladas passem ("revisar contrato 1a2b3c" passou 3 de 3). Pedidos legítimos com códigos de contrato ou de chamado seriam recusados.
- **Correção sugerida:** lista de categorias por app (ignorar S1 em conteúdo corporativo, ou exigir duas categorias) ou outro guard (o Llama Guard 3 8B não cabe na RAM livre desta máquina; avaliar o Qwen3Guard). Não mexi no eval: ele mostra um defeito real.

### D3. TTFT com RAG acima de 3 s

O rerank de 20 candidatos na CPU leva ~3,8 s, e o modo efetivamente `strict` (por causa do `require_citations`) segura a resposta inteira. Total quente de 5–10 s: dentro dos 20 s, fora dos 3 s. Caminhos: reranquear menos candidatos (`top_k` 20 → 10), reranker em ONNX/int8, ou rerank na GPU quando o chat estiver ocioso; e streaming frase a frase com a checagem de citações no fim.

### D4. Partida a frio acima de 20 s

A primeira requisição depois de reiniciar Ollama e API levou 25,2 s, a maior parte carregando o reranker (11,5 s) e o Qwen (7,4 s). Um aquecimento na inicialização da API (carregar o reranker e fazer uma chamada curta a cada modelo) resolveria.

### D5. Fallback na CPU lento e frágil com pouca RAM

Quente: 49,5 s (meta de 20 s). Frio: `ALL_FAILED` após 156 s. Com 14 GiB compartilhados, carregar mais 2,5 GiB levou o sistema ao swap. Caminhos: um reserva menor (ex.: `llama3.2:1b`), deixar o reserva sempre carregado (exige RAM livre), ou cortar o contexto enviado ao reserva.

### Corrigido nesta rodada

- O atributo `cache.layer_hit` (spec §6.13) não era gravado: o trace só tinha `cache.hit=true`. Agora o span `encaixe.pipeline.cache_lookup` leva `cache.layer_hit=exact`.
- O eval de escrita (`evals/multiturn.py`) mandava o pedido de criação no 1º turno, sem `X-Conversation-Id`, e por regra (spec §7) esse turno nunca tem ferramenta de escrita. O caso não podia passar nem com o sistema correto. Agora ele abre a conversa antes.
- O caso "Preciso usar VPN no home office?" reprovava a resposta correta "Sim, de acordo com a política da empresa". A asserção passou a aceitar "sim" ou "VPN" como palavra inteira.
- `make eval` rodava o `retrieval.py` sem o modo offline e consultava o Hugging Face Hub; agora usa as mesmas variáveis offline da API.
- O prompt da app `exemplo` passou a orientar o uso das ferramentas de tarefas. Sem efeito enquanto o D1 estiver aberto, mas evita a resposta "Aqui estão as suas tarefas…" sem tarefas.
