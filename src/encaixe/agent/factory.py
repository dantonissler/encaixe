from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.output import NativeOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from encaixe.agent.loop_guard import LoopDetected
from encaixe.core.config import LoadedApp
from encaixe.messages import LIMIT_REACHED
from encaixe.output.schemas import RespostaBase

# O Qwen3.5 "pensa" por padrão: no /v1 do Ollama ele gasta o orçamento de tokens em
# `reasoning` (conteúdo vazio, ~30 s até o primeiro token). `think: false` não tem efeito
# no /v1; `reasoning_effort: "none"` desliga. Modelos sem raciocínio ignoram o campo.
NO_REASONING: dict[str, str] = {"reasoning_effort": "none"}


def build_agent(
    app: LoadedApp, model: Model, toolsets: list[AbstractToolset[Any]]
) -> Agent[Any, RespostaBase]:
    """Agente com saída estruturada nativa.

    O system prompt NÃO é fixado aqui: o pipeline passa `instructions=built.system_prompt`
    em cada `agent.run(...)`/`run_stream(...)`. `num_ctx` vai por `extra_body.options`
    (o v2 não tem `OllamaModelSettings`).

    ATENÇÃO: no endpoint OpenAI-compatível `/v1` do Ollama, `options` costuma ser ignorado;
    o contexto efetivo é o do servidor (`OLLAMA_CONTEXT_LENGTH` no docker-compose, ou
    `PARAMETER num_ctx` num Modelfile). Logo, `num_ctx` por app não deve exceder esse valor.
    O `extra_body` é mantido por ser inofensivo; ele também leva `reasoning_effort: "none"`
    (ver `NO_REASONING`). `timeout` vai em `model_settings` (timeout HTTP por requisição ao
    modelo, ver `model_request_timeout`); o pipeline envolve a execução inteira em
    `asyncio.timeout(agent.timeout_s)`.
    """
    settings = ModelSettings(
        extra_body={"options": {"num_ctx": app.config.context.num_ctx}, **NO_REASONING},
        timeout=model_request_timeout(app),
    )
    return Agent(
        model,
        output_type=NativeOutput(app.output_model),
        toolsets=toolsets,
        retries={"output": 2},
        model_settings=settings,
    )


MIN_MODEL_REQUEST_TIMEOUT_S = 10.0


def model_request_timeout(app: LoadedApp) -> float:
    """Timeout HTTP de cada chamada a um modelo da cadeia.

    O prazo da execução (`agent.timeout_s`) é dividido entre o principal e os reservas: um
    principal travado não consome o prazo inteiro e ainda sobra tempo para o reserva.
    """
    agent_timeout = float(app.config.agent.timeout_s)
    share = agent_timeout / (1 + len(app.config.models.fallbacks))
    return max(MIN_MODEL_REQUEST_TIMEOUT_S, share)


def usage_limits_for(app: LoadedApp) -> UsageLimits:
    return UsageLimits(
        request_limit=app.config.agent.max_steps,
        tool_calls_limit=app.config.agent.max_tool_calls,
    )


def run_outcome_on_limit(
    exc: UsageLimitExceeded | LoopDetected, tools_called: list[str] | None = None
) -> RespostaBase:
    message = LIMIT_REACHED
    if tools_called:
        message += f" Ferramentas usadas: {', '.join(tools_called)}."
    return RespostaBase(resposta=message, citacoes=[])
