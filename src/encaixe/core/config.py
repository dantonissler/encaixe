from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jinja2
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from encaixe.output.schemas import RespostaBase


class AppConfigError(Exception):
    pass


class _Strict(BaseModel):
    """Chave desconhecida no app.yaml (erro de digitação) é erro na carga, não silêncio."""

    model_config = ConfigDict(extra="forbid")


class ModelsConfig(_Strict):
    primary: str
    fallbacks: list[str] = Field(default_factory=list)


class BudgetConfig(_Strict):
    # A soma das seções cabe em `context.num_ctx` (validado na carga).
    system: int = 800
    summary: int = 500
    history: int = 2500
    rag: int = 2500
    user: int = 800
    output: int = 1000


class ContextConfig(_Strict):
    num_ctx: int = 8192
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


class RagConfig(_Strict):
    enabled: bool = True
    top_k: int = 20
    rerank_top_n: int = 5
    require_citations: bool = True


class McpServerConfig(_Strict):
    name: str
    url: str
    allow: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)


class ToolsConfig(_Strict):
    native: list[str] = Field(default_factory=list)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


class AgentConfig(_Strict):
    max_steps: int = 8
    max_tool_calls: int = 12
    timeout_s: int = 60


class OutputConfig(_Strict):
    schema_path: str = Field(default="schemas.Resposta", alias="schema")
    formats: list[str] = Field(default_factory=lambda: ["json", "toon"])

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExactCacheConfig(_Strict):
    enabled: bool = True
    ttl_s: int = 3600


class CacheConfig(_Strict):
    exact: ExactCacheConfig = Field(default_factory=ExactCacheConfig)


class GuardrailsConfig(_Strict):
    input: list[str] = Field(
        default_factory=lambda: ["normalize", "prompt_injection", "content_safety", "pii"]
    )
    output: list[str] = Field(
        default_factory=lambda: ["content_safety", "pii_leak", "canary", "citations"]
    )
    on_guard_unavailable: Literal["block", "allow"] = "block"


class StreamingConfig(_Strict):
    mode: Literal["buffered_sentences", "strict"] = "buffered_sentences"


class AppConfig(_Strict):
    id: str
    models: ModelsConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


@dataclass
class LoadedApp:
    config: AppConfig
    root: Path
    system_prompt: jinja2.Template
    prompt_version: str  # hash do system.md (atributo `prompt.version` dos spans, spec §6.13)
    output_model: type[RespostaBase]
    # hash de system.md + app.yaml + arquivo do schema: entra na chave do cache exato, para
    # que mudar qualquer um deles (guards, RAG, schema...) não sirva respostas antigas.
    app_version: str = ""


def _schema_file(root: Path, ref: str) -> tuple[Path, str, str]:
    module_name, _, class_name = ref.rpartition(".")
    if not module_name or not class_name:
        raise AppConfigError(f"output.schema inválido: '{ref}' (esperado 'modulo.Classe')")
    return root / f"{module_name}.py", module_name, class_name


def _version(*parts: tuple[str, bytes]) -> str:
    """Hash sem ambiguidade de vários arquivos (nome e tamanho entram no hash)."""
    h = hashlib.sha256()
    for name, data in parts:
        h.update(name.encode("utf-8") + b"\0" + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()[:12]


def _load_schema(root: Path, ref: str) -> type[RespostaBase]:
    path, module_name, class_name = _schema_file(root, ref)
    if not path.is_file():
        raise AppConfigError(f"Arquivo de schema ausente: {path}")
    spec = importlib.util.spec_from_file_location(f"encaixe_app_{root.name}_{module_name}", path)
    if spec is None or spec.loader is None:
        raise AppConfigError(f"Não foi possível carregar {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AppConfigError(f"Erro ao importar {path}: {exc}") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AppConfigError(f"Schema '{class_name}' não existe em {path}")
    if not (isinstance(cls, type) and issubclass(cls, RespostaBase)):
        raise AppConfigError(f"Schema '{ref}' deve herdar RespostaBase")
    return cls


def validate_app_config(config: AppConfig) -> list[str]:
    """Problemas que o schema sozinho não pega (vazio = config válida): nome de guard sem
    implementação registrada e orçamento de contexto maior que `context.num_ctx`."""
    from encaixe.core.registry import available
    from encaixe.guardrails import runner  # noqa: F401 - registra os guards

    known = set(available("guard"))
    problems = []
    for stage in ("input", "output"):
        for name in getattr(config.guardrails, stage):
            if name not in known:
                problems.append(
                    f"guard desconhecido em guardrails.{stage}: '{name}' "
                    f"(disponíveis: {', '.join(sorted(known))})"
                )
    ctx = config.context
    total = sum(ctx.budget.model_dump().values())
    if total > ctx.num_ctx:
        problems.append(
            f"context.budget soma {total} tokens, mais que context.num_ctx ({ctx.num_ctx})"
        )
    return problems


def _load_app(root: Path) -> LoadedApp:
    yaml_path = root / "app.yaml"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        config = AppConfig.model_validate(raw)
    except (yaml.YAMLError, ValidationError, OSError, TypeError) as exc:
        raise AppConfigError(f"app.yaml inválido em {root}: {exc}") from exc
    if problems := validate_app_config(config):
        raise AppConfigError(f"app.yaml inválido em {root}: " + "; ".join(problems))
    prompt_path = root / "prompts" / "system.md"
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppConfigError(f"Prompt de sistema ausente: {prompt_path}") from exc
    try:
        template = jinja2.Template(prompt_text)
    except jinja2.TemplateError as exc:
        raise AppConfigError(f"Template inválido em {prompt_path}: {exc}") from exc
    output_model = _load_schema(root, config.output.schema_path)
    schema_path = _schema_file(root, config.output.schema_path)[0]
    return LoadedApp(
        config=config,
        root=root,
        system_prompt=template,
        prompt_version=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12],
        output_model=output_model,
        app_version=_version(
            ("prompts/system.md", prompt_text.encode("utf-8")),
            ("app.yaml", yaml_path.read_bytes()),
            (schema_path.name, schema_path.read_bytes()),
        ),
    )


def load_apps(apps_dir: Path) -> dict[str, LoadedApp]:
    apps: dict[str, LoadedApp] = {}
    for yaml_path in sorted(Path(apps_dir).glob("*/app.yaml")):
        app = _load_app(yaml_path.parent)
        if (other := apps.get(app.config.id)) is not None:
            raise AppConfigError(
                f"id de app repetido '{app.config.id}' em {other.root} e {app.root}"
            )
        apps[app.config.id] = app
    return apps
