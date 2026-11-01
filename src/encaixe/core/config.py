from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jinja2
import yaml
from pydantic import BaseModel, Field, ValidationError

from encaixe.output.schemas import RespostaBase


class AppConfigError(Exception):
    pass


class ModelsConfig(BaseModel):
    primary: str
    fallbacks: list[str] = Field(default_factory=list)


class BudgetConfig(BaseModel):
    system: int = 1500
    summary: int = 500
    history: int = 2500
    rag: int = 2500
    user: int = 800
    output: int = 1000


class ContextConfig(BaseModel):
    num_ctx: int = 8192
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


class RagConfig(BaseModel):
    enabled: bool = True
    top_k: int = 20
    rerank_top_n: int = 5
    require_citations: bool = True


class McpServerConfig(BaseModel):
    name: str
    url: str
    allow: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    native: list[str] = Field(default_factory=list)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)


class AgentConfig(BaseModel):
    max_steps: int = 8
    max_tool_calls: int = 12
    timeout_s: int = 60


class OutputConfig(BaseModel):
    schema_path: str = Field(default="schemas.Resposta", alias="schema")
    formats: list[str] = Field(default_factory=lambda: ["json", "toon"])

    model_config = {"populate_by_name": True}


class ExactCacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 3600


class CacheConfig(BaseModel):
    exact: ExactCacheConfig = Field(default_factory=ExactCacheConfig)


class GuardrailsConfig(BaseModel):
    input: list[str] = Field(
        default_factory=lambda: ["normalize", "prompt_injection", "content_safety", "pii"]
    )
    output: list[str] = Field(
        default_factory=lambda: ["content_safety", "pii_leak", "canary", "citations"]
    )
    on_guard_unavailable: Literal["block", "allow"] = "block"


class StreamingConfig(BaseModel):
    mode: Literal["buffered_sentences", "strict"] = "buffered_sentences"


class AppConfig(BaseModel):
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
    prompt_version: str
    output_model: type[RespostaBase]


def _load_schema(root: Path, ref: str) -> type[RespostaBase]:
    module_name, _, class_name = ref.rpartition(".")
    if not module_name or not class_name:
        raise AppConfigError(f"output.schema inválido: '{ref}' (esperado 'modulo.Classe')")
    path = root / f"{module_name}.py"
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


def _load_app(root: Path) -> LoadedApp:
    yaml_path = root / "app.yaml"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        config = AppConfig.model_validate(raw)
    except (yaml.YAMLError, ValidationError, OSError, TypeError) as exc:
        raise AppConfigError(f"app.yaml inválido em {root}: {exc}") from exc
    prompt_path = root / "prompts" / "system.md"
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppConfigError(f"Prompt de sistema ausente: {prompt_path}") from exc
    try:
        template = jinja2.Template(prompt_text)
    except jinja2.TemplateError as exc:
        raise AppConfigError(f"Template inválido em {prompt_path}: {exc}") from exc
    return LoadedApp(
        config=config,
        root=root,
        system_prompt=template,
        prompt_version=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12],
        output_model=_load_schema(root, config.output.schema_path),
    )


def load_apps(apps_dir: Path) -> dict[str, LoadedApp]:
    apps: dict[str, LoadedApp] = {}
    for yaml_path in sorted(Path(apps_dir).glob("*/app.yaml")):
        app = _load_app(yaml_path.parent)
        apps[app.config.id] = app
    return apps
