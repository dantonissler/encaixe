"""Configuração do Ollama no docker-compose.yml coerente com os modelos que as apps usam."""

from pathlib import Path

import yaml

from encaixe.core.config import load_apps
from encaixe.settings import Settings
from encaixe.warmup import plan_warmup

ROOT = Path(__file__).resolve().parents[1]


def test_ollama_keeps_every_hot_model_loaded():
    """Rodada de correções 2 da Task 15: com o guard em camadas o caminho de cada requisição
    usa 4 modelos (principal, ShieldGemma, Llama Guard, bge-m3). Sem OLLAMA_MAX_LOADED_MODELS
    o Ollama mantém 3 e troca um a cada requisição (85 cargas em 20 min, ~5 s cada)."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["ollama"]["environment"]
    plan = plan_warmup(Settings(_env_file=None), load_apps(ROOT / "apps"))
    hot = len(plan.chat_models) + len(plan.embed_models)
    assert hot == 4
    assert int(env.get("OLLAMA_MAX_LOADED_MODELS", 3)) >= hot


def _host_ip(spec: object) -> str | None:
    """IP do host de uma entrada de `ports` (forma curta ou longa do Compose)."""
    if isinstance(spec, dict):
        return spec.get("host_ip")
    parts = str(spec).split(":")
    return parts[0] if len(parts) == 3 else None


def _default(value: str) -> str:
    """Valor padrão de uma interpolação do Compose (`${VAR:-padrão}`), ou o próprio valor."""
    if value.startswith("${") and ":-" in value:
        return value[value.index(":-") + 2 : -1]
    return value


def test_every_published_port_is_loopback_only():
    """Revisão final C1: Postgres (superusuário), Redis (sem senha), Ollama, Phoenix (sem
    login) e o adaptador MCP (confia em X-User-Ref) não podem ficar na rede local. Spec §3:
    só a API e o Open WebUI ficam expostos, e por escolha do operador (README)."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    published = 0
    for name, service in compose["services"].items():
        for spec in service.get("ports", []):
            assert _host_ip(spec) == "127.0.0.1", f"{name} publica {spec!r} fora do loopback"
            published += 1
        if service.get("network_mode") == "host":
            host = _default(service.get("environment", {}).get("HOST", ""))
            assert host == "127.0.0.1", f"{name} na rede do host sem HOST=127.0.0.1 por padrão"
    assert published >= 5  # postgres, redis, ollama, phoenix, mcp-tarefas


def test_make_api_binds_loopback_by_default():
    """`make api` escuta só no loopback; API_HOST=0.0.0.0 expõe para backends em outras
    máquinas (README)."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "API_HOST ?= 127.0.0.1" in makefile
    api_recipe = makefile.split("\napi:\n", 1)[1].split("\n\n", 1)[0]
    assert "--host $(API_HOST)" in api_recipe


def test_make_migrate_runs_the_cli():
    """Revisão final I4: `python -m encaixe.db.migrate` não tem `__main__` e não fazia nada."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\nmigrate:\n", 1)[1].split("\n\n", 1)[0]
    assert recipe.strip() == "uv run encaixe migrate"


def test_make_eval_clears_the_response_cache_first():
    """Revisão final I5: sem isso o eval reaproveitava respostas em cache (TTL 1 h) e não
    passava pelos guards de saída novos."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\neval:\n", 1)[1].split("\n\n", 1)[0]
    lines = [line.strip() for line in recipe.splitlines()]
    clear = lines.index("uv run encaixe cache clear")
    promptfoo = next(i for i, line in enumerate(lines) if "$(PROMPTFOO)" in line)
    assert clear < promptfoo


def test_open_webui_reaches_the_api_on_loopback_without_publishing_ports():
    """Revisão final I6: `make api` escuta só em 127.0.0.1:8100, que um container na rede
    bridge não alcança (host.docker.internal cai no IP do gateway). Na rede do host o Open
    WebUI fala com 127.0.0.1:8100 e escuta em 127.0.0.1:38080, sem expor nada."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    webui = compose["services"]["open-webui"]
    env = webui["environment"]
    assert webui["network_mode"] == "host"
    assert "ports" not in webui and "extra_hosts" not in webui
    assert _default(env["HOST"]) == "127.0.0.1" and str(env["PORT"]) == "38080"
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:8100/v1"
