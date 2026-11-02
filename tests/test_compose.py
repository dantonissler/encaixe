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
