import shutil
from pathlib import Path

import pytest

from encaixe.core.config import AppConfigError, load_apps
from encaixe.output.schemas import RespostaBase

APPS = Path("apps")


def test_load_example_app():
    app = load_apps(APPS)["exemplo"]
    assert app.config.models.fallbacks == ["ollama:llama3.2-3b-cpu"]
    assert issubclass(app.output_model, RespostaBase)
    assert len(app.prompt_version) == 12
    out = app.system_prompt.render(canary="CANARY123", boundary="BND")
    assert "CANARY123" in out and "BND" in out and "Aurora Ltda" in out


def _copy(tmp_path: Path) -> Path:
    shutil.copytree(APPS / "exemplo", tmp_path / "exemplo")
    return tmp_path


def test_prompt_version_changes_with_prompt(tmp_path):
    root = _copy(tmp_path)
    v1 = load_apps(root)["exemplo"].prompt_version
    p = root / "exemplo" / "prompts" / "system.md"
    p.write_text(p.read_text() + "\nlinha extra\n")
    assert load_apps(root)["exemplo"].prompt_version != v1


def test_schema_not_subclass_raises(tmp_path):
    root = _copy(tmp_path)
    (root / "exemplo" / "schemas.py").write_text(
        "from pydantic import BaseModel\n\nclass Resposta(BaseModel):\n    resposta: str\n"
    )
    with pytest.raises(AppConfigError):
        load_apps(root)


def test_missing_schema_and_bad_yaml(tmp_path):
    root = _copy(tmp_path)
    (root / "exemplo" / "schemas.py").write_text("x = 1\n")
    with pytest.raises(AppConfigError):
        load_apps(root)
    (root / "exemplo" / "app.yaml").write_text("id: [unclosed\n")
    with pytest.raises(AppConfigError):
        load_apps(root)
