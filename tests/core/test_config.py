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
    assert "CANARY123" in out and "<<DADOS-" in out and "Aurora Ltda" in out


def _copy(tmp_path: Path) -> Path:
    shutil.copytree(APPS / "exemplo", tmp_path / "exemplo")
    return tmp_path


def test_prompt_version_changes_with_prompt(tmp_path):
    root = _copy(tmp_path)
    v1 = load_apps(root)["exemplo"].prompt_version
    p = root / "exemplo" / "prompts" / "system.md"
    p.write_text(p.read_text() + "\nlinha extra\n")
    assert load_apps(root)["exemplo"].prompt_version != v1


@pytest.mark.parametrize(
    "rel,extra",
    [
        ("app.yaml", "\n# só um comentário também muda a versão\n"),
        ("schemas.py", "\n\nclass Outra(Resposta):\n    pass\n"),
        ("prompts/system.md", "\nlinha extra\n"),
    ],
)
def test_app_version_covers_prompt_app_yaml_and_schema(tmp_path, rel, extra):
    """Revisão final I5: a chave do cache exato usa app_version; com só o system.md no hash,
    mudar app.yaml ou schemas.py servia respostas antigas (e o eval as reaproveitava)."""
    root = _copy(tmp_path)
    before = load_apps(root)["exemplo"]
    path = root / "exemplo" / rel
    path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")
    after = load_apps(root)["exemplo"]
    assert len(after.app_version) == 12 and after.app_version != before.app_version
    # prompt.version continua sendo o hash só do system.md
    assert (after.prompt_version != before.prompt_version) == (rel == "prompts/system.md")


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


# --- validação na carga (revisão final m2+m8) ------------------------------------------------


def _edit_yaml(root: Path, old: str, new: str) -> None:
    path = root / "exemplo" / "app.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_example_budget_fits_the_context_window():
    ctx = load_apps(APPS)["exemplo"].config.context
    assert sum(ctx.budget.model_dump().values()) <= ctx.num_ctx


@pytest.mark.parametrize(
    "old,new",
    [
        ("input: [normalize,", "input: [normalise,"),
        ("output: [content_safety,", "output: [content_safty,"),
    ],
)
def test_unknown_guard_name_fails_at_load(tmp_path, old, new):
    """Antes, um nome errado só aparecia em produção: o guard 'indisponível' bloqueava tudo."""
    root = _copy(tmp_path)
    _edit_yaml(root, old, new)
    with pytest.raises(AppConfigError, match="guard"):
        load_apps(root)


@pytest.mark.parametrize(
    "old,new",
    [
        ("streaming:\n", "streming:\n"),  # chave de topo
        ("  top_k: 10\n", "  topk: 10\n"),  # chave aninhada
        ("  exact: {enabled: true, ttl_s: 3600}", "  exact: {enabled: true, ttl: 3600}"),
    ],
)
def test_unknown_yaml_key_fails_at_load(tmp_path, old, new):
    root = _copy(tmp_path)
    _edit_yaml(root, old, new)
    with pytest.raises(AppConfigError):
        load_apps(root)


def test_budget_larger_than_num_ctx_fails_at_load(tmp_path):
    root = _copy(tmp_path)
    _edit_yaml(root, "num_ctx: 8192", "num_ctx: 4096")
    with pytest.raises(AppConfigError, match="num_ctx"):
        load_apps(root)


def test_duplicate_app_id_fails_at_load(tmp_path):
    root = _copy(tmp_path)
    shutil.copytree(root / "exemplo", root / "copia")
    with pytest.raises(AppConfigError, match="exemplo"):
        load_apps(root)
