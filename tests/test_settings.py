from pathlib import Path

import pytest

from encaixe.settings import Settings


def test_defaults_and_env_override(monkeypatch):
    monkeypatch.setenv("ENCAIXE_MAX_QUEUE", "7")
    s = Settings()
    assert s.max_queue == 7
    assert s.redis_url.endswith(":56379/0")
    assert s.embedding_dims == 1024


def _clear_docling_env(monkeypatch):
    for name in ("DOCLING_ARTIFACTS_PATH", "ENCAIXE_DOCLING_ARTIFACTS_PATH", "ENCAIXE_DOCLING_OCR"):
        monkeypatch.delenv(name, raising=False)


def test_docling_defaults_offline_artifacts_and_no_ocr(monkeypatch):
    _clear_docling_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.docling_artifacts_path == Path.home() / ".cache" / "docling" / "models"
    assert s.docling_ocr is False


@pytest.mark.parametrize("var", ["ENCAIXE_DOCLING_ARTIFACTS_PATH", "DOCLING_ARTIFACTS_PATH"])
def test_docling_artifacts_path_from_env(monkeypatch, tmp_path, var):
    _clear_docling_env(monkeypatch)
    monkeypatch.setenv(var, str(tmp_path))
    monkeypatch.setenv("ENCAIXE_DOCLING_OCR", "true")
    s = Settings(_env_file=None)
    assert s.docling_artifacts_path == tmp_path and s.docling_ocr is True


def test_docling_artifacts_path_empty_means_none(monkeypatch):
    _clear_docling_env(monkeypatch)
    monkeypatch.setenv("ENCAIXE_DOCLING_ARTIFACTS_PATH", "")
    assert Settings(_env_file=None).docling_artifacts_path is None
    assert Settings(_env_file=None, docling_artifacts_path=None).docling_artifacts_path is None
