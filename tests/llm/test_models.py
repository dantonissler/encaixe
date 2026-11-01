import pytest
from pydantic_ai.models.test import TestModel

from encaixe.llm.models import build_model


def test_parse_ollama_spec(settings):
    assert build_model("ollama:qwen3.5:4b", settings).model_name == "qwen3.5:4b"


def test_test_spec(settings):
    assert isinstance(build_model("test", settings), TestModel)


def test_unknown_prefix_raises(settings):
    with pytest.raises(ValueError):
        build_model("openai:gpt", settings)
