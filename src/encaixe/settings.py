from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENCAIXE_", env_file=".env", extra="ignore")

    env: Literal["dev", "prod"] = "dev"
    database_url: str = "postgresql://encaixe_app:encaixe_app_dev@localhost:55432/encaixe"
    database_admin_url: str = "postgresql://postgres:postgres_dev@localhost:55432/encaixe"
    redis_url: str = "redis://localhost:56379/0"
    ollama_base_url: str = "http://localhost:11434"
    otlp_endpoint: str = "http://localhost:6006/v1/traces"
    apps_dir: Path = Path("apps")
    upload_dir: Path = Path("data/uploads")
    max_concurrent_generations: int = 1
    max_queue: int = 5
    queue_timeout_s: int = 60
    injection_backend: Literal["heuristic", "model"] = "heuristic"
    injection_model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    injection_threshold: float = 0.9
    safety_model: str = "llama-guard3-1b-cpu"
    embedding_model: str = "bge-m3-cpu"
    embedding_dims: int = 1024
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    max_user_message_chars: int = 8000
    log_full_text: bool = True
    # Modelos do Docling baixados com `docling-tools models download` (execução offline).
    # Também lê DOCLING_ARTIFACTS_PATH, a variável do próprio Docling. Vazio = padrão do
    # Docling (baixa do Hugging Face no 1º uso). Diretório inexistente também cai no padrão.
    docling_artifacts_path: Path | None = Field(
        default=Path.home() / ".cache" / "docling" / "models",
        validation_alias=AliasChoices(
            "docling_artifacts_path", "ENCAIXE_DOCLING_ARTIFACTS_PATH", "DOCLING_ARTIFACTS_PATH"
        ),
    )
    # OCR desligado: os PDFs esperados têm camada de texto e os modelos de OCR não são baixados.
    docling_ocr: bool = False

    @field_validator("docling_artifacts_path", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
