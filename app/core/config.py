"""Configuração centralizada. Todos os valores vêm de variáveis de ambiente (ou .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Google Gemini API) ---
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-2.5-flash"
    # Modelos com "thinking" contam o raciocínio interno neste limite; por isso o valor é folgado.
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Embeddings e banco vetorial ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    chroma_path: Path = Path("./chroma_db")
    collection_name: str = "agri_quotes"
    data_dir: Path = Path("./data")

    # --- RAG ---
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=8, ge=1, le=30)

    # --- Histórico em memória ---
    max_history_turns: int = Field(default=5, ge=0)
    max_sessions: int = Field(default=1000, ge=1)

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
