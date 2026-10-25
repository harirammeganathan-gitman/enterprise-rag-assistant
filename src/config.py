"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise RAG Assistant"
    data_dir: Path = Path("data/index")
    qdrant_url: str = "local"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    local_model_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "LOCAL_MODEL_BASE_URL"),
    )
    local_model_name: str = Field(
        default="llama3.2",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "LOCAL_MODEL_NAME"),
    )
    local_model_api_key: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_API_KEY", "LOCAL_MODEL_API_KEY"),
    )
    api_access_key: str | None = None
    require_api_key: bool = False
    admin_username: str | None = None
    admin_password_hash: str | None = None
    admin_session_secret: str | None = None
    retrieval_candidates: int = 20
    top_k: int = 4
    cache_similarity_threshold: float = 0.92

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
