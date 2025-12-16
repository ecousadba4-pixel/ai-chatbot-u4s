from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения на основе переменных окружения."""

    database_url: str = Field(..., alias="DATABASE_URL")
    qdrant_url: AnyHttpUrl = Field(
        "http://127.0.0.1:6333", alias="QDRANT_URL"
    )
    qdrant_collection: str = Field("u4s_kb", alias="QDRANT_COLLECTION")
    embed_url: AnyHttpUrl = Field(
        "http://127.0.0.1:8011/embed", alias="EMBED_URL"
    )
    rag_facts_limit: int = Field(6, alias="RAG_FACTS_LIMIT")
    rag_files_limit: int = Field(4, alias="RAG_FILES_LIMIT")
    rag_max_context_chars: int = Field(4000, alias="RAG_MAX_CONTEXT_CHARS")
    rag_context_chars: int = Field(4000, alias="RAG_CONTEXT_CHARS")
    rag_max_snippets: int = Field(8, alias="RAG_MAX_SNIPPETS")
    rag_min_facts: int = Field(4, alias="RAG_MIN_FACTS")
    rag_score_threshold: float = Field(0.2, alias="RAG_SCORE_THRESHOLD")
    amvera_api_token: str = Field(..., alias="AMVERA_API_TOKEN")
    amvera_api_url: AnyHttpUrl = Field(
        "https://llm.amvera.ai", alias="AMVERA_API_URL"
    )
    amvera_inference_name: str = Field("deepseek", alias="AMVERA_INFERENCE_NAME")
    amvera_model: str = Field("deepseek-chat", alias="AMVERA_MODEL")
    shelter_cloud_token: str = Field(..., alias="SHELTER_CLOUD_TOKEN")

    llm_dry_run: bool = Field(False, alias="LLM_DRY_RUN")
    llm_temperature: float = Field(0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(350, alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(20.0, alias="LLM_TIMEOUT")

    max_options: int = Field(6, alias="MAX_OPTIONS")

    app_env: Literal["dev", "prod", "test"] = Field("dev", alias="APP_ENV")
    api_prefix: str = "/v1"

    request_timeout: float = 30.0
    completion_timeout: float = Field(60.0, alias="COMPLETION_TIMEOUT")
    embed_timeout: float = Field(5.0, alias="EMBED_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
