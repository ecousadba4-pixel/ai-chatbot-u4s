from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения на основе переменных окружения."""

    database_url: str = Field(..., alias="DATABASE_URL")
    qdrant_url: AnyHttpUrl = Field(..., alias="QDRANT_URL")
    deepseek_api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    shelter_cloud_token: str = Field(..., alias="SHELTER_CLOUD_TOKEN")

    llm_dry_run: bool = Field(False, alias="LLM_DRY_RUN")

    app_env: Literal["dev", "prod", "test"] = Field("dev", alias="APP_ENV")
    api_prefix: str = "/v1"

    request_timeout: float = 30.0
    completion_timeout: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
