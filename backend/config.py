from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


def _strip(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _parse_positive_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default

    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default

    return max(1, value)


def _iter_plain_origins(raw: str) -> Iterable[str]:
    for part in re.split(r"[,\s]+", raw):
        part = part.strip().strip('"').strip("'")
        if part:
            yield part


def parse_allowed_origins(raw: str | None) -> tuple[str, ...]:
    """Преобразует ALLOWED_ORIGINS в кортеж доменов."""

    if not raw:
        return ("*",)

    raw = raw.strip()
    if raw == "*":
        return ("*",)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        parsed = parsed.strip()
        return (parsed,) if parsed else ("*",)

    if isinstance(parsed, (list, tuple, set)):
        values = tuple(str(item).strip() for item in parsed if str(item).strip())
        return values or ("*",)

    values = tuple(_iter_plain_origins(raw))
    return values or ("*",)


@dataclass(frozen=True)
class AppConfig:
    amvera_api_token: str
    amvera_api_url: str
    amvera_model: str
    allowed_origins: tuple[str, ...]
    http_timeout: float = 30.0
    completion_timeout: float = 60.0
    input_max_tokens: int = 3500
    context_max_chars: int = 2500
    context_per_file_limit: int = 12
    cache_ttl: float = 180.0
    cache_max_files: int = 32
    shelter_cloud_token: str = ""
    shelter_cloud_language: str = "ru"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            amvera_api_token=_strip(os.environ.get("AMVERA_API_TOKEN")),
            amvera_api_url=_strip(
                os.environ.get("AMVERA_API_URL") or "https://llm.amvera.ai/v1"
            ),
            amvera_model=_strip(os.environ.get("AMVERA_MODEL")) or "deepseek-chat",
            allowed_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "*")),
            cache_max_files=_parse_positive_int(
                os.environ.get("CACHE_MAX_FILES"), default=32
            ),
            shelter_cloud_token=_strip(
                os.environ.get("SHELTER_CLOUD_TOKEN")
            ),
            shelter_cloud_language=_strip(
                os.environ.get("SHELTER_CLOUD_LANGUAGE") or "ru"
            ),
        )

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.amvera_api_token)


CONFIG = AppConfig.from_env()

__all__ = ["AppConfig", "CONFIG", "parse_allowed_origins"]
