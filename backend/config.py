from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

if __package__:
    from .redis_gateway import parse_redis_args
else:
    from redis_gateway import parse_redis_args


def _strip(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


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
    yandex_api_key: str
    yandex_folder_id: str
    vector_store_id: str
    allowed_origins: tuple[str, ...]
    redis_url: str = ""
    redis_args: dict[str, Any] = field(default_factory=dict)
    http_timeout: float = 30.0
    completion_timeout: float = 60.0
    input_max_tokens: int = 3500
    context_max_chars: int = 2500
    context_per_file_limit: int = 12
    cache_ttl: float = 180.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            yandex_api_key=_strip(os.environ.get("YANDEX_API_KEY")),
            yandex_folder_id=_strip(os.environ.get("YANDEX_FOLDER_ID")),
            vector_store_id=_strip(os.environ.get("VECTOR_STORE_ID")),
            allowed_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "*")),
            redis_url=_strip(os.environ.get("REDIS_URL")),
            redis_args=parse_redis_args(os.environ.get("REDIS_ARGS")),
        )

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)

    @property
    def can_use_vector_store(self) -> bool:
        return bool(self.has_api_credentials and self.vector_store_id)

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)


CONFIG = AppConfig.from_env()

__all__ = ["AppConfig", "CONFIG", "parse_allowed_origins"]
