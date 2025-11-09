"""Утилиты для работы с Redis и хранением истории."""

from __future__ import annotations

import json
import shlex
from typing import Any, Sequence
from urllib.parse import parse_qsl

try:  # pragma: no cover - опциональная зависимость в тестовой среде
    import redis
except ModuleNotFoundError:  # pragma: no cover - graceful degradation без redis
    redis = None

REDIS_HISTORY_KEY = "chat:history:{session_id}"
REDIS_HISTORY_TTL_SECONDS = 30 * 60
REDIS_MAX_MESSAGES = 50


def _coerce_arg_value(key: str, value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if key in {"db", "socket_timeout", "connect_timeout", "health_check_interval"}:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def parse_redis_args(raw: str | None) -> dict[str, Any]:
    """Преобразует REDIS_ARGS в kwargs для Redis.from_url."""

    if not raw:
        return {}

    raw = raw.strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}

    if "&" in raw or raw.startswith("?"):
        query_items = parse_qsl(raw, keep_blank_values=True)
        if query_items:
            return {key: _coerce_arg_value(key, value) for key, value in query_items}

    options: dict[str, Any] = {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = []

    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.lstrip("-")
        if not key:
            continue
        options[key] = _coerce_arg_value(key, value)

    return options


def create_redis_client(
    redis_url: str | None, redis_args: dict[str, Any] | None
) -> "redis.Redis | None":
    if redis is None or not redis_url:
        return None

    extra_args = dict(redis_args or {})
    extra_args.setdefault("decode_responses", True)

    try:
        return redis.Redis.from_url(redis_url, **extra_args)
    except Exception as error:  # pragma: no cover - защититься от ошибок конфига
        print("Redis init error:", error)
        return None


class RedisHistoryGateway:
    def __init__(
        self,
        client: "redis.Redis | None",
        *,
        ttl_seconds: int = REDIS_HISTORY_TTL_SECONDS,
        max_messages: int = REDIS_MAX_MESSAGES,
    ) -> None:
        self._client = client
        self._ttl_seconds = int(ttl_seconds)
        self._max_messages = max(1, int(max_messages))

    @property
    def max_messages(self) -> int:
        return self._max_messages

    def _key(self, session_id: str) -> str | None:
        session = (session_id or "").strip()
        if not session:
            return None
        return REDIS_HISTORY_KEY.format(session_id=session)

    def read_history(self, session_id: str) -> list[dict[str, Any]]:
        if not self._client:
            return []
        try:
            key = self._key(session_id)
            if not key:
                return []
            raw_value = self._client.get(key)
        except Exception as error:  # pragma: no cover - логируем и идем дальше
            print("Redis read_history error:", error)
            return []
        if not raw_value:
            return []
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        history: list[dict[str, Any]] = []
        for item in payload[-self._max_messages :]:
            if isinstance(item, dict):
                history.append(item)
        return history

    def write_history(
        self,
        session_id: str,
        messages: Sequence[dict[str, Any]],
        ttl: int | None = None,
    ) -> None:
        if not self._client:
            return
        key = self._key(session_id)
        if not key:
            return
        ttl_seconds = int(ttl) if ttl else self._ttl_seconds
        ttl_seconds = max(ttl_seconds, 1)
        limited: list[dict[str, Any]] = []
        for item in messages[-self._max_messages :]:
            if isinstance(item, dict):
                limited.append(item)
        try:
            payload = json.dumps(limited, ensure_ascii=False)
            self._client.setex(key, ttl_seconds, payload)
        except Exception as error:  # pragma: no cover - логируем и идем дальше
            print("Redis write_history error:", error)

    def delete_history(self, session_id: str) -> None:
        if not self._client:
            return
        key = self._key(session_id)
        if not key:
            return
        try:
            self._client.delete(key)
        except Exception as error:  # pragma: no cover - логируем и идем дальше
            print("Redis delete_history error:", error)

