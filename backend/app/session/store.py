from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import redis.asyncio as redis

from app.core.config import get_settings


class SessionStore:
    """Хранилище сессий в Redis."""

    key_prefix = "u4s:sess:"

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    async def get(self, session_id: str) -> dict[str, Any] | None:
        key = self._build_key(session_id)
        data = await self._redis.get(key)
        if data is None:
            return None
        decoded = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        return json.loads(decoded)

    async def set(self, session_id: str, data: dict[str, Any]) -> None:
        key = self._build_key(session_id)
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        await self._redis.setex(key, self._ttl_seconds, payload)

    async def delete(self, session_id: str) -> None:
        key = self._build_key(session_id)
        await self._redis.delete(key)

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except redis.RedisError:
            return False

    def _build_key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=False)
    return SessionStore(client, ttl_seconds=settings.session_ttl_seconds)


__all__ = ["SessionStore", "get_session_store"]
