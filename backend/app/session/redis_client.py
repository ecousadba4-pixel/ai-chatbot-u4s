from __future__ import annotations

import logging
from functools import lru_cache

import redis.asyncio as redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """Shared Redis client for caches and state stores."""
    settings = get_settings()
    return redis.Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )


async def close_redis_client() -> None:
    """Close shared Redis connection if it was created."""
    try:
        client = get_redis_client()
    except Exception as exc:  # pragma: no cover - defensive log
        logger.warning("Failed to obtain Redis client for closing: %s", exc)
        return

    try:
        await client.aclose()
    except Exception as exc:  # pragma: no cover - best-effort close
        logger.warning("Failed to close Redis client: %s", exc)
    finally:
        get_redis_client.cache_clear()


__all__ = ["get_redis_client", "close_redis_client"]

