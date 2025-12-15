from __future__ import annotations

import asyncio
from typing import AsyncIterator

import asyncpg

from app.core.config import get_settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    return _pool


async def lifespan_pool() -> AsyncIterator[asyncpg.Pool]:
    pool = await get_pool()
    try:
        yield pool
    finally:
        await pool.close()


async def reset_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None


__all__ = ["get_pool", "lifespan_pool", "reset_pool"]
