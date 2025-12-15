from __future__ import annotations

import asyncpg


async def fetch_hotel(pool: asyncpg.Pool) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, name, description, features_flags FROM u4s_chatbot.hotel LIMIT 1")
        return dict(row) if row else None


__all__ = ["fetch_hotel"]
