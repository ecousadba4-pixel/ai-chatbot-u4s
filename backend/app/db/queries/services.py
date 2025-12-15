from __future__ import annotations

import asyncpg


async def list_services(pool: asyncpg.Pool, *, limit: int = 20) -> list[dict]:
    sql = """
        SELECT id, name, description
        FROM u4s_chatbot.services
        ORDER BY id ASC
        LIMIT $1
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, limit)
    return [dict(row) for row in rows]


__all__ = ["list_services"]
