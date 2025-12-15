from __future__ import annotations

import asyncpg


async def list_rooms(pool: asyncpg.Pool, *, limit: int = 10) -> list[dict]:
    sql = """
        SELECT id, name, category_code, room_area, features_flags
        FROM u4s_chatbot.rooms
        ORDER BY id ASC
        LIMIT $1
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, limit)
    return [dict(row) for row in rows]


__all__ = ["list_rooms"]
