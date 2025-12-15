from __future__ import annotations

import asyncpg


async def search_faq(pool: asyncpg.Pool, *, query: str, limit: int = 5) -> list[dict]:
    sql = """
        SELECT question, answer
        FROM u4s_chatbot.faq
        WHERE question ILIKE $1 OR answer ILIKE $1
        ORDER BY similarity(question, $1) DESC
        LIMIT $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, f"%{query}%", limit)
    return [dict(row) for row in rows]


__all__ = ["search_faq"]
