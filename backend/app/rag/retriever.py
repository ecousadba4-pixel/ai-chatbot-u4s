from __future__ import annotations

import math
from typing import Iterable

import asyncpg

from app.db.queries.faq import search_faq
from app.rag.qdrant_client import QdrantClient


def _cheap_embedding(text: str, *, dim: int = 64) -> list[float]:
    """Грубый embedding без внешних зависимостей для совместимости среды."""
    text = text or ""
    vector = [0.0] * dim
    for index, char in enumerate(text.encode("utf-8")):
        vector[index % dim] += char / 255.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


async def retrieve_context(
    *,
    pool: asyncpg.Pool,
    qdrant: QdrantClient,
    question: str,
    limit: int = 5,
) -> str:
    faq_rows = await search_faq(pool, query=question, limit=limit)
    faq_section = "\n\n".join(
        f"Q: {row['question']}\nA: {row['answer']}" for row in faq_rows
    )

    vector = _cheap_embedding(question)
    qdrant_hits = await qdrant.search(collection="faq", vector=vector, limit=limit)
    rag_texts: list[str] = []
    for hit in qdrant_hits:
        payload = hit.get("payload") or {}
        text_parts = []
        for key in ("question", "answer", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                text_parts.append(value)
        if text_parts:
            rag_texts.append("\n".join(text_parts))
    rag_section = "\n\n".join(rag_texts)

    combined_parts = [part for part in (faq_section, rag_section) if part]
    return "\n\n".join(combined_parts)


__all__ = ["retrieve_context"]
