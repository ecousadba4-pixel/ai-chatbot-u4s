from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.db.pool import get_pool
from app.rag.qdrant_client import QdrantClient, get_qdrant_client
from app.rag.retriever import gather_rag_data

router = APIRouter(prefix="/knowledge", dependencies=[Depends(verify_api_key)])


class KnowledgeRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=50)


class KnowledgeResult(BaseModel):
    type: str | None = None
    title: str | None = None
    content: str = ""
    source: str | None = None
    score: float = 0.0


class KnowledgeResponse(BaseModel):
    results: list[KnowledgeResult]
    debug: dict[str, Any]


@router.post("", response_model=KnowledgeResponse)
async def knowledge_search(
    payload: KnowledgeRequest,
    pool: asyncpg.Pool = Depends(get_pool),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> KnowledgeResponse:
    rag_hits = await gather_rag_data(
        query=payload.query,
        client=qdrant,
        pool=pool,
        facts_limit=payload.limit,
        files_limit=payload.limit,
        faq_limit=min(5, payload.limit),
    )

    results: list[KnowledgeResult] = []

    for hit in rag_hits.get("facts_hits", []):
        results.append(
            KnowledgeResult(
                type=hit.get("type") or "fact",
                title=hit.get("title"),
                content=hit.get("text") or "",
                source=hit.get("source"),
                score=float(hit.get("score", 0.0) or 0.0),
            )
        )

    for hit in rag_hits.get("files_hits", []):
        results.append(
            KnowledgeResult(
                type=hit.get("type") or "file",
                title=hit.get("title"),
                content=hit.get("text") or "",
                source=hit.get("source"),
                score=float(hit.get("score", 0.0) or 0.0),
            )
        )

    for faq in rag_hits.get("faq_hits", []):
        results.append(
            KnowledgeResult(
                type="faq",
                title=faq.get("question"),
                content=faq.get("answer") or "",
                source="faq",
                score=float(faq.get("similarity", 0.0) or 0.0),
            )
        )

    results = results[: payload.limit]

    debug: dict[str, Any] = {
        "facts_hits": len(rag_hits.get("facts_hits", [])),
        "files_hits": len(rag_hits.get("files_hits", [])),
        "faq_hits": len(rag_hits.get("faq_hits", [])),
        "hits_total": rag_hits.get("hits_total", 0),
        "rag_latency_ms": rag_hits.get("rag_latency_ms", 0),
    }
    if rag_hits.get("embed_error"):
        debug["embed_error"] = rag_hits["embed_error"]

    return KnowledgeResponse(results=results, debug=debug)


__all__ = ["router"]
