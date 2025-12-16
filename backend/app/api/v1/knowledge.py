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
    request: KnowledgeRequest,
    pool: asyncpg.Pool = Depends(get_pool),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> KnowledgeResponse:
    rag_hits = await gather_rag_data(
        query=request.query,
        client=qdrant,
        pool=pool,
        facts_limit=request.limit,
        files_limit=request.limit,
        faq_limit=min(5, request.limit),
        intent="knowledge_lookup",
    )

    results: list[KnowledgeResult] = []

    qdrant_hits = rag_hits.get("qdrant_hits")
    if qdrant_hits is None:
        qdrant_hits = [
            *rag_hits.get("facts_hits", []),
            *rag_hits.get("files_hits", []),
        ]

    for hit in qdrant_hits:
        hit_payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
        content = hit_payload.get("text")
        if not content:
            for key in ("content", "chunk", "page_content", "body"):
                value = hit_payload.get(key)
                if isinstance(value, str) and value.strip():
                    content = value
                    break
        if not content:
            content = hit.get("text") or ""
        title = hit_payload.get("title") or hit.get("title")
        source = hit_payload.get("source") or hit.get("source")
        if not title:
            if isinstance(source, str) and source:
                title = source
            elif isinstance(content, str) and content:
                title = content[:60]

        results.append(
            KnowledgeResult(
                type=hit_payload.get("type") or hit.get("type"),
                title=title,
                content=content or "",
                source=source,
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

    results.sort(key=lambda item: item.score, reverse=True)
    results = results[: request.limit]

    debug: dict[str, Any] = {
        "facts_hits": len(rag_hits.get("facts_hits", [])),
        "files_hits": len(rag_hits.get("files_hits", [])),
        "qdrant_hits": len(qdrant_hits),
        "faq_hits": len(rag_hits.get("faq_hits", [])),
        "hits_total": rag_hits.get("hits_total", 0),
        "rag_latency_ms": rag_hits.get("rag_latency_ms", 0),
        "embed_latency_ms": rag_hits.get("embed_latency_ms", 0),
        "raw_qdrant_hits": rag_hits.get("raw_qdrant_hits", []),
        "sample_payload_keys": None,
        "min_score": rag_hits.get("min_score"),
        "max_score": rag_hits.get("max_score"),
        "score_threshold_used": rag_hits.get("score_threshold_used"),
        "filtered_out_count": rag_hits.get("filtered_out_count", 0),
        "expanded_queries": rag_hits.get("expanded_queries", []),
        "boosting_applied": rag_hits.get("boosting_applied", False),
        "intent_detected": rag_hits.get("intent_detected"),
        "merged_hits_count": rag_hits.get("merged_hits_count", 0),
    }
    for raw_hit in rag_hits.get("raw_qdrant_hits", []):
        payload = raw_hit.get("payload")
        if isinstance(payload, dict):
            debug["sample_payload_keys"] = list(payload.keys())
            break
    if rag_hits.get("embed_error"):
        debug["embed_error"] = rag_hits["embed_error"]

    return KnowledgeResponse(results=results, debug=debug)


__all__ = ["router"]
