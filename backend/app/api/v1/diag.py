from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.security import verify_api_key
from app.rag.qdrant_client import QdrantClient, get_qdrant_client
from app.rag.retriever import embed_query, qdrant_search

router = APIRouter(prefix="/diag", dependencies=[Depends(verify_api_key)])


class QdrantSample(BaseModel):
    scroll_samples: list[dict[str, Any]]
    search_samples: list[dict[str, Any]]


@router.get("/qdrant_sample", response_model=QdrantSample)
async def qdrant_sample(
    q: str = Query("варианты размещения", description="Тестовый запрос"),
    limit: int = Query(3, ge=1, le=10),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> QdrantSample:
    settings = get_settings()

    scroll_hits = await qdrant.scroll(collection=settings.qdrant_collection, limit=limit)
    scroll_samples = []
    for item in scroll_hits:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else None
        if payload is not None:
            scroll_samples.append(payload)
        if len(scroll_samples) >= limit:
            break

    vector = await embed_query(q)
    search_samples: list[dict[str, Any]] = []
    if vector:
        hits = await qdrant_search(
            vector, client=qdrant, limit=limit, collection=settings.qdrant_collection
        )
        for hit in hits[:limit]:
            if not isinstance(hit, dict):
                continue
            search_samples.append(
                {
                    "score": hit.get("score"),
                    "payload": hit.get("payload"),
                }
            )

    return QdrantSample(scroll_samples=scroll_samples, search_samples=search_samples)


__all__ = ["router"]
