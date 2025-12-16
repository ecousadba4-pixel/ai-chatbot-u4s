from __future__ import annotations

from typing import Any, Iterable

import logging
import time

import asyncpg
import httpx

from app.core.config import get_settings
from app.db.queries.faq import search_faq
from app.rag.qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


def _normalize_vector(vector: Any) -> list[float]:
    if isinstance(vector, list):
        floats = [float(x) for x in vector if isinstance(x, (int, float))]
        if floats:
            return floats
    return []


def _extract_embeddings(data: Any) -> list[list[float]]:
    embeddings: list[list[float]] = []

    if isinstance(data, dict):
        for key in ("embeddings", "vectors", "data", "result"):
            value = data.get(key)
            if isinstance(value, list):
                embeddings.extend(_extract_embeddings(value))
        for key in ("embedding", "vector"):
            vector = _normalize_vector(data.get(key))
            if vector:
                embeddings.append(vector)
        return embeddings

    if isinstance(data, list):
        if data and all(isinstance(x, (int, float)) for x in data):
            vector = _normalize_vector(data)
            if vector:
                embeddings.append(vector)
            return embeddings

        for item in data:
            if isinstance(item, dict):
                for key in ("embedding", "vector"):
                    vector = _normalize_vector(item.get(key))
                    if vector:
                        embeddings.append(vector)
                continue
            vector = _normalize_vector(item)
            if vector:
                embeddings.append(vector)

    return embeddings


async def embed_texts(texts: list[str]) -> tuple[list[list[float]], str | None, int]:
    settings = get_settings()

    if not texts:
        return [], None, 0

    started = time.perf_counter()
    timeout = httpx.Timeout(
        connect=2.0,
        read=settings.embed_timeout,
        write=settings.embed_timeout,
        pool=settings.embed_timeout,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(str(settings.embed_url), json={"texts": list(texts)})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - сетевые ошибки
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("Embedding request failed: %s", exc, extra={"embed_error": str(exc)})
        return [], str(exc), latency_ms
    except ValueError as exc:  # pragma: no cover - невалидный JSON
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("Failed to parse embedding response: %s", exc, extra={"embed_error": str(exc)})
        return [], str(exc), latency_ms

    latency_ms = int((time.perf_counter() - started) * 1000)

    embeddings: list[list[float]] = []
    expected_dim: int | None = None
    if isinstance(data, dict):
        dim = data.get("dim")
        if isinstance(dim, int) and dim > 0:
            expected_dim = dim

        vectors = data.get("vectors")
        if isinstance(vectors, list):
            for item in vectors:
                vector = _normalize_vector(item)
                if vector:
                    embeddings.append(vector)

            if expected_dim:
                if any(len(vec) != expected_dim for vec in embeddings):
                    logger.warning(
                        "Embedding dimension mismatch", extra={"embed_error": "dim_mismatch"}
                    )
                    return [], "dim_mismatch", latency_ms
                if expected_dim != 768:
                    logger.warning(
                        "Unexpected embedding dimension", extra={"embed_error": "unexpected_dim"}
                    )
                    return [], "unexpected_dim", latency_ms

    if not embeddings:
        embeddings = _extract_embeddings(data)

    if expected_dim and embeddings:
        if any(len(vec) != expected_dim for vec in embeddings):
            logger.warning(
                "Embedding dimension mismatch", extra={"embed_error": "dim_mismatch"}
            )
            return [], "dim_mismatch", latency_ms
        if expected_dim != 768:
            logger.warning(
                "Unexpected embedding dimension", extra={"embed_error": "unexpected_dim"}
            )
            return [], "unexpected_dim", latency_ms

    if not embeddings:
        logger.warning("Embedding service returned empty embeddings", extra={"embed_error": "empty_embeddings"})
        return [], "empty_embeddings", latency_ms
    return embeddings, None, latency_ms


async def embed_query(text: str) -> list[float]:
    """Запрашивает embedding у внешнего сервиса."""

    embeddings, _, _ = await embed_texts([text])
    return embeddings[0] if embeddings else []


def _build_filter(*, source_prefix: str | None, types: Iterable[str] | None) -> dict[str, Any] | None:
    filters = []
    if source_prefix:
        match_key = "text" if source_prefix.endswith(":") else "value"
        filters.append({"key": "payload.source", "match": {match_key: source_prefix}})
    if types:
        type_list = [item for item in types if item]
        if type_list:
            filters.append({"key": "payload.type", "match": {"any": type_list}})

    if not filters:
        return None
    return {"must": filters}


async def qdrant_search(
    vector: Iterable[float],
    *,
    client: QdrantClient,
    limit: int = 6,
    source_prefix: str | None = None,
    types: Iterable[str] | None = None,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    query_filter = _build_filter(source_prefix=source_prefix, types=types)
    return await client.search(
        collection=collection or settings.qdrant_collection,
        vector=vector,
        limit=limit,
        query_filter=query_filter,
    )


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    payload = hit.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    text = _extract_text(payload)
    title = payload.get("title") if isinstance(payload.get("title"), str) else None
    entity_id = payload.get("entity_id") if isinstance(payload.get("entity_id"), str) else None
    source = payload.get("source") if isinstance(payload.get("source"), str) else None
    type_value = payload.get("type") if isinstance(payload.get("type"), str) else None

    return {
        "score": float(hit.get("score", 0.0) or 0.0),
        "type": type_value,
        "title": title,
        "entity_id": entity_id,
        "text": text,
        "source": source,
        "payload": payload,
    }


def _deduplicate_hits(
    hits: list[dict[str, Any]], *, seen: set[str] | None = None
) -> list[dict[str, Any]]:
    known = seen if seen is not None else set()
    unique: list[dict[str, Any]] = []
    for hit in hits:
        text = hit.get("text") or ""
        title = hit.get("title") or ""
        payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
        entity_id = payload.get("entity_id") or payload.get("id")
        key = str(entity_id or f"{title}::{text[:80]}")
        if key in known:
            continue
        known.add(key)
        unique.append(hit)
    return unique


async def retrieve_context(query: str, *, client: QdrantClient) -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    try:
        vector = await embed_query(query)
    except Exception:
        return {"facts_hits": [], "files_hits": []}
    if not vector:
        return {"facts_hits": [], "files_hits": []}

    try:
        facts_raw = await qdrant_search(
            vector,
            client=client,
            limit=settings.rag_facts_limit,
            source_prefix="postgres:u4s_chatbot",
        )
    except Exception:
        facts_raw = []

    dedup_keys: set[str] = set()
    facts_hits = [_normalize_hit(item) for item in facts_raw]
    facts_hits = _deduplicate_hits(facts_hits, seen=dedup_keys)

    files_hits: list[dict[str, Any]] = []
    if len(facts_hits) < settings.rag_min_facts:
        try:
            files_raw = await qdrant_search(
                vector,
                client=client,
                limit=settings.rag_files_limit,
                source_prefix="file:",
            )
        except Exception:
            files_raw = []
        files_hits = _deduplicate_hits([
            _normalize_hit(item) for item in files_raw
        ], seen=dedup_keys)

    return {"facts_hits": facts_hits, "files_hits": files_hits}


async def gather_rag_data(
    query: str,
    *,
    client: QdrantClient,
    pool: asyncpg.Pool,
    facts_limit: int | None = None,
    files_limit: int | None = None,
    faq_limit: int = 3,
    faq_min_similarity: float = 0.35,
    intent: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    rag_started = time.perf_counter()

    expanded_queries: list[str] = []
    if intent == "lodging":
        expanded_queries = [
            f"{query} типы размещения номера домики коттеджи вместимость стоимость",
            "категории проживания домики номера коттеджи",
            "домики номера вместимость цена тариф",
        ]

    queries = [query, *expanded_queries]
    embeddings, embed_error, embed_latency_ms = await embed_texts(queries)
    if not embeddings:
        return {
            "facts_hits": [],
            "files_hits": [],
            "qdrant_hits": [],
            "faq_hits": [],
            "hits_total": 0,
            "rag_latency_ms": int((time.perf_counter() - rag_started) * 1000),
            "embed_error": embed_error,
            "embed_latency_ms": embed_latency_ms,
            "raw_qdrant_hits": [],
            "min_score": None,
            "max_score": None,
            "score_threshold_used": settings.rag_score_threshold,
            "filtered_out_count": 0,
            "expanded_queries": expanded_queries,
            "boosting_applied": False,
            "intent_detected": intent,
            "merged_hits_count": 0,
        }

    search_limit = max(
        facts_limit or settings.rag_facts_limit,
        files_limit or settings.rag_files_limit,
    )

    qdrant_raw: list[dict[str, Any]] = []
    for vector in embeddings:
        if not vector:
            continue
        try:
            qdrant_raw.extend(
                await qdrant_search(
                    vector,
                    client=client,
                    limit=search_limit,
                )
            )
        except Exception as exc:  # pragma: no cover - сеть/хранилище
            logger.error("Qdrant search failed: %s", exc)

    raw_scores = [
        float(item.get("score", 0.0) or 0.0)
        for item in qdrant_raw
        if isinstance(item, dict)
    ]
    min_score = min(raw_scores) if raw_scores else None
    max_score = max(raw_scores) if raw_scores else None

    normalized_hits = [_normalize_hit(item) for item in qdrant_raw]
    normalized_hits = _deduplicate_hits(normalized_hits)
    merged_hits_count = len(normalized_hits)

    boosting_applied = False
    if intent == "lodging":
        boosting_applied = True
        boosted_hits: list[dict[str, Any]] = []
        for hit in normalized_hits:
            payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
            type_value = payload.get("type") or hit.get("type")
            source = payload.get("source") or hit.get("source") or ""
            multiplier = 1.0
            if isinstance(type_value, str) and type_value in {"faq", "faq_ext"}:
                multiplier *= 0.85
            if isinstance(source, str) and source.startswith("knowledge/about"):
                multiplier *= 1.05
            if payload and isinstance(payload.get("subtype"), str):
                if payload.get("subtype", "").startswith("about"):
                    multiplier *= 1.05
            boosted_hit = {**hit, "score": hit.get("score", 0.0) * multiplier}
            boosted_hits.append(boosted_hit)
        normalized_hits = boosted_hits

    threshold = settings.rag_score_threshold
    filtered_hits = [
        hit for hit in normalized_hits if hit.get("score", 0.0) >= threshold
    ]
    filtered_hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    filtered_out_count = len(normalized_hits) - len(filtered_hits)

    try:
        faq_hits = await search_faq(
            pool, query=query, limit=faq_limit, min_similarity=faq_min_similarity
        )
    except Exception as exc:  # pragma: no cover - БД
        logger.error("FAQ search failed: %s", exc)
        faq_hits = []

    hits_total = len(filtered_hits) + len(faq_hits)
    rag_latency_ms = int((time.perf_counter() - rag_started) * 1000)

    return {
        "facts_hits": filtered_hits,
        "files_hits": [],
        "qdrant_hits": filtered_hits,
        "faq_hits": faq_hits,
        "hits_total": hits_total,
        "rag_latency_ms": rag_latency_ms,
        "embed_error": embed_error,
        "embed_latency_ms": embed_latency_ms,
        "raw_qdrant_hits": qdrant_raw,
        "min_score": min_score,
        "max_score": max_score,
        "score_threshold_used": threshold,
        "filtered_out_count": filtered_out_count,
        "expanded_queries": expanded_queries,
        "boosting_applied": boosting_applied,
        "intent_detected": intent,
        "merged_hits_count": merged_hits_count,
    }


async def search_hits_with_payload(
    query: str, *, client: QdrantClient
) -> dict[str, list[dict[str, Any]]]:
    settings = get_settings()
    try:
        vector = await embed_query(query)
    except Exception:
        return {"facts": [], "files": []}
    if not vector:
        return {"facts": [], "files": []}

    try:
        facts_hits = await qdrant_search(
            vector,
            client=client,
            limit=settings.rag_facts_limit,
            source_prefix="postgres:u4s_chatbot",
        )
    except Exception:
        facts_hits = []
    files_hits: list[dict[str, Any]] = []
    if len(facts_hits) < settings.rag_min_facts:
        try:
            files_hits = await qdrant_search(
                vector,
                client=client,
                limit=settings.rag_files_limit,
                source_prefix="file:",
            )
        except Exception:
            files_hits = []

    return {"facts": facts_hits, "files": files_hits}


__all__ = [
    "embed_texts",
    "embed_query",
    "qdrant_search",
    "retrieve_context",
    "gather_rag_data",
    "search_hits_with_payload",
]
