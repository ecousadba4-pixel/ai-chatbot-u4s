from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Iterable

import asyncpg

from app.core.config import get_settings
from app.db.queries.faq import search_faq
from app.rag.embed_client import get_embed_client
from app.rag.qdrant_client import QdrantClient
from app.session.redis_client import get_redis_client


logger = logging.getLogger(__name__)


class RAGCache:
    """Простой TTL-кэш для результатов RAG-поиска."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 120.0) -> None:
        self._cache: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def _make_key(self, query: str, intent: str | None) -> str:
        normalized = f"{query.strip().lower()}|{intent or ''}"
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    async def get(self, query: str, intent: str | None) -> dict[str, Any] | None:
        key = self._make_key(query, intent)
        async with self._lock:
            if key not in self._cache:
                return None
            result, ts = self._cache[key]
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return result

    async def set(self, query: str, intent: str | None, result: dict[str, Any]) -> None:
        key = self._make_key(query, intent)
        async with self._lock:
            self._cache[key] = (result, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)


class RedisRAGCache:
    """Redis-based cache for sharing RAG results across replicas."""

    def __init__(self, *, ttl_seconds: float = 120.0, prefix: str = "u4s:rag_cache:") -> None:
        self._redis = get_redis_client()
        self._ttl = int(ttl_seconds)
        self._prefix = prefix

    def _make_key(self, query: str, intent: str | None) -> str:
        normalized = f"{query.strip().lower()}|{intent or ''}"
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    async def get(self, query: str, intent: str | None) -> dict[str, Any] | None:
        key = f"{self._prefix}{self._make_key(query, intent)}"
        try:
            data = await self._redis.get(key)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Redis RAG cache get failed: %s", exc)
            return None

        if not data:
            return None

        try:
            decoded = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            return json.loads(decoded)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to decode Redis RAG cache entry: %s", exc)
            return None

    async def set(self, query: str, intent: str | None, result: dict[str, Any]) -> None:
        key = f"{self._prefix}{self._make_key(query, intent)}"
        try:
            payload = json.dumps(result, ensure_ascii=False)
            await self._redis.setex(key, self._ttl, payload)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Redis RAG cache set failed: %s", exc)


_RAG_CACHE: RAGCache | RedisRAGCache | None = None


def get_rag_cache() -> RAGCache | RedisRAGCache:
    global _RAG_CACHE
    if _RAG_CACHE is None:
        settings = get_settings()
        if settings.use_redis_cache:
            try:
                _RAG_CACHE = RedisRAGCache(ttl_seconds=settings.rag_cache_ttl)
            except Exception as exc:  # pragma: no cover - fall back to memory
                logger.warning("Falling back to in-memory RAG cache: %s", exc)
                _RAG_CACHE = RAGCache(ttl_seconds=settings.rag_cache_ttl)
        else:
            _RAG_CACHE = RAGCache(ttl_seconds=settings.rag_cache_ttl)
    return _RAG_CACHE


async def embed_texts(texts: list[str]) -> tuple[list[list[float]], str | None, int]:
    """Запрашивает embeddings через singleton клиент с кэшированием."""
    client = get_embed_client()
    return await client.embed(texts)


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


async def _safe_qdrant_search(
    vector: list[float],
    *,
    client: QdrantClient,
    limit: int,
) -> list[dict[str, Any]]:
    """Обёртка для безопасного поиска в Qdrant."""
    if not vector:
        return []
    try:
        return await qdrant_search(vector, client=client, limit=limit)
    except Exception as exc:  # pragma: no cover
        logger.error("Qdrant search failed: %s", exc)
        return []


async def _safe_faq_search(
    pool: asyncpg.Pool,
    query: str,
    limit: int,
    min_similarity: float,
) -> list[dict]:
    """Обёртка для безопасного поиска в FAQ."""
    try:
        return await search_faq(pool, query=query, limit=limit, min_similarity=min_similarity)
    except Exception as exc:  # pragma: no cover
        logger.error("FAQ search failed: %s", exc)
        return []


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
    use_cache: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    rag_started = time.perf_counter()
    cache = get_rag_cache() if use_cache else None

    # Проверка кэша
    if use_cache and cache:
        cached = await cache.get(query, intent)
        if cached is not None:
            logger.debug("RAG cache hit for query: %s", query[:50])
            # Обновляем latency для кэшированного результата
            cached_result = {**cached}
            cached_result["rag_latency_ms"] = 0
            cached_result["embed_latency_ms"] = 0
            cached_result["cache_hit"] = True
            return cached_result

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

    # Параллельный поиск: все Qdrant-запросы + FAQ одновременно
    qdrant_tasks = [
        _safe_qdrant_search(vector, client=client, limit=search_limit)
        for vector in embeddings
        if vector
    ]
    faq_task = _safe_faq_search(pool, query, faq_limit, faq_min_similarity)

    # Запускаем всё параллельно
    all_results = await asyncio.gather(*qdrant_tasks, faq_task)

    # Разбираем результаты: все кроме последнего — Qdrant, последний — FAQ
    qdrant_results = all_results[:-1]
    faq_hits = all_results[-1]

    qdrant_raw: list[dict[str, Any]] = []
    for result in qdrant_results:
        if result:
            qdrant_raw.extend(result)

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

    hits_total = len(filtered_hits) + len(faq_hits)
    rag_latency_ms = int((time.perf_counter() - rag_started) * 1000)

    result = {
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
        "cache_hit": False,
    }

    # Сохраняем в кэш (без raw_qdrant_hits для экономии памяти)
    if cache and use_cache and hits_total > 0:
        cache_result = {k: v for k, v in result.items() if k != "raw_qdrant_hits"}
        cache_result["raw_qdrant_hits"] = []
        await cache.set(query, intent, cache_result)

    return result


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
