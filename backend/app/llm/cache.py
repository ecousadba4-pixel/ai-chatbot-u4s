"""
Семантический кэш для LLM ответов.

Кэширует ответы на основе нормализованного вопроса, intent и контекста.
Позволяет быстро отдавать повторные ответы без вызова LLM.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from app.session.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    """Нормализует запрос для кэширования."""
    normalized = query.strip().lower()
    return " ".join(normalized.split())


def _make_cache_key(
    query: str, intent: str, context: str, *, context_hash_length: int
) -> str:
    """Создаёт общий ключ для разных реализаций кэша."""
    normalized_query = _normalize_query(query)
    context_snippet = context[:context_hash_length] if context else ""
    context_hash = hashlib.md5(
        context_snippet.encode(), usedforsecurity=False
    ).hexdigest()[:12]

    key_string = f"{normalized_query}|{intent}|{context_hash}"
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


class LLMCache:
    """
    Семантический кэш для LLM ответов.
    
    Ключ кэша формируется из:
    - Нормализованного текста вопроса (lowercase, stripped)
    - Intent запроса
    - Хэша первых N символов контекста (для учёта RAG данных)
    """

    def __init__(
        self,
        max_size: int = 512,
        ttl_seconds: float = 600.0,
        context_hash_length: int = 500,
    ) -> None:
        self._cache: OrderedDict[str, tuple[str, float, dict[str, Any]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._context_hash_length = context_hash_length
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, intent: str, context: str) -> str:
        """Создаёт ключ кэша."""
        return _make_cache_key(
            query, intent, context, context_hash_length=self._context_hash_length
        )

    async def get(
        self,
        query: str,
        intent: str,
        context: str = "",
    ) -> tuple[str | None, dict[str, Any] | None]:
        """
        Получает кэшированный ответ.
        
        Returns:
            Tuple из (answer, debug_info) или (None, None) если не найдено
        """
        key = self._make_key(query, intent, context)
        
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None, None
            
            answer, ts, debug_info = self._cache[key]
            if time.time() - ts > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None, None
            
            # Обновляем позицию (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            
            logger.debug(
                "LLM cache hit for query: %s (hits=%d, misses=%d)",
                query[:50], self._hits, self._misses
            )
            
            return answer, debug_info

    async def set(
        self,
        query: str,
        intent: str,
        context: str,
        answer: str,
        debug_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Сохраняет ответ в кэш.
        
        Args:
            query: Исходный вопрос пользователя
            intent: Определённый intent
            context: RAG контекст
            answer: Ответ LLM
            debug_info: Отладочная информация (опционально)
        """
        key = self._make_key(query, intent, context)
        
        async with self._lock:
            self._cache[key] = (answer, time.time(), debug_info or {})
            self._cache.move_to_end(key)
            
            # Удаляем старые записи если превышен лимит
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def invalidate(self, query: str, intent: str, context: str = "") -> bool:
        """Удаляет запись из кэша."""
        key = self._make_key(query, intent, context)
        
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Очищает весь кэш. Возвращает количество удалённых записей."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count

    def stats(self) -> dict[str, Any]:
        """Возвращает статистику кэша."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_seconds": self._ttl,
        }


class RedisLLMCache:
    """Redis-backed кэш LLM для шаринга между инстансами."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 600.0,
        context_hash_length: int = 500,
        prefix: str = "u4s:llm_cache:",
    ) -> None:
        self._redis = get_redis_client()
        self._ttl = ttl_seconds
        self._context_hash_length = context_hash_length
        self._prefix = prefix
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._known_keys: set[str] = set()

    def _make_key(self, query: str, intent: str, context: str) -> str:
        return _make_cache_key(
            query, intent, context, context_hash_length=self._context_hash_length
        )

    def _redis_key(self, cache_key: str) -> str:
        return f"{self._prefix}{cache_key}"

    async def get(
        self, query: str, intent: str, context: str = ""
    ) -> tuple[str | None, dict[str, Any] | None]:
        cache_key = self._make_key(query, intent, context)
        redis_key = self._redis_key(cache_key)

        async with self._lock:
            try:
                raw = await self._redis.get(redis_key)
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("Redis LLM cache get failed: %s", exc)
                self._misses += 1
                return None, None

            if not raw:
                self._misses += 1
                return None, None

            try:
                decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                payload = json.loads(decoded)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to decode Redis LLM cache entry: %s", exc)
                self._misses += 1
                return None, None

            answer = payload.get("answer")
            debug_info = payload.get("debug_info") or {}
            self._hits += 1
            self._known_keys.add(redis_key)
            return answer, debug_info

    async def set(
        self,
        query: str,
        intent: str,
        context: str,
        answer: str,
        debug_info: dict[str, Any] | None = None,
    ) -> None:
        cache_key = self._make_key(query, intent, context)
        redis_key = self._redis_key(cache_key)
        payload = json.dumps(
            {"answer": answer, "debug_info": debug_info or {}}, ensure_ascii=False
        )
        async with self._lock:
            try:
                await self._redis.setex(redis_key, int(self._ttl), payload)
                self._known_keys.add(redis_key)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Redis LLM cache set failed: %s", exc)

    async def invalidate(self, query: str, intent: str, context: str = "") -> bool:
        cache_key = self._make_key(query, intent, context)
        redis_key = self._redis_key(cache_key)
        async with self._lock:
            try:
                deleted = await self._redis.delete(redis_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis LLM cache invalidate failed: %s", exc)
                return False
            self._known_keys.discard(redis_key)
            return bool(deleted)

    async def clear(self) -> int:
        """Очищает все записи с заданным префиксом."""
        removed = 0
        async with self._lock:
            try:
                async for key in self._redis.scan_iter(match=f"{self._prefix}*"):
                    removed += 1
                    await self._redis.delete(key)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Redis LLM cache clear failed: %s", exc)
            self._known_keys.clear()
            self._hits = 0
            self._misses = 0
        return removed

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._known_keys),
            "max_size": 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_seconds": self._ttl,
        }


# === Singleton ===

_LLM_CACHE: LLMCache | RedisLLMCache | None = None


def get_llm_cache() -> LLMCache | RedisLLMCache:
    """Возвращает singleton экземпляр LLM кэша."""
    global _LLM_CACHE
    if _LLM_CACHE is None:
        from app.core.config import get_settings
        settings = get_settings()
        if settings.use_redis_cache:
            try:
                _LLM_CACHE = RedisLLMCache(
                    ttl_seconds=settings.llm_cache_ttl,
                )
                logger.info("LLM cache is using Redis backend")
            except Exception as exc:  # pragma: no cover - fall back to memory
                logger.warning("Falling back to in-memory LLM cache: %s", exc)
                _LLM_CACHE = LLMCache(
                    max_size=512,
                    ttl_seconds=settings.llm_cache_ttl,
                )
        else:
            _LLM_CACHE = LLMCache(
                max_size=512,
                ttl_seconds=settings.llm_cache_ttl,
            )
    return _LLM_CACHE


def reset_llm_cache() -> None:
    """Сбрасывает singleton для тестов."""
    global _LLM_CACHE
    _LLM_CACHE = None


__all__ = ["LLMCache", "RedisLLMCache", "get_llm_cache", "reset_llm_cache"]

