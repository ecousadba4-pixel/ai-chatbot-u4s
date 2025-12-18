"""
Redis-based хранилище состояния диалога с поддержкой истории сообщений.

Обеспечивает персистентность при перезапуске сервера и поддержку
контекста диалога для LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from app.core.config import get_settings
from app.booking.slot_filling import SlotState
from app.session.redis_client import close_redis_client, get_redis_client

logger = logging.getLogger(__name__)


class RedisConversationStateStore:
    """
    Персистентное хранилище состояния диалога в Redis.
    
    Хранит:
    - Состояние бронирования (BookingContext/SlotState)
    - Историю сообщений для передачи в LLM
    """

    state_prefix = "u4s:booking_state:"
    history_prefix = "u4s:history:"

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 86400,
        max_history: int = 10,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_history = max_history

    # === Состояние бронирования ===

    def get(self, session_id: str) -> dict[str, Any] | SlotState | None:
        """
        Синхронный интерфейс для совместимости с ConversationStateStore.
        
        ВАЖНО: Этот метод НЕ должен использоваться из async кода!
        Используйте get_async() напрямую в async контексте.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
            # Если мы в async контексте - это ошибка использования!
            logger.error(
                "RedisConversationStateStore.get() called from async context! "
                "Use get_async() instead. session_id=%s",
                session_id
            )
            return None
        except RuntimeError:
            # Нет running loop - можем безопасно создать новый
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.get_async(session_id))
            finally:
                loop.close()

    def set(self, session_id: str, state: dict[str, Any] | SlotState) -> None:
        """
        Синхронный интерфейс для совместимости с ConversationStateStore.
        
        ВАЖНО: Этот метод НЕ должен использоваться из async кода!
        Используйте set_async() напрямую в async контексте.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
            # Если мы в async контексте - это ошибка использования!
            logger.error(
                "RedisConversationStateStore.set() called from async context! "
                "Use set_async() instead. session_id=%s",
                session_id
            )
            return
        except RuntimeError:
            # Нет running loop - можем безопасно создать новый
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.set_async(session_id, state))
            finally:
                loop.close()

    def clear(self, session_id: str) -> None:
        """
        Синхронный интерфейс для совместимости с ConversationStateStore.
        
        ВАЖНО: Этот метод НЕ должен использоваться из async кода!
        Используйте clear_async() напрямую в async контексте.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
            # Если мы в async контексте - это ошибка использования!
            logger.error(
                "RedisConversationStateStore.clear() called from async context! "
                "Use clear_async() instead. session_id=%s",
                session_id
            )
            return
        except RuntimeError:
            # Нет running loop - можем безопасно создать новый
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.clear_async(session_id))
            finally:
                loop.close()

    async def get_async(self, session_id: str) -> dict[str, Any] | None:
        """Асинхронное получение состояния."""
        key = f"{self.state_prefix}{session_id}"
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            decoded = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            return json.loads(decoded)
        except Exception as exc:
            logger.warning("Failed to get state from Redis: %s", exc)
            return None

    async def set_async(self, session_id: str, state: dict[str, Any] | SlotState) -> None:
        """Асинхронное сохранение состояния."""
        key = f"{self.state_prefix}{session_id}"
        try:
            if hasattr(state, "as_dict"):
                data = state.as_dict()
            elif hasattr(state, "to_dict"):
                data = state.to_dict()
            else:
                data = state
            payload = json.dumps(data, ensure_ascii=False)
            await self._redis.setex(key, self._ttl, payload)
        except Exception as exc:
            logger.error("Failed to set state in Redis: %s", exc)

    async def clear_async(self, session_id: str) -> None:
        """Асинхронная очистка состояния."""
        key = f"{self.state_prefix}{session_id}"
        try:
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Failed to clear state in Redis: %s", exc)

    # === История сообщений ===

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Получает историю сообщений для передачи в LLM.
        
        Возвращает список сообщений в формате:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        key = f"{self.history_prefix}{session_id}"
        try:
            data = await self._redis.lrange(key, 0, self._max_history * 2 - 1)
            if not data:
                return []
            
            messages = []
            for item in reversed(data):  # Redis LPUSH добавляет в начало, разворачиваем
                decoded = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
                messages.append(json.loads(decoded))
            return messages
        except Exception as exc:
            logger.warning("Failed to get history from Redis: %s", exc)
            return []

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Добавляет сообщение в историю диалога.
        
        Args:
            session_id: ID сессии
            role: "user" или "assistant"
            content: Текст сообщения
        """
        key = f"{self.history_prefix}{session_id}"
        try:
            message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            await self._redis.lpush(key, message)
            # Обрезаем историю до max_history * 2 (user + assistant пары)
            await self._redis.ltrim(key, 0, self._max_history * 2 - 1)
            await self._redis.expire(key, self._ttl)
        except Exception as exc:
            logger.warning("Failed to add message to history: %s", exc)

    async def clear_history(self, session_id: str) -> None:
        """Очищает историю диалога."""
        key = f"{self.history_prefix}{session_id}"
        try:
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Failed to clear history: %s", exc)

    async def ping(self) -> bool:
        """Проверка доступности Redis."""
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False


# === Singleton ===

_REDIS_STATE_STORE: RedisConversationStateStore | None = None


def get_redis_state_store() -> RedisConversationStateStore:
    """Возвращает singleton экземпляр Redis state store."""
    global _REDIS_STATE_STORE
    if _REDIS_STATE_STORE is None:
        settings = get_settings()
        client = get_redis_client()
        _REDIS_STATE_STORE = RedisConversationStateStore(
            client,
            ttl_seconds=settings.session_ttl_seconds,
            max_history=settings.conversation_history_limit,
        )
    return _REDIS_STATE_STORE


async def close_redis_state_store() -> None:
    """Закрывает соединение с Redis."""
    global _REDIS_STATE_STORE
    if _REDIS_STATE_STORE is not None:
        try:
            await close_redis_client()
        except Exception as exc:
            logger.warning("Failed to close Redis connection: %s", exc)
        _REDIS_STATE_STORE = None


__all__ = [
    "RedisConversationStateStore",
    "get_redis_state_store",
    "close_redis_state_store",
]

