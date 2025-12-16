from __future__ import annotations

from typing import Any, Iterable

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings


class QdrantClient:
    """Минимальный клиент Qdrant для поиска ближайших точек."""

    def __init__(self, *, base_url: str | None = None, timeout: float = 10.0) -> None:
        settings = get_settings()
        self._base_url = base_url or str(settings.qdrant_url).rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        collection: str,
        vector: Iterable[float],
        limit: int = 5,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        url = f"{self._base_url}/collections/{collection}/points/search"
        payload: dict[str, Any] = {
            "vector": list(vector),
            "limit": limit,
            "with_payload": True,
        }
        if query_filter:
            payload["filter"] = query_filter

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    return []
                result = data.get("result") or []
                if isinstance(result, list):
                    return [item for item in result if isinstance(item, dict)]
                return []
        return []

    async def scroll(
        self,
        *,
        collection: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        url = f"{self._base_url}/collections/{collection}/points/scroll"
        payload: dict[str, Any] = {"limit": limit, "with_payload": True}

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    return []
                result = data.get("result")
                if isinstance(result, dict):
                    points = result.get("points") or []
                    if isinstance(points, list):
                        return [item for item in points if isinstance(item, dict)]
                return []
        return []


_CLIENT: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QdrantClient()
    return _CLIENT


__all__ = ["QdrantClient", "get_qdrant_client"]
