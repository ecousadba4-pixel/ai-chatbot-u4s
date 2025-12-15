from __future__ import annotations

from typing import Any, Sequence

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings


class DeepSeekClient:
    """Клиент для DeepSeek API."""

    def __init__(self, *, api_key: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.deepseek_api_key
        self._timeout = timeout or settings.completion_timeout
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, *, model: str, messages: Sequence[dict[str, str]]) -> str:
        settings = get_settings()
        if settings.llm_dry_run:
            return "[LLM отключён: режим dry-run]"

        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": model, "messages": list(messages)}
        url = "https://api.deepseek.com/chat/completions"

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                response = await self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = self._extract_text(data)
                if content:
                    return content
        return ""

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
        return ""


__all__ = ["DeepSeekClient"]
