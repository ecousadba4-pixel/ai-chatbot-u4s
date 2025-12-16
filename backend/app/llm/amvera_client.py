from __future__ import annotations

from typing import Any, Sequence

import httpx
from fastapi import HTTPException
import logging
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings


class AmveraLLMClient:
    """Клиент для Amvera API c проксированным доступом к DeepSeek."""

    def __init__(
        self,
        *,
        api_token: str | None = None,
        api_url: str | None = None,
        model: str | None = None,
        inference_name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._api_token = api_token or settings.amvera_api_token
        self._api_url = (api_url or str(settings.amvera_api_url)).rstrip("/")
        self._model = model or settings.amvera_model
        self._inference_name = inference_name or settings.amvera_inference_name
        self._timeout = timeout or settings.llm_timeout or settings.completion_timeout
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        http_timeout = httpx.Timeout(
            connect=settings.request_timeout,
            read=self._timeout,
            write=settings.request_timeout,
            pool=settings.request_timeout,
        )
        self._client = httpx.AsyncClient(timeout=http_timeout)
        self._logger = logging.getLogger(__name__)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, *, model: str | None = None, messages: Sequence[dict[str, str]]) -> str:
        settings = get_settings()
        if settings.llm_dry_run:
            return "[LLM отключён: режим dry-run]"

        headers = {"X-Auth-Token": f"Bearer {self._api_token}"}
        payload = {
            "model": model or self._model,
            "messages": self._format_messages(messages),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        url = f"{self._api_url}/models/{self._inference_name}"

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                try:
                    response = await self._client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    content = self._extract_text(data)
                    if content:
                        return content
                except httpx.HTTPStatusError as exc:  # type: ignore[misc]
                    self._logger.error(
                        "Amvera request failed: status=%s, body=%s",
                        exc.response.status_code if exc.response else "unknown",
                        exc.response.text if exc.response else "",
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"LLM provider error: HTTP {exc.response.status_code if exc.response else 'unknown'}",
                    ) from exc
                except httpx.HTTPError as exc:
                    self._logger.error("Amvera request failed: %s", exc)
                    raise HTTPException(status_code=502, detail="LLM provider unreachable") from exc
                except ValueError as exc:
                    self._logger.error("Amvera response parsing failed: %s", exc)
                    raise HTTPException(status_code=502, detail="LLM provider invalid response") from exc
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
                    content = message.get("content") or message.get("text")
                    if isinstance(content, str):
                        return content.strip()
        return ""

    @staticmethod
    def _format_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        formatted: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            if not role:
                continue
            content = message.get("content") or message.get("text") or ""
            formatted.append({"role": role, "text": content})
        return formatted


__all__ = ["AmveraLLMClient"]
