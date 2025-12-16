"""Простой smoke-тест для проверки RAG и embeddings."""

from __future__ import annotations

import asyncio
import json

import httpx

from app.core.config import get_settings


async def test_embeddings(client: httpx.AsyncClient, settings) -> None:
    try:
        response = await client.post(str(settings.embed_url), json={"inputs": ["ping"]})
        print("[embed] status:", response.status_code)
        print("[embed] body:", json.dumps(response.json(), ensure_ascii=False)[:500])
    except Exception as exc:  # noqa: BLE001
        print("[embed] error:", exc)


async def test_knowledge(client: httpx.AsyncClient, settings, base_url: str) -> None:
    url = f"{base_url}{settings.api_prefix}/knowledge"
    payload = {"query": "варианты размещения", "limit": 5}
    try:
        response = await client.post(url, json=payload)
        print("[knowledge] status:", response.status_code)
        print("[knowledge] body:", json.dumps(response.json(), ensure_ascii=False)[:500])
    except Exception as exc:  # noqa: BLE001
        print("[knowledge] error:", exc)


async def test_chat(client: httpx.AsyncClient, settings, base_url: str) -> None:
    url = f"{base_url}{settings.api_prefix}/chat"
    payload = {"message": "Расскажи про доступные домики", "sessionId": "smoke"}
    try:
        response = await client.post(url, json=payload)
        print("[chat] status:", response.status_code)
        print("[chat] body:", json.dumps(response.json(), ensure_ascii=False)[:500])
    except Exception as exc:  # noqa: BLE001
        print("[chat] error:", exc)


async def main() -> None:
    settings = get_settings()
    base_url = "http://127.0.0.1:8000"

    timeout = httpx.Timeout(connect=settings.request_timeout, read=settings.request_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await test_embeddings(client, settings)
        await test_knowledge(client, settings, base_url)
        await test_chat(client, settings, base_url)


if __name__ == "__main__":
    asyncio.run(main())
