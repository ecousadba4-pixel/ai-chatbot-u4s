"""Smoke-тест для эндпоинта поиска по базе знаний."""

from __future__ import annotations

import asyncio
import json
import os

import httpx


BASE_URL = os.getenv("KNOWLEDGE_URL", "http://127.0.0.1:8104")


async def main() -> None:
    url = f"{BASE_URL}/v1/knowledge"
    payload = {"query": "тест", "limit": 5}
    timeout = httpx.Timeout(connect=5.0, read=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        print("[knowledge] error:", exc)
        return

    debug = data.get("debug", {}) if isinstance(data, dict) else {}
    results = data.get("results", []) if isinstance(data, dict) else []

    print("hits_total:", debug.get("hits_total"))
    if debug:
        print("debug:", json.dumps(debug, ensure_ascii=False))

    for item in results[:5]:
        if not isinstance(item, dict):
            continue
        print(
            "-",
            item.get("type"),
            "|",
            item.get("title"),
            "| score=",
            item.get("score"),
        )


if __name__ == "__main__":
    asyncio.run(main())
