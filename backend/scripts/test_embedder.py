"""Smoke-тест для внешнего embedder."""

from __future__ import annotations

import asyncio
import json

import httpx


EMBED_URL = "http://127.0.0.1:8011/embed"


async def main() -> None:
    payload = {"texts": ["тест"]}
    timeout = httpx.Timeout(connect=2.0, read=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(EMBED_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        print("[embedder] error:", exc)
        return

    vectors = data.get("vectors") if isinstance(data, dict) else None
    vector_count = len(vectors) if isinstance(vectors, list) else 0
    first_vector = vectors[0] if vector_count else []
    first_len = len(first_vector) if isinstance(first_vector, list) else 0
    first_numbers = []
    if isinstance(first_vector, list):
        first_numbers = [float(x) for x in first_vector[:5]]

    print("model:", data.get("model"))
    print("dim:", data.get("dim"))
    print("vectors:", vector_count)
    print("vector_len:", first_len)
    print("first_numbers:", json.dumps(first_numbers, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
