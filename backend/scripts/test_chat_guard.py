"""Простой скрипт для ручной проверки RAG guard на /v1/chat."""

import asyncio
import os
import sys

import httpx


def _response_has_hallucinations(answer: str) -> bool:
    banned = ("домик", "vip", "сауна", "баня", "коттедж")
    return any(word.lower() in answer.lower() for word in banned)


async def main() -> None:
    base_url = os.getenv("CHAT_BASE_URL", "http://127.0.0.1:8104")
    message = os.getenv("CHAT_MESSAGE", "Подскажи варианты размещения")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/v1/chat",
            json={"message": message, "sessionId": ""},
            headers={"Content-Type": "application/json"},
        )

    try:
        payload = resp.json()
    except Exception:
        print("Не удалось распарсить ответ:", resp.text)
        sys.exit(1)

    answer = payload.get("answer", "")
    debug = payload.get("debug", {})

    print("Ответ:", answer)
    print("Debug:", debug)

    hits_total = debug.get("hits_total", 0)
    if hits_total == 0:
        assert not _response_has_hallucinations(answer), "Обнаружены выдуманные варианты при пустой базе"
        assert debug.get("guard_triggered") is True, "Guard должен сработать при hits_total=0"
        assert debug.get("llm_called") is False, "LLM не должен вызываться при hits_total=0"


if __name__ == "__main__":
    asyncio.run(main())
