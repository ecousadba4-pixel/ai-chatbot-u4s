from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.amvera_client import AmveraLLMClient  # noqa: E402


def _get_prompt() -> str:
    return os.environ.get("AMVERA_TEST_PROMPT", "Привет! Расскажи что-нибудь интересное.")


async def main() -> None:
    client = AmveraLLMClient()
    try:
        response = await client.chat(messages=[{"role": "user", "text": _get_prompt()}])
        print("Ответ Amvera:")
        print(response)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
