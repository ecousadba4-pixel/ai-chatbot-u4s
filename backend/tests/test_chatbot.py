import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.redis_gateway import REDIS_MAX_MESSAGES, parse_redis_args


class DummyClient:
    def __init__(self, config):
        self.config = config
        self.calls: list[dict] = []

    def call_responses(self, payload: dict):
        self.calls.append(payload)
        return {
            "output": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Ответ"},
                    ],
                }
            ]
        }

    # Vector Store helpers (not used in this test scenario)
    def list_vector_files(self):  # pragma: no cover
        return []

    def fetch_vector_meta(self, file_id: str):  # pragma: no cover
        return {}

    def fetch_vector_content(self, file_id: str):  # pragma: no cover
        return ""


class DummyRedisGateway:
    def __init__(self, *, max_messages: int = REDIS_MAX_MESSAGES):
        self.max_messages = max_messages
        self.storage: dict[str, list[dict]] = {}

    def read_history(self, session_id: str) -> list[dict]:
        return [dict(item) for item in self.storage.get(session_id, [])]

    def write_history(self, session_id: str, messages, ttl: int | None = None) -> None:
        limited = [dict(item) for item in messages][-self.max_messages :]
        self.storage[session_id] = limited

    def delete_history(self, session_id: str) -> None:
        self.storage.pop(session_id, None)


class DummyRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload

    async def body(self):
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture()
def app_module(monkeypatch):
    module_name = "backend.app"
    for key in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "VECTOR_STORE_ID"):
        os.environ[key] = f"test-{key.lower()}"

    if module_name in sys.modules:
        del sys.modules[module_name]

    app_mod = importlib.import_module(module_name)
    importlib.reload(app_mod)

    dummy_client = DummyClient(app_mod.CONFIG)
    monkeypatch.setattr(app_mod, "CLIENT", dummy_client)

    return app_mod


def test_rag_payload_uses_vector_store(app_module):
    client: DummyClient = app_module.CLIENT  # type: ignore[assignment]

    messages = [
        {"role": "system", "content": app_module.SYSTEM_PROMPT_RAG},
        {"role": "user", "content": "Расскажи про ресторан"},
    ]
    answer = app_module.rag_via_responses(messages)
    assert answer == "Ответ"

    assert len(client.calls) == 1
    payload = client.calls[0]

    assert payload["tool_resources"]["file_search"]["vector_store_ids"] == [
        app_module.CONFIG.vector_store_id
    ]
    assert pytest.approx(payload["temperature"]) == 0.3
    assert pytest.approx(payload["top_p"]) == 0.8
    assert payload["max_output_tokens"] >= 1500


def test_vector_store_fallback_handles_api_errors(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "build_context_from_vector_store",
        lambda question: "Контекст пуст.",
    )

    def _failing_call_responses(payload):
        raise RuntimeError("Responses API HTTP 500: boom")

    monkeypatch.setattr(app_module.CLIENT, "call_responses", _failing_call_responses)

    messages = [
        {"role": "system", "content": app_module.SYSTEM_PROMPT_RAG},
        {"role": "user", "content": "Привет"},
    ]
    answer = app_module.ask_with_vector_store_context(messages)
    assert answer == "Извините, сейчас не могу ответить. Попробуйте позже."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", {}),
        (None, {}),
        ("{\"password\": \"secret\", \"ssl\": true}", {"password": "secret", "ssl": True}),
        ("password=secret&db=1", {"password": "secret", "db": 1}),
        ("password=secret db=2", {"password": "secret", "db": 2}),
    ],
)
def test_parse_redis_args_handles_multiple_formats(raw, expected):
    assert parse_redis_args(raw) == expected


def test_chat_post_merges_and_persists_history(app_module, monkeypatch):
    redis_gateway = DummyRedisGateway(max_messages=4)
    redis_gateway.storage["abc"] = [
        {"role": "user", "content": "Привет", "timestamp": 10.0},
        {"role": "assistant", "content": "Здравствуйте", "timestamp": 11.0},
    ]
    monkeypatch.setattr(app_module, "REDIS_GATEWAY", redis_gateway)

    payload = {
        "sessionId": "abc",
        "history": [
            {"role": "user", "content": "Как дела?", "timestamp": 12.0},
            {"role": "assistant", "content": "Все отлично", "timestamp": 13.0},
        ],
        "question": "Расскажи про ресторан",
    }

    response = asyncio.run(app_module.chat_post(DummyRequest(payload)))
    assert response["answer"] == "Ответ"

    history = redis_gateway.storage["abc"]
    assert [msg["role"] for msg in history] == ["user", "assistant", "user", "assistant"]
    assert history[-2]["content"] == app_module.normalize_question(payload["question"])
    assert history[-1]["content"] == "Ответ"
    assert history[-1]["timestamp"] >= history[-2]["timestamp"]


def test_chat_reset_endpoint_clears_history(app_module, monkeypatch):
    redis_gateway = DummyRedisGateway()
    redis_gateway.storage["session"] = [
        {"role": "user", "content": "Привет", "timestamp": time.time()},
    ]
    monkeypatch.setattr(app_module, "REDIS_GATEWAY", redis_gateway)

    response = asyncio.run(app_module.chat_reset(DummyRequest({"sessionId": "session"})))
    assert response == {"ok": True}
    assert "session" not in redis_gateway.storage
