import asyncio
import time
from typing import Any

import pytest

import backend.redis_gateway as redis_gateway_module

from backend.tests._helpers import DummyClient, DummyRedisGateway, DummyRequest


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
    assert redis_gateway_module.parse_redis_args(raw) == expected


def test_create_redis_client_adds_default_scheme(monkeypatch):
    captured: dict[str, Any] = {}

    class DummyRedis:
        @staticmethod
        def from_url(url: str, **kwargs: Any) -> str:
            captured["url"] = url
            captured["kwargs"] = kwargs
            return "client"

    dummy_module = type("DummyRedisModule", (), {"Redis": DummyRedis})
    monkeypatch.setattr(redis_gateway_module, "redis", dummy_module)

    client = redis_gateway_module.create_redis_client("internal-redis:6379", {"db": 2})

    assert client == "client"
    assert captured["url"] == "redis://internal-redis:6379"
    assert captured["kwargs"]["db"] == 2
    assert captured["kwargs"]["decode_responses"] is True


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
