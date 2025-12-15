import asyncio
import time

import pytest

from backend.tests._helpers import DummyClient, DummyRequest, DummyStorage


def test_rag_payload_uses_deepseek(app_module):
    client: DummyClient = app_module.CLIENT  # type: ignore[assignment]

    messages = [
        {"role": "system", "content": app_module.SYSTEM_PROMPT_RAG},
        {"role": "user", "content": "Расскажи про ресторан"},
    ]
    answer = app_module.rag_via_responses(messages)
    assert answer == "Ответ"

    assert len(client.calls) == 1
    payload = client.calls[0]

    assert payload["model"] == app_module.CONFIG.amvera_model
    assert pytest.approx(payload["temperature"]) == 0.3
    assert pytest.approx(payload["top_p"]) == 0.8


def test_vector_store_fallback_handles_api_errors(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "build_context_from_vector_store",
        lambda question: "Контекст пуст.",
    )

    def _failing_call_responses(payload):
        raise RuntimeError("Chat API HTTP 500: boom")

    monkeypatch.setattr(app_module.CLIENT, "call_chat", _failing_call_responses)

    messages = [
        {"role": "system", "content": app_module.SYSTEM_PROMPT_RAG},
        {"role": "user", "content": "Привет"},
    ]
    answer = app_module.ask_with_vector_store_context(messages)
    assert answer == "Извините, сейчас не могу ответить. Попробуйте позже."


def test_chat_post_merges_and_persists_history(app_module, monkeypatch):
    storage = DummyStorage(max_messages=4)
    storage.storage["abc"] = [
        {"role": "user", "content": "Привет", "timestamp": 10.0},
        {"role": "assistant", "content": "Здравствуйте", "timestamp": 11.0},
    ]
    monkeypatch.setattr(app_module, "HISTORY_STORAGE", storage)

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

    history = storage.storage["abc"]
    assert [msg["role"] for msg in history] == ["user", "assistant", "user", "assistant"]
    assert history[-2]["content"] == app_module.normalize_question(payload["question"])
    assert history[-1]["content"] == "Ответ"
    assert history[-1]["timestamp"] >= history[-2]["timestamp"]


def test_chat_post_returns_hint_for_empty_question(app_module, monkeypatch):
    storage = DummyStorage()
    monkeypatch.setattr(app_module, "HISTORY_STORAGE", storage)

    payload = {"sessionId": "abc", "question": "  \t\n"}

    response = asyncio.run(app_module.chat_post(DummyRequest(payload)))

    assert response["answer"] == app_module.EMPTY_QUESTION_ANSWER
    assert "abc" not in storage.storage
    assert app_module.CLIENT.calls == []


def test_chat_reset_endpoint_clears_history(app_module, monkeypatch):
    storage = DummyStorage()
    storage.storage["session"] = [
        {"role": "user", "content": "Привет", "timestamp": time.time()},
    ]
    monkeypatch.setattr(app_module, "HISTORY_STORAGE", storage)

    response = asyncio.run(app_module.chat_reset(DummyRequest({"sessionId": "session"})))
    assert response == {"ok": True}
    assert "session" not in storage.storage
