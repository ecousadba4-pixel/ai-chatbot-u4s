import importlib
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.storage import MAX_HISTORY_MESSAGES


class DummyClient:
    def __init__(self, config):
        self.config = config
        self.calls: list[dict] = []

    def call_chat(self, payload: dict):
        self.calls.append(payload)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Ответ"},
                }
            ]
        }


class DummyStorage:
    def __init__(self, *, max_messages: int = MAX_HISTORY_MESSAGES):
        self.max_messages = max_messages
        self.storage: dict[str, list[dict]] = {}
        self.context_storage: dict[str, dict] = {}

    def read_history(self, session_id: str) -> list[dict]:
        return [dict(item) for item in self.storage.get(session_id, [])]

    def write_history(self, session_id: str, messages, ttl: int | None = None) -> None:
        limited = [dict(item) for item in messages][-self.max_messages :]
        self.storage[session_id] = limited

    def delete_history(self, session_id: str) -> None:
        self.storage.pop(session_id, None)

    def read_context(self, session_id: str) -> dict:
        return dict(self.context_storage.get(session_id, {}))

    def write_context(self, session_id: str, context: dict, ttl: int | None = None) -> None:
        self.context_storage[session_id] = dict(context or {})

    def delete_context(self, session_id: str) -> None:
        self.context_storage.pop(session_id, None)


class DummyRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload

    async def body(self):  # pragma: no cover - используется в других тестах
        return json.dumps(self._payload).encode("utf-8")


# NOTE: Эта функция и связанные тесты используют устаревший API.
# Legacy модули backend.config, backend.rag, backend.conversation удалены.
# Тесты test_chatbot.py и test_booking_dialogue.py требуют обновления для нового API.
def load_app_module(monkeypatch):
    module_name = "backend.app"
    for key in ("AMVERA_API_TOKEN", "AMVERA_API_URL", "AMVERA_MODEL"):
        os.environ[key] = f"test-{key.lower()}"

    # Удалены ссылки на backend.config и backend.rag (legacy модули)
    if module_name in sys.modules:
        del sys.modules[module_name]

    app_mod = importlib.import_module(module_name)
    importlib.reload(app_mod)

    # Эти атрибуты больше не существуют в новом API
    # dummy_client = DummyClient(app_mod.CONFIG)
    # monkeypatch.setattr(app_mod, "CLIENT", dummy_client)

    return app_mod
