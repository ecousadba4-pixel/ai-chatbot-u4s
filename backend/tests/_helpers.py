import importlib
import json
import os
import sys
from pathlib import Path

from backend.redis_gateway import REDIS_MAX_MESSAGES


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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

    def list_vector_files(self):  # pragma: no cover - не используется
        return []

    def fetch_vector_meta(self, file_id: str):  # pragma: no cover - не используется
        return {}

    def fetch_vector_content(self, file_id: str):  # pragma: no cover - не используется
        return ""


class DummyRedisGateway:
    def __init__(self, *, max_messages: int = REDIS_MAX_MESSAGES):
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


def load_app_module(monkeypatch):
    module_name = "backend.app"
    for key in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID", "VECTOR_STORE_ID"):
        os.environ[key] = f"test-{key.lower()}"

    for dependency in [module_name, "backend.config", "backend.rag"]:
        if dependency in sys.modules:
            del sys.modules[dependency]

    app_mod = importlib.import_module(module_name)
    importlib.reload(app_mod)

    dummy_client = DummyClient(app_mod.CONFIG)
    monkeypatch.setattr(app_mod, "CLIENT", dummy_client)

    return app_mod
