import importlib
import os
import sys

import pytest


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

    answer = app_module.rag_via_responses("Расскажи про ресторан")
    assert answer == "Ответ"

    assert len(client.calls) == 1
    payload = client.calls[0]

    assert payload["tool_resources"]["file_search"]["vector_store_ids"] == [
        app_module.CONFIG.vector_store_id
    ]
    assert pytest.approx(payload["temperature"]) == 0.3
    assert pytest.approx(payload["top_p"]) == 0.8
    assert payload["max_output_tokens"] >= 1500
