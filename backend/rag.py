from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

import requests

from .config import CONFIG
from .conversation import (
    ChatModelMessage,
    extract_last_user_content,
    messages_to_responses_input,
    replace_system_prompt,
    trim_messages_for_model,
)

FILES_API = "https://rest-assistant.api.cloud.yandex.net/v1"
RESPONSES_API = f"{FILES_API}/responses"

SYSTEM_PROMPT_RAG = (
    "Ты — вежливый и точный AI-консьерж загородного отеля «Усадьба Четыре Сезона» (usadba4.ru). "
    "Отвечай ТОЛЬКО по базе знаний (Vector Store). Если фактов нет — напиши: «Нет данных в базе знаний».\n\n"
    "Правила:\n"
    "• Отвечай кратко и естественно, без выдуманных деталей.\n"
    "• Не используй фразы «как AI-модель» и т.п.\n"
    "• Если вопрос про услугу/объект (ресторан, баня, СПА, конюшня, прокат и т.д.), форматируй так:\n"
    "  • Наличие: Да/Нет.\n"
    "  • Часы работы: <в одной строке>.\n"
    "  • Телефон: <если указан>.\n"
    "  • Ссылка: <если есть>.\n"
    "• Для общих вопросов отвечай фактами из базы, без маркетинга.\n"
    "• Если нет релевантных фактов — «Нет данных в базе знаний»."
)


def _format_http_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except Exception:
        body = resp.text or ""
    body = (body or "").strip().replace("\n", " ")
    return (body[:800] + "…") if len(body) > 800 else (body or "<пустой ответ>")


class YandexClient:
    def __init__(self, config):
        self.config = config
        self._http = requests.Session()

    def _request(
        self,
        method: str,
        url: str,
        *,
        label: str,
        headers: dict[str, str],
        timeout: float,
        **kwargs: Any,
    ) -> requests.Response:
        resp = self._http.request(method, url, headers=headers, timeout=timeout, **kwargs)
        if resp.ok:
            return resp
        raise RuntimeError(f"{label} HTTP {resp.status_code}: {_format_http_error(resp)}")

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.config.yandex_api_key}",
            "x-folder-id": self.config.yandex_folder_id,
            "Content-Type": "application/json",
        }

    def list_vector_files(self) -> list[dict[str, Any]]:
        if not self.config.can_use_vector_store:
            return []
        resp = self._request(
            "GET",
            f"{FILES_API}/vector_stores/{self.config.vector_store_id}/files",
            label="Vector Store list",
            headers=self._json_headers(),
            timeout=self.config.http_timeout,
        )
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, list) else []

    def fetch_vector_meta(self, file_id: str) -> dict[str, Any]:
        resp = self._request(
            "GET",
            f"{FILES_API}/files/{file_id}",
            label="Vector Store file meta",
            headers=self._json_headers(),
            timeout=self.config.http_timeout,
        )
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def fetch_vector_content(self, file_id: str) -> str:
        resp = self._request(
            "GET",
            f"{FILES_API}/files/{file_id}/content",
            label="Vector Store file content",
            headers=self._json_headers(),
            timeout=self.config.http_timeout,
        )
        return resp.text

    def call_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request(
            "POST",
            RESPONSES_API,
            label="Responses API",
            headers=self._json_headers(),
            timeout=self.config.completion_timeout,
            json=payload,
        )
        data = resp.json()
        return data if isinstance(data, dict) else {}


class VectorStoreGateway:
    def __init__(self, client: YandexClient, ttl_seconds: float) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._cached: tuple[float, list[dict[str, Any]]] | None = None
        self._file_cache: dict[str, tuple[float, tuple[dict[str, Any], str]]] = {}

    def list_files(self) -> list[dict[str, Any]]:
        if not self._client.config.can_use_vector_store:
            return []
        now = time.monotonic()
        if self._cached and now - self._cached[0] < self._ttl_seconds:
            return self._cached[1]
        files = self._client.list_vector_files()
        self._cached = (now, files)
        return files

    def fetch_file(self, file_id: str) -> tuple[dict[str, Any], str]:
        if not self._client.config.can_use_vector_store:
            return {}, ""

        now = time.monotonic()
        cached = self._file_cache.get(file_id)
        if cached and now - cached[0] < self._ttl_seconds:
            return cached[1]

        meta = self._client.fetch_vector_meta(file_id)
        content = self._client.fetch_vector_content(file_id)
        stored = (meta if isinstance(meta, dict) else {}, content or "")
        self._file_cache[file_id] = (now, stored)
        return stored


CLIENT = YandexClient(CONFIG)
VECTOR_STORE = VectorStoreGateway(CLIENT, CONFIG.cache_ttl)


def _extract_responses_text(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    merged = "\n".join(text for text in texts if isinstance(text, str) and text.strip())
                    if merged.strip():
                        return merged.strip()

    blocks: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    blocks.append(text.strip())

    return "\n".join(blocks).strip()


def rag_via_responses(
    messages: Sequence[ChatModelMessage], *, client: YandexClient | None = None
) -> str:
    active_client = client or CLIENT
    if not CONFIG.can_use_vector_store:
        raise RuntimeError(
            "Responses API недоступен: не заданы YANDEX_API_KEY/YANDEX_FOLDER_ID/VECTOR_STORE_ID",
        )

    trimmed_messages = trim_messages_for_model(
        messages,
        max_tokens=CONFIG.input_max_tokens,
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": messages_to_responses_input(trimmed_messages),
        "tools": [
            {"type": "file_search"},
            {"type": "web_search"},
        ],
        "tool_resources": {
            "file_search": {
                "vector_store_ids": [CONFIG.vector_store_id],
                "top_k": 8,
            },
            "web_search": {
                "gen_search_options": {
                    "host_filters": ["usadba4.ru"],
                }
            },
        },
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 2000,
    }
    data = active_client.call_responses(payload)

    answer = _extract_responses_text(data)
    return answer or "Нет данных в базе знаний."


TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")


def _extract_keywords(text: str) -> set[str]:
    tokens = [
        token
        for token in TOKEN_SPLIT_RE.split(text.lower())
        if len(token) >= 3
    ]
    return set(tokens)


def _score_line(line: str, *, keywords: set[str], index: int) -> tuple[int, int, str]:
    line_text = line.strip()
    if not line_text:
        return (1, index, "")

    if keywords:
        line_keywords = _extract_keywords(line)
        match_count = len(line_keywords & keywords)
        if match_count == 0:
            return (1, index, "")
        return (-match_count, index, line_text)

    return (0, index, line_text)


def build_context_from_vector_store(
    question: str, *, vector_store: VectorStoreGateway | None = None
) -> str:
    if not CONFIG.can_use_vector_store:
        return "Контекст пуст."

    active_store = vector_store or VECTOR_STORE

    try:
        keywords = _extract_keywords(question)
        snippets: list[str] = []
        total_chars = 0

        for file_info in active_store.list_files():
            file_id = file_info.get("id")
            if not isinstance(file_id, str):
                continue

            meta, content = active_store.fetch_file(file_id)
            lines = content.splitlines()

            scored_hits: list[tuple[int, int, str]] = []
            for index, raw_line in enumerate(lines):
                score = _score_line(raw_line, keywords=keywords, index=index)
                if score[2]:
                    scored_hits.append(score)

            if keywords and not scored_hits:
                for index, raw_line in enumerate(lines):
                    line_text = raw_line.strip()
                    if not line_text:
                        continue
                    scored_hits.append((0, index, line_text))
                    if len(scored_hits) >= CONFIG.context_per_file_limit:
                        break

            if not scored_hits:
                continue

            scored_hits.sort()
            top_hits = [text for _, _, text in scored_hits[: CONFIG.context_per_file_limit]]
            filename = meta.get("filename") if isinstance(meta, dict) else None
            header = f"### {filename or 'file'}"
            block = "\n".join([header, *top_hits])

            snippets.append(block)
            total_chars += len(block)
            if total_chars >= CONFIG.context_max_chars:
                break

        return "\n\n".join(snippets) if snippets else "Контекст пуст."
    except Exception as error:
        print("build_context_from_vector_store ERROR:", error)
        return "Контекст пуст."


def ask_with_vector_store_context(
    messages: Sequence[ChatModelMessage],
    *,
    client: YandexClient | None = None,
    vector_store: VectorStoreGateway | None = None,
) -> str:
    if not CONFIG.has_api_credentials:
        return "Извините, база знаний сейчас недоступна."

    active_client = client or CLIENT
    question = extract_last_user_content(messages)
    context = build_context_from_vector_store(question, vector_store=vector_store or VECTOR_STORE)
    system_prompt = (
        "Ты — русскоязычный AI-консьерж отеля «Усадьба Четыре Сезона».\n"
        "Правила:\n"
        "1) Отвечай кратко, по делу, вежливо.\n"
        "2) Всегда сверяй факты с разделом CONTEXT. Если релевантных сведений нет — напиши: «не нашёл точной информации в базе».\n"
        "3) Не выдумывай цены, расписания и контакты — цитируй точный текст из CONTEXT без дополнительных ссылок и отсылок.\n"
        "4) Формат ответа: 2–3 предложения, без дополнительных списков.\n"
        "5) Используй только русский язык и не давай внешние ссылки, кроме usadba4.ru.\n"
        "Работай строго с данными ниже и не добавляй сведения вне CONTEXT.\n"
        f"CONTEXT:\n{context}"
    )

    prepared_messages = replace_system_prompt(messages, system_prompt)

    trimmed_messages = trim_messages_for_model(
        prepared_messages,
        max_tokens=CONFIG.input_max_tokens,
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": messages_to_responses_input(trimmed_messages),
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 1800,
    }

    try:
        data = active_client.call_responses(payload)
    except Exception as error:
        print("Fallback Responses API error:", error)
        return "Извините, сейчас не могу ответить. Попробуйте позже."

    answer = _extract_responses_text(data)
    return answer or "Нет данных в базе знаний."


__all__ = [
    "ask_with_vector_store_context",
    "build_context_from_vector_store",
    "CLIENT",
    "RESPONSES_API",
    "SYSTEM_PROMPT_RAG",
    "VECTOR_STORE",
    "VectorStoreGateway",
    "YandexClient",
    "rag_via_responses",
]
