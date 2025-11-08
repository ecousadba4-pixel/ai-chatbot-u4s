from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

FILES_API = "https://rest-assistant.api.cloud.yandex.net/v1"
RESPONSES_API = f"{FILES_API}/responses"
COMPLETIONS_API = "https://llm.api.cloud.yandex.net/v1/chat/completions"


# ========================
#  Конфигурация приложения
# ========================

def _strip(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _iter_plain_origins(raw: str) -> Iterable[str]:
    for part in re.split(r"[,\s]+", raw):
        part = part.strip().strip('"').strip("'")
        if part:
            yield part


def parse_allowed_origins(raw: str | None) -> tuple[str, ...]:
    """Преобразует ALLOWED_ORIGINS в кортеж доменов."""

    if not raw:
        return ("*",)

    raw = raw.strip()
    if raw == "*":
        return ("*",)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        parsed = parsed.strip()
        return (parsed,) if parsed else ("*",)

    if isinstance(parsed, (list, tuple, set)):
        values = tuple(str(item).strip() for item in parsed if str(item).strip())
        return values or ("*",)

    values = tuple(_iter_plain_origins(raw))
    return values or ("*",)


@dataclass(frozen=True)
class AppConfig:
    yandex_api_key: str
    yandex_folder_id: str
    vector_store_id: str
    allowed_origins: tuple[str, ...]
    http_timeout: float = 30.0
    completion_timeout: float = 60.0
    context_max_chars: int = 2500
    context_per_file_limit: int = 12
    context_query_limit: int = 24
    context_snippet_max_chars: int = 500
    cache_ttl: float = 180.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            yandex_api_key=_strip(os.environ.get("YANDEX_API_KEY")),
            yandex_folder_id=_strip(os.environ.get("YANDEX_FOLDER_ID")),
            vector_store_id=_strip(os.environ.get("VECTOR_STORE_ID")),
            allowed_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "*")),
        )

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)

    @property
    def can_use_vector_store(self) -> bool:
        return bool(self.has_api_credentials and self.vector_store_id)


CONFIG = AppConfig.from_env()


# ========================
#  HTTP-клиент Yandex Cloud
# ========================


def _format_http_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except Exception:
        body = resp.text or ""
    body = (body or "").strip().replace("\n", " ")
    return (body[:800] + "…") if len(body) > 800 else (body or "<пустой ответ>")


class YandexClient:
    def __init__(self, config: AppConfig) -> None:
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

    def _openai_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.yandex_api_key}",
            "OpenAI-Project": self.config.yandex_folder_id,
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

    def call_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request(
            "POST",
            COMPLETIONS_API,
            label="Completions API",
            headers=self._openai_headers(),
            timeout=self.config.completion_timeout,
            json=payload,
        )
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def query_vector_store(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        if not self.config.can_use_vector_store:
            return []

        payload: dict[str, Any] = {
            "query": query,
            "top_k": limit,
            "limit": limit,
            "return_metadata": True,
        }

        resp = self._request(
            "POST",
            f"{FILES_API}/vector_stores/{self.config.vector_store_id}:query",
            label="Vector Store query",
            headers=self._json_headers(),
            timeout=self.config.http_timeout,
            json=payload,
        )

        try:
            data = resp.json()
        except Exception:
            return []

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        if isinstance(data, dict):
            for key in ("data", "results", "matches", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]

        return []


CLIENT = YandexClient(CONFIG)


class VectorStoreGateway:
    def __init__(self, client: YandexClient, ttl_seconds: float) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._cached: tuple[float, list[dict[str, Any]]] | None = None

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
        meta = self._client.fetch_vector_meta(file_id)
        content = self._client.fetch_vector_content(file_id)
        return meta, content

    def query(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        return self._client.query_vector_store(query, limit=limit)


VECTOR_STORE = VectorStoreGateway(CLIENT, ttl_seconds=CONFIG.cache_ttl)


# ========================
#  Нормализация вопроса
# ========================

def normalize_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "Есть ли в усадьбе ресторан?"

    lower_question = question.lower()
    if "ресторан" in lower_question and not any(
        key in lower_question for key in ("час", "время", "работ", "телефон", "ссылка", "как забронировать")
    ):
        return "Есть ли в усадьбе ресторан «Калина Красная»? Укажи часы, телефон и ссылку."

    return question


# ========================
#  Основная логика: RAG
# ========================

SYSTEM_PROMPT_RAG = (
    "Ты — помощник сайта usadba4.ru. Работай ТОЛЬКО по базе знаний (Vector Store).\n"
    "Если вопрос о наличии/расписании/контактах услуги — отвечай по шаблону:\n"
    "• Да/Нет.\n"
    "• Часы: <в одной строке>.\n"
    "• Телефон: <если есть>.\n"
    "• Ссылка: <если есть>.\n"
    "Не выводи детали, не относящиеся напрямую к вопросу. "
    "Если фактов нет — ответ: \"Нет данных в базе знаний\"."
)


def rag_via_responses(question: str) -> str:
    if not CONFIG.can_use_vector_store:
        raise RuntimeError(
            "Responses API недоступен: не заданы YANDEX_API_KEY/YANDEX_FOLDER_ID/VECTOR_STORE_ID",
        )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT_RAG}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": question}]},
        ],
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [CONFIG.vector_store_id]}},
        "temperature": 0.0,
        "max_output_tokens": 600,
    }
    data = CLIENT.call_responses(payload)

    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()

    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    blocks: list[str] = []
    if isinstance(data, dict):
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    blocks.append(content["text"])

    return ("\n".join(blocks)).strip() if blocks else "Нет данных в базе знаний."


# ========================
#  Фолбэк: контекст из Vector Store
# ========================

TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")


def _extract_keywords(text: str) -> set[str]:
    tokens = [
        token
        for token in TOKEN_SPLIT_RE.split((text or "").lower())
        if len(token) >= 3
    ]
    return set(tokens)


def _collect_question_keywords(text: str) -> set[str]:
    return _extract_keywords(text)


def _collect_texts(value: Any, acc: list[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            acc.append(text)
        return

    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            _collect_texts(value["text"], acc)
        for key in ("content", "data", "value", "parts", "messages", "snippet"):
            if key in value:
                _collect_texts(value[key], acc)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_texts(item, acc)


def _extract_result_text(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""

    for key in ("content", "text", "snippet", "chunk", "passage", "value"):
        if key in item:
            parts: list[str] = []
            _collect_texts(item[key], parts)
            if parts:
                return "\n".join(parts).strip()

    document = item.get("document")
    if isinstance(document, dict):
        text = _extract_result_text(document)
        if text:
            return text

    return ""


def _extract_result_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not isinstance(item, dict):
        return metadata

    raw_metadata = item.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)

    document = item.get("document")
    if isinstance(document, dict):
        doc_meta = document.get("metadata")
        if isinstance(doc_meta, dict):
            metadata = {**doc_meta, **metadata}

    return metadata


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return 0.0
    return 0.0


def _extract_timestamp(metadata: dict[str, Any]) -> float:
    for key in ("updated_at", "timestamp", "modified_at", "created_at", "time"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                continue
            try:
                normalized = raw.replace("Z", "+00:00")
                return datetime.fromisoformat(normalized).timestamp()
            except Exception:
                try:
                    return float(raw)
                except ValueError:
                    continue
    return 0.0


def _metadata_keyword_overlap(metadata: dict[str, Any], keywords: set[str]) -> int:
    if not metadata or not keywords:
        return 0

    parts: list[str] = []
    for key in ("type", "entity", "title", "tags", "category", "label"):
        value = metadata.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)

    if not parts:
        return 0

    meta_keywords = _extract_keywords(" ".join(parts))
    return len(meta_keywords & keywords)


def _format_context_block(metadata: dict[str, Any], text: str) -> str:
    title_parts: list[str] = []
    for key in ("entity", "title", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            title_parts.append(value.strip())

    if not title_parts:
        filename = metadata.get("filename")
        if isinstance(filename, str) and filename.strip():
            title_parts.append(filename.strip())

    title = " — ".join(dict.fromkeys(title_parts)) if title_parts else ""
    header = f"### {title or 'Фрагмент'}"

    info_parts: list[str] = []
    meta_type = metadata.get("type")
    if isinstance(meta_type, str) and meta_type.strip():
        info_parts.append(f"Тип: {meta_type.strip()}")

    entity = metadata.get("entity")
    if isinstance(entity, str) and entity.strip() and entity.strip() not in title_parts:
        info_parts.append(f"Объект: {entity.strip()}")

    weight = metadata.get("weight_hint")
    if isinstance(weight, (int, float, str)):
        weight_value = str(weight).strip()
        if weight_value:
            info_parts.append(f"Вес: {weight_value}")

    updated = metadata.get("updated_at") or metadata.get("modified_at")
    if isinstance(updated, str) and updated.strip():
        info_parts.append(f"Обновлено: {updated.strip()}")

    block_lines = [header]
    if info_parts:
        block_lines.append(" | ".join(info_parts))
    block_lines.append(text)
    return "\n".join(block_lines)


def _trim_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_context_from_vector_store(question: str) -> str:
    if not CONFIG.can_use_vector_store:
        return "Контекст пуст."

    keywords = _collect_question_keywords(question)
    query = question.strip()
    if keywords:
        query = f"{question.strip()}\n\nКлючевые слова: {' '.join(sorted(keywords))}"

    try:
        raw_results = VECTOR_STORE.query(query, limit=CONFIG.context_query_limit)
    except Exception as error:
        print("vector_store.query ERROR:", error)
        raw_results = []

    if not raw_results:
        return "Контекст пуст."

    scored_results: list[tuple[tuple[Any, ...], dict[str, Any], str]] = []

    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue

        text = _extract_result_text(item)
        if not text:
            continue

        metadata = _extract_result_metadata(item)

        weight = _to_float(metadata.get("weight_hint"))
        score = _to_float(
            item.get("score")
            or item.get("similarity")
            or item.get("relevance")
            or item.get("rank")
        )
        overlap = len(_extract_keywords(text) & keywords) + _metadata_keyword_overlap(metadata, keywords)
        timestamp = _extract_timestamp(metadata)

        sort_key = (-score, -weight, -overlap, -timestamp, index)
        scored_results.append((sort_key, metadata, text))

    if not scored_results:
        return "Контекст пуст."

    scored_results.sort()

    snippets: list[str] = []
    total_chars = 0
    entity_hits: dict[str, int] = {}
    seen_texts: set[str] = set()

    for _, metadata, text in scored_results:
        snippet_text = _trim_text(text, CONFIG.context_snippet_max_chars)
        if not snippet_text or snippet_text in seen_texts:
            continue

        seen_texts.add(snippet_text)

        entity_key = "".join(
            str(metadata.get(field, "")).lower() for field in ("entity", "title", "filename")
        ) or "_default"
        hits = entity_hits.get(entity_key, 0)
        if hits >= CONFIG.context_per_file_limit:
            continue
        entity_hits[entity_key] = hits + 1

        block = _format_context_block(metadata, snippet_text)
        snippets.append(block)
        total_chars += len(block)
        if total_chars >= CONFIG.context_max_chars:
            break

    return "\n\n".join(snippets) if snippets else "Контекст пуст."


def ask_with_vector_store_context(question: str) -> str:
    if not CONFIG.has_api_credentials:
        return "Извините, база знаний сейчас недоступна."

    context = build_context_from_vector_store(question)
    system_prompt = (
        "Отвечай ТОЛЬКО на основе раздела CONTEXT ниже. "
        "Если вопрос о наличии/расписании/контактах — отвечай по шаблону:\n"
        "• Да/Нет.\n"
        "• Часы: <в одной строке>.\n"
        "• Телефон: <если есть>.\n"
        "• Ссылка: <если есть>.\n"
        "Не выводи детали, не относящиеся напрямую к вопросу. "
        "Если ответа нет в CONTEXT — напиши: 'Нет данных в базе знаний'.\n"
        f"CONTEXT:\n{context}"
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }

    data = CLIENT.call_completions(payload)
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    return "Нет данных в базе знаний."


# ========================
#  Обработка запросов
# ========================

app = FastAPI(title="U4S Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CONFIG.allowed_origins),
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


def _produce_answer(question: str, *, log_prefix: str) -> str:
    normalized = normalize_question(question)
    try:
        return rag_via_responses(normalized)
    except Exception as rag_error:
        print(f"{log_prefix} RAG error:", rag_error)
        try:
            return ask_with_vector_store_context(normalized)
        except Exception as fallback_error:
            print(f"{log_prefix} fallback error:", fallback_error)
            raise


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        raw = await request.body()
        try:
            data = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/debug/info")
def debug_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "env": {
            "HAS_API_KEY": CONFIG.has_api_credentials,
            "YANDEX_FOLDER_ID": CONFIG.yandex_folder_id,
            "VECTOR_STORE_ID": CONFIG.vector_store_id,
            "ALLOWED_ORIGINS": CONFIG.allowed_origins,
            "CAN_USE_VECTOR_STORE": CONFIG.can_use_vector_store,
        },
        "vs_files_count": 0,
        "vs_sample": [],
        "error": None,
    }
    try:
        files = VECTOR_STORE.list_files()
        info["vs_files_count"] = len(files)
        info["vs_sample"] = files[:3]
    except Exception as error:
        info["error"] = str(error)
    return info


@app.get("/api/chat")
def chat_get(q: str = "") -> dict[str, str]:
    try:
        answer = _produce_answer(q, log_prefix="GET")
        return {"answer": answer}
    except Exception as fatal_error:
        print("FATAL (GET):", fatal_error)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}


@app.post("/api/chat")
async def chat_post(request: Request) -> dict[str, str]:
    payload = await _read_json_payload(request)
    question = str(payload.get("question", "")).strip()

    try:
        answer = _produce_answer(question, log_prefix="POST")
        return {"answer": answer}
    except Exception as fatal_error:
        print("FATAL (POST):", fatal_error)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}
