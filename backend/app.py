from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

try:  # pragma: no cover - поддержка запуска без пакета
    from .redis_gateway import (
        REDIS_MAX_MESSAGES,
        RedisHistoryGateway,
        create_redis_client,
        parse_redis_args,
    )
except ImportError:  # pragma: no cover - когда модуль запускают напрямую
    from redis_gateway import (  # type: ignore
        REDIS_MAX_MESSAGES,
        RedisHistoryGateway,
        create_redis_client,
        parse_redis_args,
    )

FILES_API = "https://rest-assistant.api.cloud.yandex.net/v1"
RESPONSES_API = f"{FILES_API}/responses"


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
    redis_url: str = ""
    redis_args: dict[str, Any] = field(default_factory=dict)
    http_timeout: float = 30.0
    completion_timeout: float = 60.0
    input_max_tokens: int = 3500
    context_max_chars: int = 2500
    context_per_file_limit: int = 12
    cache_ttl: float = 180.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            yandex_api_key=_strip(os.environ.get("YANDEX_API_KEY")),
            yandex_folder_id=_strip(os.environ.get("YANDEX_FOLDER_ID")),
            vector_store_id=_strip(os.environ.get("VECTOR_STORE_ID")),
            allowed_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "*")),
            redis_url=_strip(os.environ.get("REDIS_URL")),
            redis_args=parse_redis_args(os.environ.get("REDIS_ARGS")),
        )

    @property
    def has_api_credentials(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)

    @property
    def can_use_vector_store(self) -> bool:
        return bool(self.has_api_credentials and self.vector_store_id)

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)


CONFIG = AppConfig.from_env()


# ========================
#  Хранилище истории в Redis
# ========================

REDIS_GATEWAY = RedisHistoryGateway(
    create_redis_client(CONFIG.redis_url, CONFIG.redis_args)
)


# ========================
#  Работа с историей диалога
# ========================

ChatHistoryItem = dict[str, Any]
ChatModelMessage = dict[str, str]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "on"}
    return False


def _coerce_timestamp(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return fallback
    return fallback


def _sanitize_history_messages(raw_history: Any) -> list[ChatHistoryItem]:
    if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
        return []

    sanitized: list[ChatHistoryItem] = []
    base_time = time.time()
    for index, item in enumerate(raw_history):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        timestamp = _coerce_timestamp(item.get("timestamp"), base_time + index * 1e-3)
        sanitized.append({"role": role, "content": content, "timestamp": timestamp})
    return sanitized


def _merge_histories(*histories: Sequence[ChatHistoryItem], limit: int | None = None) -> list[ChatHistoryItem]:
    combined: list[tuple[float, int, ChatHistoryItem]] = []
    order = 0
    for history in histories:
        for item in history or []:
            if not isinstance(item, dict):
                continue
            timestamp = float(item.get("timestamp", 0.0) or 0.0)
            content = str(item.get("content", "")).strip()
            role = str(item.get("role", "")).strip().lower()
            if role not in {"user", "assistant"} or not content:
                continue
            combined.append((timestamp, order, {"role": role, "content": content, "timestamp": timestamp}))
            order += 1

    combined.sort(key=lambda entry: (entry[0], entry[1]))
    merged = [item for _, _, item in combined]
    if limit is not None:
        merged = merged[-limit:]
    return merged


def _build_conversation_messages(
    history: Sequence[ChatHistoryItem],
    *,
    question: str,
) -> list[ChatModelMessage]:
    messages: list[ChatModelMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
    ]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def _extract_last_user_content(messages: Sequence[ChatModelMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _normalize_messages_for_model(
    messages: Sequence[ChatModelMessage],
) -> tuple[list[ChatModelMessage], str]:
    normalized: list[ChatModelMessage] = []
    last_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user_index = index
            break

    normalized_question = ""
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if index == last_user_index:
            content = normalize_question(content)
            normalized_question = content
        normalized.append({"role": role, "content": content})

    return normalized, normalized_question


def _estimate_tokens(text: str) -> int:
    text = text or ""
    length = len(text)
    if length == 0:
        return 0
    return max(1, (length + 3) // 4)


def _trim_messages_for_model(
    messages: Sequence[ChatModelMessage],
    *,
    max_tokens: int,
    min_assistant_messages: int = 2,
) -> list[ChatModelMessage]:
    if not messages or max_tokens <= 0:
        return list(messages)

    trimmed: list[ChatModelMessage] = []
    tokens_per_message: list[int] = []

    for message in messages:
        normalized_message = {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        trimmed.append(normalized_message)
        tokens_per_message.append(_estimate_tokens(normalized_message["content"]))

    required_indices: set[int] = set()

    if trimmed and trimmed[0].get("role") == "system":
        required_indices.add(0)

    for index in range(len(trimmed) - 1, -1, -1):
        if trimmed[index].get("role") == "user":
            required_indices.add(index)
            break

    assistant_indices = [index for index, item in enumerate(trimmed) if item.get("role") == "assistant"]
    for index in assistant_indices[-min_assistant_messages:]:
        required_indices.add(index)

    total_tokens = sum(tokens_per_message)
    if total_tokens <= max_tokens:
        return trimmed

    removed_flags = [False] * len(trimmed)

    for index in range(len(trimmed)):
        if index in required_indices:
            continue
        if total_tokens <= max_tokens:
            break
        removed_flags[index] = True
        total_tokens -= tokens_per_message[index]

    result: list[ChatModelMessage] = []
    for index, message in enumerate(trimmed):
        if not removed_flags[index]:
            result.append(message)

    return result


def _replace_system_prompt(
    messages: Sequence[ChatModelMessage], new_prompt: str
) -> list[ChatModelMessage]:
    replaced: list[ChatModelMessage] = []
    system_set = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            if not system_set:
                replaced.append({"role": "system", "content": new_prompt})
                system_set = True
            continue
        replaced.append({"role": str(role), "content": str(message.get("content", "")).strip()})

    if not system_set:
        replaced.insert(0, {"role": "system", "content": new_prompt})
    return replaced


def _messages_to_responses_input(messages: Sequence[ChatModelMessage]) -> list[dict[str, Any]]:
    payload_messages: list[dict[str, Any]] = []
    for message in messages:
        text = str(message.get("content", ""))
        payload_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": [{"type": "input_text", "text": text}],
            }
        )
    return payload_messages


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


CLIENT = YandexClient(CONFIG)


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
        result = (meta, content)
        self._file_cache[file_id] = (now, result)
        return result


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


def rag_via_responses(messages: Sequence[ChatModelMessage]) -> str:
    if not CONFIG.can_use_vector_store:
        raise RuntimeError(
            "Responses API недоступен: не заданы YANDEX_API_KEY/YANDEX_FOLDER_ID/VECTOR_STORE_ID",
        )

    trimmed_messages = _trim_messages_for_model(
        messages,
        max_tokens=CONFIG.input_max_tokens,
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": _messages_to_responses_input(trimmed_messages),
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
    data = CLIENT.call_responses(payload)

    answer = _extract_responses_text(data)
    return answer or "Нет данных в базе знаний."


# ========================
#  Фолбэк: контекст из Vector Store
# ========================

TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")


def _extract_keywords(text: str) -> set[str]:
    """Возвращает множество слов длиной от 3 символов из переданного текста."""

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


def build_context_from_vector_store(question: str) -> str:
    if not CONFIG.can_use_vector_store:
        return "Контекст пуст."

    try:
        keywords = _extract_keywords(question)
        snippets: list[str] = []
        total_chars = 0

        for file_info in VECTOR_STORE.list_files():
            file_id = file_info.get("id")
            if not isinstance(file_id, str):
                continue

            meta, content = VECTOR_STORE.fetch_file(file_id)
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


def ask_with_vector_store_context(messages: Sequence[ChatModelMessage]) -> str:
    if not CONFIG.has_api_credentials:
        return "Извините, база знаний сейчас недоступна."

    question = _extract_last_user_content(messages)
    context = build_context_from_vector_store(question)
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

    prepared_messages = _replace_system_prompt(messages, system_prompt)

    trimmed_messages = _trim_messages_for_model(
        prepared_messages,
        max_tokens=CONFIG.input_max_tokens,
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": _messages_to_responses_input(trimmed_messages),
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 1800,
    }

    try:
        data = CLIENT.call_responses(payload)
    except Exception as error:
        print("Fallback Responses API error:", error)
        return "Извините, сейчас не могу ответить. Попробуйте позже."

    answer = _extract_responses_text(data)
    return answer or "Нет данных в базе знаний."


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


def _produce_answer(messages: Sequence[ChatModelMessage], *, log_prefix: str) -> str:
    try:
        return rag_via_responses(messages)
    except Exception as rag_error:
        print(f"{log_prefix} RAG error:", rag_error)
        try:
            return ask_with_vector_store_context(messages)
        except Exception as fallback_error:
            print(f"{log_prefix} fallback error:", fallback_error)
            raise


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
        parsed = data if isinstance(data, dict) else {}
    except Exception:
        raw = await request.body()
        try:
            data = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
        except Exception:
            parsed = {}
        else:
            parsed = data if isinstance(data, dict) else {}

    session_id = str(parsed.get("sessionId", "")).strip()
    history = _sanitize_history_messages(parsed.get("history"))
    question = str(parsed.get("question", "")).strip()
    reset = _to_bool(parsed.get("reset"))

    return {
        **parsed,
        "sessionId": session_id,
        "history": history,
        "question": question,
        "reset": reset,
    }


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
        conversation = _build_conversation_messages([], question=str(q or ""))
        normalized_messages, _ = _normalize_messages_for_model(conversation)
        answer = _produce_answer(normalized_messages, log_prefix="GET")
        return {"answer": answer}
    except Exception as fatal_error:
        print("FATAL (GET):", fatal_error)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}


@app.post("/api/chat")
async def chat_post(request: Request) -> dict[str, str]:
    payload = await _read_json_payload(request)
    session_id = payload.get("sessionId", "")
    client_history: list[ChatHistoryItem] = payload.get("history", [])
    question = payload.get("question", "")
    reset_requested = bool(payload.get("reset"))

    history_limit = getattr(REDIS_GATEWAY, "max_messages", REDIS_MAX_MESSAGES)

    redis_history: list[ChatHistoryItem] = []
    if session_id:
        if reset_requested:
            REDIS_GATEWAY.delete_history(session_id)
        else:
            redis_history = REDIS_GATEWAY.read_history(session_id)

    merged_history = _merge_histories(redis_history, client_history, limit=history_limit)

    conversation = _build_conversation_messages(merged_history, question=question)
    normalized_messages, normalized_question = _normalize_messages_for_model(conversation)

    try:
        answer = _produce_answer(normalized_messages, log_prefix="POST")
    except Exception as fatal_error:
        print("FATAL (POST):", fatal_error)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}
    else:
        if session_id:
            now = time.time()
            stored_history = merged_history + [
                {"role": "user", "content": normalized_question, "timestamp": now},
                {
                    "role": "assistant",
                    "content": str(answer).strip(),
                    "timestamp": now + 1e-3,
                },
            ]
            stored_history = stored_history[-history_limit:]
            REDIS_GATEWAY.write_history(session_id, stored_history)

        return {"answer": answer}


@app.post("/api/chat/reset")
async def chat_reset(request: Request) -> dict[str, bool]:
    payload = await _read_json_payload(request)
    session_id = payload.get("sessionId", "")

    if session_id:
        REDIS_GATEWAY.delete_history(session_id)

    return {"ok": True}
