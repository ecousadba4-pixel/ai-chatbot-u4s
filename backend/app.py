from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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
    http_timeout: float = 30.0
    completion_timeout: float = 60.0
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
        "tool_resources": {
            "file_search": {
                "vector_store_ids": [CONFIG.vector_store_id],
                "top_k": 8,
            }
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


def ask_with_vector_store_context(question: str) -> str:
    if not CONFIG.has_api_credentials:
        return "Извините, база знаний сейчас недоступна."

    context = build_context_from_vector_store(question)
    system_prompt = (
        "Ты — русскоязычный AI-консьерж отеля «Усадьба Четыре Сезона».\n"
        "Правила:\n"
        "1) Отвечай кратко, по делу, вежливо.\n"
        "2) Всегда сверяй факты с разделом CONTEXT. Если релевантных сведений нет — напиши: «не нашёл точной информации в базе».\n"
        "3) Не выдумывай цены, расписания и контакты — цитируй точный текст из CONTEXT с указанием раздела (например, «(см. раздел <название>)»).\n"
        "4) Формат ответа: 2–3 предложения, затем список «Что ещё могу подсказать» с тремя пунктами.\n"
        "5) Используй только русский язык и не давай внешние ссылки, кроме usadba4.ru.\n"
        "Работай строго с данными ниже и не добавляй сведения вне CONTEXT.\n"
        f"CONTEXT:\n{context}"
    )

    payload = {
        "model": f"gpt://{CONFIG.yandex_folder_id}/yandexgpt/latest",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": question}]},
        ],
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 1800,
    }

    data = CLIENT.call_responses(payload)
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
