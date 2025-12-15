from __future__ import annotations

import json
from typing import Any, Sequence

import requests
from loguru import logger

if __package__:
    from .config import CONFIG
    from .conversation import (
        ChatModelMessage,
        extract_last_user_content,
        replace_system_prompt,
        trim_messages_for_model,
    )
else:
    from config import CONFIG
    from conversation import (
        ChatModelMessage,
        extract_last_user_content,
        replace_system_prompt,
        trim_messages_for_model,
    )

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


class AmveraClient:
    def __init__(self, config):
        self.config = config
        self._http = requests.Session()

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.amvera_api_token}",
            "Content-Type": "application/json",
        }

    def call_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.amvera_api_url.rstrip('/')}/chat/completions"
        resp = self._http.post(
            url,
            headers=self._json_headers(),
            timeout=self.config.completion_timeout,
            json=payload,
        )
        if resp.ok:
            data = resp.json()
            return data if isinstance(data, dict) else {}

        body: str
        try:
            raw = resp.json()
            body = json.dumps(raw, ensure_ascii=False)
        except Exception:
            body = resp.text or ""
        body = (body or "").strip().replace("\n", " ")
        body = (body[:800] + "…") if len(body) > 800 else (body or "<пустой ответ>")
        raise RuntimeError(f"Amvera chat HTTP {resp.status_code}: {body}")


CLIENT = AmveraClient(CONFIG)


def _extract_text_from_response(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""

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
    return ""


def _prepare_openai_messages(messages: Sequence[ChatModelMessage]) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for message in messages:
        prepared.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    return prepared


def rag_via_responses(messages: Sequence[ChatModelMessage], *, client: AmveraClient | None = None) -> str:
    active_client = client or CLIENT
    if not CONFIG.has_api_credentials:
        raise RuntimeError("Amvera LLM недоступен: не задан AMVERA_API_TOKEN")

    trimmed_messages = trim_messages_for_model(
        messages,
        max_tokens=CONFIG.input_max_tokens,
    )

    payload = {
        "model": CONFIG.amvera_model,
        "messages": _prepare_openai_messages(trimmed_messages),
        "temperature": 0.3,
        "top_p": 0.8,
    }

    data = active_client.call_chat(payload)
    answer = _extract_text_from_response(data)
    return answer or "Нет данных в базе знаний."


def build_context_from_vector_store(question: str, *, vector_store: Any | None = None) -> str:
    # Интеграция с Vector Store пока недоступна в Amvera-инфраструктуре.
    return "Контекст пуст."


def ask_with_vector_store_context(
    messages: Sequence[ChatModelMessage],
    *,
    client: AmveraClient | None = None,
    vector_store: Any | None = None,
) -> str:
    if not CONFIG.has_api_credentials:
        return "Извините, база знаний сейчас недоступна."

    active_client = client or CLIENT
    question = extract_last_user_content(messages)
    context = build_context_from_vector_store(question, vector_store=vector_store)
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
        "model": CONFIG.amvera_model,
        "messages": _prepare_openai_messages(trimmed_messages),
        "temperature": 0.3,
        "top_p": 0.8,
    }

    try:
        data = active_client.call_chat(payload)
    except Exception:
        logger.exception("Fallback chat error")
        return "Извините, сейчас не могу ответить. Попробуйте позже."

    answer = _extract_text_from_response(data)
    return answer or "Нет данных в базе знаний."


__all__ = [
    "ask_with_vector_store_context",
    "build_context_from_vector_store",
    "CLIENT",
    "SYSTEM_PROMPT_RAG",
    "rag_via_responses",
]
