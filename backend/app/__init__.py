from __future__ import annotations

import json
import time
from typing import Any, Sequence

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import CONFIG, AppConfig
from backend.conversation import (
    ChatHistoryItem,
    ChatModelMessage,
    build_conversation_messages,
    merge_histories,
    normalize_messages_for_model,
    normalize_question,
    sanitize_history_messages,
    to_bool,
)
from backend.rag import (
    CLIENT as RAG_CLIENT,
    SYSTEM_PROMPT_RAG,
    VECTOR_STORE as RAG_VECTOR_STORE,
    ask_with_vector_store_context as rag_ask_with_vector_store_context,
    build_context_from_vector_store as rag_build_context_from_vector_store,
    rag_via_responses as rag_rag_via_responses,
)
from backend.redis_gateway import (
    REDIS_MAX_MESSAGES,
    RedisHistoryGateway,
    create_redis_client,
)

from .chat.handlers import BookingIntentHandler
from .dialogue.manager import BookingDialogueManager
from .dialogue.state import (
    BRANCH_BOOKING_PRICE_CHAT,
    BRANCH_ONLINE_BOOKING_REDIRECT,
    INTENT_BOOKING_INQUIRY,
)
from .services import ShelterCloudConfig, ShelterCloudService


DEFAULT_ERROR_ANSWER = "Извините, сейчас не могу ответить. Попробуйте позже."
EMPTY_QUESTION_ANSWER = "Пожалуйста, сформулируйте вопрос."


REDIS_GATEWAY = RedisHistoryGateway(
    create_redis_client(CONFIG.redis_url, CONFIG.redis_args)
)

SHELTER_CLOUD_CONFIG = ShelterCloudConfig(
    token=CONFIG.shelter_cloud_token,
    language=CONFIG.shelter_cloud_language,
    timeout=CONFIG.http_timeout,
)
if not SHELTER_CLOUD_CONFIG.is_configured():
    logger.warning("Shelter Cloud не настроен: отсутствует токен")

SHELTER_CLOUD_SERVICE = ShelterCloudService(SHELTER_CLOUD_CONFIG)
BOOKING_DIALOGUE_MANAGER = BookingDialogueManager(
    storage=REDIS_GATEWAY,
    service=SHELTER_CLOUD_SERVICE,
)
BOOKING_INTENT_HANDLER = BookingIntentHandler(BOOKING_DIALOGUE_MANAGER)

app = FastAPI(title="U4S Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CONFIG.allowed_origins),
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


CLIENT = RAG_CLIENT
VECTOR_STORE = RAG_VECTOR_STORE


def rag_via_responses(messages: Sequence[ChatModelMessage]) -> str:
    return rag_rag_via_responses(messages, client=CLIENT)


def ask_with_vector_store_context(messages: Sequence[ChatModelMessage]) -> str:
    return rag_ask_with_vector_store_context(
        messages, client=CLIENT, vector_store=VECTOR_STORE
    )


def build_context_from_vector_store(question: str) -> str:
    return rag_build_context_from_vector_store(question, vector_store=VECTOR_STORE)


def _booking_handler() -> BookingIntentHandler:
    BOOKING_DIALOGUE_MANAGER.storage = REDIS_GATEWAY
    return BOOKING_INTENT_HANDLER


def _persist_history(
    session_id: str,
    merged_history: list[ChatHistoryItem],
    question: str,
    answer: str,
    *,
    limit: int,
) -> None:
    if not session_id:
        return
    now = time.time()
    stored_history = merged_history + [
        {"role": "user", "content": question, "timestamp": now},
        {
            "role": "assistant",
            "content": str(answer).strip(),
            "timestamp": now + 1e-3,
        },
    ]
    REDIS_GATEWAY.write_history(session_id, stored_history[-limit:])


def _produce_answer(messages: Sequence[ChatModelMessage], *, log_prefix: str) -> str:
    try:
        return rag_via_responses(messages)
    except Exception:
        logger.exception("{prefix} RAG error", prefix=log_prefix)
        try:
            return ask_with_vector_store_context(messages)
        except Exception:
            logger.exception("{prefix} fallback error", prefix=log_prefix)
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
    history = sanitize_history_messages(parsed.get("history"))
    question = str(parsed.get("question", "")).strip()
    reset = to_bool(parsed.get("reset"))

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
        conversation = build_conversation_messages(
            [],
            question=str(q or ""),
            system_prompt=SYSTEM_PROMPT_RAG,
        )
        normalized_messages, normalized_question = normalize_messages_for_model(
            conversation
        )
        if not normalized_question:
            return {"answer": EMPTY_QUESTION_ANSWER}
        answer = _produce_answer(normalized_messages, log_prefix="GET")
        return {"answer": answer}
    except Exception:
        logger.exception("FATAL (GET)")
        return {"answer": DEFAULT_ERROR_ANSWER}


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

    merged_history = merge_histories(redis_history, client_history, limit=history_limit)

    conversation = build_conversation_messages(
        merged_history,
        question=question,
        system_prompt=SYSTEM_PROMPT_RAG,
    )
    normalized_messages, normalized_question = normalize_messages_for_model(
        conversation
    )
    if not normalized_question:
        return {"answer": EMPTY_QUESTION_ANSWER}

    booking_result = _booking_handler().handle(session_id, normalized_question)
    if booking_result.handled and booking_result.answer is not None:
        answer_text = booking_result.answer
        if session_id:
            _persist_history(
                session_id,
                merged_history,
                normalized_question,
                answer_text,
                limit=history_limit,
            )
        response_payload: dict[str, str] = {"answer": answer_text}
        if booking_result.intent:
            response_payload["intent"] = booking_result.intent
        if booking_result.branch:
            response_payload["branch"] = booking_result.branch
        return response_payload

    try:
        answer = _produce_answer(normalized_messages, log_prefix="POST")
    except Exception:
        logger.exception("FATAL (POST)")
        return {"answer": DEFAULT_ERROR_ANSWER}
    else:
        if session_id:
            _persist_history(
                session_id,
                merged_history,
                normalized_question,
                str(answer),
                limit=history_limit,
            )

        return {"answer": answer}


@app.post("/api/chat/reset")
async def chat_reset(request: Request) -> dict[str, bool]:
    payload = await _read_json_payload(request)
    session_id = payload.get("sessionId", "")

    if session_id:
        REDIS_GATEWAY.delete_history(session_id)
        BOOKING_DIALOGUE_MANAGER.reset(session_id)

    return {"ok": True}

