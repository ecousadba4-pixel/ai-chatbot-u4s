from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.booking.entities import extract_booking_entities_ru
from app.chat.composer import ChatComposer
from app.chat.intent import detect_intent
from app.core.config import get_settings
from app.core.security import verify_api_key
from app.session import get_session_store

router = APIRouter(prefix="/chat", dependencies=[Depends(verify_api_key)])


def get_composer() -> ChatComposer:  # pragma: no cover - переопределяется в main
    raise RuntimeError("Composer dependency is not configured")


class ChatRequest(BaseModel):
    session_id: str = Field(default="", alias="sessionId")
    message: str


class ChatResponse(BaseModel):
    answer: str
    debug: dict[str, Any] | None = None


@router.post("", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_endpoint(
    payload: ChatRequest, composer: ChatComposer = Depends(get_composer)
) -> ChatResponse:
    settings = get_settings()
    session_store = get_session_store()

    now = datetime.now(ZoneInfo("UTC"))
    now_date = now.date()
    entities = extract_booking_entities_ru(payload.message, now_date=now_date, tz="UTC")
    session_id = payload.session_id or "anonymous"
    intent = detect_intent(payload.message, booking_entities=entities.__dict__)

    if payload.session_id:
        await session_store.get(payload.session_id)
        await session_store.set(
            payload.session_id,
            {
                "sessionId": payload.session_id,
                "last_seen": now.isoformat(),
            },
        )

    if intent == "booking_quote":
        result = await composer.handle_booking(session_id, payload.message)
    elif intent == "booking_calculation":
        result = await composer.handle_booking_calculation(
            session_id, payload.message, entities
        )
    elif intent == "knowledge_lookup":
        result = await composer.handle_knowledge(payload.message)
    else:
        result = await composer.handle_general(payload.message, intent=intent)

    debug = result.get("debug", {})
    debug.setdefault("intent", intent)
    debug.setdefault("intent_detected", intent)
    debug.setdefault("booking_entities", entities.__dict__)
    debug.setdefault("missing_fields", getattr(entities, "missing_fields", []))
    debug.setdefault("shelter_called", False)
    debug.setdefault("shelter_latency_ms", 0)
    debug.setdefault("shelter_error", None)
    if debug.get("shelter_called"):
        debug["llm_called"] = False
    response_payload: dict[str, Any] = {"answer": result.get("answer", "")}
    # В проде debug скрыт по умолчанию и включается только через INCLUDE_DEBUG.
    if settings.include_debug:
        response_payload["debug"] = debug
    return ChatResponse(**response_payload)
