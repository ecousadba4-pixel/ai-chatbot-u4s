from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.booking.entities import extract_booking_entities_ru
from app.chat.composer import ChatComposer
from app.chat.intent import detect_intent
from app.core.security import verify_api_key

router = APIRouter(prefix="/chat", dependencies=[Depends(verify_api_key)])


def get_composer() -> ChatComposer:  # pragma: no cover - переопределяется в main
    raise RuntimeError("Composer dependency is not configured")


class ChatRequest(BaseModel):
    session_id: str = Field(default="", alias="sessionId")
    message: str


class ChatResponse(BaseModel):
    answer: str
    debug: dict[str, Any]


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest, composer: ChatComposer = Depends(get_composer)
) -> ChatResponse:
    now_date = datetime.now(ZoneInfo("UTC")).date()
    entities = extract_booking_entities_ru(payload.message, now_date=now_date, tz="UTC")
    session_id = payload.session_id or "anonymous"
    intent = detect_intent(payload.message, booking_entities=entities.__dict__)

    if intent == "booking_quote":
        result = await composer.handle_booking(session_id, payload.message)
    elif intent == "booking_calculation":
        result = await composer.handle_booking_calculation(entities)
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
    return ChatResponse(answer=result.get("answer", ""), debug=debug)
