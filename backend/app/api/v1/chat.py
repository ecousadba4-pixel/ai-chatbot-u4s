from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

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
    session_id = payload.session_id or "anonymous"
    intent = detect_intent(payload.message)

    if intent == "booking_quote":
        result = await composer.handle_booking(session_id, payload.message)
    elif intent == "knowledge_lookup":
        result = await composer.handle_knowledge(payload.message)
    else:
        result = await composer.handle_general(payload.message, intent=intent)

    debug = result.get("debug", {})
    debug.setdefault("intent", intent)
    return ChatResponse(answer=result.get("answer", ""), debug=debug)
