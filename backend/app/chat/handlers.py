"""Обработчики пользовательских намерений."""

from __future__ import annotations

from dataclasses import dataclass

from ..dialogue.manager import BookingDialogueManager, DialogueResult
from ..dialogue.state import INTENT_BOOKING_INQUIRY


@dataclass
class BookingIntentHandler:
    """Роутер сообщений для сценария бронирования."""

    manager: BookingDialogueManager

    def handle(self, session_id: str, question: str) -> DialogueResult:
        result = self.manager.handle_message(session_id, question)
        if result.handled:
            return result
        return DialogueResult(False, None, None, None)

    @property
    def intent_name(self) -> str:
        return INTENT_BOOKING_INQUIRY
