from __future__ import annotations

from app.booking.entities import BookingEntities
from app.booking.models import BookingQuote
from app.chat.formatting import (
    detect_detail_mode,
    format_shelter_quote,
    format_more_offers,
    postprocess_answer,
    select_min_offer_per_room_type,
)


class ResponseFormattingService:
    """Сервис для форматирования ответов пользователю."""

    def format_booking_quote(
        self, entities: BookingEntities, offers: list[BookingQuote]
    ) -> str:
        """Форматирует предложения по бронированию."""
        return format_shelter_quote(entities, offers)

    def format_more_offers(
        self, offers: list[BookingQuote], start_index: int
    ) -> tuple[str, int]:
        """Форматирует дополнительные предложения."""
        return format_more_offers(offers, start_index)

    def select_min_offer_per_room_type(
        self, offers: list[BookingQuote]
    ) -> list[BookingQuote]:
        """Выбирает минимальное предложение для каждого типа комнаты."""
        return select_min_offer_per_room_type(offers)

    def detect_detail_mode(self, user_text: str) -> bool:
        """Определяет, нужен ли подробный ответ."""
        return detect_detail_mode(user_text)

    def postprocess_answer(self, answer: str, mode: str = "brief") -> str:
        """Постобработка ответа перед отправкой пользователю."""
        return postprocess_answer(answer, mode=mode)


__all__ = ["ResponseFormattingService"]

