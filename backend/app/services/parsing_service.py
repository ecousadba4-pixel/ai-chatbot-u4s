from __future__ import annotations

from functools import lru_cache
from datetime import date

from app.booking.entities import BookingEntities
from app.booking.fsm import BookingContext
from app.booking.parsers import (
    extract_guests,
    parse_adults,
    parse_checkin,
    parse_children_ages,
    parse_children_count,
    parse_nights,
    parse_room_type,
)
from app.booking.slot_filling import SlotFiller, SlotState


class ParsedMessageCache:
    """Кэширует результаты парсинга для одного сообщения пользователя."""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    @property
    def lowered(self) -> str:
        return self._text.lower()

    @lru_cache(maxsize=1)
    def guests(self) -> dict[str, int]:
        return extract_guests(self._text)

    @lru_cache(maxsize=None)
    def checkin(self, now_date: date | None = None) -> str | None:
        return parse_checkin(self._text, now_date=now_date)

    @lru_cache(maxsize=1)
    def nights(self) -> int | None:
        return parse_nights(self._text)

    @lru_cache(maxsize=None)
    def adults(self, allow_general_numbers: bool = True) -> int | None:
        return parse_adults(self._text, allow_general_numbers=allow_general_numbers)

    @lru_cache(maxsize=1)
    def children_count(self) -> int | None:
        return parse_children_count(self._text)

    @lru_cache(maxsize=None)
    def children_ages(self, expected: int | None = None) -> list[int]:
        return parse_children_ages(self._text, expected=expected)

    @lru_cache(maxsize=1)
    def room_type(self) -> str | None:
        return parse_room_type(self._text)


class ParsingService:
    """Сервис для парсинга сообщений и извлечения сущностей бронирования."""

    def __init__(self, slot_filler: SlotFiller) -> None:
        self._slot_filler = slot_filler

    def create_parsers(self, text: str) -> ParsedMessageCache:
        """Создаёт кэшируемые парсеры для входящего сообщения."""
        return ParsedMessageCache(text)

    def apply_entities_to_context(
        self, context: BookingContext, entities: BookingEntities
    ) -> None:
        """Применяет извлечённые сущности к контексту бронирования."""
        if not context.checkin:
            context.checkin = entities.checkin
        if not context.checkout:
            context.checkout = entities.checkout
        if context.nights is None:
            context.nights = entities.nights
        if context.adults is None:
            context.adults = entities.adults
        if context.children is None:
            context.children = entities.children
        if not context.children_ages and entities.children is not None and entities.children <= 0:
            context.children_ages = []
        if context.room_type is None:
            context.room_type = entities.room_type

    def apply_entities_from_message(
        self, context: BookingContext, parsers: ParsedMessageCache
    ) -> None:
        """Применяет сущности, извлечённые из текста сообщения, к контексту."""
        if not context.checkin:
            context.checkin = parsers.checkin()
        if context.nights is None and not context.checkout:
            context.nights = parsers.nights()
        if not context.checkout and context.checkin:
            try:
                checkin_date = date.fromisoformat(context.checkin)
            except ValueError:
                checkin_date = None
            parsed_checkout = parsers.checkin(now_date=checkin_date or date.today())
            if parsed_checkout and parsed_checkout != context.checkin:
                try:
                    checkout_date = date.fromisoformat(parsed_checkout)
                except ValueError:
                    checkout_date = None
                if checkout_date and checkin_date and checkout_date > checkin_date:
                    context.checkout = parsed_checkout
        if context.adults is None:
            # Определяем, разрешены ли общие числа в зависимости от состояния
            from app.booking.fsm import BookingState
            allow_general_adults = context.state in {
                BookingState.ASK_ADULTS,
                BookingState.ASK_CHILDREN_COUNT,
                BookingState.ASK_CHILDREN_AGES,
                BookingState.CALCULATE,
                BookingState.AWAITING_USER_DECISION,
                BookingState.CONFIRM_BOOKING,
            }
            adults = parsers.adults(allow_general_numbers=allow_general_adults)
            if adults is not None:
                context.adults = adults
        if context.children is None and context.state in {
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
        }:
            context.children = parsers.children_count()
        if (
            (context.children or 0) > 0
            and not context.children_ages
            and context.state == BookingState.ASK_CHILDREN_AGES
        ):
            context.children_ages = parsers.children_ages(expected=context.children)
        if context.room_type is None:
            context.room_type = parsers.room_type()

    def extract_slot_state(self, text: str, state: SlotState | None = None) -> SlotState:
        """Извлекает состояние слотов из текста."""
        state = state or SlotState()
        state = self._slot_filler.extract(text, state)
        return state

    def apply_children_answer(self, text: str, state: SlotState) -> None:
        """Применяет ответ о детях к состоянию слотов."""
        if state.children is not None:
            return
        lowered = text.strip().lower()
        negative_children = {"нет", "неа", "нету", "не будет", "без детей"}
        if lowered in negative_children or "нет детей" in lowered:
            state.children = 0


__all__ = ["ParsingService", "ParsedMessageCache"]

