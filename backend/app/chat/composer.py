from typing import Any, TYPE_CHECKING

import logging
import re
import time
from datetime import date, timedelta

import asyncpg

from app.core.config import Settings, get_settings
from app.booking.entities import BookingEntities
from app.booking.fsm import BookingContext, BookingState
from app.booking.models import BookingQuote, Guests
from app.booking.service import BookingQuoteService
from app.booking.slot_filling import SlotFiller, SlotState
from app.llm.amvera_client import AmveraLLMClient
from app.llm.prompts import FACTS_PROMPT
from app.llm.cache import get_llm_cache
from app.rag.context_builder import build_context
from app.rag.qdrant_client import QdrantClient
from app.rag.retriever import gather_rag_data
from app.services.parsing_service import ParsedMessageCache, ParsingService
from app.services.booking_fsm_service import BookingFsmService
from app.services.response_formatting_service import ResponseFormattingService

if TYPE_CHECKING:
    from app.session.redis_state_store import RedisConversationStateStore


logger = logging.getLogger(__name__)


class ConversationStateStore:
    def get(self, session_id: str) -> SlotState | None:
        raise NotImplementedError

    def set(self, session_id: str, state: SlotState) -> None:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryConversationStateStore(ConversationStateStore):
    def __init__(self) -> None:
        self._storage: dict[str, SlotState] = {}

    def get(self, session_id: str) -> SlotState | None:
        return self._storage.get(session_id)

    def set(self, session_id: str, state: SlotState) -> None:
        self._storage[session_id] = state

    def clear(self, session_id: str) -> None:
        self._storage.pop(session_id, None)


class ChatComposer:
    """Оркестратор для обработки сообщений чата.
    
    Координирует работу сервисов парсинга, FSM бронирования и форматирования ответов.
    """

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        qdrant: QdrantClient,
        llm: AmveraLLMClient,
        slot_filler: SlotFiller,
        booking_service: BookingQuoteService,
        store: ConversationStateStore,
        booking_fsm_store: ConversationStateStore | None = None,
        settings: Settings | None = None,
        max_state_attempts: int = 3,
    ) -> None:
        self._pool = pool
        self._qdrant = qdrant
        self._llm = llm
        self._store = store
        self._booking_store = booking_fsm_store or store
        self._settings = settings or get_settings()
        self._booking_service = booking_service  # Сохраняем для handle_booking
        
        # Инициализируем сервисы
        self._parsing_service = ParsingService(slot_filler)
        self._formatting_service = ResponseFormattingService()
        self._booking_fsm_service = BookingFsmService(
            booking_service=booking_service,
            formatting_service=self._formatting_service,
            max_state_attempts=max_state_attempts,
        )

    def has_active_booking(
        self, session_id: str, entities: BookingEntities | None = None
    ) -> bool:
        booking_context = BookingContext.from_dict(self._booking_store.get(session_id))
        if booking_context and booking_context.state not in (
            BookingState.DONE,
            BookingState.CANCELLED,
            None,
        ):
            return True

        state = self._store.get(session_id)
        if isinstance(state, SlotState) and self._has_booking_context(state):
            return True
        if entities and self._entities_have_booking_data(entities):
            return True
        return False

    async def handle_booking_calculation(
        self, session_id: str, text: str, entities: BookingEntities
    ) -> dict[str, Any]:
        """Обрабатывает расчёт бронирования через FSM."""
        # Загружаем контекст
        context_dict = self._booking_store.get(session_id)
        context = self._booking_fsm_service.load_context(context_dict)
        
        # Создаём парсеры для сообщения
        parsers = self._parsing_service.create_parsers(text)
        
        # Применяем сущности к контексту
        self._parsing_service.apply_entities_to_context(context, entities)
        self._parsing_service.apply_entities_from_message(context, parsers)
        
        # Подготавливаем debug информацию
        debug = {
            "intent": "booking_calculation",
            "booking_state": context.state.value if context.state else "",
            "booking_entities": self._booking_fsm_service.get_context_entities(context),
            "missing_fields": self._booking_fsm_service.get_missing_context_fields(context),
            "shelter_called": False,
            "shelter_latency_ms": 0,
            "shelter_error": None,
            "llm_called": False,
        }
        
        # Обрабатываем сообщение через FSM
        answer = await self._booking_fsm_service.process_message(
            session_id, text, context, parsers, debug
        )
        
        # Сохраняем контекст
        context_dict = self._booking_fsm_service.save_context(context)
        self._booking_store.set(session_id, context_dict)
        
        # Обновляем debug
        debug["booking_state"] = context.state.value if context.state else ""
        debug["booking_entities"] = self._booking_fsm_service.get_context_entities(context)
        debug["missing_fields"] = self._booking_fsm_service.get_missing_context_fields(context)
        
        return {"answer": answer, "debug": debug}

    def _has_booking_context(self, state: SlotState) -> bool:
        return bool(
            state.check_in
            or state.check_out
            or state.nights is not None
            or state.adults is not None
            or state.children is not None
        )

    def _entities_have_booking_data(self, entities: BookingEntities) -> bool:
        return bool(
            entities.checkin
            or entities.checkout
            or entities.nights
            or entities.adults is not None
            or entities.children is not None
        )

    async def _advance_booking_fsm(
        self,
        session_id: str,
        context: BookingContext,
        text: str,
        debug: dict[str, Any],
        parsers: ParsedMessageCache,
    ) -> str:
        state = context.state or BookingState.ASK_CHECKIN
        consumed_fields: set[str] = set()
        if context.nights is not None:
            consumed_fields.add("nights")
        if context.adults is not None:
            consumed_fields.add("adults")
        while True:
            if state == BookingState.ASK_CHECKIN:
                context.state = BookingState.ASK_CHECKIN
                if context.checkin:
                    state = BookingState.ASK_NIGHTS_OR_CHECKOUT
                    continue
                parsed = parsers.checkin()
                if parsed:
                    context.checkin = parsed
                    context.state = BookingState.ASK_NIGHTS_OR_CHECKOUT
                    state = BookingState.ASK_NIGHTS_OR_CHECKOUT
                    continue
                return self._ask_with_retry(context, BookingState.ASK_CHECKIN, "На какую дату планируете заезд?")

            if state == BookingState.ASK_NIGHTS_OR_CHECKOUT:
                context.state = BookingState.ASK_NIGHTS_OR_CHECKOUT
                if context.nights is not None or context.checkout:
                    state = BookingState.ASK_ADULTS
                    continue
                nights = parsers.nights()
                checkout_value = None
                try:
                    checkin_date = date.fromisoformat(context.checkin) if context.checkin else None
                except ValueError:
                    checkin_date = None
                if checkin_date:
                    parsed_checkout = parsers.checkin(now_date=checkin_date)
                    if parsed_checkout:
                        try:
                            checkout_date = date.fromisoformat(parsed_checkout)
                        except ValueError:
                            checkout_date = None
                        if checkout_date and checkout_date > checkin_date:
                            checkout_value = parsed_checkout
                if nights:
                    context.nights = nights
                    consumed_fields.add("nights")
                    state = BookingState.ASK_ADULTS
                    context.state = BookingState.ASK_ADULTS
                    continue
                if checkout_value:
                    context.checkout = checkout_value
                    state = BookingState.ASK_ADULTS
                    context.state = BookingState.ASK_ADULTS
                    continue
                return self._ask_with_retry(
                    context,
                    BookingState.ASK_NIGHTS_OR_CHECKOUT,
                    "Сколько ночей остаётесь или до какого числа?",
                )

            if state == BookingState.ASK_ADULTS:
                context.state = BookingState.ASK_ADULTS
                guests_from_text = parsers.guests()
                adults_from_text = guests_from_text.get("adults")
                children_from_text = guests_from_text.get("children")
                if adults_from_text is not None:
                    context.adults = adults_from_text
                if children_from_text is not None:
                    context.children = children_from_text
                    if children_from_text <= 0:
                        context.children_ages = []

                if context.adults is not None:
                    context.state = BookingState.ASK_CHILDREN_COUNT
                    if context.children is None and children_from_text is None:
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_COUNT,
                            "Сколько детей? Если детей нет — напишите 0.",
                        )
                    state = BookingState.ASK_CHILDREN_COUNT
                    continue
                allow_general = "nights" not in consumed_fields
                adults = parsers.adults(allow_general_numbers=allow_general)
                if adults is not None:
                    context.adults = adults
                    consumed_fields.add("adults")
                    context.state = BookingState.ASK_CHILDREN_COUNT
                    if context.children is None:
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_COUNT,
                            "Сколько детей? Если детей нет — напишите 0.",
                        )
                    state = BookingState.ASK_CHILDREN_COUNT
                    continue
                return self._ask_with_retry(context, BookingState.ASK_ADULTS, "Сколько взрослых едет?")

            if state == BookingState.ASK_CHILDREN_COUNT:
                context.state = BookingState.ASK_CHILDREN_COUNT
                guests_from_text = parsers.guests()
                children_from_text = guests_from_text.get("children")
                adults_from_text = guests_from_text.get("adults")
                if adults_from_text is not None:
                    context.adults = adults_from_text
                if children_from_text is not None:
                    context.children = children_from_text
                    if children_from_text <= 0:
                        context.children_ages = []

                lowered_input = parsers.lowered
                if context.children is not None:
                    if (context.children or 0) > 0:
                        if context.children_ages and len(context.children_ages) == context.children:
                            state = BookingState.CALCULATE
                            continue
                        if "взросл" not in lowered_input:
                            ages = parsers.children_ages(expected=context.children)
                            if ages:
                                context.children_ages = ages
                                state = BookingState.CALCULATE
                                context.state = BookingState.CALCULATE
                                continue
                        context.state = BookingState.ASK_CHILDREN_AGES
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_AGES,
                            "Уточните возраст детей (через запятую).",
                        )
                    else:
                        state = BookingState.CALCULATE
                    continue
                children = parsers.children_count()
                if children is not None:
                    context.children = children
                    if children > 0:
                        context.state = BookingState.ASK_CHILDREN_AGES
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_AGES,
                            "Уточните возраст детей (через запятую).",
                        )
                    state = BookingState.CALCULATE
                    context.state = BookingState.CALCULATE
                    continue
                return self._ask_with_retry(
                    context,
                    BookingState.ASK_CHILDREN_COUNT,
                    "Сколько детей? Если детей нет — напишите 0.",
                )

            if state == BookingState.ASK_CHILDREN_AGES:
                context.state = BookingState.ASK_CHILDREN_AGES
                if (context.children or 0) == 0:
                    state = BookingState.CALCULATE
                    continue
                if context.children_ages and len(context.children_ages) == context.children:
                    state = BookingState.CALCULATE
                    continue
                ages = parsers.children_ages(expected=context.children)
                if ages:
                    context.children_ages = ages
                    state = BookingState.CALCULATE
                    context.state = BookingState.CALCULATE
                    continue
                return self._ask_with_retry(
                    context,
                    BookingState.ASK_CHILDREN_AGES,
                    "Не услышал возраст детей, укажите числа через запятую.",
                )

            if state == BookingState.CALCULATE:
                context.state = BookingState.CALCULATE
                answer = await self._calculate_booking(context, debug)
                return answer

            if state == BookingState.AWAITING_USER_DECISION:
                context.state = BookingState.AWAITING_USER_DECISION
                return self._handle_post_quote_decision(text, context, parsers)

            if state == BookingState.CONFIRM_BOOKING:
                context.state = BookingState.CONFIRM_BOOKING
                return self._handle_confirmation(text, context, parsers)

            return self._ask_with_retry(
                context, BookingState.ASK_CHECKIN, "На какую дату планируете заезд?"
            )

    def _booking_prompt(self, question: str, context: BookingContext) -> str:
        summary = self._booking_summary(context)
        parts: list[str] = []
        if summary:
            parts.append(f"Понял: {summary}.")
        parts.append(question)
        return " ".join(parts)

    def _booking_summary(self, context: BookingContext) -> str:
        fragments: list[str] = []
        if context.checkin:
            fragments.append(f"заезд {self._format_date(context.checkin)}")
        if context.nights:
            fragments.append(f"ночей {context.nights}")
        elif context.checkout:
            fragments.append(f"выезд {self._format_date(context.checkout)}")
        if context.adults is not None:
            guests = f"взрослых {context.adults}"
            if context.children is not None:
                guests += f", детей {context.children}"
            fragments.append(guests)
        if context.room_type:
            fragments.append(f"тип {context.room_type}")
        return ", ".join(fragments)

    async def _calculate_booking(
        self, context: BookingContext, debug: dict[str, Any]
    ) -> str:
        if not context.checkin:
            context.state = BookingState.ASK_CHECKIN
            return self._booking_prompt("На какую дату планируете заезд?", context)

        try:
            checkin_date = date.fromisoformat(context.checkin)
        except ValueError:
            context.checkin = None
            context.state = BookingState.ASK_CHECKIN
            return self._booking_prompt("Укажите корректную дату заезда.", context)

        nights = context.nights
        if nights is not None and nights > 0:
            context.checkout = (checkin_date + timedelta(days=nights)).isoformat()
        elif context.checkout:
            try:
                checkout_date = date.fromisoformat(context.checkout)
            except ValueError:
                context.checkout = None
                return self._ask_with_retry(
                    context, BookingState.ASK_NIGHTS_OR_CHECKOUT, "Укажите дату выезда или количество ночей."
                )
            if checkout_date <= checkin_date:
                context.checkout = None
                return self._ask_with_retry(
                    context, BookingState.ASK_NIGHTS_OR_CHECKOUT, "Дата выезда должна быть позже даты заезда."
                )
            context.nights = (checkout_date - checkin_date).days
            nights = context.nights
        else:
            return self._ask_with_retry(
                context, BookingState.ASK_NIGHTS_OR_CHECKOUT, "Сколько ночей остаётесь или до какого числа?"
            )

        if context.adults is None:
            context.state = BookingState.ASK_ADULTS
            return self._ask_with_retry(context, BookingState.ASK_ADULTS, "Сколько взрослых едет?")

        if (context.children or 0) > 0 and not context.children_ages:
            context.state = BookingState.ASK_CHILDREN_AGES
            return self._ask_with_retry(
                context, BookingState.ASK_CHILDREN_AGES, "Не услышал возраст детей, укажите числа через запятую."
            )

        guests = Guests(
            adults=context.adults,
            children=context.children or 0,
            children_ages=context.children_ages,
        )

        try:
            started = time.perf_counter()
            offers = await self._booking_service.get_quotes(
                check_in=context.checkin,
                check_out=context.checkout,
                guests=guests,
            )
            debug["shelter_called"] = True
            debug["shelter_latency_ms"] = int((time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["shelter_called"] = True
            debug["shelter_error"] = str(exc)
            context.state = BookingState.ASK_CHECKIN
            return "Не получилось получить расчёт, давайте попробуем ещё раз. На какую дату планируете заезд?"

        if not offers:
            context.state = BookingState.DONE
            return "К сожалению, нет доступных вариантов на выбранные даты. Если хотите изменить параметры, скажите \"начнём заново\"."

        booking_entities = BookingEntities(
            checkin=context.checkin,
            checkout=context.checkout,
            adults=context.adults,
            children=context.children or 0,
            nights=nights,
            room_type=context.room_type,
            missing_fields=[],
        )
        
        # Сохраняем уникальные офферы в контексте для функции "покажи все"
        unique_offers = self._formatting_service.select_min_offer_per_room_type(offers)
        sorted_offers = sorted(unique_offers, key=lambda o: o.total_price)
        context.offers = [
            {
                "room_name": o.room_name,
                "total_price": o.total_price,
                "currency": o.currency,
                "breakfast_included": o.breakfast_included,
                "room_area": o.room_area,
                "check_in": o.check_in,
                "check_out": o.check_out,
                "guests": {"adults": o.guests.adults, "children": o.guests.children},
            }
            for o in sorted_offers
        ]
        context.last_offer_index = min(3, len(sorted_offers))  # Показали первые 3
        
        price_block = self._formatting_service.format_booking_quote(booking_entities, offers)
        context.state = BookingState.AWAITING_USER_DECISION
        return price_block

    def _handle_confirmation(
        self, text: str, context: BookingContext, parsers: ParsedMessageCache
    ) -> str:
        return self._handle_post_quote_decision(text, context, parsers)

    def _handle_post_quote_decision(
        self, text: str, context: BookingContext, parsers: ParsedMessageCache
    ) -> str:
        normalized = text.strip().lower()
        room_type = parsers.room_type()
        booking_intent = any(
            token in normalized
            for token in {
                "забронировать",
                "бронировать",
                "оформляй",
                "оформляем",
                "оформляю",
                "берем",
                "берём",
                "возьми",
            }
        )

        if room_type:
            context.room_type = room_type

        if booking_intent or room_type:
            context.state = BookingState.DONE
            selection = f"Вы выбрали тип: {context.room_type}." if context.room_type else ""
            note = (
                "Я показываю цены и варианты. Оформить бронь можно по ссылке "
                "https://usadba4.ru/bronirovanie/."
            )
            return " ".join(filter(None, [selection, note, "Если нужно изменить даты, скажите 'начнём заново'."]))

        # Обработка запроса "покажи все" / "покажи больше вариантов"
        show_more_triggers = {
            "покажи все",
            "покажи всё",
            "показать все",
            "показать всё",
            "покажи больше",
            "показать больше",
            "ещё варианты",
            "еще варианты",
            "другие варианты",
            "остальные",
            "все варианты",
        }
        if any(trigger in normalized for trigger in show_more_triggers):
            return self._show_more_offers(context)

        if "дат" in normalized:
            context.state = BookingState.ASK_CHECKIN
            context.checkin = None
            context.checkout = None
            context.nights = None
            return self._booking_prompt("Изменим даты. На какую дату планируете заезд?", context)
        if "гост" in normalized or "люд" in normalized:
            context.state = BookingState.ASK_ADULTS
            context.adults = None
            context.children = None
            context.children_ages = []
            return self._booking_prompt("Сколько взрослых едет?", context)

        context.state = BookingState.AWAITING_USER_DECISION
        return (
            "Если хотите изменить параметры, напишите новые даты или количество гостей. "
            "Чтобы забронировать, воспользуйтесь ссылкой https://usadba4.ru/bronirovanie/."
        )

    def _show_more_offers(self, context: BookingContext) -> str:
        """Показывает оставшиеся офферы из сохранённого списка."""
        if not context.offers:
            return (
                "У меня нет сохранённых вариантов. "
                "Если хотите изменить параметры, напишите новые даты или количество гостей."
            )

        start_idx = context.last_offer_index
        if start_idx >= len(context.offers):
            context.state = BookingState.AWAITING_USER_DECISION
            return (
                "Вы уже видели все доступные предложения. "
                "Если хотите изменить параметры, напишите новые даты или количество гостей."
            )

        # Восстанавливаем BookingQuote из сохранённых dict'ов
        remaining_dicts = context.offers[start_idx:]
        offers_to_show = []
        for o in remaining_dicts:
            guests_data = o.get("guests", {})
            guests = Guests(
                adults=guests_data.get("adults", 2),
                children=guests_data.get("children", 0),
            )
            offers_to_show.append(
                BookingQuote(
                    room_name=o.get("room_name", "Номер"),
                    total_price=o.get("total_price", 0),
                    currency=o.get("currency", "RUB"),
                    breakfast_included=o.get("breakfast_included", False),
                    room_area=o.get("room_area"),
                    check_in=o.get("check_in", ""),
                    check_out=o.get("check_out", ""),
                    guests=guests,
                )
            )

        text, new_index = self._formatting_service.format_more_offers(offers_to_show, 0)
        context.last_offer_index = start_idx + new_index
        context.state = BookingState.AWAITING_USER_DECISION
        return text

    def _is_cancel_command(self, normalized: str) -> bool:
        return normalized in {
            "отмена",
            "отменить",
            "стоп",
            "cancel",
            "отмени",
            "начать заново",
            "начнём заново",
            "начнем заново",
            "сброс",
            "сбросить",
        }

    def _is_back_command(self, normalized: str) -> bool:
        return normalized in {"назад", "вернись", "вернуться"}

    def _next_booking_question(self, state: SlotState) -> str | None:
        if not state.check_in:
            return "checkin"
        if state.nights is None and not state.check_out:
            return "checkout_or_nights"
        if state.adults is None:
            return "adults"
        if state.children is None:
            return "children"
        if state.children > 0 and not state.children_ages:
            return "children_ages"
        return None

    def _build_booking_prompt(
        self, state: SlotState, slot: str, prefix: str | None = None
    ) -> str:
        summary = self._summary_line(state)
        question_map = {
            "checkin": "На какую дату планируете заезд?",
            "checkout_or_nights": "Сколько ночей остаётесь или до какого числа?",
            "adults": "Сколько взрослых едет?",
            "children": "Сколько детей? Если детей нет — напишите 0.",
            "children_ages": "Уточните возраст детей (через запятую).",
        }

        prompt = question_map.get(slot, "Подскажите детали бронирования, пожалуйста.")
        parts: list[str] = []
        if summary:
            parts.append(f"Понял: {summary}.")
        if prefix:
            parts.append(prefix)
        parts.append(prompt)
        return " ".join(parts).strip()

    def _summary_line(self, state: SlotState, limit: int = 3) -> str:
        fragments: list[str] = []
        if state.check_in:
            fragments.append(f"заезд {self._format_date(state.check_in)}")
        if state.nights:
            fragments.append(f"ночей {state.nights}")
        elif state.check_out:
            fragments.append(f"выезд {self._format_date(state.check_out)}")

        if state.adults is not None:
            guests = f"взрослых {state.adults}"
            if state.children is not None:
                guests += f", детей {state.children}"
            fragments.append(guests)

        if state.room_type:
            fragments.append(f"тип {state.room_type}")

        return ", ".join(fragments[:limit])

    def _format_date(self, date_str: str) -> str:
        try:
            parsed = date.fromisoformat(date_str)
        except ValueError:
            return date_str
        month_names = [
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ]
        return f"{parsed.day} {month_names[parsed.month - 1]}"

    def _apply_children_answer(self, text: str, state: SlotState) -> None:
        if state.children is not None:
            return
        lowered = text.strip().lower()
        negative_children = {"нет", "неа", "нету", "не будет", "без детей"}
        if lowered in negative_children or "нет детей" in lowered:
            state.children = 0

    def _next_missing_slot(self, state: SlotState) -> str | None:
        for field in ("check_in", "check_out", "adults", "children"):
            if getattr(state, field) in (None, ""):
                return field
        return None

    def _question_for_slot(self, slot: str, state: SlotState) -> str:
        summary = self._summary_line(state)
        question_map = {
            "check_in": "На какую дату заезд?",
            "check_out": "До какого числа остаетесь?",
            "adults": "Сколько будет взрослых?",
            "children": "Сколько детей? Если детей нет — напишите 0.",
        }
        parts: list[str] = []
        if summary:
            parts.append(f"Понял: {summary}.")
        parts.append(question_map.get(slot, "Уточните детали бронирования."))
        return " ".join(parts)

    async def handle_booking(self, session_id: str, text: str) -> dict[str, Any]:
        state = self._store.get(session_id) or SlotState()
        state = self._parsing_service.extract_slot_state(text, state)
        self._parsing_service.apply_children_answer(text, state)
        # Используем slot_filler из зависимостей
        # TODO: передать slot_filler в ParsingService или использовать напрямую
        from app.booking.slot_filling import SlotFiller
        slot_filler = SlotFiller()
        missing = slot_filler.missing_slots(state)
        self._store.set(session_id, state)

        next_slot = self._next_missing_slot(state)
        if next_slot:
            question = self._question_for_slot(next_slot, state)
            return {
                "answer": question,
                "debug": {
                    "intent": "booking_quote",
                    "slots": state.as_dict(),
                    "missing_fields": missing,
                    "pms_called": False,
                    "offers_count": 0,
                },
            }

        guests = state.guests()
        if not guests:
            return {
                "answer": "Не удалось распознать параметры бронирования. Уточните даты и количество гостей.",
                "debug": {
                    "intent": "booking_quote",
                    "slots": state.as_dict(),
                    "missing_fields": missing,
                    "pms_called": False,
                    "offers_count": 0,
                },
            }

        offers = await self._booking_service.get_quotes(
            check_in=state.check_in or "",
            check_out=state.check_out or "",
            guests=guests,
        )
        self._store.clear(session_id)

        if not offers:
            return {
                "answer": "К сожалению, нет доступных вариантов на указанные даты.",
                "debug": {
                    "intent": "booking_quote",
                    "slots": state.as_dict(),
                    "missing_fields": [],
                    "pms_called": True,
                    "offers_count": 0,
                },
            }

        entities = BookingEntities(
            checkin=state.check_in,
            checkout=state.check_out,
            adults=state.adults or 2,
            children=state.children,
        )
        answer = self._formatting_service.format_booking_quote(entities, offers)
        return {
            "answer": answer,
            "debug": {
                "intent": "booking_quote",
                "slots": state.as_dict(),
                "missing_fields": [],
                "pms_called": True,
                "offers_count": len(offers),
            },
        }

    async def handle_general(
        self, 
        text: str, 
        *, 
        intent: str = "general",
        session_id: str = "anonymous",
    ) -> dict[str, Any]:
        """
        Обрабатывает общие вопросы с поддержкой истории диалога и кэширования.
        
        Args:
            text: Текст сообщения пользователя
            intent: Определённый intent
            session_id: ID сессии для истории диалога
        """
        detail_mode = self._formatting_service.detect_detail_mode(text)

        rag_hits = await gather_rag_data(
            query=text,
            client=self._qdrant,
            pool=self._pool,
            facts_limit=self._settings.rag_facts_limit,
            files_limit=self._settings.rag_files_limit,
            faq_limit=3,
            faq_min_similarity=0.35,
            intent=intent,
        )

        qdrant_hits = rag_hits.get("qdrant_hits")
        if qdrant_hits is None:
            qdrant_hits = [
                *rag_hits.get("facts_hits", []),
                *rag_hits.get("files_hits", []),
            ]
        faq_hits = rag_hits.get("faq_hits", [])

        hits_total = rag_hits.get("hits_total", len(qdrant_hits) + len(faq_hits))

        max_snippets = max(1, self._settings.rag_max_snippets)
        facts_hits = qdrant_hits[:max_snippets]
        files_hits: list[dict[str, Any]] = []
        context_text = build_context(
            facts_hits=facts_hits,
            files_hits=files_hits,
            faq_hits=faq_hits,
        )

        system_prompt = FACTS_PROMPT
        if context_text:
            system_prompt = f"{FACTS_PROMPT}\n\n{context_text}"

        debug: dict[str, Any] = {
            "intent": intent or "general",
            "context_length": len(context_text),
            "facts_hits": len(facts_hits),
            "files_hits": len(files_hits),
            "qdrant_hits": len(qdrant_hits),
            "faq_hits": len(faq_hits),
            "rag_min_facts": self._settings.rag_min_facts,
            "hits_total": hits_total,
            "guard_triggered": False,
            "llm_called": False,
            "llm_cache_hit": False,
            "history_used": False,
        }
        debug["rag_latency_ms"] = rag_hits.get("rag_latency_ms", 0)
        debug["embed_latency_ms"] = rag_hits.get("embed_latency_ms", 0)
        if rag_hits.get("embed_error"):
            debug["embed_error"] = rag_hits["embed_error"]
        debug["raw_qdrant_hits"] = rag_hits.get("raw_qdrant_hits", [])
        debug["score_threshold_used"] = rag_hits.get("score_threshold_used")
        debug["expanded_queries"] = rag_hits.get("expanded_queries", [])
        debug["merged_hits_count"] = rag_hits.get("merged_hits_count", 0)
        debug["boosting_applied"] = rag_hits.get("boosting_applied", False)
        debug["intent_detected"] = rag_hits.get("intent_detected") or intent

        if hits_total < self._settings.rag_min_facts:
            debug["guard_triggered"] = True
            if intent == "lodging":
                answer = (
                    "Я не нашёл подтверждённой информации о домиках или номерах в базе знаний. "
                    "Если загрузите файл или страницу с типами размещения, ценами и вместимостью, я смогу отвечать точнее."
                )
            else:
                answer = (
                    "Я не нашёл подтверждённой информации в базе знаний, поэтому не буду выдумывать. "
                    "Уточните, пожалуйста: даты заезда и выезда, количество гостей, тип размещения или бюджет? "
                    "Если вам нужна баня/сауна или дополнительные услуги — тоже сообщите. "
                    "Если вы загрузили описание номеров/домиков в базу — скажите 'покажи варианты из базы'."
                )

            final_answer = self._formatting_service.postprocess_answer(
                answer, mode="detail" if detail_mode else "brief"
            )
            return {"answer": final_answer, "debug": debug}

        # Проверяем LLM кэш
        if self._settings.llm_cache_enabled:
            llm_cache = get_llm_cache()
            cached_answer, cached_debug = await llm_cache.get(text, intent, context_text)
            if cached_answer:
                debug["llm_cache_hit"] = True
                debug["llm_called"] = False
                if cached_debug:
                    debug.update({k: v for k, v in cached_debug.items() if k not in debug})
                final_answer = self._formatting_service.postprocess_answer(
                    cached_answer,
                    mode="detail" if detail_mode else "brief",
                )
                # Сохраняем в историю даже для кэшированных ответов
                await self._save_to_history(session_id, "user", text)
                await self._save_to_history(session_id, "assistant", final_answer)
                return {"answer": final_answer, "debug": debug}

        # Получаем историю диалога
        history = await self._get_conversation_history(session_id)
        
        # Формируем сообщения с историей
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Добавляем историю (последние N сообщений)
        history_limit = min(len(history), self._settings.conversation_history_limit)
        if history_limit > 0:
            messages.extend(history[-history_limit:])
            debug["history_used"] = True
            debug["history_messages_count"] = history_limit
        
        # Текущее сообщение
        messages.append({"role": "user", "content": text})

        debug["llm_called"] = True
        try:
            llm_started = time.perf_counter()
            answer = await self._llm.chat(
                model=self._settings.amvera_model, messages=messages
            )
            debug["llm_latency_ms"] = int((time.perf_counter() - llm_started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["llm_error"] = str(exc)
            rag_answer = self._build_rag_only_answer(
                qdrant_hits=qdrant_hits,
                faq_hits=faq_hits,
                rag_hits=rag_hits,
            )
            if rag_answer:
                answer = self._formatting_service.postprocess_answer(
                    rag_answer, mode="detail" if detail_mode else "brief"
                )
                return {"answer": answer, "debug": debug}
            return {
                "answer": "Сейчас не удалось получить ответ из LLM. Попробуйте уточнить запрос чуть позже.",
                "debug": debug,
            }

        final_answer = self._formatting_service.postprocess_answer(
            answer or "Нет данных в базе знаний.",
            mode="detail" if detail_mode else "brief",
        )

        # Сохраняем в LLM кэш
        if self._settings.llm_cache_enabled and answer:
            llm_cache = get_llm_cache()
            await llm_cache.set(
                text, intent, context_text, answer,
                debug_info={"llm_latency_ms": debug.get("llm_latency_ms", 0)}
            )

        # Сохраняем в историю диалога
        await self._save_to_history(session_id, "user", text)
        await self._save_to_history(session_id, "assistant", final_answer)

        return {"answer": final_answer, "debug": debug}
    
    async def _get_conversation_history(self, session_id: str) -> list[dict[str, str]]:
        """Получает историю диалога из Redis (если доступно)."""
        if not self._settings.use_redis_state_store:
            return []
        
        try:
            # Проверяем, является ли store Redis-based
            if hasattr(self._store, "get_history"):
                return await self._store.get_history(session_id)
        except Exception as exc:
            logger.warning("Failed to get conversation history: %s", exc)
        
        return []
    
    async def _save_to_history(self, session_id: str, role: str, content: str) -> None:
        """Сохраняет сообщение в историю диалога."""
        if not self._settings.use_redis_state_store:
            return
        
        try:
            if hasattr(self._store, "add_message"):
                await self._store.add_message(session_id, role, content)
        except Exception as exc:
            logger.warning("Failed to save message to history: %s", exc)

    def _build_rag_only_answer(
        self,
        *,
        qdrant_hits: list[dict[str, Any]],
        faq_hits: list[dict[str, Any]],
        rag_hits: dict[str, Any],
    ) -> str:
        merged_hits_count = rag_hits.get("merged_hits_count")
        hits_total = rag_hits.get("hits_total")
        if merged_hits_count is None:
            merged_hits_count = len(qdrant_hits)
        if hits_total is None:
            hits_total = len(qdrant_hits) + len(faq_hits)

        if not (qdrant_hits or faq_hits):
            return ""

        if merged_hits_count < max(1, self._settings.rag_min_facts) and hits_total < 1:
            return ""

        candidates: list[tuple[int, float, str, str]] = []

        for faq in faq_hits:
            answer = (faq.get("answer") or "").strip()
            if not answer:
                continue
            # Для FAQ показываем только ответ, без вопроса
            candidates.append((0, float(faq.get("similarity", 0.0) or 0.0), answer, answer))

        for hit in qdrant_hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
            type_value = (hit.get("type") or payload.get("type") or "").strip()
            source = (hit.get("source") or payload.get("source") or "").strip()

            priority = 2
            if type_value in {"faq", "faq_ext"}:
                priority = 0
            elif source.startswith("knowledge") or source.endswith(".md") or ".md" in source:
                priority = 1

            # Извлекаем чистый текст без технических метаданных
            clean_text = self._extract_clean_text(text)
            candidates.append((priority, float(hit.get("score", 0.0) or 0.0), clean_text, clean_text))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: (item[0], -item[1]))
        selected = candidates[:4]

        answer_lines = [f"• {item[2]}" for item in selected if item[2]]

        restriction_keywords = [
            "только для проживающих",
            "только для гостей",
            "по предварительной записи",
            "по предзаказу",
            "предоплата",
            "депозит",
            "залог",
            "по запросу",
            "доступно по записи",
        ]
        important_notes: list[str] = []
        for _, _, _, raw_text in selected:
            lowered = raw_text.lower()
            for keyword in restriction_keywords:
                if keyword in lowered and keyword not in important_notes:
                    important_notes.append(keyword)
        if important_notes:
            answer_lines.append("Важно:")
            for note in important_notes[:2]:
                answer_lines.append(f"• {note}")

        return "\n".join(answer_lines)

    def _extract_clean_text(self, text: str) -> str:
        """Извлекает чистый текст без технических метаданных."""
        # Если текст содержит Q: и A:, извлекаем только ответ
        if "Q:" in text and "A:" in text:
            # Формат: "Q: вопрос? A: ответ"
            parts = text.split("A:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return text

    async def handle_knowledge(
        self, 
        text: str,
        *,
        session_id: str = "anonymous",
    ) -> dict[str, Any]:
        """
        Обрабатывает запросы к базе знаний с поддержкой истории и кэширования.
        """
        rag_hits = await gather_rag_data(
            query=text,
            client=self._qdrant,
            pool=self._pool,
            facts_limit=self._settings.rag_facts_limit,
            files_limit=self._settings.rag_files_limit,
            faq_limit=3,
            faq_min_similarity=0.35,
            intent="knowledge_lookup",
        )

        qdrant_hits = rag_hits.get("qdrant_hits") or rag_hits.get("facts_hits", [])
        faq_hits = rag_hits.get("faq_hits", [])
        facts_hits = rag_hits.get("facts_hits") or qdrant_hits
        files_hits = rag_hits.get("files_hits", [])
        total_hits = len(qdrant_hits) + len(faq_hits)

        debug: dict[str, Any] = {
            "intent": "knowledge_lookup",
            "hits_total": rag_hits.get("hits_total", total_hits),
            "facts_hits": len(rag_hits.get("facts_hits", [])),
            "files_hits": len(rag_hits.get("files_hits", [])),
            "qdrant_hits": len(qdrant_hits),
            "faq_hits": len(faq_hits),
            "rag_latency_ms": rag_hits.get("rag_latency_ms", 0),
            "embed_latency_ms": rag_hits.get("embed_latency_ms", 0),
            "raw_qdrant_hits": rag_hits.get("raw_qdrant_hits", []),
            "score_threshold_used": rag_hits.get("score_threshold_used"),
            "expanded_queries": rag_hits.get("expanded_queries", []),
            "merged_hits_count": rag_hits.get("merged_hits_count", 0),
            "boosting_applied": rag_hits.get("boosting_applied", False),
            "intent_detected": rag_hits.get("intent_detected", "knowledge_lookup"),
            "llm_cache_hit": False,
            "history_used": False,
        }
        if rag_hits.get("embed_error"):
            debug["embed_error"] = rag_hits["embed_error"]

        hits_total = debug["hits_total"]
        if hits_total < max(1, self._settings.rag_min_facts):
            fallback_answer = (
                "Я не нашёл подтверждённых сведений в базе знаний по этому вопросу. "
                "Попробуйте уточнить запрос или загрузить описание с нужной информацией."
            )
            return {
                "answer": self._finalize_short_answer(fallback_answer),
                "debug": {**debug, "guard_triggered": True, "llm_called": False},
            }

        max_snippets = max(1, self._settings.rag_max_snippets)
        context_text = build_context(
            facts_hits=facts_hits[:max_snippets],
            files_hits=files_hits[:max_snippets],
            faq_hits=faq_hits,
        )

        system_prompt_parts = [
            FACTS_PROMPT,
            (
                "Отвечай одним цельным текстом на 2–4 предложения. "
                "Используй переданный контекст только для понимания ответа и не перечисляй файлы, блоки или пары вопрос-ответ. "
                "В конце можешь добавить фразу «Если хотите — расскажу подробнее»."
            ),
        ]
        if context_text:
            system_prompt_parts.append(context_text)

        system_prompt = "\n\n".join(part for part in system_prompt_parts if part)

        # Проверяем LLM кэш
        if self._settings.llm_cache_enabled:
            llm_cache = get_llm_cache()
            cached_answer, cached_debug = await llm_cache.get(text, "knowledge_lookup", context_text)
            if cached_answer:
                debug["llm_cache_hit"] = True
                debug["llm_called"] = False
                final_answer = self._finalize_short_answer(cached_answer)
                await self._save_to_history(session_id, "user", text)
                await self._save_to_history(session_id, "assistant", final_answer)
                return {"answer": final_answer, "debug": debug}

        # Получаем историю
        history = await self._get_conversation_history(session_id)
        
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        
        history_limit = min(len(history), self._settings.conversation_history_limit)
        if history_limit > 0:
            messages.extend(history[-history_limit:])
            debug["history_used"] = True
            debug["history_messages_count"] = history_limit
        
        messages.append({"role": "user", "content": text})

        debug["llm_called"] = True
        try:
            llm_started = time.perf_counter()
            answer = await self._llm.chat(
                model=self._settings.amvera_model, messages=messages
            )
            debug["llm_latency_ms"] = int((time.perf_counter() - llm_started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["llm_error"] = str(exc)
            generic_answer = (
                "Не получилось сформировать ответ, но я продолжу искать нужные данные. "
                "Попробуйте чуть позже или уточните вопрос."
            )
            return {
                "answer": self._finalize_short_answer(generic_answer),
                "debug": debug,
            }

        final_answer = self._finalize_short_answer(
            answer or "Информация из базы пока не найдена."
        )

        # Кэшируем ответ
        if self._settings.llm_cache_enabled and answer:
            llm_cache = get_llm_cache()
            await llm_cache.set(
                text, "knowledge_lookup", context_text, answer,
                debug_info={"llm_latency_ms": debug.get("llm_latency_ms", 0)}
            )

        # Сохраняем в историю
        await self._save_to_history(session_id, "user", text)
        await self._save_to_history(session_id, "assistant", final_answer)

        return {"answer": final_answer, "debug": debug}

    def _finalize_short_answer(self, answer: str) -> str:
        cleaned = (answer or "").strip()
        if not cleaned:
            return "Информации пока нет, но могу поискать ещё. Если хотите — расскажу подробнее."

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        normalized = [sentence.strip() for sentence in sentences if sentence.strip()]

        if len(normalized) > 4:
            cleaned = " ".join(normalized[:4])
        elif normalized:
            cleaned = " ".join(normalized)

        if not cleaned.endswith(".") and not cleaned.endswith("!") and not cleaned.endswith("?"):
            cleaned = f"{cleaned}."

        if "Если хотите — расскажу подробнее." not in cleaned:
            cleaned = f"{cleaned} Если хотите — расскажу подробнее."

        return cleaned


__all__ = [
    "ConversationStateStore",
    "InMemoryConversationStateStore",
    "ChatComposer",
]
