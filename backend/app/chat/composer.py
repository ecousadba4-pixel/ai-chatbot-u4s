from typing import Any

import logging
import time
from datetime import date, datetime, timedelta

import asyncpg

from app.core.config import Settings, get_settings
from app.booking.entities import BookingEntities
from app.booking.fsm import BookingContext, BookingState, initial_booking_context
from app.booking.models import Guests
from app.booking.service import BookingQuoteService
from app.chat.formatting import (
    detect_detail_mode,
    format_shelter_quote,
    postprocess_answer,
)
from app.booking.parsers import (
    extract_guests,
    parse_adults,
    parse_checkin,
    parse_children_ages,
    parse_children_count,
    parse_nights,
    parse_room_type,
)
from app.booking.slot_filling import CHILDREN_PATTERNS, SlotFiller, SlotState
from app.llm.amvera_client import AmveraLLMClient
from app.llm.prompts import FACTS_PROMPT
from app.rag.context_builder import build_context
from app.rag.qdrant_client import QdrantClient
from app.rag.retriever import gather_rag_data


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
        self._slot_filler = slot_filler
        self._booking_service = booking_service
        self._store = store
        self._booking_store = booking_fsm_store or store
        self._settings = settings or get_settings()
        self._max_state_attempts = max_state_attempts

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
        context = self._load_booking_context(session_id)
        debug = self._booking_debug(context)
        answer = await self._handle_booking_message(
            session_id, text, context, entities, debug
        )
        debug["booking_state"] = context.state.value if context.state else ""
        debug["booking_entities"] = self._context_entities(context)
        debug["missing_fields"] = self._missing_context_fields(context)
        return {"answer": answer, "debug": debug}

    def _missing_booking_fields(self, state: SlotState) -> list[str]:
        missing: list[str] = []
        if not state.check_in:
            missing.append("checkin")
        if not state.check_out and not state.nights:
            missing.append("checkout_or_nights")
        if state.adults is None:
            missing.append("adults")
        if state.children is None:
            missing.append("children")
        if (state.children or 0) > 0 and not state.children_ages:
            missing.append("children_ages")
        if state.room_type is None:
            missing.append("room_type")
        return missing

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

    def _load_booking_context(self, session_id: str) -> BookingContext:
        context = BookingContext.from_dict(self._booking_store.get(session_id))
        if context is None:
            return initial_booking_context()
        return context

    def _save_booking_context(self, session_id: str, context: BookingContext) -> None:
        context.updated_at = datetime.utcnow().timestamp()
        self._booking_store.set(session_id, context.to_dict())

    def _booking_debug(self, context: BookingContext) -> dict[str, Any]:
        return {
            "intent": "booking_calculation",
            "booking_state": context.state.value if context.state else "",
            "booking_entities": self._context_entities(context),
            "missing_fields": self._missing_context_fields(context),
            "shelter_called": False,
            "shelter_latency_ms": 0,
            "shelter_error": None,
            "llm_called": False,
        }

    def _context_entities(self, context: BookingContext) -> dict[str, Any]:
        return {
            "checkin": context.checkin,
            "checkout": context.checkout,
            "nights": context.nights,
            "adults": context.adults,
            "children": context.children,
            "children_ages": context.children_ages,
            "room_type": context.room_type,
            "promo": context.promo,
        }

    def _missing_context_fields(self, context: BookingContext) -> list[str]:
        missing: list[str] = []
        if not context.checkin:
            missing.append("checkin")
        if not context.checkout and context.nights is None:
            missing.append("checkout_or_nights")
        if context.adults is None:
            missing.append("adults")
        if context.children is None:
            missing.append("children")
        if (context.children or 0) > 0 and not context.children_ages:
            missing.append("children_ages")
        if context.room_type is None:
            missing.append("room_type")
        return missing

    async def _handle_booking_message(
        self,
        session_id: str,
        text: str,
        context: BookingContext,
        entities: BookingEntities,
        debug: dict[str, Any],
    ) -> str:
        normalized = text.strip().lower()
        if self._is_cancel_command(normalized):
            context.state = BookingState.CANCELLED
            self._booking_store.clear(session_id)
            return "Отменяю бронирование. Если понадобится помощь, напишите."

        if context.state in (None, BookingState.DONE, BookingState.CANCELLED):
            context.state = BookingState.ASK_CHECKIN

        if self._is_back_command(normalized):
            self._go_back(context)

        logger.info(
            "BOOKING_FSM state=%s ctx=%s message=%s",
            context.state,
            context.compact(),
            text,
        )

        self._apply_entities_to_context(context, entities)
        self._apply_entities_from_message(context, text)

        answer = await self._advance_booking_fsm(session_id, context, text, debug)
        self._save_booking_context(session_id, context)
        return answer

    def _apply_entities_to_context(
        self, context: BookingContext, entities: BookingEntities
    ) -> None:
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

    def _apply_entities_from_message(self, context: BookingContext, text: str) -> None:
        if not context.checkin:
            context.checkin = parse_checkin(text)
        if context.nights is None and not context.checkout:
            context.nights = parse_nights(text)
        if not context.checkout and context.checkin:
            try:
                checkin_date = date.fromisoformat(context.checkin)
            except ValueError:
                checkin_date = None
            parsed_checkout = parse_checkin(text, now_date=checkin_date or date.today())
            if parsed_checkout and parsed_checkout != context.checkin:
                try:
                    checkout_date = date.fromisoformat(parsed_checkout)
                except ValueError:
                    checkout_date = None
                if checkout_date and checkin_date and checkout_date > checkin_date:
                    context.checkout = parsed_checkout
        if context.adults is None:
            allow_general_adults = context.state in {
                BookingState.ASK_ADULTS,
                BookingState.ASK_CHILDREN_COUNT,
                BookingState.ASK_CHILDREN_AGES,
                BookingState.ASK_ROOM_TYPE,
                BookingState.CALCULATE,
                BookingState.CONFIRM_BOOKING,
            }
            adults = parse_adults(text, allow_general_numbers=allow_general_adults)
            if adults is not None:
                context.adults = adults
        if context.children is None and context.state in {
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
        }:
            context.children = parse_children_count(text)
        if (
            (context.children or 0) > 0
            and not context.children_ages
            and context.state == BookingState.ASK_CHILDREN_AGES
        ):
            context.children_ages = parse_children_ages(text, expected=context.children)
        if context.room_type is None:
            context.room_type = parse_room_type(text)

    def _ask_with_retry(self, context: BookingContext, state: BookingState, question: str) -> str:
        attempts = context.retries.get(state.value, 0) + 1
        context.retries[state.value] = attempts
        return self._booking_prompt(question, context)

    def _go_back(self, context: BookingContext) -> None:
        previous = self._previous_state(context.state)
        if previous == BookingState.ASK_CHECKIN:
            context.checkin = None
            context.nights = None
            context.checkout = None
        if previous == BookingState.ASK_NIGHTS_OR_CHECKOUT:
            context.nights = None
            context.checkout = None
        if previous == BookingState.ASK_ADULTS:
            context.adults = None
        if previous == BookingState.ASK_CHILDREN_COUNT:
            context.children = None
            context.children_ages = []
        if previous == BookingState.ASK_CHILDREN_AGES:
            context.children_ages = []
        if previous is not None:
            context.state = previous

    def _previous_state(self, state: BookingState | None) -> BookingState | None:
        order = [
            BookingState.ASK_CHECKIN,
            BookingState.ASK_NIGHTS_OR_CHECKOUT,
            BookingState.ASK_ADULTS,
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
            BookingState.ASK_ROOM_TYPE,
            BookingState.CALCULATE,
            BookingState.CONFIRM_BOOKING,
        ]
        if state in order:
            idx = order.index(state)
            return order[idx - 1] if idx > 0 else BookingState.ASK_CHECKIN
        return BookingState.ASK_CHECKIN

    async def _advance_booking_fsm(
        self,
        session_id: str,
        context: BookingContext,
        text: str,
        debug: dict[str, Any],
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
                parsed = parse_checkin(text)
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
                nights = parse_nights(text)
                checkout_value = None
                try:
                    checkin_date = date.fromisoformat(context.checkin) if context.checkin else None
                except ValueError:
                    checkin_date = None
                if checkin_date:
                    parsed_checkout = parse_checkin(text, now_date=checkin_date)
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
                guests_from_text = extract_guests(text)
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
                adults = parse_adults(text, allow_general_numbers=allow_general)
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
                guests_from_text = extract_guests(text)
                children_from_text = guests_from_text.get("children")
                adults_from_text = guests_from_text.get("adults")
                if adults_from_text is not None:
                    context.adults = adults_from_text
                if children_from_text is not None:
                    context.children = children_from_text
                    if children_from_text <= 0:
                        context.children_ages = []

                lowered_input = text.lower()
                if context.children is not None:
                    if (context.children or 0) > 0:
                        if context.children_ages and len(context.children_ages) == context.children:
                            state = BookingState.ASK_ROOM_TYPE
                            continue
                        if "взросл" not in lowered_input:
                            ages = parse_children_ages(text, expected=context.children)
                            if ages:
                                context.children_ages = ages
                                state = BookingState.ASK_ROOM_TYPE
                                context.state = BookingState.ASK_ROOM_TYPE
                                continue
                        context.state = BookingState.ASK_CHILDREN_AGES
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_AGES,
                            "Уточните возраст детей (через запятую).",
                        )
                    else:
                        if context.room_type is None:
                            context.room_type = "Студия"
                        state = BookingState.CALCULATE
                    continue
                children = parse_children_count(text)
                if children is not None:
                    context.children = children
                    if children > 0:
                        context.state = BookingState.ASK_CHILDREN_AGES
                        return self._ask_with_retry(
                            context,
                            BookingState.ASK_CHILDREN_AGES,
                            "Уточните возраст детей (через запятую).",
                        )
                    if context.room_type is None:
                        context.room_type = "Студия"
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
                    state = BookingState.ASK_ROOM_TYPE
                    continue
                if context.children_ages and len(context.children_ages) == context.children:
                    state = BookingState.ASK_ROOM_TYPE
                    continue
                ages = parse_children_ages(text, expected=context.children)
                if ages:
                    context.children_ages = ages
                    state = BookingState.ASK_ROOM_TYPE
                    context.state = BookingState.ASK_ROOM_TYPE
                    continue
                return self._ask_with_retry(
                    context,
                    BookingState.ASK_CHILDREN_AGES,
                    "Не услышал возраст детей, укажите числа через запятую.",
                )

            if state == BookingState.ASK_ROOM_TYPE:
                context.state = BookingState.ASK_ROOM_TYPE
                if context.room_type:
                    state = BookingState.CALCULATE
                    continue
                room_type = parse_room_type(text)
                if room_type:
                    context.room_type = room_type
                    state = BookingState.CALCULATE
                    continue
                return self._ask_with_retry(
                    context,
                    BookingState.ASK_ROOM_TYPE,
                    "Какой тип размещения предпочитаете: Студия, Шале, Шале Комфорт или Семейный номер?",
                )

            if state == BookingState.CALCULATE:
                context.state = BookingState.CALCULATE
                answer = await self._calculate_booking(context, debug)
                return answer

            if state == BookingState.CONFIRM_BOOKING:
                context.state = BookingState.CONFIRM_BOOKING
                return self._handle_confirmation(text, context)

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

        if context.room_type is None:
            context.room_type = "Студия"

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
        price_block = format_shelter_quote(booking_entities, offers)
        context.state = BookingState.CONFIRM_BOOKING
        return f"{price_block}\n\nОформляем бронирование?"

    def _handle_confirmation(self, text: str, context: BookingContext) -> str:
        normalized = text.strip().lower()
        if any(token in normalized for token in {"да", "оформляй", "подтверждаю", "ок"}):
            context.state = BookingState.DONE
            return "Отлично, фиксирую бронирование. Если захотите изменить детали, скажите \"начнём заново\"."
        if any(token in normalized for token in {"нет", "отмена", "стоп", "не"}):
            context.state = BookingState.DONE
            return "Ок, если захотите изменить детали, скажите \"начнём заново\"."
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
        return self._ask_with_retry(
            context, BookingState.CONFIRM_BOOKING, "Оформляем бронирование?"
        )

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
        if state.room_type is None:
            return "room_type"
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
            "room_type": "Какой тип размещения выбрать: Студия, Шале, Шале Комфорт или Семейный номер?",
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
        state = self._slot_filler.extract(text, state)
        self._apply_children_answer(text, state)
        missing = self._slot_filler.missing_slots(state)
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

        summary_lines = []
        for offer in offers:
            line = f"{offer.room_name}: {offer.total_price:.0f} {offer.currency}"
            if offer.breakfast_included:
                line += " (завтрак включён)"
            if offer.room_area:
                line += f", площадь {offer.room_area} м²"
            summary_lines.append(line)

        answer = "\n".join(summary_lines)
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

    async def handle_general(self, text: str, *, intent: str = "general") -> dict[str, Any]:
        detail_mode = detect_detail_mode(text)

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
                    "Если вы загрузили описание номеров/домиков в базу — скажите ‘покажи варианты из базы’."
                )

            final_answer = postprocess_answer(
                answer, mode="detail" if detail_mode else "brief"
            )
            return {"answer": final_answer, "debug": debug}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

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
                answer = postprocess_answer(
                    rag_answer, mode="detail" if detail_mode else "brief"
                )
                return {"answer": answer, "debug": debug}
            return {
                "answer": "Сейчас не удалось получить ответ из LLM. Попробуйте уточнить запрос чуть позже.",
                "debug": debug,
            }

        final_answer = postprocess_answer(
            answer or "Нет данных в базе знаний.",
            mode="detail" if detail_mode else "brief",
        )

        return {"answer": final_answer, "debug": debug}

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
            question = (faq.get("question") or "").strip()
            if not answer:
                continue
            text = answer
            if question:
                text = f"{question}: {answer}"
            candidates.append((0, float(faq.get("similarity", 0.0) or 0.0), text, text))

        for hit in qdrant_hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            title = (hit.get("title") or "").strip()
            payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
            type_value = (hit.get("type") or payload.get("type") or "").strip()
            source = (hit.get("source") or payload.get("source") or "").strip()

            priority = 2
            if type_value in {"faq", "faq_ext"}:
                priority = 0
            elif source.startswith("knowledge") or source.endswith(".md") or ".md" in source:
                priority = 1

            snippet = f"{title}: {text}" if title else text
            candidates.append((priority, float(hit.get("score", 0.0) or 0.0), snippet, text))

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

    async def handle_knowledge(self, text: str) -> dict[str, Any]:
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
        }
        if rag_hits.get("embed_error"):
            debug["embed_error"] = rag_hits["embed_error"]

        if not total_hits:
            return {
                "answer": (
                    "Я не нашёл подходящих фрагментов в базе знаний. Если загрузите файл или страницу с типами домиков/номеров, я буду отвечать точнее."
                ),
                "debug": debug,
            }

        summary_lines = ["Нашёл в базе знаний:"]
        for hit in qdrant_hits[: self._settings.rag_max_snippets]:
            title = hit.get("title") or hit.get("source") or "Запись"
            text = (hit.get("text") or "").strip()
            if text:
                summary_lines.append(f"• {title}: {text[:180]}")
        for faq in faq_hits[:2]:
            question = faq.get("question") or "Вопрос"
            answer = faq.get("answer") or ""
            summary_lines.append(f"• FAQ {question}: {answer[:180]}")

        return {"answer": "\n".join(summary_lines), "debug": debug}


__all__ = [
    "ConversationStateStore",
    "InMemoryConversationStateStore",
    "ChatComposer",
]
