from typing import Any

import logging
import time
from datetime import date, timedelta

import asyncpg

from app.core.config import Settings, get_settings
from app.booking.entities import BookingEntities
from app.booking.fsm import BookingState, initial_booking_context
from app.booking.models import Guests
from app.booking.service import BookingQuoteService
from app.chat.formatting import (
    detect_detail_mode,
    format_shelter_quote,
    postprocess_answer,
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
        booking_context = self._booking_store.get(session_id)
        if self._booking_fsm_active(booking_context):
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
        booking = self._ensure_booking_context(session_id)
        debug: dict[str, Any] = {
            "intent": "booking_calculation",
            "booking_state": str(getattr(booking.get("state"), "value", "")),
            "booking_entities": dict(booking.get("entities", {})),
            "missing_fields": self._missing_booking_state_fields(booking),
            "shelter_called": False,
            "shelter_latency_ms": 0,
            "shelter_error": None,
            "llm_called": False,
        }

        if self._should_reset_flow(text):
            booking = initial_booking_context()

        if booking.get("state") == BookingState.CALCULATE:
            answer = self._handle_calculate_confirmation(session_id, text, booking)
            self._booking_store.set(session_id, booking)
            debug["booking_state"] = booking["state"].value
            return {"answer": answer, "debug": debug}

        response = await self._handle_booking_fsm(session_id, text, booking, debug)
        debug["booking_state"] = booking["state"].value
        debug["booking_entities"] = dict(booking.get("entities", {}))
        return {"answer": response, "debug": debug}

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

    def _ensure_booking_context(self, session_id: str) -> dict[str, Any]:
        context = self._booking_store.get(session_id)
        if not self._booking_fsm_active(context):
            context = initial_booking_context()
            self._booking_store.set(session_id, context)
        return context

    def _booking_fsm_active(self, context: Any) -> bool:
        return isinstance(context, dict) and isinstance(
            context.get("state"), BookingState
        )

    def _should_reset_flow(self, text: str) -> bool:
        normalized = text.strip().lower()
        reset_triggers = {"начнём заново", "отмени", "другие даты", "с начала"}
        return any(trigger in normalized for trigger in reset_triggers)

    def _booking_prompt(self, question: str, booking: dict[str, Any], prefix: str | None = None) -> str:
        summary = self._booking_summary(booking)
        parts: list[str] = []
        if summary:
            parts.append(f"Понял: {summary}.")
        if prefix:
            parts.append(prefix)
        parts.append(question)
        return " ".join(parts)

    def _booking_summary(self, booking: dict[str, Any]) -> str:
        entities = booking.get("entities", {})
        fragments: list[str] = []
        checkin = entities.get("checkin")
        if checkin:
            fragments.append(f"заезд {self._format_date(checkin)}")
        nights = entities.get("nights")
        if nights:
            fragments.append(f"ночей {nights}")
        adults = entities.get("adults")
        children = entities.get("children")
        if adults is not None:
            guests = f"взрослых {adults}"
            if children is not None:
                guests += f", детей {children}"
            fragments.append(guests)
        room_type = entities.get("room_type")
        if room_type:
            fragments.append(f"тип {room_type}")
        return ", ".join(fragments)

    def _reset_attempts(self, booking: dict[str, Any], state: BookingState) -> None:
        booking.setdefault("attempts", {})
        booking["attempts"].pop(state.value, None)

    def _increment_attempts(self, booking: dict[str, Any], state: BookingState) -> int:
        attempts = booking.setdefault("attempts", {})
        attempts[state.value] = attempts.get(state.value, 0) + 1
        return attempts[state.value]

    async def _handle_booking_fsm(
        self,
        session_id: str,
        text: str,
        booking: dict[str, Any],
        debug: dict[str, Any],
    ) -> str:
        state: BookingState = booking["state"]
        if state == BookingState.WAIT_CHECKIN:
            extracted = self._slot_filler._extract_dates(text)  # noqa: SLF001
            if extracted:
                booking["entities"]["checkin"] = extracted[0]
                booking["state"] = BookingState.WAIT_NIGHTS
                self._reset_attempts(booking, state)
                question = self._booking_prompt(
                    "Сколько ночей планируете остаться?", booking
                )
                booking["last_question"] = question
                self._booking_store.set(session_id, booking)
                return question
            question = self._booking_prompt(
                "На какую дату планируете заезд?", booking
            )
            return self._repeat_or_reset(session_id, booking, question)

        if state == BookingState.WAIT_NIGHTS:
            nights = self._slot_filler._extract_nights(text)  # noqa: SLF001
            if nights is None and text.strip().isdigit():
                nights = int(text.strip())
            if nights:
                booking["entities"]["nights"] = nights
                booking["state"] = BookingState.WAIT_ADULTS
                self._reset_attempts(booking, state)
                question = self._booking_prompt("Сколько взрослых едет?", booking)
                booking["last_question"] = question
                self._booking_store.set(session_id, booking)
                return question
            question = self._booking_prompt(
                "Уточните, пожалуйста, на сколько ночей бронирование?", booking
            )
            return self._repeat_or_reset(session_id, booking, question)

        if state == BookingState.WAIT_ADULTS:
            adults = self._slot_filler._extract_adults(  # noqa: SLF001
                text, allow_general_numbers=True
            )
            if adults is not None:
                booking["entities"]["adults"] = adults
                booking["state"] = BookingState.WAIT_CHILDREN
                self._reset_attempts(booking, state)
                question = self._booking_prompt("Будут ли дети?", booking)
                booking["last_question"] = question
                self._booking_store.set(session_id, booking)
                return question
            question = self._booking_prompt(
                "Сколько взрослых едет?", booking, prefix="Ответьте числом"
            )
            return self._repeat_or_reset(session_id, booking, question)

        if state == BookingState.WAIT_CHILDREN:
            lowered = text.strip().lower()
            negative_children = {"нет", "не будет", "без детей", "нет детей", "0"}
            children = self._slot_filler._extract_first_number(  # noqa: SLF001
                lowered, CHILDREN_PATTERNS
            )
            if lowered in negative_children:
                children = 0
            if children is not None:
                booking["entities"]["children"] = children
                next_state = (
                    BookingState.WAIT_CHILDREN_AGES if children > 0 else BookingState.WAIT_ROOM_TYPE
                )
                booking["state"] = next_state
                self._reset_attempts(booking, state)
                if children == 0:
                    booking["entities"].setdefault("room_type", "Студия")
                    booking["state"] = BookingState.CALCULATE
                    booking["last_question"] = "Оформляем бронирование?"
                    self._booking_store.set(session_id, booking)
                    return await self._calculate_booking(session_id, booking, debug)

                question_text = "Уточните возраст детей (через запятую)."
                question = self._booking_prompt(question_text, booking)
                booking["last_question"] = question
                self._booking_store.set(session_id, booking)
                return question
            question = self._booking_prompt("Будут ли дети?", booking)
            return self._repeat_or_reset(session_id, booking, question)

        if state == BookingState.WAIT_CHILDREN_AGES:
            ages = self._slot_filler._extract_children_ages(text)  # noqa: SLF001
            if ages:
                booking["entities"]["children_ages"] = ages
                booking["state"] = BookingState.WAIT_ROOM_TYPE
                self._reset_attempts(booking, state)
                question = self._booking_prompt(
                    "Какой тип размещения предпочитаете: Студия, Шале, Шале Комфорт или Семейный номер?",
                    booking,
                )
                booking["last_question"] = question
                self._booking_store.set(session_id, booking)
                return question
            question = self._booking_prompt(
                "Не услышал возраст детей, укажите числа через запятую.", booking
            )
            return self._repeat_or_reset(session_id, booking, question)

        if state == BookingState.WAIT_ROOM_TYPE:
            room_type = self._slot_filler._extract_room_type(text)  # noqa: SLF001
            if room_type:
                booking["entities"]["room_type"] = room_type
                booking["state"] = BookingState.CALCULATE
                self._reset_attempts(booking, state)
                answer = await self._calculate_booking(session_id, booking, debug)
                return answer
            question = self._booking_prompt(
                "Выберите тип размещения: Студия, Шале, Шале Комфорт или Семейный номер.",
                booking,
            )
            return self._repeat_or_reset(session_id, booking, question)

        # Default fallback for any unexpected state
        booking["state"] = BookingState.WAIT_CHECKIN
        self._booking_store.set(session_id, booking)
        return "На какую дату планируете заезд?"

    def _repeat_or_reset(
        self, session_id: str, booking: dict[str, Any], question: str
    ) -> str:
        attempts = self._increment_attempts(booking, booking["state"])
        if attempts >= self._max_state_attempts:
            booking.clear()
            booking.update(initial_booking_context())
            question = "Давайте начнём заново. На какую дату планируете заезд?"
        booking["last_question"] = question
        self._booking_store.set(session_id, booking)
        return question

    def _missing_booking_state_fields(self, booking: dict[str, Any]) -> list[str]:
        entities = booking.get("entities", {})
        missing = [key for key, value in entities.items() if value in (None, [])]
        return missing

    def _handle_calculate_confirmation(
        self, session_id: str, text: str, booking: dict[str, Any]
    ) -> str:
        normalized = text.strip().lower()
        if any(token in normalized for token in {"да", "оформляй", "подтверждаю"}):
            booking["state"] = BookingState.DONE
            return "Отлично, фиксирую бронирование. Если захотите изменить детали, скажите \"начнём заново\"."

        if any(token in normalized for token in {"нет", "пока"}):
            return "Хорошо, расчёт сохранён. Если нужно изменить даты, напишите \"начнём заново\"."

        question = booking.get("last_question") or "Оформляем бронирование?"
        return self._repeat_or_reset(session_id, booking, question)

    async def _calculate_booking(
        self, session_id: str, booking: dict[str, Any], debug: dict[str, Any]
    ) -> str:
        entities = booking["entities"]
        checkin = entities.get("checkin")
        nights = entities.get("nights")
        adults = entities.get("adults")
        children = entities.get("children") or 0

        checkout: str | None = None
        if checkin and nights:
            try:
                checkin_date = date.fromisoformat(checkin)
                checkout = (checkin_date + timedelta(days=int(nights))).isoformat()
            except ValueError:
                checkout = None

        guests = Guests(adults=adults or 0, children=children)

        if not (checkin and checkout and adults is not None):
            booking["state"] = BookingState.WAIT_CHECKIN
            self._booking_store.set(session_id, booking)
            return "Не удалось собрать данные для расчёта. На какую дату планируете заезд?"

        started = time.perf_counter()
        try:
            offers = await self._booking_service.get_quotes(
                check_in=checkin,
                check_out=checkout,
                guests=guests,
            )
            debug["shelter_called"] = True
            debug["shelter_latency_ms"] = int((time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["shelter_called"] = True
            debug["shelter_error"] = str(exc)
            booking["state"] = BookingState.WAIT_CHECKIN
            self._booking_store.set(session_id, booking)
            return "Не получилось получить расчёт, давайте попробуем ещё раз. На какую дату планируете заезд?"

        if not offers:
            booking["state"] = BookingState.DONE
            self._booking_store.set(session_id, booking)
            return "К сожалению, нет доступных вариантов на выбранные даты. Если хотите изменить параметры, скажите \"начнём заново\"."

        booking_entities = BookingEntities(
            checkin=checkin,
            checkout=checkout,
            adults=adults,
            children=children,
            nights=nights,
            room_type=entities.get("room_type"),
            missing_fields=[],
        )
        price_block = format_shelter_quote(booking_entities, offers)
        cta = "Оформляем бронирование?"
        booking["last_question"] = cta
        self._booking_store.set(session_id, booking)
        return f"{price_block}\n\n{cta}"

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
            "children": "Будут дети?",
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
            "children": "Будут дети?",
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
