from typing import Any

import time
from datetime import date, timedelta

import asyncpg

from app.core.config import Settings, get_settings
from app.booking.entities import BookingEntities
from app.booking.models import Guests
from app.booking.service import BookingQuoteService
from app.chat.formatting import (
    detect_detail_mode,
    format_shelter_quote,
    postprocess_answer,
)
from app.booking.slot_filling import SlotFiller, SlotState
from app.llm.amvera_client import AmveraLLMClient
from app.llm.prompts import FACTS_PROMPT
from app.rag.context_builder import build_context
from app.rag.qdrant_client import QdrantClient
from app.rag.retriever import gather_rag_data


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
        settings: Settings | None = None,
    ) -> None:
        self._pool = pool
        self._qdrant = qdrant
        self._llm = llm
        self._slot_filler = slot_filler
        self._booking_service = booking_service
        self._store = store
        self._settings = settings or get_settings()

    def has_active_booking(
        self, session_id: str, entities: BookingEntities | None = None
    ) -> bool:
        state = self._store.get(session_id)
        if state and self._has_booking_context(state):
            return True
        if entities and self._entities_have_booking_data(entities):
            return True
        return False

    async def handle_booking_calculation(
        self, session_id: str, text: str, entities: BookingEntities
    ) -> dict[str, Any]:
        state = self._store.get(session_id) or SlotState()
        state = self._slot_filler.extract(text, state)
        self._apply_children_answer(text, state)

        if entities.checkin and not state.check_in:
            state.check_in = entities.checkin
        if entities.checkout and not state.check_out:
            state.check_out = entities.checkout
        if entities.nights and state.nights is None:
            state.nights = entities.nights
        if entities.adults is not None and state.adults is None:
            state.adults = entities.adults
        if entities.children is not None and state.children is None:
            state.children = entities.children
        if entities.room_type and state.room_type is None:
            state.room_type = entities.room_type

        if state.check_in and state.nights and not state.check_out:
            try:
                check_in_date = date.fromisoformat(state.check_in)
                state.check_out = (check_in_date + timedelta(days=state.nights)).isoformat()
            except ValueError:
                state.check_out = None

        if state.check_in and state.check_out and state.nights is None:
            try:
                delta = (
                    date.fromisoformat(state.check_out)
                    - date.fromisoformat(state.check_in)
                ).days
                state.nights = delta if delta > 0 else None
            except ValueError:
                state.nights = None
        state.errors = self._slot_filler._validate_dates(state)  # noqa: SLF001

        missing = self._missing_booking_fields(state)
        self._store.set(session_id, state)
        debug: dict[str, Any] = {
            "intent": "booking_calculation",
            "booking_entities": entities.__dict__,
            "missing_fields": missing,
            "shelter_called": False,
            "shelter_latency_ms": 0,
            "shelter_error": None,
            "llm_called": False,
        }

        next_slot = self._next_booking_question(state)
        if state.errors:
            question = self._build_booking_prompt(
                state, next_slot or "checkin", prefix=state.errors[0]
            )
            return {"answer": question, "debug": debug}

        if next_slot:
            question = self._build_booking_prompt(state, next_slot)
            return {"answer": question, "debug": debug}

        guests = Guests(adults=state.adults or 0, children=state.children or 0)

        started = time.perf_counter()
        try:
            offers = await self._booking_service.get_quotes(
                check_in=state.check_in or "",
                check_out=state.check_out or "",
                guests=guests,
            )
            debug["shelter_called"] = True
            debug["shelter_latency_ms"] = int(
                (time.perf_counter() - started) * 1000
            )
        except Exception as exc:  # noqa: BLE001
            debug["shelter_called"] = True
            debug["shelter_error"] = str(exc)
            return {
                "answer": "Не получилось получить расчёт, уточните, пожалуйста, детали позже.",
                "debug": debug,
            }

        self._store.clear(session_id)

        if not offers:
            return {
                "answer": "К сожалению, не удалось найти доступные варианты на указанные даты.",
                "debug": debug,
            }

        entities.checkin = state.check_in or entities.checkin
        entities.checkout = state.check_out or entities.checkout
        entities.adults = state.adults
        entities.children = state.children
        entities.missing_fields = []
        entities.room_type = state.room_type
        entities.nights = state.nights

        answer = format_shelter_quote(entities, offers)

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
