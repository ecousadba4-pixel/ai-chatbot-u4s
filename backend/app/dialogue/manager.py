"""Логика шагового диалога бронирования."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

from .state import (
    BRANCH_BOOKING_PRICE_CHAT,
    BRANCH_ONLINE_BOOKING_REDIRECT,
    DialogueContext,
    INTENT_BOOKING_INQUIRY,
    STATE_COMPLETE,
    STATE_IDLE,
    STATE_WAIT_ADULTS,
    STATE_WAIT_CHECK_IN,
    STATE_WAIT_CHECK_OUT,
    STATE_WAIT_CHILDREN,
    STATE_WAIT_CHILD_AGES,
)
from ..services import (
    ShelterCloudAvailabilityError,
    ShelterCloudService,
)


class ContextStorage(Protocol):
    def read_context(self, session_id: str) -> dict[str, Any]:
        ...

    def write_context(self, session_id: str, context: dict[str, Any], ttl: int | None = None) -> None:
        ...

    def delete_context(self, session_id: str) -> None:
        ...


class DialogueResult(NamedTuple):
    handled: bool
    answer: str | None
    intent: str | None
    branch: str | None


@dataclass
class BookingDialogueManager:
    storage: ContextStorage
    service: ShelterCloudService

    def reset(self, session_id: str) -> None:
        self.storage.delete_context(session_id)

    # ------------------------------------------------------------------
    def handle_message(self, session_id: str, question: str) -> DialogueResult:
        normalized = (question or "").strip()
        if not session_id or not normalized:
            return DialogueResult(False, None, None, None)

        context = self._load_context(session_id)
        lower_question = normalized.lower()

        if self._should_start_new_dialogue(context, lower_question):
            context = self._start_dialogue()
            self._save_context(session_id, context)
            return DialogueResult(
                True,
                "Отлично, помогу с подбором номера. Укажите дату заезда (ДД.ММ.ГГГГ).",
                context.intent,
                context.branch,
            )

        if context.intent != INTENT_BOOKING_INQUIRY:
            return DialogueResult(False, None, None, None)

        if self._wants_online_redirect(lower_question):
            context.branch = BRANCH_ONLINE_BOOKING_REDIRECT
            context.state = STATE_COMPLETE
            self._save_context(session_id, context)
            return DialogueResult(
                True,
                self._online_redirect_message(),
                context.intent,
                context.branch,
            )

        handler = {
            STATE_WAIT_CHECK_IN: self._handle_check_in,
            STATE_WAIT_CHECK_OUT: self._handle_check_out,
            STATE_WAIT_ADULTS: self._handle_adults,
            STATE_WAIT_CHILDREN: self._handle_children,
            STATE_WAIT_CHILD_AGES: self._handle_children_ages,
            STATE_COMPLETE: self._handle_complete,
        }.get(context.state, self._handle_unknown_state)

        answer = handler(context, lower_question)
        self._save_context(session_id, context)
        branch = context.branch
        intent = context.intent if context.intent else None
        handled = answer is not None
        return DialogueResult(handled, answer, intent, branch)

    # ------------------------------------------------------------------
    def _start_dialogue(self) -> DialogueContext:
        context = DialogueContext()
        context.intent = INTENT_BOOKING_INQUIRY
        context.branch = BRANCH_BOOKING_PRICE_CHAT
        context.state = STATE_WAIT_CHECK_IN
        context.booking.children_ages = []
        return context

    def _load_context(self, session_id: str) -> DialogueContext:
        payload = self.storage.read_context(session_id) if session_id else {}
        if isinstance(payload, dict):
            return DialogueContext.from_dict(payload)
        return DialogueContext()

    def _save_context(self, session_id: str, context: DialogueContext) -> None:
        if session_id:
            self.storage.write_context(session_id, context.to_dict())

    # ------------------------------------------------------------------
    @staticmethod
    def _should_start_new_dialogue(context: DialogueContext, lower_question: str) -> bool:
        if not lower_question:
            return False
        booking_keywords = [
            "заброни",
            "бронь",
            "номер",
            "номерок",
            "свободн",
            "booking",
        ]
        matches_intent = any(word in lower_question for word in booking_keywords)
        if not matches_intent:
            return False
        if context.intent != INTENT_BOOKING_INQUIRY:
            return True
        if context.state == STATE_COMPLETE:
            return True
        return context.state == STATE_IDLE

    @staticmethod
    def _wants_online_redirect(lower_question: str) -> bool:
        if not lower_question:
            return False
        return any(
            token in lower_question
            for token in ("онлайн", "сайт", "форм", "перейти", "сам", "самостоятельно")
        )

    # ------------------------------------------------------------------ обработчики состояний
    def _handle_check_in(self, context: DialogueContext, question: str) -> str | None:
        parsed_date = self._extract_date(question)
        if not parsed_date:
            return "Не смогла распознать дату. Введите, пожалуйста, дату заезда в формате ДД.ММ.ГГГГ."
        context.booking.check_in = parsed_date.isoformat()
        context.state = STATE_WAIT_CHECK_OUT
        return "Спасибо! Теперь укажите дату выезда (ДД.ММ.ГГГГ)."

    def _handle_check_out(self, context: DialogueContext, question: str) -> str | None:
        parsed_date = self._extract_date(question)
        if not parsed_date:
            return "Не смогла распознать дату. Введите, пожалуйста, дату выезда в формате ДД.ММ.ГГГГ."
        check_in = self._to_date(context.booking.check_in)
        if check_in and parsed_date <= check_in:
            return "Дата выезда должна быть позже даты заезда. Попробуйте указать другие даты."
        context.booking.check_out = parsed_date.isoformat()
        context.state = STATE_WAIT_ADULTS
        return "Сколько взрослых планирует заселиться?"

    def _handle_adults(self, context: DialogueContext, question: str) -> str | None:
        adults = self._extract_number(question)
        if adults is None or adults < 1:
            return "Нужно указать количество взрослых (минимум один)."
        context.booking.adults = adults
        context.state = STATE_WAIT_CHILDREN
        return "Сколько будет детей? Если без детей — напишите 0."

    def _handle_children(self, context: DialogueContext, question: str) -> str | None:
        children = self._extract_number(question)
        if children is None or children < 0:
            return "Укажите, пожалуйста, количество детей числом."
        context.booking.children = children
        context.booking.children_ages = []
        if children == 0:
            context.state = STATE_COMPLETE
            return self._finalize(context)
        context.state = STATE_WAIT_CHILD_AGES
        return (
            "Введите возраст детей через запятую или пробел. Например: 5, 7"
        )

    def _handle_children_ages(self, context: DialogueContext, question: str) -> str | None:
        ages = self._extract_numbers(question)
        expected = context.booking.children or 0
        if len(ages) != expected:
            return f"Нужно указать {expected} значений возраста через пробел или запятую."
        if any(age < 0 or age > 17 for age in ages):
            return "Возраст детей должен быть от 0 до 17 лет."
        context.booking.children_ages = ages
        context.state = STATE_COMPLETE
        return self._finalize(context)

    def _handle_complete(self, context: DialogueContext, question: str) -> str | None:
        # если в завершённом состоянии снова спросили про бронирование, перезапустим диалог
        if self._should_start_new_dialogue(context, question.lower()):
            new_context = self._start_dialogue()
            context.intent = new_context.intent
            context.branch = new_context.branch
            context.state = new_context.state
            context.booking = new_context.booking
            return "Готова подобрать новое бронирование. Укажите дату заезда (ДД.ММ.ГГГГ)."
        return None

    def _handle_unknown_state(self, context: DialogueContext, question: str) -> str | None:
        context.state = STATE_COMPLETE
        return None

    # ------------------------------------------------------------------ финализация
    def _finalize(self, context: DialogueContext) -> str:
        is_configured = getattr(self.service, "is_configured", None)
        if callable(is_configured) and not bool(is_configured()):
            context.branch = BRANCH_ONLINE_BOOKING_REDIRECT
            return (
                "Подбор доступных номеров сейчас недоступен. "
                "Предлагаю перейти в модуль онлайн-бронирования на сайте."
            )

        try:
            offers = self.service.fetch_availability(
                check_in=context.booking.check_in or "",
                check_out=context.booking.check_out or "",
                adults=context.booking.adults or 1,
                children=context.booking.children or 0,
                children_ages=context.booking.children_ages,
            )
        except ShelterCloudAvailabilityError as error:
            context.branch = BRANCH_ONLINE_BOOKING_REDIRECT
            return (
                "Не удалось получить доступность номеров: "
                f"{error}. Предлагаю воспользоваться модулем онлайн-бронирования."
            )

        if not offers:
            context.branch = BRANCH_ONLINE_BOOKING_REDIRECT
            return (
                "К сожалению, на выбранные даты нет свободных номеров. "
                "Попробуйте изменить параметры или воспользоваться онлайн-бронированием."
            )

        offer = offers[0]
        context.branch = BRANCH_BOOKING_PRICE_CHAT
        breakfast_note = "завтрак включён" if offer.get("breakfast_included") else "завтрак не включён"
        price_text = self._format_price(offer.get("price"), offer.get("currency"))
        return (
            f"Нашла вариант: {offer.get('name', 'номер')} — {price_text}, {breakfast_note}. "
            "Сообщите, если хотите продолжить бронирование или перейти в онлайн-модуль."
        )

    # ------------------------------------------------------------------ утилиты
    @staticmethod
    def _online_redirect_message() -> str:
        return (
            "Вы можете оформить бронирование самостоятельно в модуле онлайн-бронирования на сайте. "
            "Если понадобится помощь — я рядом!"
        )

    @staticmethod
    def _extract_date(question: str) -> dt.date | None:
        iso_pattern = re.search(r"(\d{4})-(\d{2})-(\d{2})", question)
        if iso_pattern:
            year, month, day = iso_pattern.groups()
            try:
                return dt.date(int(year), int(month), int(day))
            except ValueError:
                return None

        dot_pattern = re.search(r"(\d{2})[./](\d{2})[./](\d{4})", question)
        if dot_pattern:
            day, month, year = dot_pattern.groups()
            try:
                return dt.date(int(year), int(month), int(day))
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_number(question: str) -> int | None:
        match = re.search(r"\d+", question)
        if not match:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _extract_numbers(question: str) -> list[int]:
        values: list[int] = []
        for match in re.findall(r"\d+", question):
            try:
                values.append(int(match))
            except ValueError:
                continue
        return values

    @staticmethod
    def _format_price(price: Any, currency: Any) -> str:
        try:
            amount = float(price)
        except (TypeError, ValueError):
            return "цена уточняется"
        currency_code = str(currency or "RUB").upper()
        formatted = f"{amount:,.0f}".replace(",", " ")
        return f"{formatted} {currency_code}"

    @staticmethod
    def _to_date(value: str | None) -> dt.date | None:
        if not value:
            return None
        try:
            return dt.date.fromisoformat(value)
        except ValueError:
            return None
