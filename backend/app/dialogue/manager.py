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
from ..services import ShelterCloudAvailabilityError


class BookingAvailabilityService(Protocol):
    def is_configured(self) -> bool:
        ...

    def fetch_availability(
        self,
        *,
        check_in: str,
        check_out: str,
        adults: int,
        children: int,
        children_ages: list[int],
    ) -> list[dict[str, Any]]:
        ...


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
    service: BookingAvailabilityService

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
                (
                    "Когда планируете заехать?\n"
                    "Укажите дату в удобном для вас формате."
                ),
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
        context.cached_offers = []
        context.last_offer_index = -1
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
            return (
                "Не смогла распознать дату. Введите, пожалуйста, дату заезда, например: "
                "25.11.2025, 25-11-2025, 25 ноября 2025 или завтра."
            )
        context.booking.check_in = parsed_date.isoformat()
        context.state = STATE_WAIT_CHECK_OUT
        return (
            "Отлично! 😊 А до какого числа планируете остаться — "
            "или на сколько ночей хотите забронировать?"
        )

    def _handle_check_out(self, context: DialogueContext, question: str) -> str | None:
        parsed_date = self._extract_date(question)
        if not parsed_date:
            nights = self._extract_nights(question)
            if nights is not None:
                if nights < 1:
                    return (
                        "Количество ночей должно быть не меньше одной. Укажите дату выезда "
                        "или напишите, на сколько ночей нужна бронь."
                    )
                check_in = self._to_date(context.booking.check_in)
                if not check_in:
                    return (
                        "Сначала нужно указать дату заезда. После этого можно выбрать дату "
                        "выезда или количество ночей."
                    )
                parsed_date = check_in + dt.timedelta(days=nights)
            else:
                return (
                    "Не смогла распознать дату. Введите, пожалуйста, дату выезда (например: "
                    "28.11.2025, 28-11-2025, 28 ноября 2025, завтра, в эту пятницу) "
                    "или напишите, на сколько ночей нужна бронь."
                )
        check_in = self._to_date(context.booking.check_in)
        if check_in and parsed_date <= check_in:
            return "Дата выезда должна быть позже даты заезда. Попробуйте указать другие даты."
        context.booking.check_out = parsed_date.isoformat()
        context.state = STATE_WAIT_ADULTS
        if check_in:
            return (
                "Записала дату выезда — {date}. Сколько взрослых планирует заселиться?"
            ).format(date=parsed_date.strftime("%d.%m.%Y"))
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
        lower_question = question.lower()
        if self._wants_more_offers(lower_question):
            offers = context.cached_offers or []
            next_index = context.last_offer_index + 1
            if next_index < len(offers):
                context.last_offer_index = next_index
                offer = offers[next_index]
                breakfast_note = (
                    "завтрак включён" if offer.get("breakfast_included") else "завтрак не включён"
                )
                price_text = self._format_price(offer.get("price"), offer.get("currency"))
                return (
                    f"Нашла ещё вариант: {self._offer_name(offer)} — {price_text}, {breakfast_note}. "
                    "Сообщите, если хотите продолжить бронирование или перейти в онлайн-модуль."
                )
            if offers:
                return (
                    "Пока это все доступные предложения на выбранные даты. "
                    "Могу подобрать новые варианты — укажите другие даты или условия бронирования."
                )
        # если в завершённом состоянии снова спросили про бронирование, перезапустим диалог
        if self._should_start_new_dialogue(context, lower_question):
            new_context = self._start_dialogue()
            context.intent = new_context.intent
            context.branch = new_context.branch
            context.state = new_context.state
            context.booking = new_context.booking
            context.cached_offers = new_context.cached_offers
            context.last_offer_index = new_context.last_offer_index
            return (
                "Когда планируете заехать?\n"
                "Укажите дату в удобном для вас формате."
            )
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

        context.cached_offers = offers
        context.last_offer_index = 0
        offer = offers[0]
        context.branch = BRANCH_BOOKING_PRICE_CHAT
        breakfast_note = "завтрак включён" if offer.get("breakfast_included") else "завтрак не включён"
        price_text = self._format_price(offer.get("price"), offer.get("currency"))
        return (
            f"Нашла вариант: {self._offer_name(offer)} — {price_text}, {breakfast_note}. "
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
    def _offer_name(offer: dict[str, Any]) -> str:
        name = offer.get("name")
        return str(name) if name else "номер"

    @staticmethod
    def _wants_more_offers(lower_question: str) -> bool:
        if not lower_question:
            return False
        tokens = ("ещё", "еще", "больше", "другие", "вариант", "покаж", "покажи")
        return any(token in lower_question for token in tokens)

    @staticmethod
    def _extract_date(question: str) -> dt.date | None:
        lowered = question.lower()
        reference_date = dt.date.today()

        relative_date = BookingDialogueManager._extract_relative_date(lowered, reference_date)
        if relative_date:
            return relative_date

        iso_pattern = re.search(r"(\d{4})-(\d{2})-(\d{2})", lowered)
        if iso_pattern:
            year, month, day = iso_pattern.groups()
            try:
                return dt.date(int(year), int(month), int(day))
            except ValueError:
                return None

        dot_pattern = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", lowered)
        if dot_pattern:
            day, month, year = dot_pattern.groups()
            try:
                return dt.date(int(year), int(month), int(day))
            except ValueError:
                return None

        month_names = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }
        text_pattern = re.search(
            r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
            lowered,
        )
        if text_pattern:
            day_str, month_str, year_str = text_pattern.groups()
            month = month_names.get(month_str)
            if month:
                try:
                    return dt.date(int(year_str), month, int(day_str))
                except ValueError:
                    return None

        return None

    @staticmethod
    def _extract_relative_date(question: str, reference: dt.date) -> dt.date | None:
        if "завтра" in question:
            return reference + dt.timedelta(days=1)

        weekday_map = {
            "понедель": 0,
            "вторник": 1,
            "сред": 2,
            "четверг": 3,
            "пятниц": 4,
            "суббот": 5,
            "воскрес": 6,
        }
        weekday_pattern = re.search(
            r"(?:в\s+)?(?:(?:эту|этот|этой|этим|этой)|(?:следующую|следующий|следующем|следующей))?\s*"
            r"(понедельник|вторник|среда|среду|четверг|пятница|пятницу|суббота|субботу|воскресенье)",
            question,
        )
        if weekday_pattern:
            weekday_word = weekday_pattern.group(1)
            has_next = "следующ" in question
            has_this = any(token in question for token in ("эту", "этот", "этой", "этим"))
            for key, value in weekday_map.items():
                if key in weekday_word:
                    days_ahead = (value - reference.weekday()) % 7
                    if has_next:
                        if days_ahead == 0:
                            days_ahead = 7
                        else:
                            days_ahead += 7
                    elif has_this and days_ahead == 0:
                        return reference
                    if days_ahead == 0:
                        days_ahead = 7
                    return reference + dt.timedelta(days=days_ahead)
        return None

    @staticmethod
    def _extract_nights(question: str) -> int | None:
        match = re.search(r"(\d+)\s*(?:ноч(?:ь|и|ей)?)", question.lower())
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
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
