from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

from app.booking.fsm import BookingContext, BookingState, initial_booking_context
from app.booking.models import BookingQuote, Guests
from app.booking.service import BookingQuoteService
from app.services.parsing_service import ParsedMessageCache
from app.services.response_formatting_service import ResponseFormattingService

logger = logging.getLogger(__name__)


class BookingFsmService:
    """Сервис для управления FSM бронирования."""

    def __init__(
        self,
        booking_service: BookingQuoteService,
        formatting_service: ResponseFormattingService,
        max_state_attempts: int = 3,
    ) -> None:
        self._booking_service = booking_service
        self._formatting_service = formatting_service
        self._max_state_attempts = max_state_attempts

    def load_context(self, context_dict: dict[str, Any] | None) -> BookingContext:
        """Загружает контекст бронирования из словаря."""
        context = BookingContext.from_dict(context_dict)
        if context is None:
            return initial_booking_context()
        
        # Валидация загруженного контекста: если состояние требует checkin, но его нет,
        # возвращаемся к начальному состоянию
        if context.state in {
            BookingState.ASK_NIGHTS_OR_CHECKOUT,
            BookingState.ASK_ADULTS,
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
            BookingState.CALCULATE,
        }:
            if not context.checkin:
                logger.warning(
                    "Loaded context in state %s without checkin, resetting to ASK_CHECKIN. "
                    "Context dict: %s", context.state, context_dict
                )
                context = initial_booking_context()
        
        return context

    def save_context(self, context: BookingContext) -> dict[str, Any]:
        """Сохраняет контекст бронирования в словарь."""
        context.updated_at = datetime.utcnow().timestamp()
        context_dict = context.to_dict()
        # КРИТИЧНО: логируем сохранение для диагностики
        if context.checkin:
            logger.debug(
                "Saving context: checkin=%s, state=%s, nights=%s",
                context.checkin,
                context.state,
                context.nights,
            )
        else:
            logger.warning(
                "Saving context WITHOUT checkin: state=%s, nights=%s, context=%s",
                context.state,
                context.nights,
                context.compact(),
            )
        return context_dict

    def is_cancel_command(self, normalized: str) -> bool:
        """Проверяет, является ли команда командой отмены."""
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

    def is_back_command(self, normalized: str) -> bool:
        """Проверяет, является ли команда командой возврата назад."""
        return normalized in {"назад", "вернись", "вернуться"}

    def go_back(self, context: BookingContext) -> None:
        """Возвращает FSM на предыдущее состояние."""
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
        """Возвращает предыдущее состояние FSM."""
        order = [
            BookingState.ASK_CHECKIN,
            BookingState.ASK_NIGHTS_OR_CHECKOUT,
            BookingState.ASK_ADULTS,
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
            BookingState.CALCULATE,
            BookingState.AWAITING_USER_DECISION,
            BookingState.CONFIRM_BOOKING,
        ]
        if state in order:
            idx = order.index(state)
            return order[idx - 1] if idx > 0 else BookingState.ASK_CHECKIN
        return BookingState.ASK_CHECKIN

    async def process_message(
        self,
        session_id: str,
        text: str,
        context: BookingContext,
        parsers: ParsedMessageCache,
        debug: dict[str, Any],
    ) -> str:
        """Обрабатывает сообщение в контексте FSM бронирования."""
        normalized = text.strip().lower()
        
        if self.is_cancel_command(normalized):
            context.state = BookingState.CANCELLED
            return "Отменяю бронирование. Если понадобится помощь, напишите."

        if context.state in (None, BookingState.DONE, BookingState.CANCELLED):
            context.state = BookingState.ASK_CHECKIN

        if self.is_back_command(normalized):
            self.go_back(context)

        # КРИТИЧНО: валидация контекста перед обработкой
        # Если состояние требует checkin, но его нет, возвращаемся к начальному состоянию
        if context.state in {
            BookingState.ASK_NIGHTS_OR_CHECKOUT,
            BookingState.ASK_ADULTS,
            BookingState.ASK_CHILDREN_COUNT,
            BookingState.ASK_CHILDREN_AGES,
            BookingState.CALCULATE,
        }:
            if not context.checkin:
                logger.warning(
                    "Context validation failed: state %s requires checkin but it's missing. "
                    "Resetting to ASK_CHECKIN. Context: %s",
                    context.state,
                    context.compact(),
                )
                context.state = BookingState.ASK_CHECKIN

        logger.info(
            "BOOKING_FSM state=%s ctx=%s message=%s",
            context.state,
            context.compact(),
            text,
        )

        answer = await self._advance_booking_fsm(session_id, context, text, debug, parsers)
        return answer

    async def _advance_booking_fsm(
        self,
        session_id: str,
        context: BookingContext,
        text: str,
        debug: dict[str, Any],
        parsers: ParsedMessageCache,
    ) -> str:
        """Продвигает FSM вперёд на основе текущего состояния."""
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
                return self._ask_with_retry(
                    context, BookingState.ASK_CHECKIN, "На какую дату планируете заезд?"
                )

            if state == BookingState.ASK_NIGHTS_OR_CHECKOUT:
                context.state = BookingState.ASK_NIGHTS_OR_CHECKOUT
                # КРИТИЧНО: проверяем наличие checkin перед обработкой
                # Если checkin потерялся, возвращаемся к запросу даты заезда
                if not context.checkin:
                    logger.warning(
                        "Lost checkin date in ASK_NIGHTS_OR_CHECKOUT state, returning to ASK_CHECKIN. "
                        "Context: %s", context.compact()
                    )
                    context.state = BookingState.ASK_CHECKIN
                    state = BookingState.ASK_CHECKIN
                    continue
                
                if context.nights is not None or context.checkout:
                    state = BookingState.ASK_ADULTS
                    continue
                nights = parsers.nights()
                checkout_value = None
                try:
                    checkin_date = date.fromisoformat(context.checkin) if context.checkin else None
                except ValueError:
                    checkin_date = None
                    # Если checkin невалидный, возвращаемся к запросу даты
                    logger.warning(
                        "Invalid checkin date format in ASK_NIGHTS_OR_CHECKOUT: %s", context.checkin
                    )
                    context.checkin = None
                    context.state = BookingState.ASK_CHECKIN
                    state = BookingState.ASK_CHECKIN
                    continue
                
                if checkin_date:
                    parsed_checkout = parsers.checkin(now_date=checkin_date)
                    if parsed_checkout:
                        try:
                            checkout_date = date.fromisoformat(parsed_checkout)
                        except ValueError:
                            checkout_date = None
                        if checkout_date and checkin_date and checkout_date > checkin_date:
                            checkout_value = parsed_checkout
                if nights:
                    context.nights = nights
                    consumed_fields.add("nights")
                    # Перед переходом к следующему состоянию убеждаемся, что checkin сохранен
                    if not context.checkin:
                        logger.warning(
                            "Lost checkin date after extracting nights, returning to ASK_CHECKIN"
                        )
                        context.state = BookingState.ASK_CHECKIN
                        state = BookingState.ASK_CHECKIN
                        continue
                    state = BookingState.ASK_ADULTS
                    context.state = BookingState.ASK_ADULTS
                    continue
                if checkout_value:
                    context.checkout = checkout_value
                    # Перед переходом к следующему состоянию убеждаемся, что checkin сохранен
                    if not context.checkin:
                        logger.warning(
                            "Lost checkin date after extracting checkout, returning to ASK_CHECKIN"
                        )
                        context.state = BookingState.ASK_CHECKIN
                        state = BookingState.ASK_CHECKIN
                        continue
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
                # КРИТИЧНО: проверяем наличие checkin перед обработкой
                if not context.checkin:
                    logger.warning(
                        "Lost checkin date in ASK_ADULTS state, returning to ASK_CHECKIN. "
                        "Context: %s", context.compact()
                    )
                    context.state = BookingState.ASK_CHECKIN
                    state = BookingState.ASK_CHECKIN
                    continue
                
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
                return self._ask_with_retry(
                    context, BookingState.ASK_ADULTS, "Сколько взрослых едет?"
                )

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

    def _ask_with_retry(
        self, context: BookingContext, state: BookingState, question: str
    ) -> str:
        """Задаёт вопрос с учётом количества попыток."""
        attempts = context.retries.get(state.value, 0) + 1
        context.retries[state.value] = attempts
        return self._booking_prompt(question, context)

    def _booking_prompt(self, question: str, context: BookingContext) -> str:
        """Формирует промпт с вопросом и кратким резюме."""
        summary = self._booking_summary(context)
        parts: list[str] = []
        if summary:
            parts.append(f"Понял: {summary}.")
        parts.append(question)
        return " ".join(parts)

    def _booking_summary(self, context: BookingContext) -> str:
        """Формирует краткое резюме текущего контекста."""
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

    def _format_date(self, date_str: str) -> str:
        """Форматирует дату для отображения."""
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

    async def _calculate_booking(
        self, context: BookingContext, debug: dict[str, Any]
    ) -> str:
        """Выполняет расчёт бронирования."""
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

        from app.booking.entities import BookingEntities
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
        """Обрабатывает подтверждение бронирования."""
        return self._handle_post_quote_decision(text, context, parsers)

    def is_general_question(self, text: str) -> bool:
        """
        Определяет, является ли сообщение общим вопросом (не связанным с бронированием).
        
        Возвращает True, если пользователь спрашивает об услугах, инфраструктуре и т.д.,
        а не выбирает номер или меняет параметры бронирования.
        """
        normalized = text.strip().lower()
        
        # Короткие ответы — точно не общие вопросы
        if len(normalized) < 5:
            return False
        
        # Вопросительные слова и конструкции
        question_markers = [
            "есть ли", "есть", "можно ли", "можно", "как ", "где ", "когда ",
            "сколько стоит", "что включено", "какие ", "какой ", "какая ",
            "работает ли", "работает", "входит ли", "входит", "включён",
            "включен", "доступн", "предоставля", "предлага",
        ]
        
        # Ключевые слова об услугах и инфраструктуре (не о бронировании)
        service_keywords = [
            "баня", "сауна", "бассейн", "спа", "массаж",
            "еда", "питание", "ресторан", "кафе", "завтрак", "обед", "ужин",
            "меню", "кухня", "заказать еду", "доставка еды", "room service",
            "парковка", "стоянка", "wi-fi", "wifi", "вай-фай", "интернет",
            "детская", "площадка", "анимация", "развлечения",
            "трансфер", "такси", "аэропорт",
            "животные", "питомцы", "собака", "кошка", "с собакой",
            "курение", "курить", "балкон", "терраса",
            "кондиционер", "отопление", "камин",
            "велосипед", "прокат", "аренда",
            "экскурсии", "туры", "достопримечательности",
            "пляж", "река", "озеро", "рыбалка",
            "спортзал", "фитнес", "теннис",
            "прачечная", "химчистка", "глажка",
            "аптека", "магазин", "банкомат",
            "заезд ", "выезд ", "время заезда", "время выезда",
            "check-in", "check-out", "расчётный час",
        ]
        
        # Проверяем вопросительные маркеры
        has_question = any(marker in normalized for marker in question_markers)
        
        # Проверяем ключевые слова услуг
        has_service_keyword = any(keyword in normalized for keyword in service_keywords)
        
        # Вопрос об услугах — это общий вопрос
        if has_question and has_service_keyword:
            return True
        
        # Просто упоминание услуги с вопросительной интонацией (? в конце)
        if has_service_keyword and "?" in text:
            return True
        
        # Фразы типа "а есть баня?", "расскажи про ..." — общие вопросы
        if normalized.startswith(("а ", "а есть", "расскажи", "подскажи", "скажи")):
            return True
        
        return False

    def _handle_post_quote_decision(
        self, text: str, context: BookingContext, parsers: ParsedMessageCache
    ) -> str:
        """Обрабатывает решение пользователя после показа предложений."""
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

        # Проверяем, является ли сообщение общим вопросом
        if self.is_general_question(text):
            # Возвращаем специальный маркер для делегирования в RAG
            # Формат: "__DELEGATE_TO_GENERAL__" + исходный текст
            return f"__DELEGATE_TO_GENERAL__{text}"

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

    def get_context_entities(self, context: BookingContext) -> dict[str, Any]:
        """Возвращает сущности из контекста для отладки."""
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

    def get_missing_context_fields(self, context: BookingContext) -> list[str]:
        """Возвращает список отсутствующих полей в контексте."""
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
        return missing


__all__ = ["BookingFsmService"]

