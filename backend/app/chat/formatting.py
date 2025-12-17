from __future__ import annotations

from datetime import date
import re
from typing import Iterable

from app.booking.entities import BookingEntities
from app.booking.models import BookingQuote


def format_money_rub(amount: float, currency: str | None) -> str:
    currency_code = (currency or "RUB").upper()
    formatted_amount = f"{amount:,.0f}".replace(",", " ")
    if currency_code == "RUB":
        return f"{formatted_amount} ₽"
    return f"{formatted_amount} {currency_code}"


def format_date_ddmm(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        parsed = date.fromisoformat(date_str)
    except ValueError:
        return date_str
    return parsed.strftime("%d.%m")


def _calculate_nights(entities: BookingEntities) -> int | None:
    if entities.nights:
        return entities.nights
    if entities.checkin and entities.checkout:
        try:
            check_in_date = date.fromisoformat(entities.checkin)
            check_out_date = date.fromisoformat(entities.checkout)
            delta = (check_out_date - check_in_date).days
            return delta if delta > 0 else None
        except ValueError:
            return None
    return None


def _format_header(entities: BookingEntities) -> str:
    check_in = format_date_ddmm(entities.checkin)
    check_out = format_date_ddmm(entities.checkout)
    nights = _calculate_nights(entities)

    header_parts = [f"На даты {check_in}–{check_out}"]
    if nights:
        header_parts.append(f"({nights} ночи)")

    guests = [f"для {entities.adults} взрослых"]
    if entities.children:
        guests.append(f"и {entities.children} детей")

    header_parts.append(" ".join(guests))
    header_parts.append("доступны варианты:")

    return " ".join(header_parts)


def _format_offer(offer: BookingQuote) -> str:
    # Название номера с площадью в скобках
    name_part = f"🏠 {offer.room_name}"
    if offer.room_area:
        name_part += f" ({offer.room_area:g} м²)"
    
    # Цена с завтраком в скобках
    price_part = f"— {format_money_rub(offer.total_price, offer.currency)}"
    if offer.breakfast_included:
        price_part += " (завтрак включён)"
    
    return f"{name_part}\n{price_part}"


def select_min_offer_per_room_type(
    offers: Iterable[BookingQuote],
) -> list[BookingQuote]:
    """Группирует предложения по типу комнаты и берёт минимальную цену."""

    best_by_room: dict[str, tuple[BookingQuote, tuple[float, int, int]]] = {}
    for idx, offer in enumerate(offers):
        room_key = offer.room_name.lower()
        has_breakfast = 0 if bool(offer.breakfast_included) else 1
        score = (offer.total_price, has_breakfast, idx)
        current = best_by_room.get(room_key)
        if current is None or score < current[1]:
            best_by_room[room_key] = (offer, score)

    unique_offers = [item[0] for item in best_by_room.values()]
    return sorted(unique_offers, key=lambda item: item.total_price)


def format_shelter_quote(
    entities: BookingEntities, offers: Iterable[BookingQuote]
) -> str:
    max_display = 3  # показываем только 3 варианта

    unique_offers = select_min_offer_per_room_type(offers)
    sorted_offers = sorted(unique_offers, key=lambda item: item.total_price)
    formatted_offers = [_format_offer(offer) for offer in sorted_offers[:max_display]]

    parts = [_format_header(entities), "\n\n".join(formatted_offers)]

    remaining = len(sorted_offers) - len(formatted_offers)
    if remaining > 0:
        parts.append(f"Ещё доступно {remaining} вариантов. Показать все?")

    return "\n\n".join(filter(None, parts))


def format_more_offers(
    offers: list[BookingQuote],
    start_index: int,
) -> tuple[str, int]:
    """
    Форматирует оставшиеся офферы начиная с start_index.
    
    Возвращает (форматированный_текст, новый_индекс).
    """
    if start_index >= len(offers):
        return "Вы уже видели все доступные предложения.", start_index
    
    remaining_offers = offers[start_index:]
    if not remaining_offers:
        return "Вы уже видели все доступные предложения.", start_index
    
    formatted = [_format_offer(offer) for offer in remaining_offers]
    new_index = len(offers)
    
    text = "Показываю ещё варианты:\n\n" + "\n\n".join(formatted)
    return text, new_index


__all__ = [
    "format_shelter_quote",
    "format_more_offers",
    "select_min_offer_per_room_type",
    "format_money_rub",
    "format_date_ddmm",
    "detect_detail_mode",
    "postprocess_answer",
]


DETAIL_TRIGGERS = {
    "подробнее",
    "детали",
    "расскажи",
    "перечисли",
    "условия",
    "стоимость",
    "цены",
    "сколько",
    "что входит",
    "включено",
    "как добраться",
    "расписание",
    "время",
    "телефон",
    "адрес",
}


def detect_detail_mode(user_text: str) -> bool:
    """Определяет, нужен ли подробный ответ."""

    lowered = (user_text or "").lower()
    if any(trigger in lowered for trigger in DETAIL_TRIGGERS):
        return True

    if lowered.count("?") >= 2:
        return True

    connectors = re.findall(r"\b(?:и|а ещё|а еще)\b", lowered)
    if len(connectors) >= 2:
        return True

    comma_count = lowered.count(",")
    if comma_count >= 2 and "?" in lowered:
        return True

    return False


def _collapse_blank_lines(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped == "" and cleaned_lines and cleaned_lines[-1] == "":
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _remove_booking_cta(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if "забронировать" in lowered and "?" in lowered:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _collect_bullets(text: str, max_bullets: int = 3) -> list[str]:
    bullet_prefixes = ("-", "•", "*", "—", "–")
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(bullet_prefixes):
            continue
        bullet_text = stripped.lstrip("-•*—– ")

        if len(re.findall(r"\d+\s*[₽р]", bullet_text)) > 2:
            continue
        if len(re.findall(r"\d", bullet_text)) >= 6 and "," in bullet_text:
            continue

        bullets.append(f"• {bullet_text.strip()}")
        if len(bullets) >= max_bullets:
            break
    return bullets


def _build_brief_answer(text: str) -> str:
    sentences = _extract_sentences(text)
    summary_sentences = sentences[:2]

    bullets = _collect_bullets(text)

    parts: list[str] = []
    if summary_sentences:
        parts.append(" ".join(summary_sentences[:2]))
    if bullets:
        parts.append("\n".join(bullets))

    if not parts:
        return text

    brief_answer = "\n".join(parts)
    hint = "Если нужны детали — напишите «подробнее»."
    return f"{brief_answer}\n{hint}".strip()


def postprocess_answer(answer: str, mode: str = "brief") -> str:
    """Нормализует ответ перед отдачей пользователю."""

    cleaned = _collapse_blank_lines(answer)
    cleaned = _remove_booking_cta(cleaned)

    if mode == "detail":
        return cleaned

    long_answer = len(cleaned) > 700 or len(cleaned.splitlines()) > 5
    if not long_answer:
        return cleaned

    return _build_brief_answer(cleaned)
