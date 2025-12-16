from __future__ import annotations

from datetime import date
from typing import Iterable

from app.booking.entities import BookingEntities
from app.booking.models import BookingQuote
from app.core.config import get_settings


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
    lines = [f"🏠 {offer.room_name}"]
    lines.append(f"— {format_money_rub(offer.total_price, offer.currency)}")
    if offer.room_area:
        lines.append(f"— {offer.room_area:g} м²")
    if offer.breakfast_included:
        lines.append("— завтрак включён")
    return "\n".join(lines)


def format_shelter_quote(
    entities: BookingEntities, offers: Iterable[BookingQuote]
) -> str:
    settings = get_settings()
    max_options = getattr(settings, "max_options", 6)

    sorted_offers = sorted(offers, key=lambda item: item.total_price)
    formatted_offers = [_format_offer(offer) for offer in sorted_offers[:max_options]]

    parts = [_format_header(entities), "\n\n".join(formatted_offers)]

    remaining = len(sorted_offers) - len(formatted_offers)
    if remaining > 0:
        parts.append(f"…и ещё {remaining} вариантов. Сказать \"покажи ещё\"?")

    parts.append("Нужно оформить бронирование?")

    return "\n\n".join(filter(None, parts))


__all__ = [
    "format_shelter_quote",
    "format_money_rub",
    "format_date_ddmm",
]
