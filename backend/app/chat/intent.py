from __future__ import annotations

import re
from typing import Iterable

BOOKING_PATTERNS = [
    r"забронировать",
    r"бронир",
    r"вариант[аы]? на даты",
    r"дата (заезда|выезда)",
]

BOOKING_CALC_PATTERNS = [
    r"заезд",
    r"выезд",
    r"на \d+\s*(?:ноч|дн)",
    r"ноч(и|ей)",
    r"стоимость проживания",
    r"брон",
    r"забронировать",
]

KNOWLEDGE_PATTERNS = [
    r"поиск по базе",
    r"покажи .*баз[ае]",
    r"что есть в базе",
    r"из базы знаний",
]

LODGING_KEYWORDS = {
    "размещение",
    "проживание",
    "домик",
    "домики",
    "коттедж",
    "коттеджи",
    "номер",
    "номера",
    "вместимость",
    "цена",
    "стоимость",
    "тариф",
    "тарифы",
    "категори",
    "тип",
    "типы",
}


def detect_intent(text: str, booking_entities: dict | None = None) -> str:
    normalized = text.lower()

    booking_entities = booking_entities or {}
    has_price_markers = any(
        marker in normalized
        for marker in [
            "сколько стоит",
            "цена",
            "стоимость",
            "рассчитай",
            "посчитай",
            "тариф",
            "стоимость проживания",
        ]
    )
    has_dates = any(
        booking_entities.get(field)
        for field in ("checkin", "checkout", "nights")
    )
    has_booking_calc_markers = any(
        re.search(pattern, normalized) for pattern in BOOKING_CALC_PATTERNS
    )

    if has_price_markers or has_dates or has_booking_calc_markers:
        return "booking_calculation"

    for pattern in KNOWLEDGE_PATTERNS:
        if re.search(pattern, normalized):
            return "knowledge_lookup"
    for pattern in BOOKING_PATTERNS:
        if re.search(pattern, normalized):
            return "booking_quote"
    if any(keyword in normalized for keyword in LODGING_KEYWORDS):
        return "lodging"
    return "general"


__all__ = ["detect_intent"]
