from __future__ import annotations

import re
from typing import Iterable

BOOKING_PATTERNS = [
    r"забронировать",
    r"бронир",
    r"вариант[аы]? на даты",
    r"дата (заезда|выезда)",
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


def detect_intent(text: str) -> str:
    normalized = text.lower()
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
