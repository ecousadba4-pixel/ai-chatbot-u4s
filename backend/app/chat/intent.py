from __future__ import annotations

import re
from typing import Iterable

BOOKING_PATTERNS = [
    r"забронировать",
    r"бронир",
    r"стоимость",
    r"сколько будет стоить",
    r"вариант[аы]? на даты",
    r"дата (заезда|выезда)",
]


def detect_intent(text: str) -> str:
    normalized = text.lower()
    for pattern in BOOKING_PATTERNS:
        if re.search(pattern, normalized):
            return "booking_quote"
    return "general"


__all__ = ["detect_intent"]
