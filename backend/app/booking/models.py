from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable


@dataclass
class Guests:
    adults: int
    children: int = 0
    children_ages: list[int] = field(default_factory=list)


@dataclass
class Dates:
    check_in: date
    check_out: date


@dataclass
class BookingQuote:
    room_name: str
    total_price: float
    currency: str
    breakfast_included: bool
    room_area: float | None
    check_in: str
    check_out: str
    guests: Guests


__all__ = ["Guests", "Dates", "BookingQuote"]
