from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BookingState(Enum):
    ASK_CHECKIN = "ask_checkin"
    ASK_NIGHTS_OR_CHECKOUT = "ask_nights_or_checkout"
    ASK_ADULTS = "ask_adults"
    ASK_CHILDREN_COUNT = "ask_children_count"
    ASK_CHILDREN_AGES = "ask_children_ages"
    ASK_ROOM_TYPE = "ask_room_type"
    CALCULATE = "calculate"
    CONFIRM_BOOKING = "confirm_booking"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass
class BookingContext:
    checkin: str | None = None
    nights: int | None = None
    checkout: str | None = None
    adults: int | None = None
    children: int | None = None
    children_ages: list[int] = field(default_factory=list)
    room_type: str | None = None
    promo: str | None = None
    state: BookingState | None = None
    retries: dict[str, int] = field(default_factory=dict)
    updated_at: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkin": self.checkin,
            "nights": self.nights,
            "checkout": self.checkout,
            "adults": self.adults,
            "children": self.children,
            "children_ages": list(self.children_ages),
            "room_type": self.room_type,
            "promo": self.promo,
            "state": self.state.value if self.state else None,
            "retries": dict(self.retries),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> BookingContext | None:
        if not isinstance(raw, dict):
            return None
        state = raw.get("state")
        booking_state = BookingState(state) if state else None
        return cls(
            checkin=raw.get("checkin"),
            nights=raw.get("nights"),
            checkout=raw.get("checkout"),
            adults=raw.get("adults"),
            children=raw.get("children"),
            children_ages=list(raw.get("children_ages") or []),
            room_type=raw.get("room_type"),
            promo=raw.get("promo"),
            state=booking_state,
            retries=dict(raw.get("retries") or {}),
            updated_at=raw.get("updated_at", datetime.utcnow().timestamp()),
        )

    def compact(self) -> dict[str, Any]:
        return {
            "state": self.state.value if self.state else None,
            "checkin": self.checkin,
            "nights": self.nights,
            "checkout": self.checkout,
            "adults": self.adults,
            "children": self.children,
            "children_ages": self.children_ages,
            "room_type": self.room_type,
        }


def initial_booking_context() -> BookingContext:
    return BookingContext(state=BookingState.ASK_CHECKIN)


__all__ = ["BookingState", "BookingContext", "initial_booking_context"]
