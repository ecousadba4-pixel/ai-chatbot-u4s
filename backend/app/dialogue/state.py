"""Состояния и модели диалогов."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


INTENT_BOOKING_INQUIRY = "booking_inquiry"
BRANCH_BOOKING_PRICE_CHAT = "booking_price_chat"
BRANCH_ONLINE_BOOKING_REDIRECT = "online_booking_redirect"

STATE_IDLE = "idle"
STATE_WAIT_CHECK_IN = "wait_check_in"
STATE_WAIT_CHECK_OUT = "wait_check_out"
STATE_WAIT_ADULTS = "wait_adults"
STATE_WAIT_CHILDREN = "wait_children"
STATE_WAIT_CHILD_AGES = "wait_child_ages"
STATE_COMPLETE = "complete"


@dataclass
class BookingData:
    check_in: str | None = None
    check_out: str | None = None
    adults: int | None = None
    children: int | None = None
    children_ages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_in": self.check_in,
            "check_out": self.check_out,
            "adults": self.adults,
            "children": self.children,
            "children_ages": list(self.children_ages),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BookingData":
        payload = payload or {}
        ages = payload.get("children_ages") or []
        if isinstance(ages, list):
            normalized_ages = []
            for age in ages:
                try:
                    normalized_ages.append(int(age))
                except (TypeError, ValueError):
                    continue
        else:
            normalized_ages = []
        return cls(
            check_in=payload.get("check_in"),
            check_out=payload.get("check_out"),
            adults=payload.get("adults"),
            children=payload.get("children"),
            children_ages=normalized_ages,
        )


@dataclass
class DialogueContext:
    intent: str = ""
    branch: str | None = None
    state: str = STATE_IDLE
    booking: BookingData = field(default_factory=BookingData)
    cached_offers: list[dict[str, Any]] = field(default_factory=list)
    last_offer_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "branch": self.branch,
            "state": self.state,
            "booking": self.booking.to_dict(),
            "cached_offers": list(self.cached_offers),
            "last_offer_index": self.last_offer_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DialogueContext":
        payload = payload or {}
        booking_payload = payload.get("booking")
        if isinstance(booking_payload, dict):
            booking = BookingData.from_dict(booking_payload)
        else:
            booking = BookingData()
        last_offer_index_value = payload.get("last_offer_index")
        if last_offer_index_value is None:
            normalized_last_index = -1
        else:
            try:
                normalized_last_index = int(last_offer_index_value)
            except (TypeError, ValueError):
                normalized_last_index = -1
        return cls(
            intent=str(payload.get("intent", "")),
            branch=payload.get("branch"),
            state=str(payload.get("state", STATE_IDLE)),
            booking=booking,
            cached_offers=list(payload.get("cached_offers") or []),
            last_offer_index=normalized_last_index,
        )

    def reset(self) -> None:
        self.intent = ""
        self.branch = None
        self.state = STATE_IDLE
        self.booking = BookingData()
        self.cached_offers = []
        self.last_offer_index = -1
