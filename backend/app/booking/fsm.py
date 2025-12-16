from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class BookingState(Enum):
    START = "start"
    WAIT_CHECKIN = "wait_checkin"
    WAIT_NIGHTS = "wait_nights"
    WAIT_ADULTS = "wait_adults"
    WAIT_CHILDREN = "wait_children"
    WAIT_CHILDREN_AGES = "wait_children_ages"
    WAIT_ROOM_TYPE = "wait_room_type"
    CALCULATE = "calculate"
    DONE = "done"


def initial_booking_context() -> Dict[str, Any]:
    return {
        "state": BookingState.WAIT_CHECKIN,
        "entities": {
            "checkin": None,
            "nights": None,
            "adults": None,
            "children": None,
            "children_ages": None,
            "room_type": None,
        },
        "last_question": None,
        "attempts": {},
    }


__all__ = ["BookingState", "initial_booking_context"]
