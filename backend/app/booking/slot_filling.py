from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict

from app.booking.models import Guests

DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
INT_RE = re.compile(r"(\d+)")


@dataclass
class SlotState:
    check_in: str | None = None
    check_out: str | None = None
    adults: int | None = None
    children: int | None = None
    children_ages: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_in": self.check_in,
            "check_out": self.check_out,
            "adults": self.adults,
            "children": self.children,
            "children_ages": self.children_ages,
        }

    def guests(self) -> Guests | None:
        if self.check_in and self.check_out and self.adults:
            return Guests(
                adults=self.adults,
                children=self.children or 0,
                children_ages=self.children_ages,
            )
        return None


class SlotFiller:
    REQUIRED = ("check_in", "check_out", "adults")
    OPTIONAL = ("children", "children_ages")

    def extract(self, text: str, state: SlotState | None = None) -> SlotState:
        state = state or SlotState()
        lowered = text.lower()
        dates = DATE_RE.findall(text)
        if dates:
            if not state.check_in and len(dates) >= 1:
                state.check_in = dates[0]
            if not state.check_out and len(dates) >= 2:
                state.check_out = dates[1]
        adults_match = re.search(r"(взросл\w*\s+(\d+))", lowered)
        if adults_match and not state.adults:
            try:
                state.adults = int(adults_match.group(2))
            except (TypeError, ValueError):
                pass
        elif not state.adults:
            numbers = INT_RE.findall(text)
            if numbers:
                state.adults = int(numbers[0])
        children_match = re.search(r"(дет(ей|и)\s+(\d+))", lowered)
        if children_match and state.children is None:
            try:
                state.children = int(children_match.group(3))
            except (TypeError, ValueError):
                pass
        return state

    def missing_slots(self, state: SlotState) -> list[str]:
        missing: list[str] = []
        for field_name in self.REQUIRED:
            if getattr(state, field_name) in (None, ""):
                missing.append(field_name)
        return missing

    def prompt_for_missing(self, missing: list[str]) -> str:
        prompts: dict[str, str] = {
            "check_in": "Укажите дату заезда в формате ГГГГ-ММ-ДД",
            "check_out": "Укажите дату выезда в формате ГГГГ-ММ-ДД",
            "adults": "Сколько взрослых будет в бронировании?",
        }
        questions = [prompts.get(slot, slot) for slot in missing]
        return "; ".join(questions)


__all__ = ["SlotFiller", "SlotState"]
