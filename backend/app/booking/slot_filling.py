from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict

from app.booking.models import Guests

MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}

DATE_ISO_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
DATE_DOTTED_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*(20\d{2})?",
    re.IGNORECASE,
)
ADULT_PATTERNS = [
    re.compile(r"(?:взросл\w*|adult\w*)\s*[:=]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:взросл\w*|adult\w*)", re.IGNORECASE),
]
CHILDREN_PATTERNS = [
    re.compile(
        r"(\d+)\s*(?:дет(?:ей|и)|реб(?:е|ё)н(?:ок|ка)?|child\w*|kid\w*)(?!\s*(?:лет|года|год))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:дет(?:ей|и)|реб(?:е|ё)н(?:ок|ка)?|child\w*|kid\w*)\s*[:=]?\s*(\d+)(?!\s*(?:лет|года|год))",
        re.IGNORECASE,
    ),
]
AGE_BLOCK_PATTERNS = [
    re.compile(r"возраст(?:\s+дет(?:ей|и))?[:=]?\s*(?P<ages>[\d\s,;]+)", re.IGNORECASE),
    re.compile(r"дет(?:ей|и|ям)?\s+(?P<ages>[\d\s,;]+)\s*(?:лет|года|год)?", re.IGNORECASE),
    re.compile(r"реб(?:е|ё)н(?:ок|ка|ку|ком|ке)?\s+(?P<ages>[\d\s,;]+)\s*(?:лет|года|год)?", re.IGNORECASE),
]
AGE_RE = re.compile(r"(\d{1,2})\s*(?:лет|года|год)", re.IGNORECASE)


@dataclass
class SlotState:
    check_in: str | None = None
    check_out: str | None = None
    adults: int | None = None
    children: int | None = None
    children_ages: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_in": self.check_in,
            "check_out": self.check_out,
            "adults": self.adults,
            "children": self.children,
            "children_ages": self.children_ages,
            "errors": self.errors,
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
        state.errors = []
        lowered = text.lower()

        dates = self._extract_dates(text)
        if dates:
            if not state.check_in and len(dates) >= 1:
                state.check_in = dates[0]
            if not state.check_out and len(dates) >= 2:
                state.check_out = dates[1]

        if state.adults is None:
            state.adults = self._extract_first_number(lowered, ADULT_PATTERNS)

        if state.children is None:
            state.children = self._extract_first_number(lowered, CHILDREN_PATTERNS)

        ages = self._extract_children_ages(lowered)
        if ages:
            state.children_ages = ages
            if state.children is None:
                state.children = len(ages)

        state.errors = self._validate_dates(state)
        return state

    def missing_slots(self, state: SlotState) -> list[str]:
        missing: list[str] = []
        for field_name in self.REQUIRED:
            if getattr(state, field_name) in (None, ""):
                missing.append(field_name)
        return missing

    def clarification(self, state: SlotState) -> str | None:
        missing = self.missing_slots(state)
        if state.errors or missing:
            return self.prompt_with_errors(state.errors, missing)
        return None

    def prompt_for_missing(self, missing: list[str]) -> str:
        prompts: dict[str, str] = {
            "check_in": "Укажите дату заезда в формате ГГГГ-ММ-ДД",
            "check_out": "Укажите дату выезда в формате ГГГГ-ММ-ДД",
            "adults": "Сколько взрослых будет в бронировании?",
        }
        questions = [prompts.get(slot, slot) for slot in missing]
        return "; ".join(questions)

    def prompt_with_errors(self, errors: list[str], missing: list[str]) -> str:
        messages: list[str] = []
        messages.extend(errors)
        if missing:
            messages.append(self.prompt_for_missing(missing))
        return "; ".join(messages)

    def _extract_dates(self, text: str) -> list[str]:
        matches: list[tuple[int, date]] = []
        for regex, parser in (
            (DATE_ISO_RE, self._parse_iso_date),
            (DATE_DOTTED_RE, self._parse_dotted_date),
            (DATE_TEXT_RE, self._parse_text_date),
        ):
            for match in regex.finditer(text):
                parsed = parser(match)
                if parsed:
                    matches.append((match.start(), parsed))

        matches.sort(key=lambda item: item[0])
        result: list[str] = []
        seen: set[str] = set()
        for _, parsed_date in matches:
            iso = parsed_date.isoformat()
            if iso not in seen:
                seen.add(iso)
                result.append(iso)
        return result

    def _parse_iso_date(self, match: re.Match[str]) -> date | None:
        try:
            return date.fromisoformat("-".join(match.groups()))
        except ValueError:
            return None

    def _parse_dotted_date(self, match: re.Match[str]) -> date | None:
        day, month, year = match.groups()
        try:
            return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").date()
        except ValueError:
            return None

    def _parse_text_date(self, match: re.Match[str]) -> date | None:
        day_raw, month_raw, year_raw = match.groups()
        month_key = month_raw.lower()[:5]
        month = MONTHS.get(month_key)
        if not month:
            return None
        year = int(year_raw) if year_raw else date.today().year
        try:
            return date(year, month, int(day_raw))
        except ValueError:
            return None

    def _extract_first_number(self, text: str, patterns: list[re.Pattern[str]]) -> int | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                for group in reversed(match.groups()):
                    if group and group.isdigit():
                        try:
                            return int(group)
                        except (TypeError, ValueError):
                            return None
        return None

    def _extract_children_ages(self, text: str) -> list[int]:
        ages: list[int] = []
        for pattern in AGE_BLOCK_PATTERNS:
            for match in pattern.finditer(text):
                block = match.group("ages")
                ages.extend(self._split_age_block(block))

        for match in AGE_RE.finditer(text):
            value = int(match.group(1))
            ages.append(value)

        deduped: list[int] = []
        for age in ages:
            if age not in deduped:
                deduped.append(age)
        return deduped

    def _split_age_block(self, block: str) -> list[int]:
        return [int(item) for item in re.split(r"[\s,;]+", block) if item.isdigit()]

    def _validate_dates(self, state: SlotState) -> list[str]:
        errors: list[str] = []
        check_in_date: date | None = None
        check_out_date: date | None = None

        if state.check_in:
            try:
                check_in_date = date.fromisoformat(state.check_in)
            except ValueError:
                errors.append("Дата заезда указана неверно. Используйте формат ГГГГ-ММ-ДД.")
                state.check_in = None

        if state.check_out:
            try:
                check_out_date = date.fromisoformat(state.check_out)
            except ValueError:
                errors.append("Дата выезда указана неверно. Используйте формат ГГГГ-ММ-ДД.")
                state.check_out = None

        if check_in_date and check_out_date and check_out_date <= check_in_date:
            errors.append("Дата выезда должна быть позже даты заезда. Уточните дату выезда.")
            state.check_out = None

        return errors


__all__ = ["SlotFiller", "SlotState"]
