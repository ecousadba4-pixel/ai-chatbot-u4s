from __future__ import annotations

import re
from datetime import date
from typing import Optional

from app.booking.slot_filling import (
    AGE_BLOCK_PATTERNS,
    AGE_RE,
    ADULT_PATTERNS,
    CHILDREN_PATTERNS,
    DATE_DOTTED_RE,
    DATE_DOTTED_SHORT_RE,
    DATE_ISO_RE,
    DATE_TEXT_RE,
    MONTHS,
    NUMBER_WORD_PATTERN,
    RUS_NUMBER_WORDS,
    SlotFiller,
)


_slot_filler = SlotFiller()


_ZERO_TOKENS = {
    "0",
    "нет",
    "не будет",
    "без",
    "без детей",
    "без ребенка",
    "без ребёнка",
    "нет детей",
}

_NUMBER_WORDS: dict[int, set[str]] = {
    0: {"ноль", "нуль"},
    1: {"один", "одна", "одно", "одного", "одной"},
    2: {"два", "две", "двое", "двух"},
    3: {"три", "трое", "трех", "трёх"},
    4: {"четыре", "четверо", "четырех", "четырёх"},
    5: {"пять", "пятеро", "пяти"},
    6: {"шесть", "шестерых", "шести"},
    7: {"семь", "семерых", "семи"},
    8: {"восемь", "восьмерых", "восьми"},
    9: {"девять", "девятерых", "девяти"},
    10: {"десять", "десятерых", "десяти"},
}


def normalize_int(text: str) -> Optional[int]:
    if not text:
        return None

    lowered = text.strip().lower()
    if lowered in _ZERO_TOKENS:
        return 0

    digit_match = re.search(r"\d+", lowered)
    if digit_match:
        try:
            return int(digit_match.group())
        except ValueError:
            return None

    for value, variants in _NUMBER_WORDS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", lowered):
                return value

    mapped = RUS_NUMBER_WORDS.get(lowered)
    if mapped is not None:
        return mapped

    return None


def extract_guests(text: str) -> dict[str, int]:
    lowered = text.strip().lower()
    result: dict[str, int] = {}

    plus_match = re.search(
        r"(?P<adults>[\w-]+)\s*\+\s*(?P<children>[\w-]+)", lowered
    )
    if plus_match:
        adults = normalize_int(plus_match.group("adults"))
        children = normalize_int(plus_match.group("children"))
        if adults is not None:
            result["adults"] = adults
        if children is not None:
            result["children"] = children
        return result

    for pattern, field in (
        (re.compile(r"(?P<value>(?:\d+|[а-яё-]+))\s*(?:взросл\w*|adult\w*)"), "adults"),
        (
            re.compile(
                r"(?:взросл\w*|adult\w*)[^\dа-яё]*(?P<value>(?:\d+|[а-яё-]+))"
            ),
            "adults",
        ),
        (
            re.compile(
                r"(?P<value>(?:\d+|[а-яё-]+))\s*(?:дет(?:ей|и)|реб(?:е|ё)н(?:ок|ка)?|child\w*|kid\w*)"
            ),
            "children",
        ),
        (
            re.compile(
                r"(?:дет(?:ей|и)|реб(?:е|ё)н(?:ок|ка)?|child\w*|kid\w*)[^\dа-яё]*(?P<value>(?:\d+|[а-яё-]+))"
            ),
            "children",
        ),
    ):
        match = pattern.search(lowered)
        if match:
            value = normalize_int(match.group("value"))
            if value is not None:
                result[field] = value

    return result


def parse_checkin(text: str, now_date: date | None = None) -> str | None:
    today = now_date or date.today()
    dates = _extract_dates_with_future(text, today)
    return dates[0].isoformat() if dates else None


def parse_nights(text: str) -> int | None:
    lowered = text.strip().lower()
    match = re.search(
        rf"(?P<value>\d+|{NUMBER_WORD_PATTERN})\s*(?:ноч(?:и|ей)?|дн(?:я|ей)?)(?!\s*(?:назад|спустя))",
        lowered,
    )
    if match:
        return _parse_number_token(match.group("value"))

    if lowered.isdigit():
        return int(lowered)

    simple = _parse_number_token(lowered)
    if simple is not None and re.fullmatch(rf"({NUMBER_WORD_PATTERN}|\d+)", lowered):
        return simple
    return None


def parse_adults(text: str, *, allow_general_numbers: bool = True) -> int | None:
    for pattern in ADULT_PATTERNS:
        match = pattern.search(text)
        if match:
            value = _parse_number_token(match.group(1))
            if value is not None:
                return value
    if allow_general_numbers:
        return _parse_number_token(text.strip())
    return None


def parse_children_count(text: str) -> int | None:
    lowered = text.strip().lower()
    if lowered in {"да", "будут", "есть"}:
        return None

    if lowered in _ZERO_TOKENS:
        return 0

    if re.fullmatch(rf"(\d+|{NUMBER_WORD_PATTERN})", lowered):
        normalized = normalize_int(lowered)
        if normalized is not None:
            return normalized

    children = _slot_filler._extract_first_number(lowered, CHILDREN_PATTERNS)  # noqa: SLF001
    return children


def parse_children_ages(text: str, *, expected: int | None = None) -> list[int]:
    ages: list[int] = []
    for pattern in AGE_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            block = match.group("ages")
            ages.extend(_split_ages(block))

    for match in AGE_RE.finditer(text):
        ages.append(int(match.group(1)))

    if not ages:
        ages = _split_ages(text)

    filtered = [age for age in ages if 0 <= age <= 17]
    if expected is not None and filtered and len(filtered) != expected:
        return []
    return filtered


def parse_room_type(text: str) -> str | None:
    return _slot_filler._extract_room_type(text.lower())  # noqa: SLF001


def _split_ages(block: str) -> list[int]:
    return [int(item) for item in re.split(r"[\s,;]+", block) if item.isdigit()]


def _parse_number_token(token: str | None) -> int | None:
    return normalize_int(token or "")


def _extract_dates_with_future(text: str, today: date) -> list[date]:
    matches: list[tuple[int, date]] = []
    for regex, parser in (
        (DATE_ISO_RE, _parse_iso_date),
        (DATE_DOTTED_RE, _parse_dotted_date),
        (DATE_DOTTED_SHORT_RE, _parse_dotted_date),
        (DATE_TEXT_RE, _parse_text_date),
    ):
        for match in regex.finditer(text):
            parsed = parser(match, today)
            if parsed:
                matches.append((match.start(), parsed))

    matches.sort(key=lambda item: item[0])
    result: list[date] = []
    seen: set[str] = set()
    for _, parsed_date in matches:
        if parsed_date.isoformat() in seen:
            continue
        seen.add(parsed_date.isoformat())
        result.append(parsed_date)
    return result


def _parse_iso_date(match: re.Match[str], _today: date) -> date | None:
    try:
        return date.fromisoformat("-".join(match.groups()))
    except ValueError:
        return None


def _parse_dotted_date(match: re.Match[str], today: date) -> date | None:
    groups = match.groups()
    if len(groups) == 3:
        day, month, year = groups
    else:
        day, month = groups
        year = str(today.year)
    try:
        parsed = date.fromisoformat(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    except ValueError:
        return None
    return _ensure_future(parsed, today)


def _parse_text_date(match: re.Match[str], today: date) -> date | None:
    day_raw, month_raw, year_raw = match.groups()
    lowered_month = month_raw.lower()
    month = next((value for key, value in MONTHS.items() if lowered_month.startswith(key)), None)
    if not month:
        return None
    year = int(year_raw) if year_raw else today.year
    try:
        parsed = date(year, month, int(day_raw))
    except ValueError:
        return None
    return _ensure_future(parsed, today)


def _ensure_future(parsed: date, today: date) -> date:
    if parsed < today:
        try:
            return parsed.replace(year=parsed.year + 1)
        except ValueError:
            return parsed
    return parsed


__all__ = [
    "parse_checkin",
    "parse_nights",
    "parse_adults",
    "parse_children_count",
    "parse_children_ages",
    "parse_room_type",
    "normalize_int",
    "extract_guests",
]
