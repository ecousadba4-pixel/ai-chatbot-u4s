from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from zoneinfo import ZoneInfo

MONTHS = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "мая": 5,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}


DATE_RANGE_TEXT_RE = re.compile(
    r"(?:с\s*)?(?P<start>\d{1,2})\s*(?:[-–]|по)?\s*(?P<end>\d{1,2})\s+"
    r"(?P<month>[а-яА-ЯёЁ]+)(?:\s+(?P<year>\d{4}))?",
    re.IGNORECASE,
)
DATE_RANGE_NUMERIC_RE = re.compile(
    r"(?P<start>\d{1,2})\s*[-–]\.?(?P<end>\d{1,2})[./](?P<month>\d{1,2})(?:[./](?P<year>\d{4}))?",
    re.IGNORECASE,
)

DATE_ISO_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
DATE_DOTTED_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*(20\d{2})?",
    re.IGNORECASE,
)

ADULTS_RE = re.compile(
    r"(?P<adults>\d+)\s*(?:взр\b|взросл\w*|adult\w*)",
    re.IGNORECASE,
)
ADULTS_ONLY_RE = re.compile(r"(?P<adults>\d+)\s*\+\s*(?P<children>\d+)")
CHILDREN_RE = re.compile(
    r"(?P<children>\d+)\s*(?:дет(?:ей|и)|реб(?:е|ё)н(?:ок|ка)?|child\w*)",
    re.IGNORECASE,
)


@dataclass
class BookingEntities:
    checkin: str | None
    checkout: str | None
    adults: int | None
    children: int
    nights: int | None
    missing_fields: list[str]


def _parse_yearless(day: int, month: int, *, current: date) -> date | None:
    try:
        candidate = date(current.year, month, day)
    except ValueError:
        return None
    if candidate < current:
        try:
            return date(current.year + 1, month, day)
        except ValueError:
            return None
    return candidate


def _parse_iso(match: re.Match[str]) -> date | None:
    try:
        return date.fromisoformat("-".join(match.groups()))
    except ValueError:
        return None


def _parse_dotted(match: re.Match[str]) -> date | None:
    day, month, year = match.groups()
    try:
        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_text(match: re.Match[str], *, current: date) -> date | None:
    day_raw, month_raw, year_raw = match.groups()
    month_key = month_raw.lower()[:3]
    month = MONTHS.get(month_key)
    if not month:
        return None
    if year_raw:
        try:
            return date(int(year_raw), month, int(day_raw))
        except ValueError:
            return None
    return _parse_yearless(int(day_raw), month, current=current)


def _extract_dates(text: str, *, current: date) -> list[date]:
    lowered = text.lower()

    range_match = DATE_RANGE_TEXT_RE.search(lowered)
    if range_match:
        start_day = int(range_match.group("start"))
        end_day = int(range_match.group("end"))
        month_key = range_match.group("month")[:3].lower()
        month = MONTHS.get(month_key)
        year_raw = range_match.group("year")
        if month:
            if year_raw:
                try:
                    year_value = int(year_raw)
                except ValueError:
                    year_value = None
                if year_value:
                    try:
                        checkin = date(year_value, month, start_day)
                        checkout = date(year_value, month, end_day)
                        return [checkin, checkout]
                    except ValueError:
                        pass
            checkin = _parse_yearless(start_day, month, current=current)
            checkout = _parse_yearless(end_day, month, current=current)
            if checkin and checkout:
                return [checkin, checkout]

    numeric_match = DATE_RANGE_NUMERIC_RE.search(lowered)
    if numeric_match:
        start_day = int(numeric_match.group("start"))
        end_day = int(numeric_match.group("end"))
        month_raw = numeric_match.group("month")
        try:
            month = int(month_raw)
        except ValueError:
            month = None
        year_raw = numeric_match.group("year")
        year_value = int(year_raw) if year_raw and year_raw.isdigit() else None
        if month and 1 <= month <= 12:
            if year_value:
                try:
                    checkin = date(year_value, month, start_day)
                    checkout = date(year_value, month, end_day)
                    return [checkin, checkout]
                except ValueError:
                    pass
            checkin = _parse_yearless(start_day, month, current=current)
            checkout = _parse_yearless(end_day, month, current=current)
            if checkin and checkout:
                return [checkin, checkout]

    matches: list[tuple[int, date]] = []
    for regex, parser in (
        (DATE_ISO_RE, _parse_iso),
        (DATE_DOTTED_RE, _parse_dotted),
    ):
        for match in regex.finditer(text):
            parsed = parser(match)
            if parsed:
                matches.append((match.start(), parsed))

    for match in DATE_TEXT_RE.finditer(text):
        parsed = _parse_text(match, current=current)
        if parsed:
            matches.append((match.start(), parsed))

    matches.sort(key=lambda item: item[0])
    seen: set[str] = set()
    dates: list[date] = []
    for _, parsed_date in matches:
        iso = parsed_date.isoformat()
        if iso not in seen:
            seen.add(iso)
            dates.append(parsed_date)
    return dates[:2]


def _extract_guests(text: str) -> tuple[int | None, int]:
    adults: int | None = None
    children = 0

    plus_match = ADULTS_ONLY_RE.search(text)
    if plus_match:
        try:
            adults = int(plus_match.group("adults"))
            children = int(plus_match.group("children"))
        except ValueError:
            adults = None
            children = 0

    adults_match = ADULTS_RE.search(text)
    if adults_match:
        try:
            adults = int(adults_match.group("adults"))
        except ValueError:
            adults = None

    children_match = CHILDREN_RE.search(text)
    if children_match:
        try:
            children = int(children_match.group("children"))
        except ValueError:
            children = 0

    return adults, max(0, children)


def extract_booking_entities_ru(
    text: str, now_date: date | None = None, tz: str | None = None
) -> BookingEntities:
    current = now_date
    if current is None:
        if tz:
            try:
                current = datetime.now(ZoneInfo(tz)).date()
            except Exception:  # noqa: BLE001
                current = date.today()
        else:
            current = date.today()

    dates = _extract_dates(text, current=current)
    checkin = dates[0].isoformat() if len(dates) >= 1 else None
    checkout = dates[1].isoformat() if len(dates) >= 2 else None

    adults, children = _extract_guests(text.lower())

    nights: int | None = None
    if checkin and checkout:
        try:
            delta = date.fromisoformat(checkout) - date.fromisoformat(checkin)
            nights = delta.days if delta.days > 0 else None
        except ValueError:
            nights = None

    missing_fields: list[str] = []
    if not checkin:
        missing_fields.append("checkin")
    if not checkout:
        missing_fields.append("checkout")
    if adults is None:
        missing_fields.append("adults")

    return BookingEntities(
        checkin=checkin,
        checkout=checkout,
        adults=adults,
        children=children,
        nights=nights,
        missing_fields=missing_fields,
    )


__all__ = ["extract_booking_entities_ru", "BookingEntities"]
