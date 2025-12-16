import sys
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.booking.entities import extract_booking_entities_ru


def _parse(message: str, today: date):
    return extract_booking_entities_ru(message, now_date=today)


def test_parses_december_19_without_truncation():
    today = date(2025, 12, 1)
    entities = _parse("19 декабря на 2 ночи", today)

    checkin = date.fromisoformat(entities.checkin)
    assert checkin.day == 19
    assert checkin.month == 12


def test_parses_first_of_december():
    today = date(2025, 11, 30)
    entities = _parse("1 декабря", today)

    checkin = date.fromisoformat(entities.checkin)
    assert checkin.day == 1
    assert checkin.month == 12


def test_parses_november_abbreviation():
    today = date(2025, 10, 1)
    entities = _parse("29 ноя", today)

    checkin = date.fromisoformat(entities.checkin)
    assert checkin.month == 11


def test_parses_day_with_leading_zero():
    today = date(2025, 11, 30)
    entities = _parse("на 03 декабря", today)

    checkin = date.fromisoformat(entities.checkin)
    assert checkin.day == 3


def test_parses_day_with_suffix():
    today = date(2025, 12, 1)
    entities = _parse("заезд 19-го декабря", today)

    checkin = date.fromisoformat(entities.checkin)
    assert checkin.day == 19
