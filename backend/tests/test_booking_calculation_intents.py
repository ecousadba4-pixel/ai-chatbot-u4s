import sys
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.booking.entities import extract_booking_entities_ru
from app.chat.intent import detect_intent


def _detect(message: str, today: date):
    entities = extract_booking_entities_ru(message, now_date=today)
    intent = detect_intent(message, booking_entities=entities.__dict__)
    return intent, entities


def test_booking_calculation_flow_detects_missing_fields():
    today = date(2024, 12, 1)

    intent, entities = _detect("на 19 декабря на 2 дня", today)
    assert intent == "booking_calculation"
    assert set(entities.missing_fields) == {"adults", "room_type"}

    intent, entities = _detect("заезд 10.01 на 3 ночи, 2 взрослых", today)
    assert intent == "booking_calculation"
    assert set(entities.missing_fields) == {"room_type"}

    intent, entities = _detect("хочу шале на 5 января на 2 ночи", today)
    assert intent == "booking_calculation"
    assert set(entities.missing_fields) == {"adults"}
