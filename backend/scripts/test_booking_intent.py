from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.booking.entities import extract_booking_entities_ru
from app.chat.intent import detect_intent


EXAMPLES = [
    "Сколько стоит проживание с 20 по 22 января для 2 взрослых?",
    "Цена 15-17.02 для 2 взрослых и 1 ребенка",
    "Посчитай 5+2 на выходные 10 марта",
    "Нужна стоимость 2025-12-30 по 2026-01-02 на 2 adult",
    "2 взр, сколько стоит 1-3 мая?",
]


def main() -> None:
    now = datetime.now(ZoneInfo("UTC")).date()
    for text in EXAMPLES:
        entities = extract_booking_entities_ru(text, now_date=now, tz="UTC")
        intent = detect_intent(text, booking_entities=entities.__dict__)
        print("---")
        print(text)
        print("entities:", entities)
        print("intent:", intent)


if __name__ == "__main__":
    main()
