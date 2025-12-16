import asyncio
import os
import sys
from datetime import date
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("AMVERA_API_TOKEN", "test-token")
os.environ.setdefault("SHELTER_CLOUD_TOKEN", "test-shelter")

from app.booking.entities import extract_booking_entities_ru
from app.booking.models import BookingQuote, Guests
from app.chat.composer import ChatComposer, InMemoryConversationStateStore
from app.booking.slot_filling import SlotFiller


class DummyBookingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get_quotes(self, *, check_in: str, check_out: str, guests: Guests):
        self.calls.append({"check_in": check_in, "check_out": check_out, "guests": guests})
        return [
            BookingQuote(
                room_name="Студия",
                total_price=12000,
                currency="RUB",
                breakfast_included=True,
                room_area=None,
                check_in=check_in,
                check_out=check_out,
                guests=guests,
            )
        ]


def test_booking_calculation_fsm_linear_flow(monkeypatch):
    store = InMemoryConversationStateStore()
    fsm_store = InMemoryConversationStateStore()
    slot_filler = SlotFiller()
    booking_service = DummyBookingService()

    composer = ChatComposer(
        pool=None,  # type: ignore[arg-type]
        qdrant=None,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        slot_filler=slot_filler,
        booking_service=booking_service,
        store=store,
        booking_fsm_store=fsm_store,
    )

    today = date(2024, 12, 1)
    
    class FixedDate(date):
        @classmethod
        def today(cls):  # noqa: D401
            return today

    monkeypatch.setattr("app.booking.slot_filling.date", FixedDate)
    entities = extract_booking_entities_ru("хочу рассчитать", now_date=today)

    response = asyncio.run(
        composer.handle_booking_calculation("fsm", "хочу рассчитать", entities)
    )
    assert "какую дату" in response["answer"].lower()

    entities = extract_booking_entities_ru("19 декабря", now_date=today)
    response = asyncio.run(
        composer.handle_booking_calculation("fsm", "19 декабря", entities)
    )
    assert "сколько ночей" in response["answer"].lower()

    response = asyncio.run(
        composer.handle_booking_calculation("fsm", "2", entities)
    )
    assert "сколько взрослых" in response["answer"].lower()

    response = asyncio.run(
        composer.handle_booking_calculation("fsm", "двое", entities)
    )
    assert "будут ли дети" in response["answer"].lower()

    response = asyncio.run(
        composer.handle_booking_calculation("fsm", "нет", entities)
    )
    assert "оформляем бронирование" in response["answer"].lower()
    assert "студия" in response["answer"].lower()

    assert booking_service.calls
    last_call = booking_service.calls[-1]
    assert last_call["check_in"] == "2024-12-19"
    assert last_call["check_out"] == "2024-12-21"
    guests: Guests = last_call["guests"]  # type: ignore[assignment]
    assert guests.adults == 2
    assert guests.children == 0
