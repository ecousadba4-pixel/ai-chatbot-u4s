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
from app.booking.fsm import BookingContext, BookingState
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


@pytest.fixture()
def booking_fsm_env(monkeypatch):
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
    monkeypatch.setattr("app.booking.parsers.date", FixedDate)
    monkeypatch.setattr("app.booking.entities.date", FixedDate)

    def make_entities(message: str):
        return extract_booking_entities_ru(message, now_date=today)

    return composer, booking_service, make_entities, fsm_store


def test_booking_calculation_fsm_linear_flow(booking_fsm_env):
    composer, booking_service, make_entities, _fsm_store = booking_fsm_env

    def send(message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation("fsm", message, entities))

    response = send("хочу рассчитать")
    assert "какую дату" in response["answer"].lower()

    response = send("19 декабря")
    assert "сколько ночей" in response["answer"].lower()

    response = send("2")
    assert "сколько взрослых" in response["answer"].lower()

    response = send("двое")
    assert "сколько детей" in response["answer"].lower()

    response = send("нет")
    assert "оформляем бронирование" in response["answer"].lower()
    assert "студия" in response["answer"].lower()

    assert booking_service.calls
    last_call = booking_service.calls[-1]
    assert last_call["check_in"] == "2024-12-19"
    assert last_call["check_out"] == "2024-12-21"
    guests: Guests = last_call["guests"]  # type: ignore[assignment]
    assert guests.adults == 2
    assert guests.children == 0


def test_children_step_accepts_number_and_moves_forward(booking_fsm_env):
    composer, _booking_service, make_entities, fsm_store = booking_fsm_env

    def send(message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation("fsm", message, entities))

    send("хочу рассчитать")
    send("19 декабря")
    send("2")
    send("2")
    response = send("2")

    assert "возраст дет" in response["answer"].lower()
    context = BookingContext.from_dict(fsm_store.get("fsm"))
    assert context
    assert context.children == 2
    assert context.state == BookingState.ASK_CHILDREN_AGES


def test_children_step_handles_zero_and_yes(booking_fsm_env):
    composer, _booking_service, make_entities, fsm_store = booking_fsm_env

    def send(session_id: str, message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation(session_id, message, entities))

    session_zero = "fsm-zero"
    send(session_zero, "хочу рассчитать")
    send(session_zero, "19 декабря")
    send(session_zero, "2")
    send(session_zero, "2")

    response_zero = send(session_zero, "нет")
    assert "оформляем" in response_zero["answer"].lower()
    context_zero = BookingContext.from_dict(fsm_store.get(session_zero))
    assert context_zero
    assert context_zero.children == 0

    session_yes = "fsm-yes"
    send(session_yes, "хочу рассчитать")
    send(session_yes, "19 декабря")
    send(session_yes, "2")
    send(session_yes, "2")
    response_yes = send(session_yes, "да")
    assert "сколько детей" in response_yes["answer"].lower()
    context_yes = BookingContext.from_dict(fsm_store.get(session_yes))
    assert context_yes
    assert context_yes.state == BookingState.ASK_CHILDREN_COUNT


def test_combined_guests_parsing_skips_children_question(booking_fsm_env):
    composer, _booking_service, make_entities, fsm_store = booking_fsm_env

    def send(message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation("fsm", message, entities))

    send("хочу рассчитать")
    send("19 декабря")
    send("2")
    response = send("2 взрослых и 2 детей")

    assert "возраст дет" in response["answer"].lower()
    context = BookingContext.from_dict(fsm_store.get("fsm"))
    assert context
    assert context.adults == 2
    assert context.children == 2
    assert context.state == BookingState.ASK_CHILDREN_AGES


def test_repeat_children_value_does_not_reset(booking_fsm_env):
    composer, _booking_service, make_entities, fsm_store = booking_fsm_env

    def send(message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation("fsm", message, entities))

    send("хочу рассчитать")
    send("19 декабря")
    send("2")
    send("2")
    send("2")
    response = send("2")

    assert "возраст дет" in response["answer"].lower()
    context = BookingContext.from_dict(fsm_store.get("fsm"))
    assert context
    assert context.children == 2
    assert context.state == BookingState.ASK_CHILDREN_AGES


def test_reset_happens_only_on_explicit_command(booking_fsm_env):
    composer, _booking_service, make_entities, fsm_store = booking_fsm_env

    def send(message: str):
        entities = make_entities(message)
        return asyncio.run(composer.handle_booking_calculation("fsm", message, entities))

    send("хочу рассчитать")
    send("19 декабря")
    send("2")
    intermediate = send("что?")
    assert "какую дату" not in intermediate["answer"].lower()

    cancel_response = send("начать заново")
    assert "отменяю бронирование" in cancel_response["answer"].lower()
    assert fsm_store.get("fsm") is None
