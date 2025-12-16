import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.booking.entities import BookingEntities
from app.booking.models import BookingQuote, Guests
from app.chat.formatting import format_shelter_quote
from app.core.config import get_settings


def _reset_settings_cache():
    try:
        get_settings.cache_clear()
    except AttributeError:
        pass


def _prepare_settings_env(monkeypatch, max_options: str) -> None:
    monkeypatch.setenv("MAX_OPTIONS", max_options)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    monkeypatch.setenv("AMVERA_API_TOKEN", "test-amvera")
    monkeypatch.setenv("SHELTER_CLOUD_TOKEN", "test-shelter")
    _reset_settings_cache()


def test_format_shelter_quote_renders_readable_blocks(monkeypatch):
    _prepare_settings_env(monkeypatch, "6")

    entities = BookingEntities(
        checkin="2025-01-20",
        checkout="2025-01-22",
        adults=2,
        children=1,
        nights=2,
        missing_fields=[],
    )
    guests = Guests(adults=2, children=1)
    offers = [
        BookingQuote(
            room_name="Стандарт",
            total_price=25000,
            currency="RUB",
            breakfast_included=False,
            room_area=30,
            check_in=entities.checkin or "",
            check_out=entities.checkout or "",
            guests=guests,
        ),
        BookingQuote(
            room_name="Эконом",
            total_price=19230,
            currency="RUB",
            breakfast_included=True,
            room_area=None,
            check_in=entities.checkin or "",
            check_out=entities.checkout or "",
            guests=guests,
        ),
    ]

    answer = format_shelter_quote(entities, offers)

    assert (
        answer
        == "На даты 20.01–22.01 (2 ночи) для 2 взрослых и 1 детей доступны варианты:\n\n"
        "🏠 Эконом\n"
        "— 19 230 ₽\n"
        "— завтрак включён\n\n"
        "🏠 Стандарт\n"
        "— 25 000 ₽\n"
        "— 30 м²\n\n"
        "Нужно оформить бронирование?"
    )

    _reset_settings_cache()


def test_format_shelter_quote_respects_limit_and_currency(monkeypatch):
    _prepare_settings_env(monkeypatch, "2")

    entities = BookingEntities(
        checkin="2025-03-01",
        checkout="2025-03-04",
        adults=1,
        children=0,
        nights=None,
        missing_fields=[],
    )
    guests = Guests(adults=1, children=0)
    offers = [
        BookingQuote(
            room_name="Дорм",
            total_price=4500,
            currency="EUR",
            breakfast_included=None,  # type: ignore[arg-type]
            room_area=None,
            check_in=entities.checkin or "",
            check_out=entities.checkout or "",
            guests=guests,
        ),
        BookingQuote(
            room_name="Стандарт",
            total_price=5000,
            currency="USD",
            breakfast_included=None,  # type: ignore[arg-type]
            room_area=None,
            check_in=entities.checkin or "",
            check_out=entities.checkout or "",
            guests=guests,
        ),
        BookingQuote(
            room_name="Люкс",
            total_price=4700,
            currency="RUB",
            breakfast_included=False,
            room_area=40,
            check_in=entities.checkin or "",
            check_out=entities.checkout or "",
            guests=guests,
        ),
    ]

    answer = format_shelter_quote(entities, offers)

    assert (
        answer
        == "На даты 01.03–04.03 (3 ночи) для 1 взрослых доступны варианты:\n\n"
        "🏠 Дорм\n"
        "— 4 500 EUR\n\n"
        "🏠 Люкс\n"
        "— 4 700 ₽\n"
        "— 40 м²\n\n"
        "…и ещё 1 вариантов. Сказать \"покажи ещё\"?\n\n"
        "Нужно оформить бронирование?"
    )

    _reset_settings_cache()
