import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.booking.slot_filling import SlotFiller


def test_extracts_dates_and_normalizes_iso():
    filler = SlotFiller()
    state = filler.extract("Заезд 01.12.2024, выезд 05/12/2024")

    assert state.check_in == "2024-12-01"
    assert state.check_out == "2024-12-05"
    assert state.errors == []


def test_detects_checkout_before_checkin_and_prompts_again():
    filler = SlotFiller()
    state = filler.extract("Заезд 10-02-2025, выезд 08-02-2025")

    message = filler.clarification(state)

    assert "выезда должна быть позже" in message.lower()
    assert "укажите дату выезда" in message.lower()
    assert state.check_out is None
    assert "check_out" in filler.missing_slots(state)


def test_extracts_guest_numbers_with_roles():
    filler = SlotFiller()
    text = "Нужно 2 взрослых, 2 детей 5 и 7 лет"

    state = filler.extract(text)

    assert state.adults == 2
    assert state.children == 2
    assert state.children_ages == [5, 7]


def test_infers_children_count_from_ages_when_missing():
    filler = SlotFiller()
    text = "2 взрослых и детям 4,6 лет"

    state = filler.extract(text)

    assert state.adults == 2
    assert state.children == 2
    assert state.children_ages == [4, 6]
