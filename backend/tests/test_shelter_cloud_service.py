import pytest

from backend.app.dialogue.manager import BookingDialogueManager
from backend.app.services.shelter_cloud import ShelterCloudService


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {
                "data": [
                    [
                        {
                            "id": 6930,
                            "name": "Стандарт",
                            "roomArea": 32,
                            "availableRooms": 3,
                        }
                    ],
                    [],
                    [],
                    [],
                    [],
                    [
                        {
                            "roomCategoryID": 6930,
                            "tariffID": 23974,
                            "price": 5000,
                            "currency": "rub",
                        }
                    ],
                ]
            },
            {
                "name": "Стандарт",
                "price": 5000.0,
                "currency": "RUB",
                "breakfast_included": True,
                "room_area": 32,
            },
        ),
        (
            {
                "rooms": [
                    {
                        "name": "Стандарт",
                        "roomArea": 20,
                        "rates": [
                            {
                                "price": {
                                    "amount": 3500,
                                    "currency": "eur",
                                },
                                "mealPlan": {"breakfastIncluded": True},
                            }
                        ],
                    }
                ]
            },
            {
                "name": "Стандарт",
                "price": 3500.0,
                "currency": "EUR",
                "breakfast_included": True,
                "room_area": 20,
            },
        ),
    ],
)
def test_extract_offers(payload, expected):
    offers = ShelterCloudService._extract_offers(payload)
    assert offers
    offer = offers[0]
    assert offer == expected


def test_offer_name_includes_room_area():
    offer = {"name": "Стандарт", "room_area": 25}
    assert BookingDialogueManager._offer_name(offer) == "Стандарт (25 м²)"


def test_offer_name_without_area():
    offer = {"name": "Стандарт"}
    assert BookingDialogueManager._offer_name(offer) == "Стандарт"
