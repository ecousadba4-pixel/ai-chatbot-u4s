import asyncio
import datetime as dt

from backend.app.services import ShelterCloudAvailabilityError
from backend.tests._helpers import DummyRedisGateway, DummyRequest
from backend.app.dialogue import manager as manager_module


class DummyShelterCloudService:
    def __init__(self, offers):
        self.offers = offers
        self.calls = []

    def fetch_availability(self, **kwargs):
        self.calls.append(kwargs)
        return self.offers


def _prepare_booking(app_module, monkeypatch, offers=None):
    redis_gateway = DummyRedisGateway()
    monkeypatch.setattr(app_module, "REDIS_GATEWAY", redis_gateway)

    service = DummyShelterCloudService(offers or [])
    monkeypatch.setattr(app_module, "SHELTER_CLOUD_SERVICE", service)
    app_module.BOOKING_DIALOGUE_MANAGER.storage = redis_gateway
    app_module.BOOKING_DIALOGUE_MANAGER.service = service

    return redis_gateway, service


def test_booking_flow_success(app_module, monkeypatch):
    redis_gateway, service = _prepare_booking(
        app_module,
        monkeypatch,
        offers=[
            {
                "name": "Стандарт",
                "price": 12345,
                "currency": "RUB",
                "breakfast_included": True,
            }
        ],
    )

    session_id = "booking"

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Хочу забронировать номер"})
        )
    )

    assert response["answer"].startswith("Когда планируете заехать?")
    assert response["intent"] == "booking_inquiry"
    assert response["branch"] == "booking_price_chat"

    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "Заезд 10.10.2024"}))
    )
    assert "до какого числа планируете остаться" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "Выезд 12.10.2024"}))
    )
    assert "сколько взрослых" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "2 взрослых"}))
    )
    assert "сколько будет детей" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "1 ребенок"}))
    )
    assert "возраст детей" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "5 лет"}))
    )

    answer = response["answer"]
    assert "Нашла 1 вариант" in answer
    assert "1. Стандарт" in answer
    assert "12 345" in answer
    assert "площадь не указана" in answer
    assert "(завтрак включён)" in answer
    assert response["branch"] == "booking_price_chat"

    assert service.calls
    last_call = service.calls[-1]
    assert last_call["check_in"] == "2024-10-10"
    assert last_call["check_out"] == "2024-10-12"
    assert last_call["adults"] == 2
    assert last_call["children"] == 1
    assert last_call["children_ages"] == [5]

    context = redis_gateway.context_storage[session_id]
    assert context["intent"] == "booking_inquiry"
    assert context["booking"]["adults"] == 2
    assert context["booking"]["children_ages"] == [5]


def test_booking_flow_handles_no_rooms(app_module, monkeypatch):
    redis_gateway, service = _prepare_booking(app_module, monkeypatch, offers=[])

    session_id = "no-rooms"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Нужно забронировать номер"})
        )
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "01.12.2024"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "05.12.2024"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "2"}))
    )
    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "0"}))
    )

    assert "нет свободных номеров" in response["answer"].lower()
    assert response["branch"] == "online_booking_redirect"


def test_booking_accepts_various_date_formats(app_module, monkeypatch):
    redis_gateway, _service = _prepare_booking(app_module, monkeypatch, offers=[])

    class FixedDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2025, 11, 19)

    monkeypatch.setattr(manager_module.dt, "date", FixedDate)

    session_id = "formats"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "нужно забронировать"})
        )
    )

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "заезд 25 ноября 2025"})
        )
    )
    assert "до какого числа планируете остаться" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "на 3 ночи"})
        )
    )
    assert "28.11.2025" in response["answer"]
    assert "взросл" in response["answer"].lower()

    context = redis_gateway.context_storage[session_id]
    assert context["booking"]["check_in"] == "2025-11-25"
    assert context["booking"]["check_out"] == "2025-11-28"


def test_booking_accepts_relative_dates(app_module, monkeypatch):
    redis_gateway, _service = _prepare_booking(app_module, monkeypatch, offers=[])

    class FixedDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2025, 11, 19)

    monkeypatch.setattr(manager_module.dt, "date", FixedDate)

    session_id = "relative"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "хочу бронь"})
        )
    )

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "завтра"})
        )
    )
    assert "до какого числа планируете остаться" in response["answer"].lower()

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "в эту пятницу"})
        )
    )
    assert "21.11.2025" in response["answer"]

    context = redis_gateway.context_storage[session_id]
    assert context["booking"]["check_in"] == "2025-11-20"
    assert context["booking"]["check_out"] == "2025-11-21"


def test_booking_flow_handles_api_errors(app_module, monkeypatch):
    redis_gateway = DummyRedisGateway()
    monkeypatch.setattr(app_module, "REDIS_GATEWAY", redis_gateway)

    class FailingShelterService:
        def fetch_availability(self, **kwargs):
            raise ShelterCloudAvailabilityError("API недоступен")

    service = FailingShelterService()
    monkeypatch.setattr(app_module, "SHELTER_CLOUD_SERVICE", service)
    app_module.BOOKING_DIALOGUE_MANAGER.storage = redis_gateway
    app_module.BOOKING_DIALOGUE_MANAGER.service = service

    session_id = "error"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Хочу забронировать"})
        )
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "12.01.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "14.01.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "2"}))
    )
    response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "0"}))
    )

    assert "не удалось получить доступность" in response["answer"].lower()
    assert "онлайн-бронир" in response["answer"].lower()
    assert response["branch"] == "online_booking_redirect"


def test_booking_online_redirect_branch(app_module, monkeypatch):
    redis_gateway, service = _prepare_booking(app_module, monkeypatch, offers=[])

    session_id = "redirect"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Хочу забронировать номер"})
        )
    )

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Лучше забронирую онлайн"})
        )
    )

    assert "онлайн-бронир" in response["answer"].lower()
    assert response["branch"] == "online_booking_redirect"


def test_booking_more_offer_requests(app_module, monkeypatch):
    redis_gateway, service = _prepare_booking(
        app_module,
        monkeypatch,
        offers=[
            {
                "name": "Стандарт",
                "price": 10000,
                "currency": "RUB",
                "breakfast_included": True,
            },
            {
                "name": "Люкс",
                "price": 15000,
                "currency": "RUB",
                "breakfast_included": False,
                "room_area": 45,
            },
            {
                "name": "Семейный",
                "price": 18000,
                "currency": "RUB",
                "breakfast_included": True,
                "room_area": 55,
            },
            {
                "name": "Пентхаус",
                "price": 35000,
                "currency": "RUB",
                "breakfast_included": True,
                "room_area": 120,
            },
        ],
    )

    session_id = "more-offers"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Нужно забронировать номер"})
        )
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "10.12.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "12.12.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "2"}))
    )
    final_response = asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "0"}))
    )

    assert "Нашла 4 варианта" in final_response["answer"]
    assert "1. Стандарт" in final_response["answer"]
    assert "2. Люкс" in final_response["answer"]
    assert "3. Семейный" in final_response["answer"]
    assert final_response["branch"] == "booking_price_chat"

    more_response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "покажи больше вариантов"})
        )
    )

    assert "Показываю ещё варианты" in more_response["answer"]
    assert "4. Пентхаус" in more_response["answer"]
    assert "(завтрак включён)" in more_response["answer"]
    assert more_response["branch"] == "booking_price_chat"

    no_more_response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "покажи больше вариантов"})
        )
    )

    assert "все доступные предложения" in no_more_response["answer"].lower()
    assert no_more_response["branch"] == "booking_price_chat"

    context = redis_gateway.context_storage[session_id]
    assert context["last_offer_index"] == 3


def test_booking_shares_online_form_link(app_module, monkeypatch):
    _prepare_booking(
        app_module,
        monkeypatch,
        offers=[
            {
                "name": "Стандарт",
                "price": 12345,
                "currency": "RUB",
                "breakfast_included": True,
            }
        ],
    )

    session_id = "booking-link"

    asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Нужно забронировать номер"})
        )
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "01.02.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "03.02.2025"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "2"}))
    )
    asyncio.run(
        app_module.chat_post(DummyRequest({"sessionId": session_id, "question": "0"}))
    )

    response = asyncio.run(
        app_module.chat_post(
            DummyRequest({"sessionId": session_id, "question": "Хочу забронировать этот номер"})
        )
    )

    assert response["answer"].startswith(
        "Отлично! Для продолжения бронирования перейдите по ссылке:"
    )
    assert "<https://usadba4.ru/bronirovanie/>" in response["answer"]
    assert response["branch"] == "booking_price_chat"
