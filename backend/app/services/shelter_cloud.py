"""Интеграция с API Shelter Cloud."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


class ShelterCloudError(RuntimeError):
    """Базовая ошибка взаимодействия с Shelter Cloud."""


class ShelterCloudAuthenticationError(ShelterCloudError):
    """Ошибка конфигурации или авторизации Shelter Cloud."""


class ShelterCloudAvailabilityError(ShelterCloudError):
    """Ошибка при запросе доступности номеров."""


@dataclass(frozen=True)
class ShelterCloudConfig:
    base_url: str
    token: str
    language: str = "ru"
    timeout: float = 30.0

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)


class ShelterCloudService:
    """Клиент Shelter Cloud с использованием online API токена."""

    def __init__(
        self,
        config: ShelterCloudConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()

    def is_configured(self) -> bool:
        """Возвращает, настроен ли сервис."""

        return self._config.is_configured()

    # ---- общие HTTP-хелперы ---------------------------------------------

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._config.is_configured():
            raise ShelterCloudAuthenticationError("Shelter Cloud не настроен")

        url = urljoin(self._config.base_url.rstrip("/") + "/", path.lstrip("/"))
        body = {
            "token": self._config.token,
            "language": self._config.language or "ru",
            **payload,
        }

        try:
            response = self._session.post(
                url,
                json=body,
                timeout=self._config.timeout,
            )
        except Exception as error:  # pragma: no cover - сеть может быть недоступна
            logger.exception("Shelter Cloud request error")
            raise ShelterCloudAvailabilityError(str(error)) from error

        if response.status_code >= 400:
            logger.error(
                "Shelter Cloud HTTP %s at %s: %s",
                response.status_code,
                path,
                response.text,
            )
            raise ShelterCloudAvailabilityError(
                f"HTTP_{response.status_code}: {response.text.strip()}"
            )

        return self._safe_json(response)

    # ---- доступность номеров --------------------------------------------

    def fetch_hotel_params(self) -> dict[str, Any]:
        """Возвращает параметры отеля из Shelter Cloud."""

        data = self._request("api/online/getHotelParams", payload={})
        if isinstance(data.get("data"), dict):
            return data["data"]
        return data

    def fetch_availability(
        self,
        *,
        check_in: str,
        check_out: str,
        adults: int,
        children: int,
        children_ages: Iterable[int],
    ) -> list[dict[str, Any]]:
        payload = {
            "dateFrom": check_in,
            "dateTo": check_out,
            "adults": adults,
            "children": children,
        }
        if children:
            payload["childrenAges"] = [max(0, int(age)) for age in children_ages]

        data = self._request("api/online/getAvailability", payload=payload)
        offers = self._extract_offers(data)
        offers.sort(key=lambda item: item.get("price", float("inf")))
        return offers

    # ---- утилиты --------------------------------------------------------

    @staticmethod
    def _safe_json(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"data": payload}
        return {}

    @staticmethod
    def _extract_offers(payload: dict[str, Any]) -> list[dict[str, Any]]:
        offers: list[dict[str, Any]] = []
        rooms: list[Any] = []
        raw_rooms = payload.get("rooms") or payload.get("items")
        if isinstance(raw_rooms, list):
            rooms = raw_rooms
        else:
            data_block = payload.get("data")
            if isinstance(data_block, list):
                for chunk in data_block:
                    if isinstance(chunk, list):
                        rooms.extend(item for item in chunk if isinstance(item, dict))
                    elif isinstance(chunk, dict):
                        rooms.append(chunk)
        if not rooms:
            return offers

        for room in rooms:
            if not isinstance(room, dict):
                continue
            room_name = str(
                room.get("name")
                or room.get("roomName")
                or room.get("title")
                or "Номер"
            ).strip()
            rates = room.get("rates") or room.get("offers") or []
            if not isinstance(rates, list):
                rates = []
            for rate in rates:
                if not isinstance(rate, dict):
                    continue
                price_info = rate.get("total") or rate.get("price") or {}
                amount = price_info.get("amount") or price_info.get("value")
                try:
                    price_value = float(amount)
                except (TypeError, ValueError):
                    continue
                currency = str(price_info.get("currency") or "RUB").upper()
                breakfast = False
                meal_plan = rate.get("mealPlan") or {}
                if isinstance(meal_plan, dict):
                    breakfast = bool(
                        meal_plan.get("breakfastIncluded")
                        or meal_plan.get("includesBreakfast")
                    )
                else:
                    breakfast = bool(
                        rate.get("breakfastIncluded") or rate.get("includesBreakfast")
                    )

                offers.append(
                    {
                        "name": room_name,
                        "price": price_value,
                        "currency": currency,
                        "breakfast_included": breakfast,
                    }
                )
        return offers


class ShelterCloudOfflineService:
    """Простейшая офлайн-имитация Shelter Cloud."""

    def __init__(
        self,
        offers: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._offers = list(offers) if offers is not None else list(DEFAULT_OFFLINE_OFFERS)

    @staticmethod
    def is_configured() -> bool:
        return True

    def fetch_availability(
        self,
        *,
        check_in: str,
        check_out: str,
        adults: int,
        children: int,
        children_ages: Iterable[int],
    ) -> list[dict[str, Any]]:
        nights = self._calc_nights(check_in, check_out)
        guests = max(1, int(adults) + int(children))
        offers: list[dict[str, Any]] = []
        for offer in self._offers:
            price_per_night = float(offer.get("price_per_night", offer.get("price", 0)) or 0)
            total_price = max(price_per_night, 0.0) * max(nights, 1)
            extra_fee = max(0, guests - 2) * float(offer.get("extra_guest_fee", 0) or 0)
            adjusted_price = total_price + extra_fee
            offers.append(
                {
                    "name": offer.get("name", "Номер"),
                    "price": round(adjusted_price, 2) if adjusted_price else offer.get("price"),
                    "currency": offer.get("currency", "RUB"),
                    "breakfast_included": bool(offer.get("breakfast_included", True)),
                }
            )
        offers.sort(key=lambda item: item.get("price", float("inf")))
        return offers

    @staticmethod
    def _calc_nights(check_in: str, check_out: str) -> int:
        try:
            start = dt.date.fromisoformat(check_in)
            end = dt.date.fromisoformat(check_out)
        except ValueError:
            return 1
        delta = (end - start).days
        return delta if delta > 0 else 1


DEFAULT_OFFLINE_OFFERS: tuple[dict[str, Any], ...] = (
    {
        "name": "Стандартный номер",
        "price_per_night": 9900,
        "currency": "RUB",
        "breakfast_included": True,
        "extra_guest_fee": 1200,
    },
    {
        "name": "Семейный дом",
        "price_per_night": 16800,
        "currency": "RUB",
        "breakfast_included": True,
        "extra_guest_fee": 0,
    },
)


__all__ = [
    "ShelterCloudService",
    "ShelterCloudConfig",
    "ShelterCloudError",
    "ShelterCloudAuthenticationError",
    "ShelterCloudAvailabilityError",
    "ShelterCloudOfflineService",
]
