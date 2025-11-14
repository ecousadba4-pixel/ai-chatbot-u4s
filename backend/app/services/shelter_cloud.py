"""Интеграция с API Shelter Cloud."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import requests

from loguru import logger

AVAILABILITY_URL = "https://pms.frontdesk24.ru/api/online/getVariants"
HOTEL_PARAMS_URL = "https://pms.frontdesk24.ru/api/online/getHotelParams"


class ShelterCloudError(RuntimeError):
    """Базовая ошибка взаимодействия с Shelter Cloud."""


class ShelterCloudAuthenticationError(ShelterCloudError):
    """Ошибка конфигурации или авторизации Shelter Cloud."""


class ShelterCloudAvailabilityError(ShelterCloudError):
    """Ошибка при запросе доступности номеров."""


@dataclass(frozen=True)
class ShelterCloudConfig:
    token: str
    language: str = "ru"
    timeout: float = 30.0

    def is_configured(self) -> bool:
        return bool(self.token)


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
        url: str,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._config.is_configured():
            raise ShelterCloudAuthenticationError("Shelter Cloud не настроен")

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
                "Shelter Cloud HTTP {status} at {url}: {body}",
                status=response.status_code,
                url=url,
                body=response.text,
            )
            raise ShelterCloudAvailabilityError(
                f"HTTP_{response.status_code}: {response.text.strip()}"
            )

        return self._safe_json(response)

    # ---- доступность номеров --------------------------------------------

    def fetch_hotel_params(self) -> dict[str, Any]:
        """Возвращает параметры отеля из Shelter Cloud."""

        data = self._request(HOTEL_PARAMS_URL, payload={})
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
        room_payload: dict[str, Any] = {
            "adults": adults,
        }
        if children:
            normalized_ages = [str(max(0, int(age))) for age in children_ages][:children]
            if normalized_ages:
                room_payload["kidsAges"] = ",".join(normalized_ages)

        payload = {
            "dateFrom": check_in,
            "dateTo": check_out,
            "rooms": [room_payload],
        }

        data = self._request(AVAILABILITY_URL, payload=payload)
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

        chunked_data = payload.get("data")
        if isinstance(chunked_data, list):
            offers.extend(ShelterCloudService._extract_from_chunked_payload(chunked_data))
            if offers:
                return offers

        rooms: list[Any] = []
        raw_rooms = payload.get("rooms") or payload.get("items")
        if isinstance(raw_rooms, list):
            rooms = raw_rooms
        else:
            if isinstance(chunked_data, list):
                for chunk in chunked_data:
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
            room_area = room.get("roomArea") or room.get("area")
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
                        "room_area": room_area,
                    }
                )
        return offers

    @staticmethod
    def _extract_from_chunked_payload(chunks: list[Any]) -> list[dict[str, Any]]:
        offers: list[dict[str, Any]] = []
        categories: dict[int, dict[str, Any]] = {}
        variants: list[dict[str, Any]] = []

        for chunk in chunks:
            if isinstance(chunk, list):
                for item in chunk:
                    if not isinstance(item, dict):
                        continue
                    if "roomCategoryID" in item and "price" in item:
                        variants.append(item)
                    elif "roomCategoryID" in item and "priceRub" in item:
                        variants.append(item)
                    elif "id" in item and (
                        "availableRooms" in item or "availableBeds" in item
                    ):
                        category_key = ShelterCloudService._normalize_category_id(item.get("id"))
                        if category_key is not None:
                            categories[category_key] = item
            elif isinstance(chunk, dict):
                if "roomCategoryID" in chunk and (
                    "price" in chunk or "priceRub" in chunk
                ):
                    variants.append(chunk)

        for variant in variants:
            category_key = ShelterCloudService._normalize_category_id(
                variant.get("roomCategoryID")
            )
            category: dict[str, Any] | None = (
                categories.get(category_key) if category_key is not None else None
            )
            room_name = str(
                (category or {}).get("name")
                or variant.get("roomName")
                or variant.get("name")
                or "Номер"
            ).strip()
            room_area = (category or {}).get("roomArea")

            price_value: float | None = None
            for key in ("price", "priceRub", "priceWithoutDiscount"):
                amount = variant.get(key)
                if amount is None:
                    continue
                try:
                    price_value = float(amount)
                    break
                except (TypeError, ValueError):
                    continue
            if price_value is None:
                continue

            currency = str(variant.get("currency") or "RUB").upper()

            offers.append(
                {
                    "name": room_name,
                    "price": price_value,
                    "currency": currency,
                    "breakfast_included": True,
                    "room_area": room_area,
                }
            )

        return offers

    @staticmethod
    def _normalize_category_id(value: Any) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text
__all__ = [
    "ShelterCloudService",
    "ShelterCloudConfig",
    "ShelterCloudError",
    "ShelterCloudAuthenticationError",
    "ShelterCloudAvailabilityError",
]
