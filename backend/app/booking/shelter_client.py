from __future__ import annotations

from typing import Any, Iterable

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.booking.models import BookingQuote, Guests

AVAILABILITY_URL = "https://pms.frontdesk24.ru/api/online/getVariants"
HOTEL_PARAMS_URL = "https://pms.frontdesk24.ru/api/online/getHotelParams"


class ShelterCloudError(RuntimeError):
    """Базовая ошибка взаимодействия с Shelter Cloud."""


class ShelterCloudAuthenticationError(ShelterCloudError):
    """Ошибка конфигурации или авторизации Shelter Cloud."""


class ShelterCloudAvailabilityError(ShelterCloudError):
    """Ошибка при запросе доступности номеров."""


class ShelterCloudService:
    def __init__(self, *, token: str | None = None, language: str = "ru") -> None:
        settings = get_settings()
        self._token = token or settings.shelter_cloud_token
        self._language = language
        self._client = httpx.AsyncClient()

    def is_configured(self) -> bool:
        return bool(self._token)

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_hotel_params(self) -> dict[str, Any]:
        data = await self._request(HOTEL_PARAMS_URL, payload={})
        if isinstance(data.get("data"), dict):
            return data["data"]
        return data

    async def fetch_availability(
        self,
        *,
        check_in: str,
        check_out: str,
        guests: Guests,
    ) -> list[BookingQuote]:
        payload: dict[str, Any] = {
            "dateFrom": check_in,
            "dateTo": check_out,
            "rooms": [self._room_payload(guests)],
        }
        data = await self._request(AVAILABILITY_URL, payload=payload)
        offers = self._extract_offers(data, guests=guests, dates=(check_in, check_out))
        offers.sort(key=lambda item: item.total_price)
        return offers

    async def _request(self, url: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_configured():
            raise ShelterCloudAuthenticationError("Shelter Cloud не настроен")

        body = {
            "token": self._token,
            "language": self._language,
            **payload,
        }

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
        ):
            with attempt:
                response = await self._client.post(url, json=body)
                response.raise_for_status()
                return self._safe_json(response)
        return {}

    @staticmethod
    def _room_payload(guests: Guests) -> dict[str, Any]:
        payload = {"adults": guests.adults}
        if guests.children:
            normalized = [str(max(0, int(age))) for age in guests.children_ages][: guests.children]
            if normalized:
                payload["kidsAges"] = ",".join(normalized)
        return payload

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"data": payload}
        return {}

    def _extract_offers(
        self, payload: dict[str, Any], *, guests: Guests, dates: tuple[str, str]
    ) -> list[BookingQuote]:
        offers: list[BookingQuote] = []
        chunked_data = payload.get("data")
        rooms: list[Any] = []

        if isinstance(chunked_data, list):
            rooms.extend(self._extract_from_chunked_payload(chunked_data))
        if not rooms:
            raw_rooms = payload.get("rooms") or payload.get("items")
            if isinstance(raw_rooms, list):
                rooms.extend(raw_rooms)

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
                quote = self._build_quote(rate, room_name, room_area, guests, dates)
                if quote:
                    offers.append(quote)
        return offers

    def _build_quote(
        self,
        rate: dict[str, Any],
        room_name: str,
        room_area: Any,
        guests: Guests,
        dates: tuple[str, str],
    ) -> BookingQuote | None:
        if not isinstance(rate, dict):
            return None

        price_value, currency = self._extract_price(rate)
        if price_value is None:
            return None

        breakfast = self._is_breakfast_included(rate)

        return BookingQuote(
            room_name=room_name,
            total_price=price_value,
            currency=currency,
            breakfast_included=breakfast,
            room_area=room_area if isinstance(room_area, (int, float)) else None,
            check_in=dates[0],
            check_out=dates[1],
            guests=guests,
        )

    def _extract_from_chunked_payload(self, chunks: list[Any]) -> list[dict[str, Any]]:
        rooms: list[dict[str, Any]] = []
        categories: dict[int, dict[str, Any]] = {}
        variants: list[dict[str, Any]] = []

        for chunk in chunks:
            if isinstance(chunk, list):
                for item in chunk:
                    if not isinstance(item, dict):
                        continue
                    if "roomCategoryID" in item and (
                        "price" in item or "priceRub" in item
                    ):
                        variants.append(item)
                    elif "id" in item and (
                        "availableRooms" in item or "availableBeds" in item
                    ):
                        key = self._normalize_category_id(item.get("id"))
                        if key is not None:
                            categories[key] = item
            elif isinstance(chunk, dict):
                if "roomCategoryID" in chunk and (
                    "price" in chunk or "priceRub" in chunk
                ):
                    variants.append(chunk)

        for variant in variants:
            category_key = self._normalize_category_id(variant.get("roomCategoryID"))
            category = categories.get(category_key) if category_key is not None else None
            room_name = str(
                (category or {}).get("name")
                or variant.get("roomName")
                or variant.get("name")
                or "Номер"
            ).strip()
            room_area = (category or {}).get("roomArea")
            price_value, currency = self._extract_variant_price(variant)
            if price_value is None:
                continue

            rooms.append(
                {
                    "name": room_name,
                    "roomArea": room_area,
                    "rates": [
                        {
                            "total": {"amount": price_value, "currency": currency},
                            "mealPlan": {"breakfastIncluded": True},
                        }
                    ],
                }
            )
        return rooms

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

    @staticmethod
    def _extract_price(rate: dict[str, Any]) -> tuple[float | None, str]:
        price_info = rate.get("total") or rate.get("price") or {}
        if not isinstance(price_info, dict):
            return None, "RUB"

        amount = price_info.get("amount") or price_info.get("value")
        price_value = ShelterCloudService._to_float(amount)
        if price_value is None:
            return None, "RUB"

        currency = str(price_info.get("currency") or "RUB").upper()
        return price_value, currency

    @staticmethod
    def _is_breakfast_included(rate: dict[str, Any]) -> bool:
        meal_plan = rate.get("mealPlan") or {}
        if isinstance(meal_plan, dict):
            return bool(
                meal_plan.get("breakfastIncluded")
                or meal_plan.get("includesBreakfast")
            )
        return bool(rate.get("breakfastIncluded") or rate.get("includesBreakfast"))

    @staticmethod
    def _extract_variant_price(variant: dict[str, Any]) -> tuple[float | None, str]:
        price_value: float | None = None
        for key in ("price", "priceRub", "priceWithoutDiscount"):
            price_value = ShelterCloudService._to_float(variant.get(key))
            if price_value is not None:
                break
        if price_value is None:
            return None, "RUB"

        currency = str(variant.get("currency") or "RUB").upper()
        return price_value, currency

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "ShelterCloudService",
    "ShelterCloudError",
    "ShelterCloudAuthenticationError",
    "ShelterCloudAvailabilityError",
]
