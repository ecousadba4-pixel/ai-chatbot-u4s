"""Интеграция с API Shelter Cloud."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


class ShelterCloudError(RuntimeError):
    """Базовая ошибка взаимодействия с Shelter Cloud."""


class ShelterCloudAuthenticationError(ShelterCloudError):
    """Не удалось получить access token."""


class ShelterCloudAvailabilityError(ShelterCloudError):
    """Ошибка при запросе доступности номеров."""


@dataclass(frozen=True)
class ShelterCloudConfig:
    base_url: str
    client_id: str
    client_secret: str
    timeout: float = 30.0

    def is_configured(self) -> bool:
        return bool(self.base_url and self.client_id and self.client_secret)


class ShelterCloudService:
    """Клиент Shelter Cloud с кэшированием токена."""

    def __init__(
        self,
        config: ShelterCloudConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._token: str | None = None
        self._token_expire_at: float = 0.0

    # ---- авторизация -----------------------------------------------------

    def authorize(self, *, force: bool = False) -> str:
        if not self._config.is_configured():
            raise ShelterCloudAuthenticationError("Shelter Cloud не настроен")

        now = time.time()
        if not force and self._token and now < self._token_expire_at - 5.0:
            return self._token

        token_url = urljoin(self._config.base_url.rstrip("/") + "/", "oauth/token")
        payload = {
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "grant_type": "client_credentials",
        }
        try:
            response = self._session.post(token_url, data=payload, timeout=self._config.timeout)
        except Exception as error:  # pragma: no cover - сеть может быть недоступна
            logger.exception("Shelter Cloud auth request error")
            raise ShelterCloudAuthenticationError(str(error)) from error

        if response.status_code >= 400:
            logger.error("Shelter Cloud auth HTTP %s: %s", response.status_code, response.text)
            raise ShelterCloudAuthenticationError(
                f"AUTH_HTTP_{response.status_code}: {response.text.strip()}"
            )

        data = self._safe_json(response)
        token = str(data.get("access_token", "")).strip()
        expires_in = float(data.get("expires_in", 0) or 0)
        if not token:
            raise ShelterCloudAuthenticationError("Пустой access token")

        self._token = token
        self._token_expire_at = now + max(expires_in, 30.0)
        return token

    # ---- доступность номеров --------------------------------------------

    def fetch_availability(
        self,
        *,
        check_in: str,
        check_out: str,
        adults: int,
        children: int,
        children_ages: Iterable[int],
    ) -> list[dict[str, Any]]:
        token = self.authorize()

        availability_url = urljoin(
            self._config.base_url.rstrip("/") + "/",
            "api/v1/availability",
        )
        params = {
            "arrivalDate": check_in,
            "departureDate": check_out,
            "adults": adults,
            "children": children,
        }
        if children:
            params["childrenAges"] = ",".join(str(max(0, int(age))) for age in children_ages)

        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = self._session.get(
                availability_url,
                params=params,
                headers=headers,
                timeout=self._config.timeout,
            )
        except Exception as error:  # pragma: no cover - сеть может быть недоступна
            logger.exception("Shelter Cloud availability request error")
            raise ShelterCloudAvailabilityError(str(error)) from error

        if response.status_code >= 400:
            logger.error(
                "Shelter Cloud availability HTTP %s: %s",
                response.status_code,
                response.text,
            )
            raise ShelterCloudAvailabilityError(
                f"AVAILABILITY_HTTP_{response.status_code}: {response.text.strip()}"
            )

        data = self._safe_json(response)
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
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _extract_offers(payload: dict[str, Any]) -> list[dict[str, Any]]:
        offers: list[dict[str, Any]] = []
        rooms = payload.get("rooms") or payload.get("items") or []
        if not isinstance(rooms, list):
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
                    breakfast = bool(rate.get("breakfastIncluded") or rate.get("includesBreakfast"))

                offers.append(
                    {
                        "name": room_name,
                        "price": price_value,
                        "currency": currency,
                        "breakfast_included": breakfast,
                    }
                )
        return offers


__all__ = [
    "ShelterCloudService",
    "ShelterCloudConfig",
    "ShelterCloudError",
    "ShelterCloudAuthenticationError",
    "ShelterCloudAvailabilityError",
]
