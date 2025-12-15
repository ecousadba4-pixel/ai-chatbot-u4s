from __future__ import annotations

from typing import Sequence

from fastapi import HTTPException, status

from app.booking.models import BookingQuote, Guests
from app.booking.shelter_client import (
    ShelterCloudAvailabilityError,
    ShelterCloudAuthenticationError,
    ShelterCloudService,
)


class BookingQuoteService:
    def __init__(self, shelter: ShelterCloudService) -> None:
        self._shelter = shelter

    async def get_quotes(
        self, *, check_in: str, check_out: str, guests: Guests
    ) -> list[BookingQuote]:
        try:
            offers = await self._shelter.fetch_availability(
                check_in=check_in, check_out=check_out, guests=guests
            )
        except ShelterCloudAuthenticationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        except ShelterCloudAvailabilityError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
        return offers[:3]


__all__ = ["BookingQuoteService"]
