from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.db.pool import get_pool
from app.db.queries.hotel import fetch_hotel
from app.db.queries.rooms import list_rooms
from app.db.queries.services import list_services

router = APIRouter(prefix="/facts", dependencies=[Depends(verify_api_key)])


@router.get("/hotel")
async def hotel_info(pool=Depends(get_pool)) -> dict:
    hotel = await fetch_hotel(pool)
    return {"hotel": hotel}


@router.get("/rooms")
async def rooms_info(pool=Depends(get_pool)) -> dict:
    return {"rooms": await list_rooms(pool)}


@router.get("/services")
async def services_info(pool=Depends(get_pool)) -> dict:
    return {"services": await list_services(pool)}
