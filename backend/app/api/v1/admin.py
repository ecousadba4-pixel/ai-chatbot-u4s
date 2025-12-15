from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/admin")


@router.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
