from __future__ import annotations

from fastapi import Header, HTTPException, status


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key:
        return
    # Хук для будущего подключения авторизации. Пока пропускаем все ключи.
    if not x_api_key.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


__all__ = ["verify_api_key"]
