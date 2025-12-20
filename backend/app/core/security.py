from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None)
) -> None:
    # Пропускаем OPTIONS запросы (preflight) - они обрабатываются CORS middleware
    if request.method == "OPTIONS":
        return
    
    if not x_api_key:
        return
    # Хук для будущего подключения авторизации. Пока пропускаем все ключи.
    if not x_api_key.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


__all__ = ["verify_api_key"]
