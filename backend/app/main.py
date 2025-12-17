from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from fastapi import Depends, FastAPI

from app.api.v1 import admin, chat, diag, facts, knowledge, rag_search
from app.booking.service import BookingQuoteService
from app.booking.shelter_client import ShelterCloudService
from app.booking.slot_filling import SlotFiller
from app.chat.composer import ChatComposer, InMemoryConversationStateStore
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.pool import get_pool
from app.llm.amvera_client import AmveraLLMClient
from app.rag.embed_client import close_embed_client, get_embed_client
from app.rag.qdrant_client import QdrantClient, get_qdrant_client
from app.session import get_session_store
from app.session.redis_state_store import get_redis_state_store, close_redis_state_store

logger = logging.getLogger(__name__)

settings = get_settings()
setup_logging()

# Выбираем хранилище состояния в зависимости от конфигурации
if settings.use_redis_state_store:
    shared_state_store = get_redis_state_store()
    state_store = shared_state_store
    booking_state_store = shared_state_store
    logger.info("Using Redis state store for conversation state")
else:
    state_store = InMemoryConversationStateStore()
    booking_state_store = InMemoryConversationStateStore()
    logger.info("Using in-memory state store for conversation state")

slot_filler = SlotFiller()
qdrant_client = get_qdrant_client()
llm_client = AmveraLLMClient()
shelter_service = ShelterCloudService()
booking_service = BookingQuoteService(shelter_service)


async def _warmup_connections() -> None:
    """Прогрев соединений и health check при старте."""
    logger.info("Warming up connections and health checks...")
    
    health_status: dict[str, bool] = {
        "embed": False,
        "qdrant": False,
        "redis": False,
        "postgres": False,
    }
    
    async def warmup_embed() -> None:
        """Прогрев embed клиента."""
        try:
            embed_client = get_embed_client()
            await embed_client.embed(["warmup test"])
            health_status["embed"] = True
            logger.info("✓ Embed client ready")
        except Exception as exc:
            logger.error("✗ Embed client failed: %s", exc)

    async def warmup_qdrant() -> None:
        """Прогрев Qdrant клиента."""
        try:
            await qdrant_client.scroll(collection=settings.qdrant_collection, limit=1)
            health_status["qdrant"] = True
            logger.info("✓ Qdrant client ready")
        except Exception as exc:
            logger.error("✗ Qdrant client failed: %s", exc)

    async def warmup_redis() -> None:
        """Прогрев Redis."""
        try:
            session_store = get_session_store()
            if await session_store.ping():
                health_status["redis"] = True
                logger.info("✓ Redis ready")
            else:
                logger.warning("⚠ Redis ping returned False")
        except Exception as exc:
            logger.error("✗ Redis failed: %s", exc)

    async def warmup_postgres() -> None:
        """Прогрев PostgreSQL."""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["postgres"] = True
            logger.info("✓ PostgreSQL ready")
        except Exception as exc:
            logger.error("✗ PostgreSQL failed: %s", exc)

    # Параллельный прогрев всех сервисов
    await asyncio.gather(
        warmup_embed(),
        warmup_qdrant(),
        warmup_redis(),
        warmup_postgres(),
        return_exceptions=True
    )

    # Проверка критических сервисов
    critical_services = ["embed", "qdrant", "postgres"]
    failed_critical = [svc for svc in critical_services if not health_status[svc]]
    
    if failed_critical:
        logger.error(
            "CRITICAL: Some essential services are not available: %s",
            ", ".join(failed_critical)
        )
    
    logger.info(
        "Warmup complete. Status: embed=%s, qdrant=%s, redis=%s, postgres=%s",
        "✓" if health_status["embed"] else "✗",
        "✓" if health_status["qdrant"] else "✗",
        "✓" if health_status["redis"] else "✗",
        "✓" if health_status["postgres"] else "✗",
    )


async def lifespan(app: FastAPI):
    pool = await get_pool()
    warmup_task: asyncio.Task | None = None

    # Прогрев соединений (в фоне, если включено)
    if settings.enable_startup_warmup:
        def _log_warmup_result(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                logger.error("Warmup task failed: %s", exc)

        warmup_task = asyncio.create_task(_warmup_connections())
        warmup_task.add_done_callback(_log_warmup_result)
    else:
        logger.info("Startup warmup is disabled via configuration")

    try:
        yield
    finally:
        if warmup_task:
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

        # Закрываем все соединения
        await pool.close()
        await qdrant_client.close()
        await llm_client.close()
        await shelter_service.close()
        await close_embed_client()
        
        # Закрываем Redis state store если используется
        if settings.use_redis_state_store:
            await close_redis_state_store()
            logger.info("Redis state store closed")


def composer_dependency(pool=Depends(get_pool)) -> ChatComposer:
    return ChatComposer(
        pool=pool,
        qdrant=qdrant_client,
        llm=llm_client,
        slot_filler=slot_filler,
        booking_service=booking_service,
        store=state_store,
        booking_fsm_store=booking_state_store,
        settings=settings,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="U4S Chat API", lifespan=lifespan)
    api_prefix = settings.api_prefix

    app.dependency_overrides[chat.get_composer] = composer_dependency

    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(facts.router, prefix=api_prefix)
    app.include_router(knowledge.router, prefix=api_prefix)
    app.include_router(rag_search.router, prefix=api_prefix)
    app.include_router(diag.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    return app


app = create_app()
