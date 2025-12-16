from __future__ import annotations

from fastapi import Depends, FastAPI

from app.api.v1 import admin, chat, facts, knowledge, rag_search
from app.booking.service import BookingQuoteService
from app.booking.shelter_client import ShelterCloudService
from app.booking.slot_filling import SlotFiller
from app.chat.composer import ChatComposer, InMemoryConversationStateStore
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.pool import get_pool
from app.llm.amvera_client import AmveraLLMClient
from app.rag.qdrant_client import QdrantClient, get_qdrant_client

settings = get_settings()
setup_logging()

state_store = InMemoryConversationStateStore()
slot_filler = SlotFiller()
qdrant_client = get_qdrant_client()
llm_client = AmveraLLMClient()
shelter_service = ShelterCloudService()
booking_service = BookingQuoteService(shelter_service)


async def lifespan(app: FastAPI):
    pool = await get_pool()
    try:
        yield
    finally:
        await pool.close()
        await qdrant_client.close()
        await llm_client.close()
        await shelter_service.close()


def composer_dependency(pool=Depends(get_pool)) -> ChatComposer:
    return ChatComposer(
        pool=pool,
        qdrant=qdrant_client,
        llm=llm_client,
        slot_filler=slot_filler,
        booking_service=booking_service,
        store=state_store,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="U4S Chat API", lifespan=lifespan)
    api_prefix = settings.api_prefix

    app.dependency_overrides[chat.get_composer] = composer_dependency

    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(facts.router, prefix=api_prefix)
    app.include_router(knowledge.router, prefix=api_prefix)
    app.include_router(rag_search.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    return app


app = create_app()
