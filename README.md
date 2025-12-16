# AI Chatbot U4S — новый backend

## Архитектура
- **FastAPI** (`backend/app/main.py`) — единая точка входа, роуты в `app/api/v1`.
- **PostgreSQL (schema `u4s_chatbot`)** — источник фактов. Асинхронный пул `asyncpg` в `app/db/pool.py`, запросы в `app/db/queries/*`.
- **RAG / поиск** — минимальный клиент Qdrant (`app/rag/qdrant_client.py`) и сбор контекста (`app/rag/retriever.py`). FAQ из Postgres + совпадения из Qdrant формируют контекст для LLM.
- **LLM** — Amvera API к DeepSeek (`app/llm/amvera_client.py`), промпты в `app/llm/prompts.py`. Переменная `LLM_DRY_RUN` позволяет отключить реальные вызовы.
- **Бронирование** — модуль `app/booking`: модели (`models.py`), слоты (`slot_filling.py`), PMS клиент Shelter Cloud (`shelter_client.py`) и сервис расчёта (`service.py`).
- **Чат** — маршрутизация интентов (`app/chat/intent.py`) и объединение пайплайнов (`app/chat/composer.py`).

## Пайплайн /v1/chat
1. Определение интента. Ключевые слова бронирования переводят запрос в сценарий `booking_quote`, иначе используется общий RAG-поток.
2. **Слот-филлинг** для бронирования: обязательные слоты `check_in`, `check_out`, `adults`, опциональные `children`, `children_ages`. Недостающие слоты запрашиваются у пользователя, состояние хранится в `InMemoryConversationStateStore`.
3. При готовности слотов вызывается `BookingQuoteService`, который обращается к Shelter Cloud и возвращает топ-3 предложений. Ответ включает CTA «Оформить бронирование?» и отладочный блок `{intent, slots, pms_called, offers_count}`.
4. Для общего интента формируется контекст из фактов PostgreSQL + Qdrant, далее вызывается Amvera (DeepSeek) с системным промптом `FACTS_PROMPT`.

## Конфигурация окружения
- `DATABASE_URL` — строка подключения `asyncpg`.
- `QDRANT_URL` — базовый URL кластера Qdrant.
- `AMVERA_API_TOKEN` — токен доступа к Amvera API.
- `AMVERA_API_URL` — базовый URL Amvera API (по умолчанию `https://llm.amvera.ai`).
- `AMVERA_INFERENCE_NAME` — имя inference-эндпоинта Amvera (`llama`, `gpt`, `deepseek`, `qwen`).
- `AMVERA_MODEL` — модель, указанная в Amvera (например `deepseek-chat`).
- `SHELTER_CLOUD_TOKEN` — токен PMS Frontdesk24.
- `LLM_DRY_RUN` — `true/false` для отключения реальных запросов в LLM.
- `LLM_TEMPERATURE` — температура генерации (по умолчанию 0.1 для минимальных галлюцинаций).
- `LLM_MAX_TOKENS` — максимальная длина ответа от LLM (по умолчанию 350 токенов).
- `RAG_MAX_SNIPPETS` — сколько сниппетов фактов/файлов включать в контекст (по умолчанию 8).
- `RAG_CONTEXT_CHARS` / `RAG_MAX_CONTEXT_CHARS` — лимит символов контекста, обрезает слишком длинные фрагменты (по умолчанию 4000).
- `RAG_MIN_FACTS` — минимальное число совпадений, ниже которого срабатывает guard.

## RAG guard против выдумок
- Если суммарное количество попаданий (facts + files + FAQ) ниже `RAG_MIN_FACTS`, вызов LLM блокируется.
- Пользователь получает безопасный ответ: «Я не нашёл подтверждённой информации в базе знаний, поэтому не буду выдумывать...» и набор уточняющих вопросов.
- В debug приходят поля `hits_total`, `guard_triggered=true`, `llm_called=false`, чтобы быстро увидеть причину.

## Отличия Facts vs RAG
- **Facts**: структурированные данные из PostgreSQL (FAQ, номера, услуги). Используются напрямую либо для обогащения контекста.
- **RAG**: дополнительные совпадения из Qdrant по компактным embedding без внешних SDK. Контекст из фактов и RAG передаётся в Amvera (DeepSeek), что даёт актуальные ответы без галлюцинаций.

## Запуск
```bash
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --app-dir backend
```

## Проверка подключения к Amvera
```bash
AMVERA_API_TOKEN=... \
AMVERA_API_URL=https://llm.amvera.ai \
AMVERA_INFERENCE_NAME=deepseek \
AMVERA_MODEL=deepseek-chat \
python backend/scripts/test_amvera_llm.py
```

## Расширение actions
Новые сценарии действий (actions) можно подключать через маршрутизатор интентов (`app/chat/intent.py`) и реализовывать в `app/chat/composer.py`, сохраняя собственное состояние диалога и изолированные клиенты внешних систем.
