FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 1. Копируем зависимости и устанавливаем
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# 2. Копируем код приложения (сохраняем пакет backend)
COPY backend ./backend

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

