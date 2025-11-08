FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 1. Копируем зависимости и устанавливаем
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# 2. Копируем код приложения
COPY backend/app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

