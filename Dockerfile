FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 1) Ставим зависимости
# Вариант A: requirements.txt в backend/
COPY backend/requirements*.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 2) Копируем код
COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]

