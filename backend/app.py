# backend/app.py
import os
import re
import json
from typing import List

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ========================
#  Переменные окружения
# ========================
YANDEX_API_KEY   = os.environ.get("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
VECTOR_STORE_ID  = os.environ.get("VECTOR_STORE_ID", "")

# YC endpoints
FILES_API     = "https://rest-assistant.api.cloud.yandex.net/v1"
RESPONSES_API = f"{FILES_API}/responses"
COMPL_API     = "https://llm.api.cloud.yandex.net/v1/chat/completions"

# Разрешённые домены для CORS
def parse_allowed_origins(raw: str) -> List[str]:
    """Возвращает список доменов из переменной окружения ALLOWED_ORIGINS.

    Поддерживает разделители запятая, пробел, табы и переводы строк.
    Пустая строка, а также значение вида "*" трактуется как разрешение для всех
    доменов.
    """

    if not raw or raw.strip() == "*":
        return ["*"]

    parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
    return parts or ["*"]


ALLOWED_ORIGINS = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "*"))


def _is_set(value: str) -> bool:
    return bool(value and value.strip())


def can_use_vector_store() -> bool:
    return all((_is_set(YANDEX_API_KEY), _is_set(YANDEX_FOLDER_ID), _is_set(VECTOR_STORE_ID)))


def can_call_completions() -> bool:
    return all((_is_set(YANDEX_API_KEY), _is_set(YANDEX_FOLDER_ID)))

# ========================
#  Приложение FastAPI
# ========================
app = FastAPI(title="U4S Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ========================
#  Вспомогательные хелперы
# ========================
def yc_json_headers():
    """Заголовки для REST Assistant/Vector Store (Api-Key + x-folder-id)."""
    return {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
        "Content-Type": "application/json",
    }

def yc_openai_headers():
    """Заголовки для chat.completions (Bearer + OpenAI-Project)."""
    return {
        "Authorization": f"Bearer {YANDEX_API_KEY}",
        "OpenAI-Project": YANDEX_FOLDER_ID,
        "Content-Type": "application/json",
    }

def format_http_error(resp: requests.Response) -> str:
    """Возвращает укороченный текст ошибки от API для логов."""
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except Exception:
        body = resp.text or ""
    body = (body or "").strip().replace("\n", " ")
    return (body[:800] + "…") if len(body) > 800 else (body or "<пустой ответ>")

def ensure_ok(resp: requests.Response, label: str) -> None:
    """Бросает подробную ошибку, если HTTP-ответ не 2xx."""
    if resp.ok:
        return
    details = format_http_error(resp)
    raise RuntimeError(f"{label} HTTP {resp.status_code}: {details}")

def vs_list_files():
    """Список файлов в Vector Store (для диагностики и фолбэка)."""
    if not can_use_vector_store():
        return []
    url = f"{FILES_API}/vector_stores/{VECTOR_STORE_ID}/files"
    r = requests.get(url, headers=yc_json_headers(), timeout=30)
    ensure_ok(r, "Vector Store list")
    return r.json().get("data", [])

# ========================
#  Нормализация вопроса
# ========================
def normalize_question(q: str) -> str:
    """Подсказываем модели фокус, если вопрос общий."""
    ql = (q or "").lower()
    if "ресторан" in ql and not any(k in ql for k in ("час", "время", "работ", "телефон", "ссылка", "как забронировать")):
        return "Есть ли в усадьбе ресторан «Калина Красная»? Укажи часы, телефон и ссылку."
    return q or "Есть ли в усадьбе ресторан?"

# ========================
#  Основная логика: RAG
# ========================
SYSTEM_PROMPT_RAG = (
    "Ты — помощник сайта usadba4.ru. Работай ТОЛЬКО по базе знаний (Vector Store).\n"
    "Если вопрос о наличии/расписании/контактах услуги — отвечай по шаблону:\n"
    "• Да/Нет.\n"
    "• Часы: <в одной строке>.\n"
    "• Телефон: <если есть>.\n"
    "• Ссылка: <если есть>.\n"
    "Не выводи детали, не относящиеся напрямую к вопросу. "
    "Если фактов нет — ответ: \"Нет данных в базе знаний\"."
)

def rag_via_responses(question: str) -> str:
    """RAG через Responses API + Vector Store."""
    if not can_use_vector_store():
        raise RuntimeError("Responses API недоступен: не заданы YANDEX_API_KEY/YANDEX_FOLDER_ID/VECTOR_STORE_ID")
    payload = {
        "model": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT_RAG}]},
            {"role": "user",   "content": [{"type": "input_text", "text": question}]},
        ],
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
        "temperature": 0.0,          # максимально детерминированно
        "max_output_tokens": 600,
    }
    r = requests.post(RESPONSES_API, headers=yc_json_headers(), json=payload, timeout=60)
    ensure_ok(r, "Responses API")
    data = r.json()

    # Прямой текст (некоторые версии API возвращают field output_text)
    if data.get("output_text"):
        return data["output_text"].strip()

    # Разбор по блокам (универсальный путь)
    out = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                out.append(c["text"])
    return ("\n".join(out)).strip() if out else "Нет данных в базе знаний."

# ========================
#  Фолбэк: контекст из VS
# ========================
PRIMARY_KEYS = [
    "ресторан «калина красная»", 'ресторан "калина красная"', "ресторан калинка красная",
    "ресторан", "часы работы", "работает ежедневно", "телефон ресторана", "онлайн-заказ", "ссылка", "доставка"
]
SECONDARY_KEYS = ["кафе", "завтрак", "завтраки", "меню", "кухня"]
ALL_KEYS = list({*PRIMARY_KEYS, *SECONDARY_KEYS})

def build_context_from_vs(question: str) -> str:
    """Собираем релевантные строки из всех файлов VS, отдавая приоритет ключам про наличие/часы/телефон ресторана.
       Исключаем «завтрак»/«завтраки», если их нет в самом вопросе.
    """
    if not can_use_vector_store():
        return "Контекст пуст."
    try:
        ql = (question or "").lower()
        exclude_breakfast = ("завтрак" not in ql and "завтраки" not in ql)

        snips = []
        total_chars = 0
        files = vs_list_files()

        for f in files:
            fid = f["id"]

            meta_resp = requests.get(f"{FILES_API}/files/{fid}", headers=yc_json_headers(), timeout=30)
            ensure_ok(meta_resp, "Vector Store file meta")
            meta = meta_resp.json()

            text_resp = requests.get(f"{FILES_API}/files/{fid}/content", headers=yc_json_headers(), timeout=30)
            ensure_ok(text_resp, "Vector Store file content")
            text = text_resp.text
            lines = text.splitlines()

            # Сначала собираем все совпадения
            hits = []
            for ln in lines:
                l = ln.lower()
                if any(k in l for k in ALL_KEYS):
                    if exclude_breakfast and ("завтрак" in l or "завтраки" in l):
                        continue
                    hits.append(ln.strip())

            if not hits:
                continue

            # Приоритизация: primary -> 0, secondary -> 1, прочее -> 2
            def score(line: str) -> int:
                l = line.lower()
                if any(k in l for k in PRIMARY_KEYS):
                    return 0
                if any(k in l for k in SECONDARY_KEYS):
                    return 1
                return 2

            hits.sort(key=score)

            # Ограничим до 12 строк на документ, чтобы не раздувать контекст
            top = hits[:12]
            block = f"### {meta.get('filename','file')}\n" + "\n".join(top)

            if block:
                snips.append(block)
                total_chars += len(block)
                if total_chars > 2500:  # общий лимит контекста ~2.5k символов
                    break

        return "\n\n".join(snips) if snips else "Контекст пуст."
    except Exception as e:
        print("build_context_from_vs ERROR:", e)
        return "Контекст пуст."

def ask_with_vs_context(question: str) -> str:
    """Фолбэк: обычный chat.completions, но с явным контекстом из VS и строгим форматом ответа."""
    if not can_call_completions():
        return "Извините, база знаний сейчас недоступна."
    context = build_context_from_vs(question)
    sys = (
        "Отвечай ТОЛЬКО на основе раздела CONTEXT ниже. "
        "Если вопрос о наличии/расписании/контактах — отвечай по шаблону:\n"
        "• Да/Нет.\n"
        "• Часы: <в одной строке>.\n"
        "• Телефон: <если есть>.\n"
        "• Ссылка: <если есть>.\n"
        "Не выводи детали, не относящиеся напрямую к вопросу. "
        "Если ответа нет в CONTEXT — напиши: 'Нет данных в базе знаний'.\n"
        f"CONTEXT:\n{context}"
    )
    payload = {
        "model": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user",   "content": question},
        ],
        "temperature": 0.0,   # строго по контенту
        "max_tokens": 400,
    }
    r = requests.post(COMPL_API, headers=yc_openai_headers(), json=payload, timeout=60)
    ensure_ok(r, "Completions API")
    return r.json()["choices"][0]["message"]["content"]

# ========================
#  Роуты API
# ========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/info")
def debug_info():
    """Быстрая проверка: видит ли контейнер переменные и файлы VS."""
    info = {
        "env": {
            "HAS_API_KEY": bool(YANDEX_API_KEY),
            "YANDEX_FOLDER_ID": YANDEX_FOLDER_ID,
            "VECTOR_STORE_ID": VECTOR_STORE_ID,
            "ALLOWED_ORIGINS": ALLOWED_ORIGINS,
            "CAN_USE_VECTOR_STORE": can_use_vector_store(),
            "CAN_CALL_COMPLETIONS": can_call_completions(),
        },
        "vs_files_count": 0,
        "vs_sample": [],
        "error": None,
    }
    try:
        files = vs_list_files()
        info["vs_files_count"] = len(files)
        info["vs_sample"] = files[:3]
    except Exception as e:
        info["error"] = str(e)
    return info

@app.get("/api/chat")
def chat_get(q: str = ""):
    """GET-вариант: удобно тестировать из адресной строки/curl -G."""
    question = normalize_question(q)
    try:
        try:
            answer = rag_via_responses(question)  # 1) RAG
        except Exception as e:
            print("RAG error (GET):", e)
            answer = ask_with_vs_context(question)  # 2) VS-context fallback
        return {"answer": answer}
    except Exception as e:
        print("FATAL (GET):", e)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}

@app.post("/api/chat")
async def chat_post(request: Request):
    """POST-вариант: читает JSON вручную, чтобы избежать 400 на прокси."""
    try:
        # Попытка 1 — штатный парсер
        try:
            data = await request.json()
        except Exception:
            # Попытка 2 — сырое тело + json.loads
            raw = await request.body()
            data = json.loads(raw.decode("utf-8", errors="ignore") or "{}")

        q = (data.get("question") or "").strip() if isinstance(data, dict) else ""
        question = normalize_question(q)

        try:
            answer = rag_via_responses(question)  # 1) RAG
        except Exception as e:
            print("RAG error (POST):", e)
            answer = ask_with_vs_context(question)  # 2) VS-context fallback
        return {"answer": answer}

    except Exception as e:
        print("FATAL (POST):", e)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}
