# backend/app.py
import os
import json
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
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

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
    body: str
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except (ValueError, json.JSONDecodeError):
        body = resp.text or ""
    body = body.strip().replace("\n", " ")
    if len(body) > 800:
        body = body[:800] + "…"
    return body or "<пустой ответ>"


def ensure_ok(resp: requests.Response, label: str) -> None:
    """Бросает подробную ошибку, если HTTP-ответ не 2xx."""
    if resp.ok:
        return
    details = format_http_error(resp)
    raise RuntimeError(f"{label} HTTP {resp.status_code}: {details}")


def vs_list_files():
    """Список файлов в Vector Store (для диагностики и фолбэка)."""
    url = f"{FILES_API}/vector_stores/{VECTOR_STORE_ID}/files"
    r = requests.get(url, headers=yc_json_headers(), timeout=30)
    ensure_ok(r, "Vector Store list")
    return r.json().get("data", [])

# ========================
#  Основная логика: RAG
# ========================
SYSTEM_PROMPT_RAG = (
    "Ты — помощник сайта usadba4.ru.\n"
    "1) СНАЧАЛА выполни поиск по базе знаний (Vector Store) и используй только найденные фрагменты.\n"
    '2) Отвечай ТОЛЬКО фактами из фрагментов, без догадок и допущений. Если фрагментов нет — ответь: "Нет данных в базе знаний".\n'
    "3) Часы/цены/телефоны передавай дословно и в одной строке."
)

def rag_via_responses(question: str) -> str:
    """RAG через Responses API + Vector Store."""
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
    if "output_text" in data and data["output_text"]:
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
KEYS_FOR_CONTEXT = [
    "ресторан", "калина красная", "кафе", "доставка", "телефон", "часы",
    "баня", "чан", "бронь", "тариф", "спа", "сауна", "хамам", "бассейн",
    "цена", "стоимость", "завтрак", "контакты", "меню", "онлайн-заказ", "работает ежедневно"
]

def build_context_from_vs() -> str:
    """Собираем релевантные строки из всех файлов VS по ключевым словам."""
    try:
        snips = []
        files = vs_list_files()
        for f in files:
            fid = f["id"]
            meta_resp = requests.get(
                f"{FILES_API}/files/{fid}", headers=yc_json_headers(), timeout=30
            )
            ensure_ok(meta_resp, "Vector Store file meta")
            meta = meta_resp.json()

            text_resp = requests.get(
                f"{FILES_API}/files/{fid}/content", headers=yc_json_headers(), timeout=30
            )
            ensure_ok(text_resp, "Vector Store file content")
            text = text_resp.text
            lines = [ln for ln in text.splitlines() if any(k in ln.lower() for k in KEYS_FOR_CONTEXT)]
            if lines:
                snips.append(f"### {meta.get('filename','file')}\n" + "\n".join(lines))
        return "\n\n".join(snips) or "Контекст пуст."
    except Exception as e:
        print("build_context_from_vs ERROR:", e)
        return "Контекст пуст."

def ask_with_vs_context(question: str) -> str:
    """Фолбэк: обычный chat.completions, но с явным контекстом из VS."""
    context = build_context_from_vs()
    sys = (
        "Отвечай ТОЛЬКО на основе раздела CONTEXT ниже. "
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
        "max_tokens": 600,
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
    question = (q or "").strip() or "Есть ли в усадьбе ресторан?"
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
        if not q:
            q = "Есть ли в усадьбе ресторан?"

        try:
            answer = rag_via_responses(q)  # 1) RAG
        except Exception as e:
            print("RAG error (POST):", e)
            answer = ask_with_vs_context(q)  # 2) VS-context fallback
        return {"answer": answer}

    except Exception as e:
        print("FATAL (POST):", e)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}


