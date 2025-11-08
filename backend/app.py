import os, json, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# === ENV ===
YANDEX_API_KEY   = os.environ["YANDEX_API_KEY"]
YANDEX_FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
VECTOR_STORE_ID  = os.environ.get("VECTOR_STORE_ID", "")

RESPONSES_URL = "https://rest-assistant.api.cloud.yandex.net/v1/responses"
COMPL_URL     = "https://llm.api.cloud.yandex.net/v1/chat/completions"
FILES_URL     = "https://rest-assistant.api.cloud.yandex.net/v1"

def _headers_json_api():
    return {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
        "Content-Type": "application/json",
    }

def _headers_openai():
    return {
        "Authorization": f"Bearer {YANDEX_API_KEY}",
        "OpenAI-Project": YANDEX_FOLDER_ID,
        "Content-Type": "application/json",
    }

SYSTEM_PROMPT = (
    "Ты — помощник сайта usadba4.ru.\n"
    "1) СНАЧАЛА выполни поиск по базе знаний (Vector Store) и используй только найденные фрагменты.\n"
    "2) Отвечай ТОЛЬКО фактами из фрагментов, без догадок и допущений.\n"
    "3) Часы/цены/телефоны передавай дословно и в одной строке.\n"
    "4) Если фрагментов нет — ответь: \"Нет данных в базе знаний\"."
)

def rag_via_responses(question: str) -> str:
    payload = {
        "model": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "input_text", "text": question}]},
        ],
        "tools": [{"type": "file_search"}],
        "tool_resources": {"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
        "temperature": 0.0,        # максимально детерминированно
        "max_output_tokens": 500,
    }
    r = requests.post(RESPONSES_URL, headers=_headers_json_api(),
                      data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data.get("output_text")
    if text:
        return text
    out = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                out.append(c["text"])
    return "\n".join(out).strip()

def build_context_from_vs() -> str:
    def get(url): return requests.get(url, headers=_headers_json_api(), timeout=30)
    files = get(f"{FILES_URL}/vector_stores/{VECTOR_STORE_ID}/files").json().get("data", [])
    keys = ["ресторан","калина красная","кафе","доставка","телефон","часы",
            "баня","чан","бронь","тариф","спа","сауна","хамам","бассейн","цена"]
    snips = []
    for f in files:
        fid = f["id"]
        meta = get(f"{FILES_URL}/files/{fid}").json()
        text = get(f"{FILES_URL}/files/{fid}/content").text
        lines = [ln for ln in text.splitlines() if any(k in ln.lower() for k in keys)]
        if lines:
            snips.append(f"### {meta.get('filename','file')}\n" + "\n".join(lines))
    return "\n\n".join(snips) or "Контекст пуст."

def ask_with_context(question: str) -> str:
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
        "temperature": 0.0,  # строго по контексту
        "max_tokens": 500,
    }
    r = requests.post(COMPL_URL, headers=_headers_openai(),
                      data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ==== FastAPI app ====
app = FastAPI()

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    question: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/chat")
def chat(body: ChatIn):
    q = (body.question or "").strip()
    if not q:
        q = "Есть ли в усадьбе ресторан?"
    try:
        try:
            ans = rag_via_responses(q)     # 1) прямой RAG
        except Exception as e:
            print("RAG error:", e)
            ans = ask_with_context(q)      # 2) fallback
        return {"answer": ans}
    except Exception as e:
        print("FATAL:", e)
        return {"answer": "Извините, сейчас не могу ответить. Попробуйте позже."}
