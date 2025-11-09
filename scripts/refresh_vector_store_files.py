# scripts/refresh_vector_store_files.py
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI
import requests

BASE_URL = "https://rest-assistant.api.cloud.yandex.net/v1"


def update_chunking_strategy(*, api_key: str, folder_id: str, search_index_id: str,
                             max_chunk_tokens: int, overlap_tokens: int) -> None:
    """Вызывает SearchIndex.Update для настройки чанков."""

    url = f"{BASE_URL}/searchIndices:update"
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }
    payload = {
        "searchIndexId": search_index_id,
        "chunkingStrategy": {
            "staticStrategy": {
                "maxChunkSizeTokens": max_chunk_tokens,
                "chunkOverlapTokens": overlap_tokens,
            }
        },
    }

    print("\n🧱 Обновляю параметры разбивки на чанки…")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 300:
        body = resp.text
        try:
            data = resp.json()
            body = str(data)
        except Exception:
            pass
        raise RuntimeError(
            f"SearchIndex.Update HTTP {resp.status_code}: {body[:500]}"
        )

    data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
    status = data.get("status") if isinstance(data, dict) else None
    print(
        "   ✅ Параметры чанков обновлены"
        + (f" (status={status})" if status else "")
    )

def mask(s: str, keep=4):
    if not s:
        return ""
    return (s[:keep] + "…" + s[-keep:]) if len(s) > keep * 2 else "…"

def wait_ready(client: OpenAI, vs_id: str, timeout_sec: int = 900, poll_sec: int = 2):
    print("⏳ Ожидаю готовности индекса…")
    t0 = time.time()
    while True:
        cur = client.vector_stores.retrieve(vs_id)
        status = (getattr(cur, "status", "") or "").lower()
        if status in ("completed", "ready", "succeeded"):
            print(f"  ✅ Готово: {vs_id} (status={status})")
            return
        if status in ("failed", "error"):
            raise RuntimeError(f"Индекс не собрался: {cur}")
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Не дождался готовности за {timeout_sec} c")
        time.sleep(poll_sec)

def upload_file_tuple(client: OpenAI, src: Path):
    # Загружаем кортежем (filename, fileobj, mime)
    try:
        with open(src, "rb") as fh:
            uploaded = client.files.create(file=(src.name, fh, "application/json"),
                                           purpose="assistants")
        return uploaded.id, "application/json"
    except Exception as e1:
        print(f"   ⚠️ Не удалось как application/json: {e1}")
        with open(src, "rb") as fh:
            uploaded = client.files.create(file=(src.name, fh, "text/plain"),
                                           purpose="assistants")
        return uploaded.id, "text/plain"

def main():
    ap = argparse.ArgumentParser(description="Soft-refresh files in Yandex AI Studio Vector Store")
    ap.add_argument("--vs-id", required=True, help="vector_store_id (сохранится тот же)")
    ap.add_argument("--kb", required=True, help="Путь к kb.jsonl")
    ap.add_argument("--folder-id", required=True, help="YANDEX_FOLDER_ID")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--chunk-size", type=int, default=512, help="Размер чанка в токенах")
    ap.add_argument("--chunk-overlap", type=int, default=128, help="Перекрытие чанков в токенах")
    args = ap.parse_args()

    api_key = os.environ.get("YANDEX_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("YANDEX_API_KEY is empty")

    kb_file = Path(args.kb).expanduser().resolve()
    if not kb_file.exists() or kb_file.stat().st_size == 0:
        raise SystemExit(f"Файл не найден или пуст: {kb_file}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL, project=args.folder_id)

    print(f"➡️  Vector Store: {args.vs_id}")
    print(f"➡️  KB file     : {kb_file}")
    print(f"🔐 FOLDER      : {args.folder_id}")
    print(f"🔑 KEY         : {mask(api_key)}")

    # Проверка стора
    vs = client.vector_stores.retrieve(args.vs_id)
    print(f"   ✅ Найден стор: name={getattr(vs, 'name','')}, status={getattr(vs,'status','unknown')}")

    update_chunking_strategy(
        api_key=api_key,
        folder_id=args.folder_id,
        search_index_id=args.vs_id,
        max_chunk_tokens=args.chunk_size,
        overlap_tokens=args.chunk_overlap,
    )

    # Удаляем старые файлы
    print("\n🧹 Удаляю старые файлы из стора…")
    cursor = None
    total_deleted = 0
    while True:
        lst = client.vector_stores.files.list(vector_store_id=args.vs_id, limit=100, after=cursor) if cursor \
              else client.vector_stores.files.list(vector_store_id=args.vs_id, limit=100)
        for f in lst.data:
            try:
                client.vector_stores.files.delete(vector_store_id=args.vs_id, file_id=f.id)
                total_deleted += 1
            except Exception as de:
                print(f"   ⚠️ Не удалил file_id={f.id}: {de}")
        cursor = getattr(lst, "last_id", None)
        if not cursor or len(lst.data) < 100:
            break
    print(f"   ✅ Удалено файлов: {total_deleted}")

    # Загружаем новый kb.jsonl
    print("\n📂 Загружаю новый kb.jsonl в AI Studio Files…")
    file_id, used_mime = upload_file_tuple(client, kb_file)
    print(f"   ✅ Загружен: file_id={file_id}, mime={used_mime}")

    # Привязываем к сто́ру
    print("\n➕ Привязываю файл к Vector Store…")
    client.vector_stores.files.create(vector_store_id=args.vs_id, file_id=file_id)
    print("   ✅ Файл привязан, началась индексация")

    # Ждем готовности
    wait_ready(client, args.vs_id, timeout_sec=args.timeout)

    print("\n🎉 Готово! Vector Store обновлён и сохранил тот же ID.")
    print(f"vector_store_id = {args.vs_id}")

if __name__ == "__main__":
    main()
