import os
import time
import json
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request as urllib_request
from urllib import error as urllib_error

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - fallback для сред без requests
    requests = None  # type: ignore[assignment]

# Используем Files & Vector Store API, а не Assistant API
BASE_URL = "https://assistant.api.cloud.yandex.net/foundation-models/v1"
FILES_BASE_URL = "https://assistant.api.cloud.yandex.net/foundation-models/v1"


def _parse_json_response(
    status_code: int, text: str, content_type: str
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if content_type.startswith("application/json"):
        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict):
                data = maybe_json
        except ValueError:
            data = {}
    return data


def _http_post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    timeout: int,
) -> Tuple[int, str, str, Dict[str, Any]]:
    """Делает POST с JSON, используя requests или стандартную библиотеку."""

    if requests is not None:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        content_type = resp.headers.get("Content-Type", "")
        text = resp.text
        data = _parse_json_response(resp.status_code, text, content_type)
        return resp.status_code, text, content_type, data

    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            content_type = resp.headers.get("Content-Type", "")
            text = resp.read().decode("utf-8", "replace")
    except urllib_error.HTTPError as err:
        status_code = err.code
        content_type = err.headers.get("Content-Type", "") if err.headers else ""
        text = err.read().decode("utf-8", "replace")

    data = _parse_json_response(status_code, text, content_type)

    return status_code, text, content_type, data


def _http_get_json(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: int,
) -> Tuple[int, str, str, Dict[str, Any]]:
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout)
        content_type = resp.headers.get("Content-Type", "")
        text = resp.text
        data = _parse_json_response(resp.status_code, text, content_type)
        return resp.status_code, text, content_type, data

    req = urllib_request.Request(url, headers=headers, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            content_type = resp.headers.get("Content-Type", "")
            text = resp.read().decode("utf-8", "replace")
    except urllib_error.HTTPError as err:
        status_code = err.code
        content_type = err.headers.get("Content-Type", "") if err.headers else ""
        text = err.read().decode("utf-8", "replace")

    data = _parse_json_response(status_code, text, content_type)
    return status_code, text, content_type, data


def mask(s: str, keep=4):
    if not s:
        return ""
    return (s[:keep] + "…" + s[-keep:]) if len(s) > keep * 2 else "…"


def _vector_store_headers(api_key: str, folder_id: str) -> Dict[str, str]:
    return {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
    }


def get_vector_store(api_key: str, folder_id: str, vs_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/vectorStores/{vs_id}"
    status_code, body, _, data = _http_get_json(
        url,
        _vector_store_headers(api_key, folder_id),
        timeout=60,
    )
    if status_code >= 300:
        raise RuntimeError(
            f"Не удалось получить VectorStore {vs_id}: HTTP {status_code}: {body[:500]}"
        )
    if not isinstance(data, dict):
        raise RuntimeError("Ответ vectorStores/get не является JSON-объектом")
    return data


def wait_ready(
    api_key: str,
    folder_id: str,
    vs_id: str,
    *,
    timeout_sec: int = 900,
    poll_sec: int = 2,
) -> None:
    print("⏳ Ожидаю готовности векторного хранилища…")
    t0 = time.time()
    while True:
        cur = get_vector_store(api_key, folder_id, vs_id)
        status = (cur.get("status") or "").lower() if isinstance(cur, dict) else ""
        if status in {"completed", "ready", "succeeded"}:
            print(f"  ✅ Готово: {vs_id} (status={status})")
            return
        if status in {"failed", "error"}:
            raise RuntimeError(f"Хранилище не собралось: {json.dumps(cur, ensure_ascii=False)}")
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Не дождался готовности за {timeout_sec} c")
        time.sleep(poll_sec)


def _collect_file_ids(store_data: Dict[str, Any]) -> List[str]:
    candidates: Iterable[Any] = []
    for key in ("files", "sourceFiles", "attachedFiles"):
        items = store_data.get(key)
        if isinstance(items, list):
            candidates = items
            break
    ids: List[str] = []
    for item in candidates:
        if isinstance(item, dict):
            file_id: Optional[str] = (
                item.get("id")
                or item.get("fileId")
                or item.get("file_id")
                or item.get("fileID")
            )
            if isinstance(file_id, str) and file_id:
                ids.append(file_id)
    return ids


def remove_files(api_key: str, folder_id: str, vs_id: str, file_ids: List[str]) -> None:
    if not file_ids:
        return
    payload = {"fileIds": file_ids}
    url = f"{BASE_URL}/vectorStores/{vs_id}:removeFiles"
    status_code, body, _, _ = _http_post_json(
        url,
        _vector_store_headers(api_key, folder_id),
        payload,
        timeout=60,
    )
    if status_code >= 300:
        raise RuntimeError(
            f"Не удалось удалить файлы {file_ids}: HTTP {status_code}: {body[:500]}"
        )


def add_file_to_vector_store(api_key: str, folder_id: str, vs_id: str, file_id: str) -> None:
    payload = {"fileIds": [file_id]}
    url = f"{BASE_URL}/vectorStores/{vs_id}:addFiles"
    status_code, body, _, _ = _http_post_json(
        url,
        _vector_store_headers(api_key, folder_id),
        payload,
        timeout=60,
    )
    if status_code >= 300:
        raise RuntimeError(
            f"Не удалось привязать файл {file_id}: HTTP {status_code}: {body[:500]}"
        )


def upload_file(api_key: str, folder_id: str, src: Path) -> Tuple[str, str]:
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
    }
    content = src.read_bytes()
    mime = "application/json"
    payload = {
        "name": src.name,
        "mimeType": mime,
        "content": base64.b64encode(content).decode("ascii"),
    }
    url = f"{FILES_BASE_URL}/files"
    status_code, body, _, data = _http_post_json(
        url,
        headers,
        payload,
        timeout=300,
    )
    if status_code >= 300:
        raise RuntimeError(
            f"Не удалось загрузить файл {src}: HTTP {status_code}: {body[:500]}"
        )
    if not isinstance(data, dict):
        raise RuntimeError("Ответ при загрузке файла не является JSON-объектом")
    file_id = data.get("id") or data.get("fileId")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("В ответе на загрузку файла нет идентификатора")
    return file_id, mime


def main():
    ap = argparse.ArgumentParser(description="Refresh files in Yandex AI Studio Vector Store (vectorStores API)")
    ap.add_argument(
        "--vs-id",
        help="vector_store_id (можно задать через переменную окружения VECTOR_STORE_ID)",
    )
    ap.add_argument("--kb", required=True, help="Путь к kb.jsonl")
    ap.add_argument("--folder-id", required=True, help="YANDEX_FOLDER_ID")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    vs_id_env = os.environ.get("VECTOR_STORE_ID", "").strip()
    vs_id = (args.vs_id or vs_id_env).strip()
    if not vs_id:
        raise SystemExit(
            "vector_store_id не задан: используйте --vs-id или переменную окружения VECTOR_STORE_ID"
        )

    api_key = os.environ.get("YANDEX_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("YANDEX_API_KEY is empty")

    kb_file = Path(args.kb).expanduser().resolve()
    if not kb_file.exists() or kb_file.stat().st_size == 0:
        raise SystemExit(f"Файл не найден или пуст: {kb_file}")

    source = "--vs-id" if args.vs_id else "VECTOR_STORE_ID"
    print(f"➡️  Vector Store: {vs_id} (источник: {source})")
    print(f"➡️  KB file     : {kb_file}")
    print(f"🔐 FOLDER      : {args.folder_id}")
    print(f"🔑 KEY         : {mask(api_key)}")

    # Проверка хранилища
    vs = get_vector_store(api_key, args.folder_id, vs_id)
    name = ""
    if isinstance(vs, dict):
        name = str(vs.get("name", ""))
    status = str(vs.get("status", "unknown")) if isinstance(vs, dict) else "unknown"
    print(f"   ✅ Найдено хранилище: name={name}, status={status}")

    # Удаляем старые файлы
    print("\n🧹 Удаляю старые файлы из хранилища…")
    current_vs = get_vector_store(api_key, args.folder_id, vs_id)
    existing_file_ids = _collect_file_ids(current_vs if isinstance(current_vs, dict) else {})
    if existing_file_ids:
        try:
            remove_files(api_key, args.folder_id, vs_id, existing_file_ids)
            print(f"   ✅ Удалено файлов: {len(existing_file_ids)}")
        except Exception as err:
            print(f"   ⚠️ Ошибка при удалении: {err}")
    else:
        print("   ℹ️  Хранилище пусто, удалять нечего")

    # Загружаем новый kb.jsonl
    print("\n📂 Загружаю новый kb.jsonl в AI Studio Files…")
    file_id, used_mime = upload_file(api_key, args.folder_id, kb_file)
    print(f"   ✅ Загружен: file_id={file_id}, mime={used_mime}")

    # Привязываем к хранилищу
    print("\n➕ Привязываю файл к Vector Store…")
    add_file_to_vector_store(api_key, args.folder_id, vs_id, file_id)
    print("   ✅ Файл привязан, началась индексация")

    # Ждем готовности
    wait_ready(api_key, args.folder_id, vs_id, timeout_sec=args.timeout)

    print("\n🎉 Готово! Vector Store обновлён и сохранил тот же ID.")
    print(f"vector_store_id = {vs_id}")


if __name__ == "__main__":
    main()
