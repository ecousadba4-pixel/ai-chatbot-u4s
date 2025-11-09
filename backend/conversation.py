from __future__ import annotations

import time
from typing import Any, Sequence

ChatHistoryItem = dict[str, Any]
ChatModelMessage = dict[str, str]


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "on"}
    return False


def coerce_timestamp(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return fallback
    return fallback


def sanitize_history_messages(raw_history: Any) -> list[ChatHistoryItem]:
    if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
        return []

    sanitized: list[ChatHistoryItem] = []
    base_time = time.time()
    for index, item in enumerate(raw_history):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        timestamp = coerce_timestamp(item.get("timestamp"), base_time + index * 1e-3)
        sanitized.append({"role": role, "content": content, "timestamp": timestamp})
    return sanitized


def merge_histories(*histories: Sequence[ChatHistoryItem], limit: int | None = None) -> list[ChatHistoryItem]:
    combined: list[tuple[float, int, ChatHistoryItem]] = []
    order = 0
    for history in histories:
        for item in history or []:
            if not isinstance(item, dict):
                continue
            timestamp = float(item.get("timestamp", 0.0) or 0.0)
            content = str(item.get("content", "")).strip()
            role = str(item.get("role", "")).strip().lower()
            if role not in {"user", "assistant"} or not content:
                continue
            combined.append((timestamp, order, {"role": role, "content": content, "timestamp": timestamp}))
            order += 1

    combined.sort(key=lambda entry: (entry[0], entry[1]))
    merged = [item for _, _, item in combined]
    if limit is not None:
        merged = merged[-limit:]
    return merged


def build_conversation_messages(
    history: Sequence[ChatHistoryItem],
    *,
    question: str,
    system_prompt: str,
) -> list[ChatModelMessage]:
    messages: list[ChatModelMessage] = [
        {"role": "system", "content": system_prompt},
    ]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def extract_last_user_content(messages: Sequence[ChatModelMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def normalize_question(question: str) -> str:
    question = str(question or "").strip()
    if not question:
        return ""

    lowered = question.lower()
    if "ресторан" in lowered and "кал" in lowered:
        return "Есть ли в усадьбе ресторан «Калина Красная»? Укажи часы, телефон и ссылку."

    return question


def normalize_messages_for_model(
    messages: Sequence[ChatModelMessage],
) -> tuple[list[ChatModelMessage], str]:
    normalized: list[ChatModelMessage] = []
    last_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user_index = index
            break

    normalized_question = ""
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if index == last_user_index:
            content = normalize_question(content)
            normalized_question = content
        normalized.append({"role": role, "content": content})

    return normalized, normalized_question


def estimate_tokens(text: str) -> int:
    text = text or ""
    length = len(text)
    if length == 0:
        return 0
    return max(1, (length + 3) // 4)


def trim_messages_for_model(
    messages: Sequence[ChatModelMessage],
    *,
    max_tokens: int,
    min_assistant_messages: int = 2,
) -> list[ChatModelMessage]:
    if not messages or max_tokens <= 0:
        return list(messages)

    trimmed: list[ChatModelMessage] = []
    tokens_per_message: list[int] = []

    for message in messages:
        normalized_message = {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        trimmed.append(normalized_message)
        tokens_per_message.append(estimate_tokens(normalized_message["content"]))

    required_indices: set[int] = set()

    if trimmed and trimmed[0].get("role") == "system":
        required_indices.add(0)

    for index in range(len(trimmed) - 1, -1, -1):
        if trimmed[index].get("role") == "user":
            required_indices.add(index)
            break

    assistant_indices = [index for index, item in enumerate(trimmed) if item.get("role") == "assistant"]
    for index in assistant_indices[-min_assistant_messages:]:
        required_indices.add(index)

    total_tokens = sum(tokens_per_message)
    if total_tokens <= max_tokens:
        return trimmed

    removed_flags = [False] * len(trimmed)

    for index in range(len(trimmed)):
        if index in required_indices:
            continue
        if total_tokens <= max_tokens:
            break
        removed_flags[index] = True
        total_tokens -= tokens_per_message[index]

    result: list[ChatModelMessage] = []
    for index, message in enumerate(trimmed):
        if not removed_flags[index]:
            result.append(message)

    return result


def replace_system_prompt(
    messages: Sequence[ChatModelMessage], new_prompt: str
) -> list[ChatModelMessage]:
    replaced: list[ChatModelMessage] = []
    system_set = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            if not system_set:
                replaced.append({"role": "system", "content": new_prompt})
                system_set = True
            continue
        replaced.append({"role": str(role), "content": str(message.get("content", "")).strip()})

    if not system_set:
        replaced.insert(0, {"role": "system", "content": new_prompt})
    return replaced


def messages_to_responses_input(messages: Sequence[ChatModelMessage]) -> list[dict[str, Any]]:
    payload_messages: list[dict[str, Any]] = []
    for message in messages:
        text = str(message.get("content", ""))
        payload_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": [{"type": "input_text", "text": text}],
            }
        )
    return payload_messages


__all__ = [
    "ChatHistoryItem",
    "ChatModelMessage",
    "build_conversation_messages",
    "coerce_timestamp",
    "estimate_tokens",
    "extract_last_user_content",
    "merge_histories",
    "messages_to_responses_input",
    "normalize_messages_for_model",
    "normalize_question",
    "replace_system_prompt",
    "sanitize_history_messages",
    "to_bool",
    "trim_messages_for_model",
]
