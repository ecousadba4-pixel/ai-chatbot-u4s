from __future__ import annotations

import re
from typing import Iterable

from app.core.config import get_settings


def _is_technical_title(title: str) -> bool:
    """Проверяет, является ли title техническим (файл, блок и т.д.)."""
    if not title:
        return False
    # Технические паттерны: "file.txt", "блок N", "chunk N"
    technical_patterns = [
        r"\.txt",
        r"\.md",
        r"\.json",
        r"блок\s*\d+",
        r"chunk\s*\d+",
        r"^file:",
        r"^postgres:",
    ]
    lower_title = title.lower()
    for pattern in technical_patterns:
        if re.search(pattern, lower_title, re.IGNORECASE):
            return True
    return False


def _extract_answer_from_qa(text: str) -> str:
    """Извлекает ответ из Q/A формата."""
    if "Q:" in text and "A:" in text:
        parts = text.split("A:", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text


def _format_source_suffix(hit: dict) -> str:
    source = hit.get("source") or ""
    type_value = hit.get("type") or ""
    entity_id = hit.get("entity_id") or ""

    parts = [f"source={source}" if source else None, f"type={type_value}" if type_value else None]
    if entity_id:
        parts.append(f"id={entity_id}")
    suffix = " ".join([part for part in parts if part])
    return f" [{suffix}]" if suffix else ""


def _format_line(text: str, hit: dict) -> str:
    suffix = _format_source_suffix(hit)
    return f"- {text}{suffix}".strip()


def _collect_section_lines(
    *,
    title: str,
    lines: Iterable[str],
    builder: list[str],
    max_chars: int,
) -> None:
    pending = list(lines)
    if not pending:
        return

    for chunk in (title, *pending):
        if not chunk:
            continue
        if sum(len(line) for line in builder) + len(builder) + len(chunk) > max_chars:
            return
        builder.append(chunk)


def build_context(
    *,
    facts_hits: list[dict],
    files_hits: list[dict],
    faq_hits: list[dict] | None = None,
) -> str:
    settings = get_settings()
    max_chars = settings.rag_max_context_chars
    lines: list[str] = []

    faq_lines = []
    for item in faq_hits or []:
        question = item.get("question") or ""
        answer = item.get("answer") or ""
        if not (question and answer):
            continue
        # Формируем простой формат для LLM
        faq_lines.append(f"- Вопрос: {question}\n  Ответ: {answer}")

    _collect_section_lines(
        title="### FAQ",
        lines=faq_lines,
        builder=lines,
        max_chars=max_chars,
    )

    fact_lines = []
    for hit in facts_hits:
        text = hit.get("text") or ""
        if not text:
            continue
        title = hit.get("title") or ""
        # Пропускаем технические title (файлы, блоки)
        if _is_technical_title(title):
            title = ""
        # Извлекаем чистый текст из Q/A формата
        clean_text = _extract_answer_from_qa(text)
        prefix = f"{title}: " if title else ""
        fact_lines.append(_format_line(f"{prefix}{clean_text}", hit))

    _collect_section_lines(
        title="### Контекст (факты)",
        lines=fact_lines,
        builder=lines,
        max_chars=max_chars,
    )

    file_lines = []
    for hit in files_hits:
        text = hit.get("text") or ""
        if not text:
            continue
        title = hit.get("title") or ""
        # Пропускаем технические title
        if _is_technical_title(title):
            title = ""
        clean_text = _extract_answer_from_qa(text)
        prefix = f"{title}: " if title else ""
        file_lines.append(_format_line(f"{prefix}{clean_text}", hit))

    _collect_section_lines(
        title="### Контекст (описания)",
        lines=file_lines,
        builder=lines,
        max_chars=max_chars,
    )

    return "\n".join(lines)


__all__ = ["build_context"]
