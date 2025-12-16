from __future__ import annotations

from typing import Iterable

from app.core.config import get_settings


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
    max_chars = settings.rag_context_chars or settings.rag_max_context_chars
    lines: list[str] = []

    faq_lines = []
    for item in faq_hits or []:
        question = item.get("question") or ""
        answer = item.get("answer") or ""
        if not (question and answer):
            continue
        faq_lines.append(f"- Q: {question}\n  A: {answer}")

    _collect_section_lines(
        title="### FAQ точное совпадение",
        lines=faq_lines,
        builder=lines,
        max_chars=max_chars,
    )

    fact_lines = []
    for hit in facts_hits:
        text = hit.get("text") or ""
        title = hit.get("title") or ""
        prefix = f"{title}: " if title else ""
        fact_lines.append(_format_line(f"{prefix}{text}", hit))

    _collect_section_lines(
        title="### Контекст (факты)",
        lines=fact_lines,
        builder=lines,
        max_chars=max_chars,
    )

    file_lines = []
    for hit in files_hits:
        text = hit.get("text") or ""
        title = hit.get("title") or ""
        prefix = f"{title}: " if title else ""
        file_lines.append(_format_line(f"{prefix}{text}", hit))

    _collect_section_lines(
        title="### Контекст (описания)",
        lines=file_lines,
        builder=lines,
        max_chars=max_chars,
    )

    return "\n".join(lines)


__all__ = ["build_context"]
