"""
Сервис для работы с RAG (Retrieval Augmented Generation).

Объединяет поиск по Qdrant и Postgres FAQ, формирование контекста и вызов LLM.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import asyncpg

from app.chat.formatting import detect_detail_mode, postprocess_answer
from app.core.config import Settings, get_settings
from app.llm.amvera_client import AmveraLLMClient
from app.llm.prompts import FACTS_PROMPT
from app.rag.context_builder import build_context
from app.rag.qdrant_client import QdrantClient
from app.rag.retriever import gather_rag_data

logger = logging.getLogger(__name__)


class RAGService:
    """
    Сервис RAG-поиска и генерации ответов.

    Координирует:
    - Поиск по векторной базе (Qdrant)
    - Поиск по FAQ (Postgres)
    - Формирование контекста
    - Генерацию ответа через LLM
    """

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        qdrant: QdrantClient,
        llm: AmveraLLMClient,
        settings: Settings | None = None,
    ) -> None:
        self._pool = pool
        self._qdrant = qdrant
        self._llm = llm
        self._settings = settings or get_settings()

    async def search_and_answer(
        self,
        text: str,
        *,
        intent: str = "knowledge_lookup",
    ) -> dict[str, Any]:
        """
        Поиск по базе знаний и генерация ответа.
        """
        rag_hits = await gather_rag_data(
            query=text,
            client=self._qdrant,
            pool=self._pool,
            facts_limit=self._settings.rag_facts_limit,
            files_limit=self._settings.rag_files_limit,
            faq_limit=3,
            faq_min_similarity=0.35,
            intent= intent,
        )

        qdrant_hits = rag_hits.get("qdrant_hits") or rag_hits.get("facts_hits", [])
        faq_hits = rag_hits.get("faq_hits", [])
        facts_hits = rag_hits.get("facts_hits") or qdrant_hits
        files_hits = rag_hits.get("files_hits", [])
        total_hits = rag_hits.get("hits_total", len(qdrant_hits) + len(faq_hits))

        debug = self._build_debug(rag_hits, intent, qdrant_hits, faq_hits)

        # Прямая отдача по FAQ, если уверенное совпадение
        faq_answer = self._extract_faq_answer(faq_hits)
        if faq_answer:
            debug["faq_direct"] = True
            debug["llm_called"] = False
            return {"answer": self._finalize_short_answer(faq_answer), "debug": debug}

        # Guard по минимальному числу фактов
        if total_hits < max(1, self._settings.rag_min_facts):
            debug["guard_triggered"] = True
            fallback_answer = (
                "Я не нашёл подтверждённых сведений в базе знаний по этому вопросу. "
                "Попробуйте уточнить запрос или загрузить описание с нужной информацией."
            )
            return {
                "answer": self._finalize_short_answer(fallback_answer),
                "debug": {**debug, "llm_called": False},
            }

        # Строим контекст
        max_snippets = max(1, self._settings.rag_max_snippets)
        context_text = build_context(
            facts_hits=facts_hits[:max_snippets],
            files_hits=files_hits[:max_snippets],
            faq_hits=faq_hits,
        )

        system_prompt = self._build_system_prompt(context_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        debug["llm_called"] = True

        try:
            llm_started = time.perf_counter()
            answer = await self._llm.chat(
                model=self._settings.amvera_model,
                messages=messages,
            )
            debug["llm_latency_ms"] = int((time.perf_counter() - llm_started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["llm_error"] = str(exc)
            generic_answer = (
                "Не получилось сформировать ответ, но я продолжу искать нужные данные. "
                "Попробуйте чуть позже или уточните вопрос."
            )
            return {
                "answer": self._finalize_short_answer(generic_answer),
                "debug": debug,
            }

        final_answer = self._finalize_short_answer(
            answer or "Информация из базы пока не найдена."
        )

        return {"answer": final_answer, "debug": debug}

    async def general_answer(
        self,
        text: str,
        *,
        intent: str = "general",
    ) -> dict[str, Any]:
        """
        Обработка общих вопросов через RAG + LLM.
        """
        detail_mode = detect_detail_mode(text)

        rag_hits = await gather_rag_data(
            query=text,
            client=self._qdrant,
            pool=self._pool,
            facts_limit=self._settings.rag_facts_limit,
            files_limit=self._settings.rag_files_limit,
            faq_limit=3,
            faq_min_similarity=0.35,
            intent=intent,
        )

        qdrant_hits = rag_hits.get("qdrant_hits")
        if qdrant_hits is None:
            qdrant_hits = [
                *rag_hits.get("facts_hits", []),
                *rag_hits.get("files_hits", []),
            ]
        faq_hits = rag_hits.get("faq_hits", [])
        hits_total = rag_hits.get("hits_total", len(qdrant_hits) + len(faq_hits))

        debug = self._build_debug(rag_hits, intent, qdrant_hits, faq_hits)

        # Если есть уверенное совпадение FAQ — отвечаем сразу
        faq_answer = self._extract_faq_answer(faq_hits)
        if faq_answer:
            debug["faq_direct"] = True
            debug["llm_called"] = False
            answer = postprocess_answer(faq_answer, mode="brief")
            return {"answer": answer, "debug": debug}

        max_snippets = max(1, self._settings.rag_max_snippets)
        facts_hits = qdrant_hits[:max_snippets]

        context_text = build_context(
            facts_hits=facts_hits,
            files_hits=[],
            faq_hits=faq_hits,
        )

        system_prompt = FACTS_PROMPT
        if context_text:
            system_prompt = f"{FACTS_PROMPT}\n\n{context_text}"

        debug["context_length"] = len(context_text)

        # Guard: недостаточно данных
        if hits_total < self._settings.rag_min_facts:
            debug["guard_triggered"] = True
            if intent == "lodging":
                answer = (
                    "Я не нашёл подтверждённой информации о домиках или номерах в базе знаний. "
                    "Если загрузите файл или страницу с типами размещения, ценами и вместимостью, я смогу отвечать точнее."
                )
            else:
                answer = (
                    "Я не нашёл подтверждённой информации в базе знаний, поэтому не буду выдумывать. "
                    "Уточните, пожалуйста: даты заезда и выезда, количество гостей, тип размещения или бюджет? "
                    "Если вам нужна баня/сауна или дополнительные услуги — тоже сообщите. "
                    "Если вы загрузили описание номеров/домиков в базу — скажите 'покажи варианты из базы'."
                )

            final_answer = postprocess_answer(
                answer, mode="detail" if detail_mode else "brief"
            )
            return {"answer": final_answer, "debug": debug}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        debug["llm_called"] = True

        try:
            llm_started = time.perf_counter()
            answer = await self._llm.chat(
                model=self._settings.amvera_model,
                messages=messages,
            )
            debug["llm_latency_ms"] = int((time.perf_counter() - llm_started) * 1000)
        except Exception as exc:  # noqa: BLE001
            debug["llm_error"] = str(exc)
            rag_answer = self._build_rag_only_answer(qdrant_hits, faq_hits, rag_hits)
            if rag_answer:
                answer = postprocess_answer(
                    rag_answer, mode="detail" if detail_mode else "brief"
                )
                return {"answer": answer, "debug": debug}
            return {
                "answer": "Сейчас не удалось получить ответ из LLM. Попробуйте уточнить запрос чуть позже.",
                "debug": debug,
            }

        final_answer = postprocess_answer(
            answer or "Нет данных в базе знаний.",
            mode="detail" if detail_mode else "brief",
        )

        return {"answer": final_answer, "debug": debug}

    def _extract_faq_answer(self, faq_hits: list[dict[str, Any]]) -> str | None:
        """
        Возвращает лучший ответ из FAQ, если он есть и с нормальной похожестью.
        """
        if not faq_hits:
            return None

        best = faq_hits[0]
        answer = (best.get("answer") or "").strip()
        if not answer:
            return None

        similarity = float(best.get("similarity") or 0.0)
        if similarity < 0.35:
            return None

        return answer

    def _build_debug(
        self,
        rag_hits: dict[str, Any],
        intent: str,
        qdrant_hits: list[dict[str, Any]],
        faq_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Формирует отладочную информацию."""
        return {
            "intent": intent,
            "hits_total": rag_hits.get("hits_total", len(qdrant_hits) + len(faq_hits)),
            "facts_hits": len(rag_hits.get("facts_hits", [])),
            "files_hits": len(rag_hits.get("files_hits", [])),
            "qdrant_hits": len(qdrant_hits),
            "faq_hits": len(faq_hits),
            "rag_latency_ms": rag_hits.get("rag_latency_ms", 0),
            "embed_latency_ms": rag_hits.get("embed_latency_ms", 0),
            "raw_qdrant_hits": rag_hits.get("raw_qdrant_hits", []),
            "score_threshold_used": rag_hits.get("score_threshold_used"),
            "expanded_queries": rag_hits.get("expanded_queries", []),
            "merged_hits_count": rag_hits.get("merged_hits_count", 0),
            "boosting_applied": rag_hits.get("boosting_applied", False),
            "intent_detected": rag_hits.get("intent_detected", intent),
            "embed_error": rag_hits.get("embed_error"),
            "guard_triggered": False,
            "llm_called": False,
            "rag_min_facts": self._settings.rag_min_facts,
        }

    def _build_system_prompt(self, context_text: str) -> str:
        """Формирует системный промпт для LLM."""
        parts = [
            FACTS_PROMPT,
            (
                "Отвечай одним цельным текстом на 2–4 предложения. "
                "Используй переданный контекст только для понимания ответа и не перечисляй файлы, блоки или пары вопрос-ответ. "
                "В конце можешь добавить фразу «Если хотите — расскажу подробнее»."
            ),
        ]
        if context_text:
            parts.append(context_text)

        return "\n\n".join(part for part in parts if part)

    def _finalize_short_answer(self, answer: str) -> str:
        """Финализирует короткий ответ."""
        cleaned = (answer or "").strip()
        if not cleaned:
            return "Информации пока нет, но могу поискать ещё. Если хотите — расскажу подробнее."

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        normalized = [s.strip() for s in sentences if s.strip()]

        if len(normalized) > 4:
            cleaned = " ".join(normalized[:4])
        elif normalized:
            cleaned = " ".join(normalized)

        if not cleaned.endswith((".", "!", "?")):
            cleaned = f"{cleaned}."

        if "Если хотите — расскажу подробнее." not in cleaned:
            cleaned = f"{cleaned} Если хотите — расскажу подробнее."

        return cleaned

    def _build_rag_only_answer(
        self,
        qdrant_hits: list[dict[str, Any]],
        faq_hits: list[dict[str, Any]],
        rag_hits: dict[str, Any],
    ) -> str:
        """Формирует ответ только из RAG-данных (без LLM)."""
        merged_hits_count = rag_hits.get("merged_hits_count", len(qdrant_hits))
        hits_total = rag_hits.get("hits_total", len(qdrant_hits) + len(faq_hits))

        if not (qdrant_hits or faq_hits):
            return ""

        if merged_hits_count < max(1, self._settings.rag_min_facts) and hits_total < 1:
            return ""

        candidates: list[tuple[int, float, str, str]] = []

        for faq in faq_hits:
            answer = (faq.get("answer") or "").strip()
            question = (faq.get("question") or "").strip()
            if not answer:
                continue
            text = f"{question}: {answer}" if question else answer
            candidates.append((0, float(faq.get("similarity", 0.0) or 0.0), text, text))

        for hit in qdrant_hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            title = (hit.get("title") or "").strip()
            payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}
            type_value = (hit.get("type") or payload.get("type") or "").strip()
            source = (hit.get("source") or payload.get("source") or "").strip()

            priority = 2
            if type_value in {"faq", "faq_ext"}:
                priority = 0
            elif source.startswith("knowledge") or ".md" in source:
                priority = 1

            snippet = f"{title}: {text}" if title else text
            candidates.append((priority, float(hit.get("score", 0.0) or 0.0), snippet, text))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: (item[0], -item[1]))
        selected = candidates[:4]

        answer_lines = [f"• {item[2]}" for item in selected if item[2]]

        return "\n".join(answer_lines)


__all__ = ["RAGService"]

