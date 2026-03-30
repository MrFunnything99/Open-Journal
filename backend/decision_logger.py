"""Append-only decision log for observability; never raises into user-facing paths."""
from __future__ import annotations

import json
from typing import Any

import vec_store

_TEXT_CAP = 32768


def _cap_text(val: str | None, max_len: int = _TEXT_CAP) -> tuple[str | None, bool]:
    if val is None:
        return None, False
    if len(val) <= max_len:
        return val, False
    return val[:max_len], True


def _jsonify(val: Any) -> str | None:
    if val is None:
        return None
    try:
        return json.dumps(val, ensure_ascii=False, default=str)
    except Exception:
        return str(val)[:_TEXT_CAP]


class DecisionLogger:
    """Structured inserts into decision_log; failures are swallowed."""

    @staticmethod
    def _write(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        action_type: str,
        input_summary: str | None = None,
        retrieved_items: str | None = None,
        llm_prompt_summary: str | None = None,
        llm_response: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
        search_api_calls: str | None = None,
    ) -> int | None:
        truncated_any = False
        pairs = [
            ("input_summary", input_summary),
            ("retrieved_items", retrieved_items),
            ("llm_prompt_summary", llm_prompt_summary),
            ("llm_response", llm_response),
            ("final_output", final_output),
            ("reasoning_notes", reasoning_notes),
            ("search_api_calls", search_api_calls),
        ]
        cleaned: dict[str, str | None] = {}
        for key, raw in pairs:
            cap, trunc = _cap_text(raw)
            cleaned[key] = cap
            truncated_any = truncated_any or trunc
        if truncated_any:
            note = (cleaned["reasoning_notes"] or "") + " [truncated_field_32kb]"
            cleaned["reasoning_notes"], _ = _cap_text(note.strip())

        try:
            return vec_store.decision_log_insert(
                instance_id=instance_id or "",
                session_id=session_id,
                action_type=action_type,
                input_summary=cleaned["input_summary"],
                retrieved_items=cleaned["retrieved_items"],
                llm_prompt_summary=cleaned["llm_prompt_summary"],
                llm_response=cleaned["llm_response"],
                final_output=cleaned["final_output"],
                reasoning_notes=cleaned["reasoning_notes"],
                duration_ms=duration_ms,
                model_used=_cap_text(model_used, 2048)[0] if model_used else None,
                search_api_calls=cleaned["search_api_calls"],
            )
        except Exception as e:
            print("[backend] decision_log insert failed:", e)
            return None

    @staticmethod
    def log_context_retrieval(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        query: str,
        retrieved_items: list[dict] | None,
        final_output: str,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> int | None:
        inp = (query or "")[:8000]
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="context_retrieval",
            input_summary=inp,
            retrieved_items=_jsonify(retrieved_items),
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
        )

    @staticmethod
    def log_recommendation(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        retrieved_items: list[dict] | None = None,
        llm_prompt_summary: str | None = None,
        llm_response: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
        search_api_calls: list[dict] | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="recommendation",
            input_summary=input_summary,
            retrieved_items=_jsonify(retrieved_items),
            llm_prompt_summary=llm_prompt_summary,
            llm_response=llm_response,
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
            search_api_calls=_jsonify(search_api_calls),
        )

    @staticmethod
    def log_extraction(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        llm_response: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="extraction",
            input_summary=input_summary,
            llm_response=llm_response,
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
        )

    @staticmethod
    def log_feedback_processing(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        llm_prompt_summary: str | None = None,
        llm_response: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="feedback_processing",
            input_summary=input_summary,
            llm_prompt_summary=llm_prompt_summary,
            llm_response=llm_response,
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
        )

    @staticmethod
    def log_profile_update(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        llm_prompt_summary: str | None = None,
        llm_response: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="profile_update",
            input_summary=input_summary,
            llm_prompt_summary=llm_prompt_summary,
            llm_response=llm_response,
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
        )

    @staticmethod
    def log_writing_hint(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        retrieved_items: list[dict] | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="writing_hint",
            input_summary=input_summary,
            retrieved_items=_jsonify(retrieved_items),
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
        )

    @staticmethod
    def log_link_search(
        *,
        instance_id: str = "",
        session_id: str | None = None,
        input_summary: str | None = None,
        final_output: str | None = None,
        reasoning_notes: str | None = None,
        duration_ms: int | None = None,
        model_used: str | None = None,
        search_api_calls: list[dict] | None = None,
    ) -> int | None:
        return DecisionLogger._write(
            instance_id=instance_id,
            session_id=session_id,
            action_type="link_search",
            input_summary=input_summary,
            final_output=final_output,
            reasoning_notes=reasoning_notes,
            duration_ms=duration_ms,
            model_used=model_used,
            search_api_calls=_jsonify(search_api_calls),
        )


def log_chat_interviewer_retrieval(
    *,
    instance_id: str,
    session_id: str | None,
    query: str,
    context_chars: int,
    model_used: str,
    duration_ms: int | None = None,
) -> None:
    DecisionLogger._write(
        instance_id=instance_id,
        session_id=session_id,
        action_type="context_retrieval",
        input_summary=f"chat_query={(query or '')[:2000]} context_chars={context_chars}",
        reasoning_notes="interviewer_node retrieval for chat",
        duration_ms=duration_ms,
        model_used=model_used,
    )