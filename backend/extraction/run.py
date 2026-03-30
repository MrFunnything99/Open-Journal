from __future__ import annotations

import json
import re
from typing import Any

from extraction.llm_client import completion_for_extraction
from extraction.schema import ExtractionV1

SCHEMA_VERSION = "ExtractionV1"
_EXTRACTION_VALIDATE_FAILS = 0


_EXTRACTION_PROMPT_TEMPLATE = """You are a journal memory extractor. Given a journal session transcript, extract ONLY simple, factual, structured data.

Transcript:
---
{transcript}
---

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "events": ["short description of an event or activity", ...],
  "people": ["person 1", ...],
  "activities": ["activity 1", ...],
  "topics": ["concrete topic 1", ...],
  "emotions": ["emotion 1", ...],
  "facts": [
    {{"text": "short factual phrase", "scope": "entry", "confidence": 0.9}},
    {{"text": "stable preference or identity-level fact", "scope": "global", "confidence": 0.85}}
  ]
}}

Rules:
- events/people/activities/topics/emotions: situational → always entry-level lists (no scope field).
- facts: each object MUST have "text", "scope" ("entry" | "global"), and "confidence" (0.0–1.0).
  - Use "entry" for things tied to THIS session only (what happened today).
  - Use "global" ONLY for stable traits: long-term job, relationship, persistent likes/dislikes, stated goals.
- If uncertain, OMIT rather than guess. Keep text SHORT.
- Avoid clinical diagnoses or identity labels ("you are depressed"). Stick to user-stated facts.
"""


def _extract_json_object(text: str) -> str | None:
    if "{" not in text or "}" not in text:
        return None
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_journal_transcript(transcript: str) -> tuple[dict[str, Any], str]:
    """
    Run LLM extraction + Pydantic validation.
    Returns (legacy-shaped dict for save_session_data, raw_llm_output).
    """
    global _EXTRACTION_VALIDATE_FAILS
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(transcript=(transcript or "").strip()[:120_000])
    raw = (completion_for_extraction(prompt) or "").strip()
    if not raw:
        return {"summary": "", "facts": [], "metadata": {}, "structured_facts": []}, ""

    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:].lstrip()
    text = text.strip()
    json_str = _extract_json_object(text) or text

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Repair pass: minimal cleanup
        json_str2 = re.sub(r",\s*}", "}", json_str)
        json_str2 = re.sub(r",\s*]", "]", json_str2)
        try:
            data = json.loads(json_str2)
        except json.JSONDecodeError:
            return {"summary": "", "facts": [], "metadata": {}, "structured_facts": []}, raw

    if not isinstance(data, dict):
        return {"summary": "", "facts": [], "metadata": {}, "structured_facts": []}, raw

    try:
        model = ExtractionV1.model_validate(
            {
                "events": data.get("events"),
                "people": data.get("people"),
                "activities": data.get("activities"),
                "topics": data.get("topics"),
                "emotions": data.get("emotions"),
                "facts": data.get("facts"),
            }
        )
    except Exception as e:
        _EXTRACTION_VALIDATE_FAILS += 1
        print("[backend] ExtractionV1 validation failed:", e, "| total_fails=", _EXTRACTION_VALIDATE_FAILS)
        return {"summary": "", "facts": [], "metadata": {}, "structured_facts": []}, raw

    metadata = {
        "events": model.events,
        "people": model.people,
        "activities": model.activities,
        "topics": model.topics,
        "emotions": model.emotions,
    }
    segments: list[str] = []
    if model.events:
        segments.append("Events: " + "; ".join(model.events))
    if model.people:
        segments.append("People: " + ", ".join(model.people))
    if model.activities:
        segments.append("Activities: " + ", ".join(model.activities))
    if model.topics:
        segments.append("Topics: " + ", ".join(model.topics))
    if model.emotions:
        segments.append("Emotions: " + ", ".join(model.emotions))
    summary_str = " | ".join(segments)

    flat_facts = [f.text for f in model.facts]
    structured = [f.model_dump() for f in model.facts]

    return {
        "summary": summary_str,
        "facts": flat_facts,
        "metadata": metadata,
        "structured_facts": structured,
    }, raw
