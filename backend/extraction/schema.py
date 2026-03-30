from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExtractedFact(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    scope: Literal["entry", "global"] = "entry"
    confidence: float = Field(0.8, ge=0.0, le=1.0)


class ExtractionV1(BaseModel):
    """Structured memory from one journal transcript."""

    events: list[str] = Field(default_factory=list, max_length=80)
    people: list[str] = Field(default_factory=list, max_length=80)
    activities: list[str] = Field(default_factory=list, max_length=80)
    topics: list[str] = Field(default_factory=list, max_length=80)
    emotions: list[str] = Field(default_factory=list, max_length=80)
    facts: list[ExtractedFact] = Field(default_factory=list, max_length=120)

    @field_validator("events", "people", "activities", "topics", "emotions", mode="before")
    @classmethod
    def _string_lists(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for x in v:
            if isinstance(x, (str, int, float)):
                s = str(x).strip()
                if s:
                    out.append(s[:400])
        return out[:80]

    @field_validator("facts", mode="before")
    @classmethod
    def _facts(cls, v: object) -> list:
        if not isinstance(v, list):
            return []
        out: list[ExtractedFact | dict] = []
        for x in v:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    out.append({"text": t[:500], "scope": "entry", "confidence": 0.75})
            elif isinstance(x, dict):
                out.append(x)
        return out[:120]
