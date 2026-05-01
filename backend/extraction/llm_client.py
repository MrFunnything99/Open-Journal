"""Completion backend for extraction, routed through Tinfoil."""
from __future__ import annotations


def completion_for_extraction(prompt: str) -> str:
    from library import _call_helper_llm

    return (_call_helper_llm(prompt) or "").strip()
