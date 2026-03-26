"""Pluggable completion backend for extraction (Gemini / OpenRouter path vs local Ollama)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _ollama_complete(prompt: str) -> str:
    host = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    model = (os.getenv("OLLAMA_MODEL") or "llama3.2").strip()
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return (data.get("response") or "").strip()


def completion_for_extraction(prompt: str) -> str:
    backend = (os.getenv("EXTRACTION_LLM_BACKEND") or "").strip().lower()
    if backend in ("ollama", "local", "local_ollama"):
        try:
            return _ollama_complete(prompt)
        except urllib.error.URLError as e:
            print("[backend] Ollama extraction error:", e)
            return ""
    from library import _call_gemini

    return (_call_gemini(prompt) or "").strip()
