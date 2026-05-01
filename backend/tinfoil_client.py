"""Single Tinfoil inference client for Selfmeridian.

All model inference, embeddings, and speech-to-text should go through this module.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from typing import Any

TINFOIL_BASE_URL = "https://inference.tinfoil.sh/v1"
DEFAULT_TINFOIL_CHAT_MODEL = "kimi-k2-6"
DEFAULT_TINFOIL_CHAT_FALLBACK_MODEL = "deepseek-v4-pro"
DEFAULT_TINFOIL_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TINFOIL_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
TINFOIL_EMBEDDING_DIM = 768


def tinfoil_api_configured() -> bool:
    return bool((os.getenv("TINFOIL_API_KEY") or "").strip())


def tinfoil_chat_model() -> str:
    return (os.getenv("TINFOIL_CHAT_MODEL") or DEFAULT_TINFOIL_CHAT_MODEL).strip() or DEFAULT_TINFOIL_CHAT_MODEL


def tinfoil_chat_fallback_model() -> str:
    model = (os.getenv("TINFOIL_CHAT_FALLBACK_MODEL") or DEFAULT_TINFOIL_CHAT_FALLBACK_MODEL).strip()
    return model or DEFAULT_TINFOIL_CHAT_FALLBACK_MODEL


def tinfoil_embedding_model() -> str:
    return (os.getenv("TINFOIL_EMBEDDING_MODEL") or DEFAULT_TINFOIL_EMBEDDING_MODEL).strip() or DEFAULT_TINFOIL_EMBEDDING_MODEL


def tinfoil_transcription_model() -> str:
    model = (os.getenv("TINFOIL_TRANSCRIPTION_MODEL") or DEFAULT_TINFOIL_TRANSCRIPTION_MODEL).strip()
    return model or DEFAULT_TINFOIL_TRANSCRIPTION_MODEL


def _api_key() -> str:
    key = (os.getenv("TINFOIL_API_KEY") or "").strip()
    if not key:
        raise ValueError("TINFOIL_API_KEY is not configured")
    return key


def _post_json(path: str, payload: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{TINFOIL_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        detail = body[:500]
        try:
            data = json.loads(body) if body else {}
            err = data.get("error")
            detail = err.get("message") if isinstance(err, dict) else (err or data.get("message") or detail)
        except Exception:
            pass
        raise ValueError(f"Tinfoil API error ({e.code}): {detail}") from e


def _message_content(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts).strip()
    return ""


def normalize_chat_response(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice0.get("message") if isinstance(choice0, dict) else None
    return _message_content(msg if isinstance(msg, dict) else None)


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": (model or tinfoil_chat_model()).strip(),
        "messages": messages,
        "max_tokens": int(max_tokens) if max_tokens is not None else 8192,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    return _post_json("/chat/completions", payload, timeout_sec=timeout_sec)


def chat_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_sec: float = 120.0,
) -> str:
    data = chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )
    return normalize_chat_response(data)


def embeddings(texts: list[str], *, model: str | None = None, timeout_sec: float = 120.0) -> list[list[float]]:
    if not texts:
        return []
    payload: dict[str, Any] = {
        "model": (model or tinfoil_embedding_model()).strip(),
        "input": [t if (t or "").strip() else " " for t in texts],
        "encoding_format": "float",
    }
    data = _post_json("/embeddings", payload, timeout_sec=timeout_sec)
    rows = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
    out: list[list[float]] = []
    for row in rows:
        emb = row.get("embedding")
        if not isinstance(emb, list):
            raise ValueError("Tinfoil embeddings response missing embedding")
        out.append([float(x) for x in emb])
    if len(out) != len(texts):
        raise ValueError(f"Tinfoil embeddings: expected {len(texts)} vectors, got {len(out)}")
    return out


def _multipart_field(name: str, value: str, boundary: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _multipart_file(name: str, filename: str, content_type: str, data: bytes, boundary: str) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    return header + data + b"\r\n"


def transcribe_audio(
    audio_bytes: bytes,
    *,
    filename: str = "audio.wav",
    mime_type: str | None = None,
    model: str | None = None,
    timeout_sec: float = 180.0,
) -> str:
    fname = filename or "audio.wav"
    ext = os.path.splitext(fname)[1].lower().lstrip(".")
    if ext not in {"mp3", "wav"}:
        raise ValueError("Tinfoil Whisper transcription supports mp3 and wav audio. Convert recordings to wav before upload.")
    content_type = mime_type or mimetypes.guess_type(fname)[0] or ("audio/mpeg" if ext == "mp3" else "audio/wav")
    boundary = f"----selfmeridian-tinfoil-{uuid.uuid4().hex}"
    body = b"".join(
        [
            _multipart_field("model", (model or tinfoil_transcription_model()).strip(), boundary),
            _multipart_file("file", fname, content_type, audio_bytes, boundary),
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    req = urllib.request.Request(
        f"{TINFOIL_BASE_URL}/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"Tinfoil transcription error ({e.code}): {body_text[:500]}") from e
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return (raw or "").strip()
    text = data.get("text")
    return str(text).strip() if text is not None else ""


def encode_audio_b64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("ascii")
