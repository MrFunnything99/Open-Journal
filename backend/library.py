"""
Library for Selfmeridian: gist_facts (semantic) and episodic_log (episodic) memory,
and consumed_content (library). Uses SQLite + sqlite-vec for vector storage.

Embeddings: Perplexity (`_embed_texts` → PERPLEXITY_API_KEY); without a key, placeholder vectors match EMBEDDING_DIM so library rows still persist (semantic search degraded).
Extraction / chat helpers: OpenRouter (OPENROUTER_API_KEY + OPENROUTER_GEMINI_MODEL, default Gemini 3 Pro preview)
when GEMINI_VIA_OPENROUTER is enabled, else `_get_gemini_client` + `generate_content` (GEMINI_API_KEY) — never used for embeddings.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import urllib.error
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Gemini client for extraction / helpers only — embeddings use Perplexity (`_embed_texts`).
_gemini_client: genai.Client | None = None
_PERPLEXITY_EMBED_FALLBACK_WARNED = False

PPLX_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/embeddings"
PPLX_CONTEXTUAL_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/contextualizedembeddings"
# Context model: use contextualized endpoint (one chunk per pseudo-document for unrelated texts).
PPLX_EMBED_BATCH_DOCS = 480
DEFAULT_PERPLEXITY_EMBEDDING_MODEL = "pplx-embed-context-v1-4b"
PPLX_SEARCH_URL = "https://api.perplexity.ai/search"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_GEMINI_MODEL = "google/gemini-3-pro-preview"


def _openrouter_gemini_disabled_explicitly() -> bool:
    v = (os.getenv("GEMINI_VIA_OPENROUTER") or "").strip().lower()
    return v in ("0", "false", "no", "off")


def openrouter_gemini_enabled() -> bool:
    """When True, `_call_gemini` uses OpenRouter instead of the Google Gemini SDK."""
    if _openrouter_gemini_disabled_explicitly():
        return False
    return bool((os.getenv("OPENROUTER_API_KEY") or "").strip())


def openrouter_gemini_model() -> str:
    return (os.getenv("OPENROUTER_GEMINI_MODEL") or DEFAULT_OPENROUTER_GEMINI_MODEL).strip()


def gemini_extraction_backend() -> str:
    """Startup label: how library extraction/helpers resolve the chat model."""
    if openrouter_gemini_enabled():
        return f"openrouter ({openrouter_gemini_model()})"
    if (os.getenv("GEMINI_API_KEY") or "").strip():
        return f"google ({(os.getenv('GEMINI_CHAT_MODEL') or 'gemini-3-flash-preview').strip()})"
    return "none (set OPENROUTER_API_KEY or GEMINI_API_KEY)"


def _openrouter_normalize_message_content(msg: dict | None) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
                elif block.get("type") == "text":
                    tx = block.get("text")
                    if isinstance(tx, str) and tx.strip():
                        parts.append(tx.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts).strip() if parts else ""
    return str(content).strip()


def _openrouter_chat_completion(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    timeout_sec: float | None = None,
) -> str:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return ""
    eff_timeout = float(timeout_sec) if timeout_sec is not None else float(
        os.getenv("OPENROUTER_GEMINI_TIMEOUT_SEC", "75")
    )
    m = (model or openrouter_gemini_model()).strip() or DEFAULT_OPENROUTER_GEMINI_MODEL
    temp_raw = os.getenv("OPENROUTER_GEMINI_TEMPERATURE")
    if temperature is not None:
        temp = float(temperature)
    elif temp_raw is not None and str(temp_raw).strip() != "":
        temp = float(temp_raw)
    else:
        temp = 0.7
    payload: dict = {
        "model": m,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://selfmeridian.local"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "SelfMeridian"),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
            raw_text = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            raw_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw_text = ""
        try:
            err_j = json.loads(raw_text) if raw_text else {}
            detail = err_j.get("error", {}).get("message") if isinstance(err_j.get("error"), dict) else err_j.get("error")
        except Exception:
            detail = raw_text[:500] if raw_text else None
        print("[backend] OpenRouter chat error:", status, detail or "")
        return ""
    if status < 200 or status >= 300:
        print("[backend] OpenRouter chat error: HTTP", status)
        return ""
    try:
        data = json.loads(raw_text) if raw_text else {}
    except json.JSONDecodeError:
        print("[backend] OpenRouter chat: invalid JSON response")
        return ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    msg0 = choices[0].get("message") if isinstance(choices[0], dict) else None
    return _openrouter_normalize_message_content(msg0 if isinstance(msg0, dict) else None)


def _ensure_storage() -> None:
    """Ensure SQLite + sqlite-vec DB is initialized."""
    import vec_store

    vec_store.ensure_db()


def _get_gemini_client() -> genai.Client:
    """Lazy Gemini SDK client for `generate_content` only (extraction, recommendations helpers). Not used for embeddings."""
    global _gemini_client
    if _gemini_client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is required for Gemini extraction and helper calls (generate_content). "
                "Embeddings use PERPLEXITY_API_KEY only. See https://ai.google.dev/gemini-api/docs/get-started"
            )
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _decode_perplexity_int8_b64(b64: str) -> list[float]:
    """Decode Perplexity base64_int8 embedding and L2-normalize for cosine search in sqlite-vec."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    n = float(np.linalg.norm(arr))
    if n > 0:
        arr = arr / n
    return arr.tolist()


def _perplexity_post_json(url: str, payload: dict, api_key: str, timeout_sec: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print("[backend] Perplexity embeddings HTTP error:", e.code, err_body[:500])
        raise


def _placeholder_embeddings(texts: list[str], dim: int) -> list[list[float]]:
    """L2-normalized pseudo-vectors (same dim as vec tables) when Perplexity is unavailable."""
    out: list[list[float]] = []
    u32_max = float(2**32 - 1)
    for i, t in enumerate(texts):
        seed = hashlib.blake2b(f"{i}\0{t}".encode(), digest_size=64).digest()
        buf = bytearray(seed)
        while len(buf) < dim * 4:
            seed = hashlib.blake2b(seed, digest_size=64).digest()
            buf.extend(seed)
        words = struct.unpack(f"{dim}I", bytes(buf[: dim * 4]))
        floats = [(w / u32_max) * 2.0 - 1.0 for w in words]
        n = math.sqrt(sum(x * x for x in floats))
        if n < 1e-12:
            floats = [1.0 / math.sqrt(dim)] * dim
        else:
            floats = [x / n for x in floats]
        out.append(floats)
    return out


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts via Perplexity (default: pplx-embed-context-v1-4b on contextualized API).
    Each unrelated string is sent as a single-chunk \"document\" so the context model applies per text.
    Vectors are L2-normalized float32 for sqlite-vec cosine distance.
    """
    if not texts:
        return []
    api_key = (os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "").strip()
    if not api_key:
        global _PERPLEXITY_EMBED_FALLBACK_WARNED
        if not _PERPLEXITY_EMBED_FALLBACK_WARNED:
            _PERPLEXITY_EMBED_FALLBACK_WARNED = True
            print(
                "[backend] PERPLEXITY_API_KEY unset — using placeholder embeddings "
                "(library and memory rows still save; semantic retrieval is degraded)."
            )
        dim = int(os.getenv("EMBEDDING_DIM", "2560"))
        return _placeholder_embeddings(texts, dim)
    model = os.getenv("PERPLEXITY_EMBEDDING_MODEL", DEFAULT_PERPLEXITY_EMBEDDING_MODEL).strip()
    use_contextual = "context" in model.lower()

    def _sanitize(t: str) -> str:
        s = (t or "").strip()
        return s if s else " "

    out: list[list[float]] = []
    for start in range(0, len(texts), PPLX_EMBED_BATCH_DOCS):
        batch = [_sanitize(t) for t in texts[start : start + PPLX_EMBED_BATCH_DOCS]]
        if use_contextual:
            payload = {
                "model": model,
                "input": [[t] for t in batch],
                "encoding_format": "base64_int8",
            }
            body = _perplexity_post_json(PPLX_CONTEXTUAL_EMBEDDINGS_URL, payload, api_key)
            docs = sorted(body.get("data") or [], key=lambda x: x.get("index", 0))
            if len(docs) != len(batch):
                raise ValueError(
                    f"Perplexity contextualized embeddings: expected {len(batch)} documents, got {len(docs)}"
                )
            for doc in docs:
                chunks = sorted(doc.get("data") or [], key=lambda x: x.get("index", 0))
                if not chunks or "embedding" not in chunks[0]:
                    raise ValueError("Perplexity contextualized response missing embedding chunk")
                out.append(_decode_perplexity_int8_b64(chunks[0]["embedding"]))
        else:
            payload = {
                "model": model,
                "input": batch,
                "encoding_format": "base64_int8",
            }
            body = _perplexity_post_json(PPLX_EMBEDDINGS_URL, payload, api_key)
            rows = sorted(body.get("data") or [], key=lambda x: x.get("index", 0))
            if len(rows) != len(batch):
                raise ValueError(
                    f"Perplexity embeddings: expected {len(batch)} vectors, got {len(rows)}"
                )
            for row in rows:
                if "embedding" not in row:
                    raise ValueError("Perplexity embeddings response missing embedding")
                out.append(_decode_perplexity_int8_b64(row["embedding"]))

    return out


def _gemini_response_to_text(result) -> str:
    """
    Extract full text from a GenerateContentResponse, including parts that have thought=True.
    The SDK's .text property skips thought parts, which can leave extraction (e.g. JSON) empty.
    """
    text = getattr(result, "text", "") or ""
    if text and text.strip():
        return text.strip()
    try:
        if not getattr(result, "candidates", None) or not result.candidates:
            return ""
        c0 = result.candidates[0]
        if not getattr(c0, "content", None) or not getattr(c0.content, "parts", None):
            return ""
        out = []
        for part in c0.content.parts:
            ptext = getattr(part, "text", None)
            if isinstance(ptext, str) and ptext.strip():
                out.append(ptext)
        return "\n".join(out).strip() if out else ""
    except Exception as e:
        print("[backend] _gemini_response_to_text fallback error:", e)
        return ""


def _call_gemini(prompt: str) -> str:
    """
    Call the primary chat model for extraction/helpers: OpenRouter (Gemini 3 Pro preview by default)
    when OPENROUTER_API_KEY is set and GEMINI_VIA_OPENROUTER is not disabled; otherwise Google Gemini SDK
    (GEMINI_CHAT_MODEL, default gemini-3-flash-preview).
    Returns the response text (empty string on failure).
    For Google SDK responses, uses full text including thought parts so JSON extraction is not lost.
    """
    if openrouter_gemini_enabled():
        try:
            return _openrouter_chat_completion(prompt)
        except Exception as e:
            print("[backend] _call_gemini (OpenRouter) error:", e)
            return ""
    try:
        client = _get_gemini_client()
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
        result = client.models.generate_content(model=model, contents=prompt)
        text = _gemini_response_to_text(result)
        return text.strip()
    except Exception as e:
        print("[backend] _call_gemini error:", e)
        return ""


def _call_gemini_with_google_search(prompt: str) -> str:
    """
    Same as _call_gemini but with Google Search grounding so the model can use real-time web results.
    When using OpenRouter, there is no Google Search tool — this calls the same completion as _call_gemini.
    Falls back to _call_gemini if grounding is unavailable (e.g. SDK or model support).
    """
    if openrouter_gemini_enabled():
        return _call_gemini(prompt)
    try:
        from google.genai import types
        client = _get_gemini_client()
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        result = client.models.generate_content(model=model, contents=prompt, config=config)
        text = _gemini_response_to_text(result)
        return text.strip()
    except Exception as e:
        print("[backend] _call_gemini_with_google_search fallback to plain _call_gemini:", e)
        return _call_gemini(prompt)


def run_library_interview(
    messages: list[dict],
    library_items: list[dict],
    new_user_message: str,
) -> tuple[str, list[dict]]:
    """
    Run one turn of the library interview agent. Asks about books (and optionally other items);
    when the user has given enough, the model may output SAVE_NOTE|item_id|note to save a short
    factual note. Returns (assistant_reply_cleaned, notes_saved) where notes_saved is
    [{"item_id": str, "note": str}, ...].
    """

    books_text = "\n".join(
        f"- id={item.get('id', '')} | {item.get('title', '?')} | {item.get('author', '') or '(no author)'}"
        + (f" | current_note: {item.get('note', '')[:100]}" if (item.get('note') or "").strip() else "")
        for item in library_items
    )
    if not books_text.strip():
        return "You don't have any books in your library yet. Add some with the + button, then we can talk about them.", []

    system = f"""You are a warm, curious interviewer who loves talking about books. Your goal is to go through the user's library and capture what they liked (or didn't) about each book—short, factual notes only—so their recommendation system can do better. Make the conversation fun and stimulating, not a checklist.

The user's library (each has an id; use it exactly when saving a note):
{books_text}

How to ask (vary your approach):
- Use what you know: Draw on the book's themes, the author's style, genre, or cultural context to ask specific questions. For example: "Dune's politics and ecology are both huge—did one side pull you in more?" or "Van der Kolk gets pretty science-heavy in places—did that land for you or feel like a slog?" or "That one has a famously divisive ending—where did you land?"
- Go beyond "how was it?": Ask about a particular character, scene, or idea; the ending; what surprised them; what they'd tell a friend; whether it lived up to the hype; how it compares to something else they've read or to the film/series if there is one.
- Mix it up: Sometimes be playful or wry, sometimes thoughtful. You can reference the author's other work, the genre, or a common criticism and ask for their take. One question per turn—make it count.
- Comparative questions are great: "How does that stack up to [another book they have]?" or "More of a page-turner or more of a slow burn?"

Rules:
- Ask about one book at a time (or compare two when it fits). One short, specific question or follow-up per message.
- When the user has given you enough about a specific book (what they liked, didn't like, key takeaway), save it by outputting exactly one line: SAVE_NOTE|item_id|short note
  - item_id must be the exact id from the list above.
  - The note must be SHORT (one sentence or a few phrases), FACTUAL (only what the user said), and NO hallucination. Examples: "Loved the worldbuilding; found the ending rushed." or "Helpful for understanding trauma; heavy read."
- You may output at most one SAVE_NOTE per message. After SAVE_NOTE, continue the conversation (e.g. ask about the next book or wrap up).
- Do not invent or assume anything in the note. If the user only said "it was good", the note could be "User said they liked it." Keep it brief.
- Tone: conversational, curious, occasionally fun or witty—never stiff or repetitive.
"""

    parts = [system, "\n\n--- Conversation ---\n"]
    for m in messages:
        role = "User" if (m.get("role") == "user") else "Assistant"
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}\n")
    parts.append(f"User: {new_user_message.strip()}\n")
    parts.append("Assistant:")

    prompt = "\n".join(parts)
    raw = _call_gemini(prompt)
    if not raw:
        return "I had a small hiccup. Want to try again?", []

    notes_saved: list[dict] = []
    out_lines: list[str] = []
    for line in raw.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("SAVE_NOTE|"):
            rest = line_stripped[len("SAVE_NOTE|") :]
            pipe_count = rest.count("|")
            if pipe_count >= 1:
                idx = rest.index("|")
                item_id = rest[:idx].strip()
                note = rest[idx + 1 :].strip()
                if item_id and note and len(note) <= 2000:
                    try:
                        update_consumed(item_id, note=note)
                        notes_saved.append({"item_id": item_id, "note": note})
                    except Exception as e:
                        print("[backend] run_library_interview update_consumed error:", e)
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines).strip()
    return cleaned if cleaned else "Anything else you want to add about your reading?", notes_saved


def wipe_memory() -> None:
    """Clear gist and episodic memory (consumed_content kept)."""
    import vec_store

    _ensure_storage()
    vec_store.wipe_memory()


def add_consumed(
    content_type: str,
    title: str,
    *,
    author: str | None = None,
    url: str | None = None,
    liked: bool = True,
    note: str | None = None,
    date_completed: str | None = None,
    instance_id: str = "",
    id_override: str | None = None,
) -> None:
    """
    Record that the user has read/listened to a recommendation (book, podcast, article, research).
    Stored in SQLite+sqlite-vec for the recommendation agent. Optional note and date_completed for Library UI.
    """
    import vec_store

    _ensure_storage()
    ts = datetime.utcnow().isoformat() + "Z"
    ts_safe = ts.replace(":", "-").replace(".", "-")
    doc = f"User consumed {content_type}: {title}"
    if author:
        doc += f" by {author}"
    doc += f". Liked: {'yes' if liked else 'no'}."
    if date_completed:
        doc += f" Completed: {date_completed}."
    if note:
        doc += f" Note: {note}"
    emb = _embed_texts([doc])[0]
    uid = (id_override or "").strip()
    if not uid:
        uid = f"consumed_{ts_safe}_{abs(hash(title + content_type)) % 10**8}"
    vec_store.add_consumed(
        uid,
        doc,
        emb,
        type_=content_type,
        title=title,
        author=(author or "")[:300],
        url=(url or "")[:500],
        liked=liked,
        timestamp=ts,
        note=(note or "")[:2000],
        date_completed=(date_completed or "")[:50],
        instance_id=instance_id or "",
    )


def apply_library_tool_items(items: object, instance_id: str = "") -> tuple[int, list[str]]:
    """
    Validate structured items from the chat agent (LLM tool calls) and insert into the library.
    Returns (count_added, short labels for confirmation).
    """
    if not isinstance(items, list):
        return 0, []
    added = 0
    labels: list[str] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        ctype = str(raw.get("type", "")).lower().strip()
        if ctype not in ("book", "podcast", "article", "research"):
            continue
        title = (raw.get("title") or "").strip()
        if not title:
            continue
        author = (raw.get("author") or "").strip() or None
        url = (raw.get("url") or "").strip() or None
        liked = raw.get("liked", True)
        if isinstance(liked, str):
            liked = liked.strip().lower() in ("true", "1", "yes")
        note = (raw.get("note") or "").strip() or None
        try:
            add_consumed(
                ctype,
                title,
                author=author,
                url=url,
                liked=bool(liked),
                note=note,
                instance_id=instance_id,
            )
            added += 1
            labels.append(
                f"{ctype}: {title[:80]}" + (f" ({author[:60]})" if author else "")
            )
        except Exception as e:
            print("[backend] apply_library_tool_items add_consumed error:", e)
            continue
    return added, labels


def process_library_note(text: str, type_filter: str | None = None, instance_id: str = "") -> int:
    """
    Agent that organizes freeform 'library' notes into structured consumed items.
    If type_filter is set ("book" | "podcast" | "article" | "research"), only that type is extracted.
    Returns the number of items added.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    type_hint = (type_filter or "").strip().lower()
    if type_hint not in ("book", "podcast", "article", "research"):
        type_hint = None
    type_instruction = (
        f' Only output items of type "{type_hint}". Ignore any other media types in the text.'
        if type_hint
        else ""
    )
    prompt = f"""You are a reading library organizer. The user will paste a list of titles or short notes (one per line or comma-separated).
Your job is to turn this into a structured list of items for a recommendation system.{type_instruction}

TEXT:
---
{cleaned}
---

Return ONLY valid JSON (no markdown, no comments) that is an array of objects:
[
  {{"type": "book" | "podcast" | "article" | "research", "title": "...", "author": "...", "url": "", "liked": true, "note": ""}},
  ...
]

Rules:
- Each line or item in the text should become one object.
- **Title and author formatting (critical for books):** Use canonical, properly formatted titles and authors so recommenders are not confused. For books: look up the real title and author if the user typed casually (e.g. "dune" → title "Dune", author "Frank Herbert"; "body keeps the score" → title "The Body Keeps the Score", author "Bessel van der Kolk"). Use title case for titles and full author name(s). For podcasts/articles/research, use consistent title case and full names where applicable.
- "url" leave empty unless you know a real link. "liked" default true. "note" leave empty for now.
"""
    try:
        text_out = _call_gemini(prompt)
        text_out = str(text_out).strip()
        if text_out.startswith("```"):
            text_out = text_out.split("```")[1]
            if text_out.startswith("json"):
                text_out = text_out[4:]
        data = json.loads(text_out)
    except Exception as e:
        print("[backend] process_library_note parse error:", e)
        return 0
    if not isinstance(data, list):
        return 0
    added = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        ctype = str(item.get("type", "")).lower()
        if ctype not in ("book", "podcast", "article", "research"):
            continue
        if type_hint and ctype != type_hint:
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        author = (item.get("author") or "").strip() or None
        url = (item.get("url") or "").strip() or None
        liked = bool(item.get("liked", True))
        note = (item.get("note") or "").strip() or None
        try:
            add_consumed(ctype, title, author=author, url=url, liked=liked, note=note, instance_id=instance_id)
            added += 1
        except Exception as e:
            print("[backend] process_library_note add_consumed error:", e)
            continue
    return added


def get_consumed_context(max_items: int = 80, instance_id: str = "") -> str:
    """
    Return a single string describing what the user has consumed, liked/disliked,
    and notes — for injection into recommendation prompts and Perplexity query builders.
    """
    import vec_store

    _ensure_storage()
    try:
        rows = vec_store.get_consumed_context_rows(max_items=max_items, instance_id=instance_id)
        if not rows:
            return "The user has not marked any books, podcasts, or articles as read/listened yet."
        lines = []
        for m in rows:
            t = m.get("type", "item")
            title = m.get("title", "?")
            author = m.get("author", "")
            liked = m.get("liked", True)
            note = (m.get("note") or "").strip()
            line = f"- {t}: {title}"
            if author:
                line += f" ({author})"
            if liked:
                line += " — enjoyed / liked"
            else:
                line += " — did not enjoy (avoid similar items, authors, outlets, or angles; respect any note below)"
            if note:
                line += f". User note: {note}"
            lines.append(line)
        return (
            "What the user has already consumed (books, podcasts, articles, research). "
            "Do NOT recommend the same titles/URLs again. "
            "Honor **liked** items as taste signals; honor **did not enjoy** rows and **User note** as signals to steer away (themes, tone, subject matter) while staying useful.\n"
            + "\n".join(lines)
        )
    except Exception:
        return "The user has not marked any recommendations as consumed yet."


def _parse_date_completed(s: str) -> tuple[int, int, int]:
    """Return (year, month, day) for sorting; (0,0,0) for empty so it sorts last."""
    s = (s or "").strip()
    if not s:
        return (0, 0, 0)
    parts = s.replace("-", " ").split()
    y = m = d = 0
    if len(parts) >= 1 and parts[0].isdigit():
        y = int(parts[0])
    if len(parts) >= 2 and parts[1].isdigit():
        m = int(parts[1])
    if len(parts) >= 3 and parts[2].isdigit():
        d = int(parts[2])
    return (y, m, d)


def list_consumed(max_items: int = 200, instance_id: str = "") -> dict[str, list[dict]]:
    """
    Return consumed items grouped by type for the Library UI, sorted by date_completed (newest first).
    Each item: { "id", "title", "author", "date_completed", "note" }.
    """
    import vec_store

    _ensure_storage()
    out: dict[str, list[dict]] = {"books": [], "podcasts": [], "articles": [], "research": []}
    try:
        rows = vec_store.list_consumed_rows(max_items=max_items, instance_id=instance_id)
        type_to_key = {"book": "books", "podcast": "podcasts", "article": "articles", "research": "research"}
        for r in rows:
            key = type_to_key.get(r["type"])
            if not key:
                continue
            if not (r.get("title") or "").strip():
                continue
            out[key].append({
                "id": r.get("id", ""),
                "title": r.get("title", "?"),
                "author": r.get("author", ""),
                "date_completed": r.get("date_completed", ""),
                "note": r.get("note", ""),
            })
        for key in out:
            out[key].sort(key=lambda x: _parse_date_completed(x.get("date_completed") or ""), reverse=True)
        return out
    except Exception as e:
        print("[backend] list_consumed error:", e)
        return out


def update_consumed(
    item_id: str,
    *,
    date_completed: str | None = None,
    note: str | None = None,
    title: str | None = None,
    author: str | None = None,
    url: str | None = None,
    instance_id: str = "",
) -> bool:
    """Update metadata for a consumed item. Returns True if updated."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.update_consumed(
            item_id,
            date_completed=date_completed,
            note=note,
            title=title,
            author=author,
            url=url,
            instance_id=instance_id,
        )
    except Exception as e:
        print("[backend] update_consumed error:", e)
        return False


def delete_consumed(item_id: str, instance_id: str = "") -> bool:
    """Remove a consumed item by id. Returns True if deleted."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.delete_consumed(item_id, instance_id=instance_id)
    except Exception as e:
        print("[backend] delete_consumed error:", e)
        return False


def _extract_session_data(transcript: str) -> dict:
    """Use Gemini LLM to extract structured, factual memory from transcript."""
    prompt = f"""You are a journal memory extractor. Given a journal session transcript, extract ONLY simple, factual, structured data.

Transcript:
---
{transcript}
---

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "events": ["short description of an event or activity", ...],
  "people": ["person 1", "person 2", ...],
  "activities": ["activity 1", "activity 2", ...],
  "topics": ["concrete topic 1", "concrete topic 2", ...],
  "emotions": ["emotion 1", "emotion 2", ...],
  "facts": ["short factual statement about the user", ...]
}}

Rules:
- Use SHORT phrases, not paragraphs. Do NOT write narrative summaries.
- events: concrete events or activities ("argument with dad", "job applications", "treadmill workout"), NOT abstract psychological labels.
- people: names or simple references ("Dad", "Colin"). Do NOT invent people.
- activities: simple activity labels ("reading", "journaling", "job search", "exercise").
- topics: concrete topics ("therapy", "family conflict", "career"), NOT abstract ideas like "mental health dysregulation" or "emotional dysregulation".
- emotions: plain emotion words tied to specific events ("anxious", "hopeful", "frustrated", "calm"). Do NOT convert emotions into numbers.
- facts: hard, verifiable facts about the user (job, relationships, stable preferences, ongoing projects). Empty list if none.
- If you are uncertain about something, OMIT it instead of guessing.
- Avoid speculative interpretation and abstract psychology language.
"""
    text = _call_gemini(prompt)
    text = (text or "").strip()
    if not text:
        return {"summary": "", "facts": [], "metadata": {}}
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # If model returned reasoning + JSON, extract the first complete JSON object
    json_str = text
    if "{" in text and "}" in text:
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = text[start : i + 1]
                        break
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[backend] _extract_session_data JSON error:", e)
        return {"summary": "", "facts": [], "metadata": {}}
    if not isinstance(data, dict):
        return {"summary": "", "facts": [], "metadata": {}}
    # Normalize top-level lists
    events = data.get("events") or []
    people = data.get("people") or []
    activities = data.get("activities") or []
    topics = data.get("topics") or []
    emotions = data.get("emotions") or []
    facts = data.get("facts") or []

    if not isinstance(events, list):
        events = []
    if not isinstance(people, list):
        people = []
    if not isinstance(activities, list):
        activities = []
    if not isinstance(topics, list):
        topics = []
    if not isinstance(emotions, list):
        emotions = []
    if not isinstance(facts, list):
        facts = []

    # Build structured metadata object used elsewhere
    metadata = {
        "events": [str(x) for x in events if isinstance(x, (str, int, float)) and str(x).strip()],
        "people": [str(x) for x in people if isinstance(x, (str, int, float)) and str(x).strip()],
        "activities": [str(x) for x in activities if isinstance(x, (str, int, float)) and str(x).strip()],
        "topics": [str(x) for x in topics if isinstance(x, (str, int, float)) and str(x).strip()],
        "emotions": [str(x) for x in emotions if isinstance(x, (str, int, float)) and str(x).strip()],
    }

    # Derive a compact, non-narrative summary string for embeddings / RAG
    segments: list[str] = []
    if metadata["events"]:
        segments.append("Events: " + "; ".join(metadata["events"]))
    if metadata["people"]:
        segments.append("People: " + ", ".join(metadata["people"]))
    if metadata["activities"]:
        segments.append("Activities: " + ", ".join(metadata["activities"]))
    if metadata["topics"]:
        segments.append("Topics: " + ", ".join(metadata["topics"]))
    if metadata["emotions"]:
        segments.append("Emotions: " + ", ".join(metadata["emotions"]))
    summary_str = " | ".join(segments)

    return {
        "summary": summary_str,
        "facts": [str(x) for x in facts if isinstance(x, (str, int, float)) and str(x).strip()],
        "metadata": metadata,
    }


def _get_person_passages(person_name: str, max_passages: int = 40) -> list[str]:
    """
    Hybrid retrieval for a person: combine keyword and vector search over gist + episodic docs.
    Returns a list of short passages for downstream agents.
    """
    import vec_store

    _ensure_storage()
    person = (person_name or "").strip()
    if not person:
        return []
    person_l = person.lower()
    passages: list[str] = []

    # Keyword over gist facts
    try:
        for item in vec_store.list_gist_with_ids():
            doc = (item.get("document") or "").strip()
            if doc and person_l in doc.lower():
                passages.append(doc)
    except Exception:
        pass

    # Keyword + metadata over episodic summaries
    try:
        for item in vec_store.list_episodic_with_ids():
            doc = (item.get("document") or "").strip()
            if doc and person_l in doc.lower():
                passages.append(doc)
            meta_json = item.get("metadata_json")
            if not meta_json:
                continue
            try:
                meta = json.loads(meta_json)
            except Exception:
                continue
            events = meta.get("events") or []
            if isinstance(events, list):
                for e in events:
                    s = str(e)
                    if s and person_l in s.lower():
                        passages.append(s)
    except Exception:
        pass

    # Vector search from gist + episodic
    try:
        emb = _embed_texts([person])[0]
        for doc in vec_store.query_gist(emb, k=8):
            if doc:
                passages.append(doc)
        for doc in vec_store.query_episodic(emb, k=8):
            if doc:
                passages.append(doc)
    except Exception:
        pass

    # De-duplicate and trim
    seen: set[str] = set()
    unique: list[str] = []
    for p in passages:
        s = p.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        unique.append(s)
        if len(unique) >= max_passages:
            break
    return unique


def run_relationship_summary_agent(person_id: int, person_name: str) -> str:
    """RelationshipSummaryAgent: infer implied emotional tone toward a person.

    Uses a simple cache so we don't re-call the LLM on every detail view.
    """
    import vec_store

    _ensure_storage()
    # If we already have a cached summary for this person, return it.
    cached = vec_store.get_person_ai_summary(person_id)
    if cached and (cached.get("summary") or "").strip():
        return (cached.get("summary") or "").strip()

    passages = _get_person_passages(person_name)
    if not passages:
        return ""
    joined = "\n\n".join(f"- {p}" for p in passages)
    prompt = f"""You are RelationshipSummaryAgent.

You analyze how the user writes about other people in their journals.

Analyze these journal passages mentioning {person_name}.

Focus on:
- emotional tone
- patterns of interaction
- admiration or conflict
- trust or anxiety
- attachment signals

Write a concise summary (2–4 sentences) describing the user's implied feelings about this person.
Use neutral, observational language. Do NOT invent events; only summarize what is implied.

Passages:
---
{joined}
---
"""
    text = _call_gemini(prompt)
    text = (text or "").strip()
    # Very light post-processing: cap at 4 sentences
    if not text:
        return ""
    parts = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not parts:
        return ""
    summary = ". ".join(parts[:4])
    if not summary.endswith("."):
        summary += "."
    try:
        vec_store.set_person_ai_summary(person_id, summary)
    except Exception as e:
        print("[backend] run_relationship_summary_agent cache error:", e)
    return summary


def run_person_facts_agent(person_id: int, person_name: str) -> list[dict]:
    """
    PersonFactsAgent: extract stable factual statements about a person from journal passages.
    Returns list of {id, fact_text, confidence, source_journal_id, created_at} after storing to DB.
    """
    import vec_store

    _ensure_storage()
    # If we already have stored facts for this person, return them without re-running the LLM.
    try:
        existing = vec_store.list_person_facts(person_id)
    except Exception:
        existing = []
    if existing:
        return existing

    passages = _get_person_passages(person_name)
    if not passages:
        return []
    joined = "\n\n".join(f"- {p}" for p in passages)
    prompt = f"""You are PersonFactsAgent.

The following passages are from a journal. The journal writer (the "user") often talks about themselves AND about other people. You must extract ONLY facts that describe **{person_name}** — the other person — NOT the journal writer.

CRITICAL RULES:
- Each fact must be ABOUT {person_name} (their age, job, school, hobbies, traits, projects, role in the user's life). The SUBJECT of the fact must be {person_name}.
- Do NOT include any fact that describes the journal writer / user (e.g. "I work at X", "I go to UNCA", "I like hiking"). Those are facts about the user, not about {person_name}.
- If the passage only describes what the user did or who the user is, leave it out. Only include facts that clearly describe {person_name} (e.g. "{person_name} works at Google", "{person_name} is in grad school", "My friend {person_name} is 25").
- When in doubt, omit. Include only facts that unambiguously describe {person_name}.

Focus on stable, factual information about {person_name}:
- age, occupation, school
- projects, hobbies, interests
- role or relationship (e.g. coworker, roommate, sibling) only if it describes {person_name}

Avoid emotional interpretation and speculation.

Return ONLY valid JSON with this structure (no markdown, no extra text):
{{
  "facts": [
    {{
      "fact_text": "19-year-old UNCA student",
      "confidence": 0.9,
      "source_id": "summary:12"
    }},
    ...
  ]
}}

Passages (from the user's journal; extract only facts about {person_name}, not about the user):
---
{joined}
---
"""
    raw = _call_gemini(prompt)
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        print("[backend] run_person_facts_agent JSON error:", e)
        return []
    facts = data.get("facts") or []
    if not isinstance(facts, list):
        return []
    # Normalize minimal fields for storage
    cleaned: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        text = (f.get("fact_text") or "").strip()
        if not text:
            continue
        conf = f.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_f = None
        src = (f.get("source_id") or "").strip()
        cleaned.append(
            {
                "fact_text": text,
                "confidence": conf_f,
                "source_id": src,
            }
        )
    try:
        vec_store.replace_person_facts(person_id, cleaned)
    except Exception as e:
        print("[backend] run_person_facts_agent store error:", e)
    try:
        return vec_store.list_person_facts(person_id)
    except Exception:
        return []


def run_people_grouping_agent() -> None:
    """
    Use an LLM to propose social groups (e.g., UNC Charlotte, CPCC, Mentors) and
    assign people to them, then store results in person_groups. This runs offline
    and is triggered explicitly from the API, so it can be relatively heavy.
    """
    import vec_store

    _ensure_storage()
    try:
        people = vec_store.list_people_with_groups()
    except Exception as e:
        print("[backend] run_people_grouping_agent people error:", e)
        return
    if not people:
        return

    # For richer descriptors, include any stored person facts.
    person_facts: dict[int, list[str]] = {}
    for p in people:
        pid = p["id"]
        try:
            facts = vec_store.list_person_facts(pid)
        except Exception:
            facts = []
        person_facts[pid] = [f.get("fact_text", "") for f in facts if f.get("fact_text")]

    lines = []
    for p in people:
        pid = p["id"]
        name = p["name"]
        facts_blob = "; ".join(person_facts.get(pid, [])) or "(no extra facts)"
        lines.append(f"- id: {pid}, name: {name}, details: {facts_blob}")
    people_blob = "\n".join(lines)

    prompt = f"""You are SocialGroupingAgent.

You are given a list of people mentioned in a user's journal, with some factual descriptors.
Your job is to organize them into meaningful social groups that will be used to draw a graph.

Rules:
- Create 3–15 groups that feel natural and interpretable to the user.
- Typical group examples: universities or schools (e.g. "UNC Charlotte", "CPCC"), workplaces,
  friend clusters, family, mentors/therapists, healthcare providers, clubs, etc.
- Prefer SHORT, human-readable group names (1–3 words). Reuse existing names like university
  or workplace names when obvious from the data.
- People can belong to multiple groups (e.g. "Mentors" and "UNC Charlotte").
- If you are unsure about a person, put them in a generic group like "Other" or "Misc".
- Do NOT invent biographical facts; only infer groups that are clearly suggested by the descriptors.

Return ONLY valid JSON with this structure (no markdown, no comments):
{{
  "groups": [
    {{
      "name": "UNC Charlotte",
      "members": [1, 2, 3]
    }},
    {{
      "name": "Mentors",
      "members": [4, 5]
    }}
  ]
}}

People:
{people_blob}
"""
    raw = _call_gemini(prompt)
    raw = (raw or "").strip()
    if not raw:
        return
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        print("[backend] run_people_grouping_agent JSON error:", e)
        return
    groups = data.get("groups") or []
    if not isinstance(groups, list):
        return

    # Build mapping person_id -> set of group names, then write back via vec_store.set_person_groups.
    assignments: dict[int, set[str]] = {}
    for g in groups:
        if not isinstance(g, dict):
            continue
        gname = (g.get("name") or "").strip()
        if not gname:
            continue
        members = g.get("members") or []
        if not isinstance(members, list):
            continue
        for mid in members:
            try:
                pid = int(mid)
            except (TypeError, ValueError):
                continue
            if pid not in assignments:
                assignments[pid] = set()
            assignments[pid].add(gname)

    for pid, gset in assignments.items():
        try:
            vec_store.set_person_groups(pid, sorted(gset))
        except Exception as e:
            print("[backend] run_people_grouping_agent set_person_groups error:", e)


def save_session_data(session_id: str, transcript: str, entry_date: str | None = None, instance_id: str = "") -> dict:
    """
    Extract summary + facts + metadata from transcript via Gemini; embed with Perplexity; save to SQLite+sqlite-vec.
    Returns {"summary": str, "facts": list[str]} for callers (e.g. LightRAG feed).
    Episodic row also stores metadata_json for future time-series / pattern analysis; metadata is not embedded.
    If entry_date is provided (ISO date or datetime), use it as the stored timestamp; otherwise use now.
    """
    import vec_store

    _ensure_storage()
    data = _extract_session_data(transcript)
    summary = data.get("summary", "")
    facts = data.get("facts", [])
    metadata = data.get("metadata") or {}
    if not summary and not facts:
        print("[backend] save_session_data: extraction returned no summary or facts (check Gemini response)")

    ts = datetime.utcnow().isoformat() + "Z"
    if entry_date:
        try:
            s = entry_date.strip()[:26].replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except Exception:
            pass

    if summary:
        summary_emb = _embed_texts([summary])[0]
        metadata_json = json.dumps(metadata) if metadata else None
        vec_store.add_episodic(session_id, ts, summary, summary_emb, metadata_json=metadata_json, instance_id=instance_id or "")
        print("[backend] save_session_data: saved 1 episodic summary")

    # Deduplicate facts: if a new fact is very similar to an existing one, update it instead of inserting
    GIST_SIMILARITY_THRESHOLD = 0.85  # cosine similarity above this => update existing fact
    if facts:
        fact_embs = _embed_texts(facts)
        n_existing = vec_store.gist_count(instance_id=instance_id or "")
        for fact, emb in zip(facts, fact_embs):
            if n_existing > 0:
                nearest = vec_store.query_gist_nearest(emb, k=1)
                if nearest:
                    _doc_id, existing_doc, distance = nearest[0]
                    similarity = 1.0 - distance
                    if similarity >= GIST_SIMILARITY_THRESHOLD:
                        # Only update if the new fact has more content (avoids regression to weaker facts)
                        if len(fact) > len(existing_doc):
                            vec_store.update_gist(_doc_id, fact, emb)
                        continue
            vec_store.add_gist(session_id, ts, fact, emb, instance_id=instance_id or "")
            n_existing += 1
        print("[backend] save_session_data: saved %d gist facts" % len(facts))

    return {"summary": summary, "facts": facts}


def _parse_iso_date(ts: str) -> datetime | None:
    """Parse ISO timestamp to datetime; return None if invalid."""
    if not ts or not ts.strip():
        return None
    try:
        s = ts.strip()[:26].replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _recency_boost(days_ago: float) -> float:
    """Return boost for recency: 0.4 for last 30 days, 0.2 for 31–90 days, 0 otherwise (40% more weight on present)."""
    if days_ago <= 30:
        return 0.4
    if days_ago <= 90:
        return 0.2
    return 0.0


def get_relevant_context(query: str, top_k_gist: int = 8, top_k_episodic: int = 5, instance_id: str = "") -> str:
    """
    Embed the query, retrieve relevant gist facts and episodic summaries from SQLite+sqlite-vec,
    with ~40% more weight on recent journals. Return a single string for the interviewer's context.
    """
    import vec_store

    if not query or not query.strip():
        return "None."
    _ensure_storage()
    query_emb = _embed_texts([query.strip()])[0]
    now = datetime.now(timezone.utc)

    def rerank_with_recency(
        items: list[tuple[str, str]],
        k: int,
    ) -> list[tuple[str, str]]:
        scored: list[tuple[float, tuple[str, str]]] = []
        for rank, (doc, ts) in enumerate(items):
            days_ago = 999.0
            dt = _parse_iso_date(ts)
            if dt:
                try:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days_ago = max(0, (now - dt).total_seconds() / 86400)
                except Exception:
                    pass
            boost = _recency_boost(days_ago)
            # Lower rank = better similarity; add recency boost so recent items rank higher
            score = -rank + boost
            scored.append((score, (doc, ts)))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:k]]

    parts = []
    try:
        if vec_store.gist_count(instance_id=instance_id) > 0:
            count = vec_store.gist_count(instance_id=instance_id)
            fetch_k = min(max(top_k_gist * 2, 10), count)
            raw = vec_store.query_gist_with_timestamp(query_emb, fetch_k, instance_id=instance_id)
            items = rerank_with_recency(raw, top_k_gist)
            if items:
                lines = []
                for doc, ts in items:
                    dt = _parse_iso_date(ts)
                    if dt:
                        lines.append(f"- {doc} (from {dt.strftime('%Y-%m-%d')})")
                    else:
                        lines.append(f"- {doc}")
                parts.append("Facts and details from the user's life and journals:\n" + "\n".join(lines))
    except Exception:
        try:
            docs = vec_store.query_gist(query_emb, min(top_k_gist, vec_store.gist_count(instance_id=instance_id)), instance_id=instance_id)
            if docs:
                parts.append("Facts and details from the user's life and journals:\n" + "\n".join(f"- {d}" for d in docs))
        except Exception:
            pass

    try:
        if vec_store.episodic_count(instance_id=instance_id) > 0:
            count = vec_store.episodic_count(instance_id=instance_id)
            fetch_k = min(max(top_k_episodic * 2, 8), count)
            raw = vec_store.query_episodic_with_timestamp(query_emb, fetch_k, instance_id=instance_id)
            items = rerank_with_recency(raw, top_k_episodic)
            if items:
                lines = []
                for doc, ts in items:
                    dt = _parse_iso_date(ts)
                    if dt:
                        lines.append(f"- {doc} (from {dt.strftime('%Y-%m-%d')})")
                    else:
                        lines.append(f"- {doc}")
                parts.append("Relevant journal summaries (more recent entries favored):\n" + "\n".join(lines))
            elif not items and vec_store.episodic_count(instance_id=instance_id) > 0:
                docs = vec_store.query_episodic(query_emb, min(top_k_episodic, vec_store.episodic_count(instance_id=instance_id)), instance_id=instance_id)
                if docs:
                    parts.append("Relevant journal summaries:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        try:
            docs = vec_store.query_episodic(query_emb, min(top_k_episodic, vec_store.episodic_count(instance_id=instance_id)), instance_id=instance_id)
            if docs:
                parts.append("Relevant journal summaries:\n" + "\n".join(f"- {d}" for d in docs))
        except Exception:
            pass

    if not parts:
        return "None."
    return "\n\n".join(parts)


def get_memory_for_visualization(instance_id: str = "") -> tuple[list[str], list[str]]:
    """
    Return (gist_facts, episodic_summaries) as lists of document strings for diagram generation.
    """
    import vec_store

    _ensure_storage()
    gist_docs: list[str] = []
    episodic_docs: list[str] = []

    try:
        gist_docs = vec_store.get_all_gist(instance_id=instance_id)
    except Exception:
        pass

    try:
        episodic_docs = vec_store.get_all_episodic(instance_id=instance_id)
    except Exception:
        pass

    return (gist_docs, episodic_docs)


def get_memory_for_date(date_iso: str, instance_id: str = "") -> tuple[list[dict], list[dict]]:
    """Return (episodic_summaries, gist_facts) for the given date (YYYY-MM-DD)."""
    import vec_store

    _ensure_storage()
    episodic: list[dict] = []
    gist: list[dict] = []
    try:
        episodic = vec_store.get_episodic_for_date(date_iso, instance_id=instance_id)
    except Exception as e:
        print("[backend] get_episodic_for_date error:", e)
    try:
        gist = vec_store.get_gist_for_date(date_iso, instance_id=instance_id)
    except Exception as e:
        print("[backend] get_gist_for_date error:", e)
    return (episodic, gist)


def generate_day_summary(
    date_iso: str,
    raw_transcript: str | None,
    episodic: list[dict],
    gist: list[dict],
) -> str:
    """Use the LLM to produce a short summary/highlights for the day from raw journal and DB memory."""
    date_display = date_iso[:10] if date_iso else "this day"
    episodic_blob = "\n".join(f"- {m.get('document', '')}" for m in episodic if m.get("document"))
    gist_blob = "\n".join(f"- {m.get('document', '')}" for m in gist if m.get("document"))
    raw_section = ""
    if raw_transcript and raw_transcript.strip():
        raw_section = f"""
RAW JOURNAL ENTRY (user and AI conversation) for this day:
---
{raw_transcript.strip()[:8000]}
---
"""
    prompt = f"""You are summarizing a single day of the user's journaling life. Given the date and any raw journal transcript plus what is stored in the memory DB for that day, return exactly TWO separate paragraphs with these exact headings.

Date: {date_display}
{raw_section}
MEMORY DB — Episodic summaries for this day:
{episodic_blob or "(none)"}

MEMORY DB — Gist facts for this day:
{gist_blob or "(none)"}

Use exactly this format (include the headings and a blank line after each heading):

Objective

[One paragraph: facts only. What happened, what was said or done, concrete events or topics. No interpretation or feelings. Just the facts.]

Story of the day

[One paragraph: synthesis. The narrative of the day — themes, how things fit together, what it added up to. The story of the day.]

If there is no journal and no DB content for this day, say so briefly under Objective and under Story of the day write something like "No recorded content to synthesize." Do not invent details."""
    out = _call_gemini(prompt)
    return (out or "No summary generated.").strip()


def list_memory_facts(instance_id: str = "") -> list[dict]:
    """Return all gist facts with id, document, session_id, timestamp for Memory UI."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.list_gist_with_ids(instance_id=instance_id)
    except Exception as e:
        print("[backend] list_memory_facts error:", e)
        return []


def list_memory_summaries(instance_id: str = "") -> list[dict]:
    """Return all episodic summaries with id, document, session_id, timestamp for Memory UI."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.list_episodic_with_ids(instance_id=instance_id)
    except Exception as e:
        print("[backend] list_memory_summaries error:", e)
        return []


def get_person_events(person_name: str) -> list[dict]:
    """
    Return episodic events involving the given person name.
    Derived from memory_episodic.metadata_json (events + people lists).
    """
    items = list_memory_summaries()
    if not person_name or not person_name.strip():
        return []
    target = person_name.strip()
    target_l = target.lower()
    results: list[dict] = []
    for item in items:
        meta_json = item.get("metadata_json")
        if not meta_json:
            continue
        try:
            meta = json.loads(meta_json)
        except Exception:
            continue
        people = meta.get("people") or []
        if not isinstance(people, list) or target not in people:
            continue
        events = meta.get("events") or []
        if not isinstance(events, list):
            continue
        # Keep only events that clearly mention this person's name to avoid random, unrelated actions.
        clean_events: list[str] = []
        for e in events:
            if not isinstance(e, (str, int, float)):
                continue
            s = str(e).strip()
            if not s:
                continue
            # Require the person's name to appear in the event text.
            if target_l not in s.lower():
                continue
            clean_events.append(s)
        if not clean_events:
            continue
        results.append(
            {
                "summary_id": item.get("id"),
                "timestamp": item.get("timestamp") or "",
                "events": clean_events,
            }
        )
    return results


def update_memory_fact(fact_id: int, document: str) -> bool:
    """Update a gist fact by id; re-embeds and updates vec store. Returns True if found."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return False
    doc = document.strip()
    emb = _embed_texts([doc])[0]
    return vec_store.update_gist(fact_id, doc, emb)


def update_memory_summary(
    summary_id: int, document: str, metadata: dict | None = None
) -> bool:
    """Update an episodic summary by id; re-embeds and updates vec store. Optionally update metadata_json."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return False
    doc = document.strip()
    emb = _embed_texts([doc])[0]
    metadata_json = json.dumps(metadata) if metadata is not None else None
    return vec_store.update_episodic(summary_id, doc, emb, metadata_json=metadata_json)


def delete_memory_fact(fact_id: int) -> bool:
    """Delete a gist fact by id. Returns True if found and deleted."""
    import vec_store

    _ensure_storage()
    return vec_store.delete_gist(fact_id)


def delete_memory_summary(summary_id: int) -> bool:
    """Delete an episodic summary by id. Returns True if found and deleted."""
    import vec_store

    _ensure_storage()
    return vec_store.delete_episodic(summary_id)


def add_memory_fact(document: str, session_id: str | None = None, instance_id: str = "") -> int | None:
    """Add a single user-created fact; returns new id or None on failure."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return None
    doc = document.strip()
    sid = session_id or "user"
    ts = datetime.utcnow().isoformat() + "Z"
    emb = _embed_texts([doc])[0]
    return vec_store.add_gist(sid, ts, doc, emb, instance_id=instance_id or "")


def add_memory_summary(document: str, session_id: str | None = None, instance_id: str = "") -> int | None:
    """Add a single user-created summary; returns new id or None on failure."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return None
    doc = document.strip()
    sid = session_id or "user"
    ts = datetime.utcnow().isoformat() + "Z"
    emb = _embed_texts([doc])[0]
    return vec_store.add_episodic(sid, ts, doc, emb, instance_id=instance_id or "")


def generate_memory_mermaid(gist_facts: list[str], episodic_summaries: list[str]) -> str:
    """
    Use LLM to generate a Mermaid diagram (mindmap or flowchart) from vector DB content.
    Returns the raw Mermaid code string.
    """
    if not gist_facts and not episodic_summaries:
        return """mindmap
  root((Your memory))
    Empty
    Start journaling to see facts and themes here"""

    facts_blob = "\n".join(f"- {f}" for f in (gist_facts or [])[:50])
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_summaries or [])[:30])

    prompt = f"""You are a visualization expert. Create a single Mermaid diagram that represents this person's journaled memory in a satisfying, visual way.

FACTS ABOUT THE USER (from their journals):
{facts_blob or "(none)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none)"}

Instructions:
- Output ONLY valid Mermaid code. No markdown code fence, no explanation.
- Use a mindmap diagram with root "My journal" or "Memory" and organize facts and themes into clear branches (e.g. Work, Relationships, Health, Goals, Emotions). Keep node labels SHORT (a few words) so the diagram stays readable.
- If you have both facts and summaries, group related items under thematic branches. Make it feel personal and reflective of what they wrote.
- Use simple labels; avoid long sentences. Use parentheses for the root: root((My journal)).
- Maximum 30–40 nodes total to keep the diagram clean."""

    code = _call_gemini(prompt)
    code = (code or "").strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code or """mindmap
  root((My journal))
    Add entries to see your memory here"""


def _parse_recommendation_json(text: str, default: list) -> list:
    """Parse LLM JSON output into list of dicts with title, author, reason, url."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return default
    if not isinstance(raw, list):
        return default
    return [
        {
            "title": isinstance(x, dict) and x.get("title") or str(x),
            "author": isinstance(x, dict) and x.get("author") or "",
            "reason": isinstance(x, dict) and x.get("reason") or "",
            "url": isinstance(x, dict) and x.get("url") or "",
        }
        for x in raw
    ]


def _pplx_search_api_key() -> str:
    return (os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "").strip()


def _perplexity_search_api(
    query: str,
    *,
    max_results: int = 8,
    search_recency_filter: str | None = "month",
    country: str | None = None,
    timeout_sec: float = 45.0,
) -> list[dict]:
    """Call Perplexity Search POST /search; return list of {title, url, snippet, date}."""
    api_key = _pplx_search_api_key()
    if not api_key or not (query or "").strip():
        return []
    payload: dict = {
        "query": query.strip(),
        "max_results": max(1, min(20, int(max_results))),
    }
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if country and len(str(country).strip()) >= 2:
        payload["country"] = str(country).strip()[:2].upper()
    try:
        body = _perplexity_post_json(PPLX_SEARCH_URL, payload, api_key, timeout_sec=timeout_sec)
    except Exception as e:
        print("[backend] Perplexity Search error:", e)
        return []
    raw_list = body.get("results") or []
    out: list[dict] = []
    for r in raw_list:
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if not title or not url:
            continue
        out.append({
            "title": title[:500],
            "url": url[:2000],
            "snippet": snippet[:2500],
            "date": r.get("date"),
        })
    return out


def _parse_json_string_list(text: str) -> list[str]:
    """Extract a JSON array of strings from Gemini (or similar) output."""
    if not text or not isinstance(text, str):
        return []
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    if "[" in t and "]" in t:
        t = t[t.index("[") : t.rindex("]") + 1]
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]


def _recent_journals_block(recent_summaries_blob: str) -> str:
    if not (recent_summaries_blob or "").strip():
        return ""
    return f"""
══════════════════════════════════════════════════════════════════
HIGHEST PRIORITY — MOST RECENT JOURNAL SESSIONS (weight heavily; most alive for the user now)
══════════════════════════════════════════════════════════════════
{recent_summaries_blob}

"""


def _merge_perplexity_queries(
    queries: list[str],
    *,
    max_per_query: int = 6,
    search_recency_filter: str | None = "month",
    country: str | None = None,
    max_total_hits: int = 24,
    timeout_per_query: float = 30.0,
) -> list[dict]:
    """Run Perplexity Search for each query in parallel threads; merge with URL dedupe (stable query order)."""
    clean = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not clean:
        return []

    def _one(q: str) -> list[dict]:
        return _perplexity_search_api(
            q,
            max_results=max_per_query,
            search_recency_filter=search_recency_filter,
            country=country,
            timeout_sec=timeout_per_query,
        )

    merge_cap = float(os.getenv("RECOMMENDATIONS_PPLX_MERGE_WAIT_CAP_SEC", "52"))
    merge_wait = min(timeout_per_query + 12.0, merge_cap)
    n_workers = min(8, len(clean))
    rows_per_query: list[list[dict]] = [[] for _ in clean]
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        future_to_i = {executor.submit(_one, q): i for i, q in enumerate(clean)}
        futures_list = list(future_to_i.keys())
        futures_wait(futures_list, timeout=merge_wait)
        for fut, i in future_to_i.items():
            if not fut.done():
                continue
            try:
                rows_per_query[i] = fut.result(timeout=0)
            except Exception as e:
                print("[backend] Perplexity merge worker error:", e)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    seen_urls: set[str] = set()
    merged: list[dict] = []
    for rows in rows_per_query:
        for row in rows:
            u = (row.get("url") or "").strip().lower()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            merged.append(row)
            if len(merged) >= max_total_hits:
                return merged
    return merged


def _search_hits_context_lines(hits: list[dict], max_chars: int = 10000) -> str:
    buf: list[str] = []
    n = 0
    for i, h in enumerate(hits[:35], 1):
        t = h.get("title") or ""
        u = h.get("url") or ""
        s = (h.get("snippet") or "")[:450]
        line = f"{i}. {t} | {u}\n   {s}"
        if n + len(line) > max_chars:
            break
        buf.append(line)
        n += len(line) + 1
    return "\n".join(buf)


def _filter_hits_basic(hits: list[dict], consumed_lower: str) -> list[dict]:
    """Drop paywalled domains, video/replay junk, and obvious consumed URL/title overlaps."""
    out: list[dict] = []
    for h in hits:
        url = h.get("url") or ""
        title = h.get("title") or ""
        if _is_paywalled_domain(url):
            continue
        if _is_video_or_replay_result(title, url):
            continue
        tl = title.lower()
        ul = url.lower()
        if tl and tl in consumed_lower:
            continue
        # Light overlap: skip if URL path appears quoted in consumed
        if ul and ul in consumed_lower:
            continue
        out.append(h)
    return out


def _books_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Book recommendations via Perplexity Search + Gemini structuring (no duplicate titles vs consumed)."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are a book curator. Based on this person's journal-derived memory and consumed library, suggest 3–5 books (title + author + one-sentence reason). No URLs.

FACTS:
{facts_blob or "(none yet)"}

SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title": "...", "author": "...", "reason": "..."}}, ...]"""
        return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])

    q_prompt = f"""Suggest 3 short **web search** queries to discover **nonfiction or fiction books** (lists, reviews, "best books on…") for this person. Queries must help find real book titles and authors on the open web.
Respect **did not enjoy** rows in CONSUMED — steer away from similar themes or authors they rejected.

FACTS:
{facts_blob or "(none)"}

SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY a JSON array of 3 strings. Example: ["best psychology books for anxiety 2024", "literary fiction grief healing"]"""
    queries = _parse_json_string_list(_call_gemini(q_prompt))
    if len(queries) < 2:
        queries = ["best nonfiction books personal growth highly rated", "literary fiction book recommendations deep characters"]
    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        queries[:4],
        max_per_query=5,
        search_recency_filter="year",
        max_total_hits=22,
    )
    hits = _filter_hits_basic(hits, consumed_lower)
    blob = _search_hits_context_lines(hits)
    if not blob.strip():
        prompt = f"""Suggest 3–5 books for this person (title, author, reason). {recent_section}{consumed[:4000]}
facts: {facts_blob[:2000]}
summaries: {summaries_blob[:2000]}
Return ONLY JSON array: [{{"title","author","reason","url"}}] url optional."""
        return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])

    curate = f"""You are a book curator. Pick **3–5 distinct books** for this user using ONLY information grounded in SEARCH_RESULTS (real books mentioned there). Map each to canonical **title** and **author** as in the source; **url** must be copied exactly from SEARCH_RESULTS when present (review, publisher, Goodreads, etc.) or "".
Tie each **reason** to the user's themes in CONTEXT; keep reasons one sentence, honest.

CONTEXT — FACTS:
{facts_blob or "(none)"}

CONTEXT — SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:5000]}

SEARCH_RESULTS:
{blob}

Rules: Do NOT output books that appear in CONSUMED as already read. Avoid books/authors clearly similar to **did not enjoy** notes.
Return ONLY JSON: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    out = _parse_recommendation_json(_call_gemini(curate), [])
    # Fallback if Gemini returned empty
    if not out:
        return _parse_recommendation_json(_call_gemini_with_google_search(curate), [])
    return out[:8]


LISTEN_NOTES_BASE = "https://listen-api.listennotes.com/api/v2"
# Set to True to skip calling the Listen Notes API (use LLM-only fallback for podcast suggestions).
PODCAST_API_PAUSED = False


def _listen_notes_search_episodes(
    query: str,
    api_key: str,
    max_results: int = 5,
    published_after_ms: int | None = None,
) -> list[dict]:
    """Call Listen Notes API search (type=episode). Returns list of {title, author, url, reason, pub_date_ms}."""
    import urllib.parse
    import urllib.request
    params = {"q": query, "type": "episode", "only_one_episode_per_podcast": 1}
    if published_after_ms is not None:
        params["published_after"] = published_after_ms
    qs = urllib.parse.urlencode(params)
    url = f"{LISTEN_NOTES_BASE}/search?{qs}"
    req = urllib.request.Request(url, headers={"X-ListenAPI-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print("[backend] Listen Notes API error:", e)
        return []
    results = data.get("results") or []
    out = []
    for r in results:
        title = (r.get("title_original") or r.get("title_highlighted") or "").strip()
        podcast = r.get("podcast") or {}
        author = (podcast.get("title_original") or podcast.get("title_highlighted") or "").strip()
        ln_url = (r.get("listennotes_url") or "").strip()
        pub_ms = r.get("pub_date_ms")
        if isinstance(pub_ms, (int, float)):
            pub_ms = int(pub_ms)
        else:
            pub_ms = 0
        if title and ln_url:
            out.append({
                "title": title[:500],
                "author": author[:300],
                "reason": f"Suggested based on your interests: {query[:80]}.",
                "url": ln_url,
                "pub_date_ms": pub_ms,
            })
        if len(out) >= max_results:
            break
    return out


def _generate_podcast_reasons(
    episodes: list[dict],
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
) -> list[str]:
    """Generate a short personalized reason (blurb) for each podcast episode. Returns list of strings in same order."""
    if not episodes:
        return []
    list_text = "\n".join(
        f"{i + 1}. Show: {e.get('author', '')} | Episode: {e.get('title', '')}"
        for i, e in enumerate(episodes)
    )
    prompt = f"""Given this user's context and these podcast episodes, write a one-sentence reason why each episode is a good suggestion for them. Be specific to their interests (from journals and consumed items). Same order as the list.

USER CONTEXT (facts and journal themes):
{facts_blob or "(none)"}

CONSUMED (what they've read/listened to and their reflections):
{consumed[:1500]}

EPISODES (show | episode title):
{list_text}

Return ONLY a JSON array of {len(episodes)} strings, one reason per episode, in the same order. No markdown. Example: ["Reason for episode 1.", "Reason for episode 2."]"""
    try:
        text = _call_gemini(prompt)
        text = str(text).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        reasons = json.loads(text)
        if isinstance(reasons, list) and len(reasons) >= len(episodes):
            return [str(r).strip() or "Matches your interests." for r in reasons[: len(episodes)]]
    except Exception as e:
        print("[backend] _generate_podcast_reasons error:", e)
    return [e.get("reason", "Matches your interests.") for e in episodes]


def _podcasts_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Dedicated podcast agent: uses Listen Notes API for real episode links when key is set."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    api_key = (os.getenv("LISTENNOTES_API_KEY") or "").strip()
    if api_key and not PODCAST_API_PAUSED:
        prompt = f"""Based on this person's journal-derived memory and what they have consumed (books, podcasts, articles, research) and their reflections on any of them, suggest 2–3 short search queries (topics or themes) to find relevant podcast episodes. Use tastes from their book notes, article reads, and research too—e.g. themes from a book they loved can become podcast queries. We will prioritize episodes from the last 3 months, so prefer queries that are likely to surface recent, timely episodes. Examples: "mindfulness sleep", "anxiety therapy", "Huberman Lab sleep".

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES (all themes):
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY a JSON array of 2–3 short search query strings, no markdown. Example: ["mindfulness and sleep", "therapy for anxiety"]"""
        try:
            text = _call_gemini(prompt)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            queries = json.loads(text)
            if not isinstance(queries, list):
                queries = []
        except Exception as e:
            print("[backend] Podcast agent query parsing:", e)
            queries = []
        consumed_lower = consumed.lower()
        seen_urls = set()
        # Prefer episodes from the last 3 months (newer = better)
        from datetime import datetime, timedelta
        published_after_ms = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)
        out = []
        per_query = 3
        for q in queries[:3]:
            if not isinstance(q, str) or not q.strip():
                continue
            for item in _listen_notes_search_episodes(
                q.strip(), api_key, max_results=per_query, published_after_ms=published_after_ms
            ):
                if item["url"] in seen_urls:
                    continue
                if item["title"].lower() in consumed_lower or (item["author"] and item["author"].lower() in consumed_lower):
                    continue
                seen_urls.add(item["url"])
                out.append(item)
        # If we got few recent episodes, fill with older ones (no date filter)
        if len(out) < 5:
            for q in queries[:3]:
                if len(out) >= 5:
                    break
                if not isinstance(q, str) or not q.strip():
                    continue
                for item in _listen_notes_search_episodes(q.strip(), api_key, max_results=per_query):
                    if item["url"] in seen_urls:
                        continue
                    if item["title"].lower() in consumed_lower or (item["author"] and item["author"].lower() in consumed_lower):
                        continue
                    seen_urls.add(item["url"])
                    out.append(item)
        # Sort by pub_date_ms descending (newer first), then take top 5
        out.sort(key=lambda x: x.get("pub_date_ms") or 0, reverse=True)
        out = out[:5]
        if out:
            reasons = _generate_podcast_reasons(out, facts_blob, summaries_blob, consumed)
            for i, item in enumerate(out):
                item["reason"] = reasons[i] if i < len(reasons) else item.get("reason", "Matches your interests.")
                item.pop("pub_date_ms", None)
            return out
    prompt = f"""You are a podcast curator. Based on this person's journal-derived memory and what they have consumed (books, podcasts, articles, research) and their reflections on any of them, suggest 3–5 specific podcast episodes (show + episode). Use their tastes from books, articles, and research too—e.g. themes from a paper or book they liked can inform podcast picks.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES (all themes):
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Rules: Use "author" for the show name and "title" for the episode. Provide direct "url" (Spotify or Apple Podcasts episode link) when you know it; otherwise leave "url" empty. Do NOT suggest items they have already consumed. For each give a short "reason" (one sentence).
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini_with_google_search(prompt)
    return _parse_recommendation_json(text, [])


def _generate_article_reasons(
    articles: list[dict],
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
) -> list[str]:
    """Generate one-sentence reasons for each article using LLM."""
    list_text = "\n".join(
        f"{i + 1}. {e.get('title', '')} | {e.get('url', '')}"
        for i, e in enumerate(articles)
    )
    prompt = f"""Given these articles, write one short sentence per article saying why it might interest the user. Base the reason on what the article is actually about; be concrete and literal. Only mention journal themes if there's a direct, obvious fit—do not stretch (e.g. don't link a sports replay to "Steinbeck" or "roommates"). Same order as the list.

USER CONTEXT (facts and journal themes):
{facts_blob or "(none)"}

CONSUMED (what they've read and their reflections):
{consumed[:1500]}

ARTICLES (title | url):
{list_text}

Return ONLY a JSON array of {len(articles)} strings, one reason per article, in the same order. No markdown. Example: ["Reason for article 1.", "Reason for article 2."]"""
    try:
        text = _call_gemini(prompt)
        text = str(text).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        reasons = json.loads(text)
        if isinstance(reasons, list) and len(reasons) >= len(articles):
            return [str(r).strip() or "Matches your interests." for r in reasons[: len(articles)]]
    except Exception as e:
        print("[backend] _generate_article_reasons error:", e)
    return [a.get("reason", "Matches your interests.") for a in articles]


# Domains we exclude from article recommendations to avoid paywalled content (Tavily + post-filter).
PAYWALLED_ARTICLE_DOMAINS = [
    "nytimes.com",
    "wsj.com",
    "washingtontimes.com",
    "washingtonpost.com",
    "theatlantic.com",
    "newyorker.com",
    "economist.com",
    "ft.com",
    "bloomberg.com",
    "barrons.com",
    "latimes.com",
    "bostonglobe.com",
    "chicagotribune.com",
    "harpers.org",
    "medium.com",
    "substack.com",  # many paywalled
]


def _is_video_or_replay_result(title: str, url: str) -> bool:
    """Skip sports replays, full-length videos, and live streams—we want articles and long-reads only."""
    t = (title or "").lower()
    if "full-length replay" in t or "full replay" in t or "game replay" in t:
        return True
    if " vs. " in t and ("tournament" in t or "game #" in t or "watch" in t):
        return True
    if "watch " in t and ("live" in t or "stream" in t or "replay" in t):
        return True
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).netloc or "").lower()
        if any(x in host for x in ("foxsports.com", "espn.com", "cbssports.com", "nbcsports.com", "bleacherreport.com")):
            if "replay" in t or "watch" in t or " vs " in t:
                return True
    except Exception:
        pass
    return False


def _is_paywalled_domain(url: str) -> bool:
    """True if the URL's host is in our paywalled-domain list (or a subdomain of one)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return False
        for domain in PAYWALLED_ARTICLE_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return True
        return False
    except Exception:
        return False


def _tavily_search_articles(
    queries: list[str],
    api_key: str,
    max_per_query: int = 4,
    topic: str = "news",
) -> list[dict]:
    """Call Tavily Search API and return list of {title, author, reason, url}. Uses topic=news for real articles. Excludes paywalled domains."""
    from urllib.parse import urlparse

    import sys
    print(f"[backend] Tavily Search API: entered with {len(queries)} queries", flush=True)
    try:
        from tavily import TavilyClient
    except ImportError as e:
        print("[backend] Tavily Search API: import failed (install with: pip install tavily-python):", e, flush=True)
        return []
    try:
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        print("[backend] Tavily Search API: client init failed:", e, flush=True)
        return []
    seen_urls: set[str] = set()
    out: list[dict] = []
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        query_str = q.strip()
        try:
            print(f"[backend] Tavily Search API: query={query_str!r} topic={topic} max_results={max_per_query}", flush=True)
            response = client.search(
                query=query_str,
                topic=topic,
                max_results=max_per_query,
                search_depth="basic",
                exclude_domains=PAYWALLED_ARTICLE_DOMAINS,
            )
        except Exception as e:
            print("[backend] Tavily search error:", e)
            continue
        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", []) or []
        print(f"[backend] Tavily Search API: got {len(results)} results for {query_str!r}", flush=True)
        for r in results:
            url = (r.get("url", "") if isinstance(r, dict) else getattr(r, "url", None)) or ""
            if not url or url in seen_urls or _is_paywalled_domain(url):
                continue
            title = (r.get("title", "") if isinstance(r, dict) else getattr(r, "title", None)) or ""
            if _is_video_or_replay_result(title, url):
                continue
            seen_urls.add(url)
            content = (r.get("content", "") if isinstance(r, dict) else getattr(r, "content", None)) or ""
            try:
                parsed = urlparse(url)
                author = parsed.netloc or ""
                if author.startswith("www."):
                    author = author[4:]
            except Exception:
                author = ""
            out.append({
                "title": title or "Article",
                "author": author,
                "reason": (content[:120] + "…") if content else "Matches your interests.",
                "url": url,
            })
    if out:
        print(f"[backend] Tavily Search API: returning {len(out)} articles (after dedupe and paywall filter)")
    return out


def _articles_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Informational / thought-provoking articles via Perplexity Search + personalized reasons."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are an article curator. Suggest 3–5 **informational** articles or essays (helpful, thought-provoking explainers—not clickbait). Real URLs only.

FACTS:
{facts_blob or "(none yet)"}

SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title","author","reason","url"}}, ...]"""
        return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])

    q_prompt = f"""Suggest 2–3 short **web search** queries for **in-depth readable articles**: explainers, essays, analysis, studies written for educated readers—not breaking headline chyrons.
Honor CONSUMED: respect **did not enjoy** and negative notes.

FACTS:
{facts_blob or "(none)"}

SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY JSON array of 2–3 strings. Example: ["long read climate adaptation solutions", "cognitive science of habit formation explainer"]"""
    queries = _parse_json_string_list(_call_gemini(q_prompt))
    if len(queries) < 2:
        queries = ["thought-provoking long read science society", "deep dive psychology well-being evidence"]
    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        queries[:3],
        max_per_query=6,
        search_recency_filter="month",
        max_total_hits=18,
    )
    hits = _filter_hits_basic(hits, consumed_lower)
    out: list[dict] = []
    for h in hits[:8]:
        url = h.get("url") or ""
        try:
            parsed = urllib.parse.urlparse(url)
            author = (parsed.netloc or "").lower()
            if author.startswith("www."):
                author = author[4:]
        except Exception:
            author = ""
        snip = (h.get("snippet") or "")[:200]
        out.append({
            "title": (h.get("title") or "Article")[:500],
            "author": author[:200],
            "reason": (snip + "…") if len(h.get("snippet") or "") > 200 else (h.get("snippet") or "Suggested read."),
            "url": url,
        })
    out = out[:6]
    if out:
        reasons = _generate_article_reasons(out, facts_blob, summaries_blob, consumed)
        for i, item in enumerate(out):
            item["reason"] = reasons[i] if i < len(reasons) else item.get("reason", "")
        print(f"[backend] Articles: Perplexity path returning {len(out)} articles.")
        return out
    return _parse_recommendation_json(_call_gemini_with_google_search(
        f"Suggest 3 articles with real URLs for this user. {recent_section}{consumed[:3000]}\nfacts:{facts_blob[:1500]}"
    ), [])


# --- Research: Semantic Scholar + PMC/PubMed (E-Utilities) ---
# Semantic Scholar: https://api.semanticscholar.org/api-docs/
# PMC/NCBI E-Utilities: https://pmc.ncbi.nlm.nih.gov/tools/developers/ (E-Utilities), https://www.ncbi.nlm.nih.gov/books/NBK25501/


def _semantic_scholar_search_papers(
    queries: list[str],
    max_per_query: int = 5,
) -> list[dict]:
    """Search Semantic Scholar Academic Graph API. Returns list of {title, author, url, year}. Optional SEMANTIC_SCHOLAR_API_KEY for higher rate limits."""
    import time
    import urllib.parse
    import urllib.request

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    # Without key: ~100 req/5 min → ~3.5s between requests to avoid 429
    delay_between_requests = 0.1 if api_key else 4.0

    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,url,authors,year,externalIds,openAccessPdf"
    seen_urls: set[str] = set()
    out: list[dict] = []
    for i, q in enumerate(queries):
        if not isinstance(q, str) or not q.strip():
            continue
        if i > 0:
            time.sleep(delay_between_requests)
        query_str = q.strip()
        params = urllib.parse.urlencode({"query": query_str, "limit": max_per_query, "fields": fields})
        url = f"{base}?{params}"
        headers = {"User-Agent": "Selfmeridian/1.0 (research recommendations)"}
        if api_key:
            headers["x-api-key"] = api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5.0)
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        data = json.loads(resp.read().decode())
                except Exception as retry_e:
                    print("[backend] Semantic Scholar search error (429 then retry):", retry_e)
                    continue
            else:
                print("[backend] Semantic Scholar search error:", e)
                continue
        except Exception as e:
            print("[backend] Semantic Scholar search error:", e)
            continue
        results = data.get("data") if isinstance(data, dict) else []
        for r in results:
            if not isinstance(r, dict):
                continue
            paper_id = r.get("paperId") or ""
            title = (r.get("title") or "").strip()
            if not title:
                continue
            url_val = r.get("url") or ""
            if not url_val:
                url_val = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
            if url_val and url_val in seen_urls:
                continue
            authors = r.get("authors") or []
            if authors and isinstance(authors[0], dict):
                name = authors[0].get("name") or ""
                author = f"{name} et al." if name else "Unknown"
            else:
                author = "Unknown"
            year = r.get("year")
            if year is not None:
                author = f"{author} ({year})" if author else str(year)
            seen_urls.add(url_val)
            out.append({"title": title, "author": author, "url": url_val, "reason": ""})
    return out


def _pmc_pubmed_search_papers(
    queries: list[str],
    max_per_query: int = 5,
) -> list[dict]:
    """Search PubMed via NCBI E-Utilities (esearch + efetch). Returns list of {title, author, url}. No API key required."""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    base_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    common = {"tool": "Selfmeridian", "email": "research@selfmeridian.com"}
    seen_urls: set[str] = set()
    out: list[dict] = []
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        query_str = q.strip()
        try:
            params_esearch = {**common, "db": "pubmed", "term": query_str, "retmax": max_per_query, "retmode": "json"}
            url_esearch = base_esearch + "?" + urllib.parse.urlencode(params_esearch)
            req = urllib.request.Request(url_esearch, headers={"User-Agent": "Selfmeridian/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print("[backend] PubMed esearch error:", e)
            continue
        id_list = []
        try:
            id_list = (data.get("esearchresult") or {}).get("idlist") or []
        except Exception:
            pass
        if not id_list:
            continue
        ids_str = ",".join(str(x) for x in id_list[: max_per_query])
        try:
            params_efetch = {**common, "db": "pubmed", "id": ids_str, "retmode": "xml"}
            url_efetch = base_efetch + "?" + urllib.parse.urlencode(params_efetch)
            req = urllib.request.Request(url_efetch, headers={"User-Agent": "Selfmeridian/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                root = ET.fromstring(resp.read())
        except Exception as e:
            print("[backend] PubMed efetch error:", e)
            continue
        def _find(el, path: str):
            """Find first child by tag path (no namespace)."""
            for tag in path.split("/"):
                if el is None:
                    return None
                for c in el:
                    if c.tag.endswith("}" + tag) or c.tag == tag:
                        el = c
                        break
                else:
                    return None
            return el

        for art in root.iter():
            if art.tag.endswith("}PubmedArticle") or art.tag == "PubmedArticle":
                try:
                    med = _find(art, "MedlineCitation")
                    art_el = _find(med, "Article") if med is not None else None
                    if art_el is None:
                        continue
                    title_el = _find(art_el, "ArticleTitle")
                    title = "".join((title_el.itertext() or [])).strip() if title_el is not None else ""
                    pmid = ""
                    pubmed_data = _find(art, "PubmedData")
                    if pubmed_data is not None:
                        id_list = _find(pubmed_data, "ArticleIdList")
                        if id_list is not None:
                            for aid in id_list:
                                if aid.tag.endswith("}ArticleId") or aid.tag == "ArticleId":
                                    if (aid.get("IdType") or aid.get("idtype")) == "pubmed":
                                        pmid = (aid.text or "").strip()
                                        break
                    if not pmid:
                        continue
                    author_el = _find(art_el, "AuthorList")
                    first_author = _find(author_el, "Author") if author_el is not None else None
                    last_el = _find(first_author, "LastName") if first_author is not None else None
                    last = (last_el.text or "").strip() if last_el is not None else ""
                    author = f"{last} et al." if last else "Unknown"
                    url_val = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if url_val in seen_urls:
                        continue
                    seen_urls.add(url_val)
                    out.append({"title": title or "Article", "author": author, "url": url_val, "reason": ""})
                except Exception as e:
                    print("[backend] PubMed parse one article:", e)
                    continue
    return out


def _generate_research_reasons(
    papers: list[dict],
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
) -> list[str]:
    """Generate one short sentence per paper; keep reasons grounded in the paper topic."""
    list_text = "\n".join(
        f"{i + 1}. {p.get('title', '')} | {p.get('url', '')}"
        for i, p in enumerate(papers)
    )
    prompt = f"""Given these research papers, write one short sentence per paper saying why it might interest the user. Base the reason on what the paper is actually about; be concrete. Only mention journal themes if there's a direct fit.
Same order as the list.

USER CONTEXT:
{facts_blob or "(none)"}

CONSUMED:
{consumed[:1500]}

PAPERS (title | url):
{list_text}

Return ONLY a JSON array of {len(papers)} strings, one reason per paper, same order. No markdown."""
    try:
        text = _call_gemini(prompt)
        text = str(text).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        if "[" in text and "]" in text:
            start, end = text.index("["), text.rindex("]") + 1
            text = text[start:end]
        reasons = json.loads(text)
        if isinstance(reasons, list) and len(reasons) >= len(papers):
            return [str(r).strip() or "Relevant to your interests." for r in reasons[: len(papers)]]
    except Exception as e:
        print("[backend] _generate_research_reasons error:", e)
    return [p.get("reason", "Relevant to your interests.") for p in papers]


SCHOLARLY_URL_MARKERS = (
    "doi.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
    "biorxiv.org",
    "medrxiv.org",
    "/pmc/articles/",
    "nature.com/articles",
    "science.org/doi",
    "cell.com",
    "plos.org",
    "frontiersin.org",
    "springer.com/article",
    "wiley.com",
    "ieee.org",
    "acm.org",
    "pnas.org",
    "journals.",
)


def _prefer_scholarly_hits(hits: list[dict]) -> list[dict]:
    good: list[dict] = []
    rest: list[dict] = []
    for h in hits:
        u = (h.get("url") or "").lower()
        if any(m in u for m in SCHOLARLY_URL_MARKERS):
            good.append(h)
        else:
            rest.append(h)
    return good + rest


def _research_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Research papers via Perplexity Search (scholarly URLs preferred) + grounded reasons."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are a research curator. Suggest 3–5 peer-reviewed or preprint papers with **verified** URLs (doi.org, PubMed, arXiv, Semantic Scholar only if certain).

FACTS:
{facts_blob or "(none yet)"}

SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title","author","reason","url"}}, ...]"""
        return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])

    q_prompt = f"""Suggest 2–3 short **web search** queries to find **peer-reviewed research papers, systematic reviews, or reputable preprints** (PubMed, arXiv, journal DOI pages). Be concrete; include methodology terms when useful (RCT, meta-analysis, cohort).
Honor CONSUMED — avoid subfields or angles the user disliked.

FACTS:
{facts_blob or "(none)"}

SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY JSON array of 2–3 strings."""
    queries = _parse_json_string_list(_call_gemini(q_prompt))
    if len(queries) < 2:
        queries = [
            "randomized controlled trial mental health well-being recent",
            "systematic review climate health intersection pubmed",
        ]
    widened = []
    for q in queries[:3]:
        widened.append(q)
        if "doi" not in q.lower() and "pubmed" not in q.lower():
            widened.append(f"{q} peer-reviewed OR systematic review")
    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        widened[:5],
        max_per_query=5,
        search_recency_filter="year",
        max_total_hits=22,
    )
    hits = _prefer_scholarly_hits(_filter_hits_basic(hits, consumed_lower))
    out: list[dict] = []
    for h in hits[:10]:
        url = h.get("url") or ""
        title = (h.get("title") or "")[:500]
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        author_guess = host or "Source"
        snip = (h.get("snippet") or "")[:180]
        out.append({
            "title": title or "Paper",
            "author": author_guess[:200],
            "reason": (snip + "…") if len(h.get("snippet") or "") > 180 else (h.get("snippet") or ""),
            "url": url,
        })
    out = out[:6]
    if out:
        reasons = _generate_research_reasons(out, facts_blob, summaries_blob, consumed)
        for i, item in enumerate(out):
            item["reason"] = reasons[i] if i < len(reasons) else item.get("reason", "")
        print(f"[backend] Research: Perplexity path returning {len(out)} papers.")
        return out
    prompt = f"""Suggest 3–5 research papers with verified doi.org / pubmed / arxiv URLs only.

{recent_section}{consumed[:4000]}
FACTS: {facts_blob[:2000]}
SUMMARIES: {summaries_blob[:2000]}
Return ONLY JSON array."""
    return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])


def _news_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """
    Uplifting / constructive news via Perplexity Search (week recency, country-aware).
    Extension point: conversational dislikes flow into `consumed` via get_consumed_context.
    """
    recent_section = _recent_journals_block(recent_summaries_blob)
    cc = (os.getenv("PERPLEXITY_NEWS_COUNTRY") or "US").strip()
    country = cc[:2].upper() if len(cc) >= 2 else "US"

    queries: list[str] = [
        "positive news breakthroughs science health environment progress this week",
        "good news clean energy infrastructure innovation milestones",
        "uplifting civic or global development stories solutions focused",
    ]

    if _pplx_search_api_key():
        gq = f"""The user benefits from **uplifting, constructive news**: scientific progress, climate/energy wins, public-health advances, humanitarian or community solutions, inspiring engineering (e.g. new reactors, grid, transit)—**not** outrage or culture-war angles.
Write **1–2 short web search queries** tailored to their **interests** in CONTEXT and (if journals mention it) **location**. Include phrases like "good news", "opens", "launches", "record", "milestones" where natural.
Respect CONSUMED **did not enjoy** lines—avoid outlets/topics they rejected.

FACTS:
{facts_blob or "(none)"}

SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:5500]}

Return ONLY a JSON array of 1–2 strings."""

        extra = _parse_json_string_list(_call_gemini(gq))
        for q in extra[:2]:
            if q and q not in queries:
                queries.append(q)

    if not _pplx_search_api_key():
        prompt = f"""Curate 3–6 **positive or solution-focused** news stories with verifiable URLs for this reader.

FACTS:
{facts_blob or "(none)"}

SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title","author","reason","url"}}, ...]"""
        return _parse_recommendation_json(_call_gemini_with_google_search(prompt), [])

    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        queries[:6],
        max_per_query=4,
        search_recency_filter="week",
        country=country,
        max_total_hits=26,
        timeout_per_query=30.0,
    )
    hits = _filter_hits_basic(hits, consumed_lower)
    out: list[dict] = []
    for h in hits[:14]:
        url = h.get("url") or ""
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            host = ""
        snip = (h.get("snippet") or "")[:200]
        out.append({
            "title": (h.get("title") or "News")[:500],
            "author": host[:200],
            "reason": (snip + "…") if len(h.get("snippet") or "") > 200 else (h.get("snippet") or ""),
            "url": url,
        })
    out = out[:12]
    if out:
        reasons = _generate_article_reasons(out, facts_blob, summaries_blob, consumed)
        for i, item in enumerate(out):
            item["reason"] = reasons[i] if i < len(reasons) else item.get("reason", "")
        print(f"[backend] News: Perplexity path returning {len(out)} items.")
        return out
    return _parse_recommendation_json(
        _call_gemini_with_google_search(
            f"Positive/solution-focused news with real URLs. {recent_section}{consumed[:3000]}\nfacts:{facts_blob[:1500]}"
        ),
        [],
    )


def generate_recommendations(instance_id: str = "") -> dict:
    """
    Run five dedicated agents (books, podcasts, articles, research, news) in parallel.
    Uses gist + episodic memory and consumed library; **recent** journal sessions get extra weight (~25%).
    Books/articles/research/news use Perplexity Search when PERPLEXITY_API_KEY is set.

    Extension: feed conversational or library **dislikes** into the same `consumed` string (e.g. via
    `add_consumed` + `get_consumed_context`) so future chat-derived preferences affect the next run
    without changing agent signatures.
    """
    gist_docs, episodic_docs = get_memory_for_visualization(instance_id=instance_id)
    consumed = get_consumed_context(instance_id=instance_id)
    facts_blob = "\n".join(f"- {f}" for f in (gist_docs or [])[:70])
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_docs or [])[:45])
    episodic_list = episodic_docs or []
    n_recent = max(8, int(len(episodic_list) * 0.25)) if episodic_list else 0
    recent_episodic = episodic_list[-n_recent:] if n_recent else []
    recent_summaries_blob = "\n".join(f"- {s}" for s in recent_episodic) if recent_episodic else ""

    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []
    news_list: list = []

    agent_timeout = float(os.getenv("RECOMMENDATIONS_AGENT_TIMEOUT_SEC", "68"))
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_books = executor.submit(_books_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_podcasts = executor.submit(_podcasts_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_articles = executor.submit(_articles_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_research = executor.submit(_research_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_news = executor.submit(_news_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        try:
            books_list = future_books.result(timeout=agent_timeout)
        except Exception as e:
            print("[backend] recommendations books_agent error:", e)
        try:
            podcasts_list = future_podcasts.result(timeout=agent_timeout)
        except Exception as e:
            print("[backend] recommendations podcasts_agent error:", e)
        try:
            articles_list = future_articles.result(timeout=agent_timeout)
        except Exception as e:
            print("[backend] recommendations articles_agent error:", e)
        try:
            research_list = future_research.result(timeout=agent_timeout)
        except Exception as e:
            print("[backend] recommendations research_agent error:", e)
        try:
            news_list = future_news.result(timeout=agent_timeout)
        except Exception as e:
            print("[backend] recommendations news_agent error:", e)

    return {
        "books": books_list if isinstance(books_list, list) else [],
        "podcasts": podcasts_list if isinstance(podcasts_list, list) else [],
        "articles": articles_list if isinstance(articles_list, list) else [],
        "research": research_list if isinstance(research_list, list) else [],
        "news": news_list if isinstance(news_list, list) else [],
    }
