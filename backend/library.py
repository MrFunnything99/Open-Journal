"""
Library for Selfmeridian: gist_facts (semantic) and episodic_log (episodic) memory,
and consumed_content (library). Uses SQLite + sqlite-vec for vector storage.

Embeddings: Perplexity (`_embed_texts` → PERPLEXITY_API_KEY); without a key, placeholder vectors match EMBEDDING_DIM so library rows still persist (semantic search degraded).
Extraction / helpers: OpenRouter chat completions only (`OPENROUTER_API_KEY`; model `OPENROUTER_EXTRACTION_MODEL`, with legacy fallback `OPENROUTER_GEMINI_MODEL`).
Books / articles / research agents: xAI Grok + OpenRouter web when `OPENROUTER_LIBRARY_WEB_MODEL` is set (default x-ai/grok-4.1-fast:online).
Tune `OPENROUTER_EXTRACTION_MAX_TOKENS` (default 8192) so OpenRouter does not reserve huge output budgets per request.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import re
import time
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

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from decision_logger import DecisionLogger

_rec_search_log: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "_rec_search_log", default=None
)


def _rec_search_log_begin() -> None:
    _rec_search_log.set([])


def _rec_search_log_snapshot() -> list[dict]:
    buf = _rec_search_log.get()
    return list(buf) if buf else []


def _rec_search_log_append(
    api: str, query: str, results_count: int, urls_returned: list[str] | None = None
) -> None:
    buf = _rec_search_log.get()
    if buf is None:
        return
    buf.append(
        {
            "api": api,
            "query": (query or "")[:800],
            "results_count": int(results_count),
            "urls_returned": (urls_returned or [])[:40],
        }
    )


_PERPLEXITY_EMBED_FALLBACK_WARNED = False

PPLX_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/embeddings"
PPLX_CONTEXTUAL_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/contextualizedembeddings"
# Context model: use contextualized endpoint (one chunk per pseudo-document for unrelated texts).
PPLX_EMBED_BATCH_DOCS = 480
DEFAULT_PERPLEXITY_EMBEDDING_MODEL = "pplx-embed-context-v1-4b"
PPLX_SEARCH_URL = "https://api.perplexity.ai/search"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_EXTRACTION_MODEL = "google/gemini-3-pro-preview"
# Books / articles / research recommendation agents: xAI Grok + OpenRouter web plugin (`:online` = native provider search).
DEFAULT_OPENROUTER_LIBRARY_WEB_MODEL = "x-ai/grok-4.1-fast:online"


def openrouter_api_configured() -> bool:
    return bool((os.getenv("OPENROUTER_API_KEY") or "").strip())


def openrouter_extraction_model() -> str:
    return (
        (os.getenv("OPENROUTER_EXTRACTION_MODEL") or os.getenv("OPENROUTER_GEMINI_MODEL") or DEFAULT_OPENROUTER_EXTRACTION_MODEL)
        .strip()
    )


def _library_recommendations_openrouter_enabled() -> bool:
    """When True, books/articles/research agents use OpenRouter (Grok + web) instead of Gemini (+ Google Search)."""
    if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        return False
    v = (os.getenv("OPENROUTER_LIBRARY_WEB_MODEL") or DEFAULT_OPENROUTER_LIBRARY_WEB_MODEL).strip().lower()
    return v not in ("0", "false", "no", "off", "disabled")


def _openrouter_library_web_model() -> str:
    return (os.getenv("OPENROUTER_LIBRARY_WEB_MODEL") or DEFAULT_OPENROUTER_LIBRARY_WEB_MODEL).strip()


def _openrouter_library_fast_model() -> str:
    """Same model family without `:online` for query-generation steps (no extra web plugin)."""
    explicit = (os.getenv("OPENROUTER_LIBRARY_FAST_MODEL") or "").strip()
    if explicit:
        return explicit
    m = _openrouter_library_web_model()
    if m.endswith(":online"):
        return m[: -len(":online")] or "x-ai/grok-4.1-fast"
    return m or "x-ai/grok-4.1-fast"


def _call_library_rec_web(prompt: str) -> str:
    if _library_recommendations_openrouter_enabled():
        try:
            to = float(os.getenv("OPENROUTER_LIBRARY_WEB_TIMEOUT_SEC", "120"))
        except ValueError:
            to = 120.0
        return _openrouter_chat_completion(prompt, model=_openrouter_library_web_model(), timeout_sec=to)
    if openrouter_api_configured():
        return _call_gemini(prompt)
    return ""


def _call_library_rec_fast(prompt: str) -> str:
    if _library_recommendations_openrouter_enabled():
        try:
            to = float(os.getenv("OPENROUTER_LIBRARY_FAST_TIMEOUT_SEC", "90"))
        except ValueError:
            to = 90.0
        return _openrouter_chat_completion(prompt, model=_openrouter_library_fast_model(), timeout_sec=to)
    if openrouter_api_configured():
        return _call_gemini(prompt)
    return ""


def library_recommendations_llm_label() -> str:
    """Short label for startup logs."""
    if _library_recommendations_openrouter_enabled():
        return (
            f"OpenRouter {_openrouter_library_web_model()} (web) + "
            f"{_openrouter_library_fast_model()} (query/planning)"
        )
    if openrouter_api_configured():
        return f"OpenRouter {openrouter_extraction_model()} (set OPENROUTER_LIBRARY_WEB_MODEL for :online web)"
    return "none (set OPENROUTER_API_KEY)"


def extraction_llm_backend() -> str:
    """Startup label: extraction / helper LLM (OpenRouter only)."""
    if openrouter_api_configured():
        return f"openrouter ({openrouter_extraction_model()})"
    return "none (set OPENROUTER_API_KEY for extraction/helpers)"


def gemini_extraction_backend() -> str:
    """Deprecated alias for startup logs; OpenRouter-only."""
    return extraction_llm_backend()


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
    max_tokens: int | None = None,
) -> str:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return ""
    eff_timeout = float(timeout_sec) if timeout_sec is not None else float(
        os.getenv("OPENROUTER_EXTRACTION_TIMEOUT_SEC") or os.getenv("OPENROUTER_GEMINI_TIMEOUT_SEC") or "75"
    )
    m = (model or openrouter_extraction_model()).strip() or DEFAULT_OPENROUTER_EXTRACTION_MODEL
    temp_raw = os.getenv("OPENROUTER_EXTRACTION_TEMPERATURE") or os.getenv("OPENROUTER_GEMINI_TEMPERATURE")
    if temperature is not None:
        temp = float(temperature)
    elif temp_raw is not None and str(temp_raw).strip() != "":
        temp = float(temp_raw)
    else:
        temp = 0.7
    if max_tokens is not None:
        try:
            eff_max = max(32, min(int(max_tokens), 65536))
        except (TypeError, ValueError):
            eff_max = 256
    else:
        _mt = (os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS") or os.getenv("OPENROUTER_GEMINI_MAX_TOKENS") or "8192").strip()
        try:
            eff_max = max(256, min(int(_mt), 65536))
        except ValueError:
            eff_max = 8192
    max_tokens = eff_max
    payload: dict = {
        "model": m,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
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


def _call_gemini(prompt: str) -> str:
    """
    Extraction / helper LLM: OpenRouter chat completions only (`OPENROUTER_API_KEY`).
    Returns empty string if OpenRouter is not configured or the request fails.
    """
    if not openrouter_api_configured():
        return ""
    try:
        return _openrouter_chat_completion(prompt)
    except Exception as e:
        print("[backend] _call_gemini (OpenRouter) error:", e)
        return ""


def _call_gemini_with_google_search(prompt: str) -> str:
    """
    Legacy name: previously Gemini + Google Search. Now identical to `_call_gemini`
    unless `OPENROUTER_LIBRARY_WEB_MODEL` enables the `:online` path via `_call_library_rec_web`.
    """
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
    isbn: str = "",
    publish_year: str = "",
    openlibrary_key: str = "",
    cover_url: str = "",
    subjects: str = "",
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
    if publish_year:
        doc += f" ({publish_year})"
    if isbn:
        doc += f" [ISBN {isbn}]"
    doc += f". Liked: {'yes' if liked else 'no'}."
    if subjects:
        doc += f" Genres/topics: {subjects}."
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
        isbn=(isbn or "")[:20],
        publish_year=(publish_year or "")[:10],
        openlibrary_key=(openlibrary_key or "")[:100],
        cover_url=(cover_url or "")[:300],
        subjects=(subjects or "")[:1000],
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


# --- Open Library HTTP (disabled): constants + helpers kept for when we re-enable catalog lookup. ---
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_USER_AGENT = os.getenv(
    "OPENLIBRARY_USER_AGENT",
    "SelfMeridian/1.0 (https://github.com/MrFunnything99/Open-Journal)",
)
OPENLIBRARY_THROTTLE_SEC = 0.2


def _openlibrary_http_timeout_sec() -> float:
    """Unused while Open Library queries are stubbed out."""
    try:
        v = float(os.getenv("OPENLIBRARY_HTTP_TIMEOUT_SEC", "3.5"))
    except ValueError:
        v = 3.5
    return max(1.0, min(v, 30.0))


_DERIVATIVE_TITLE_PATTERNS = [
    "summary of", "summary:", "study guide", "workbook", "sparknotes",
    "cliffsnotes", "cliff's notes", "coles notes", "masterclass",
    "analysis of", "review of", "companion to", "[adaptation]",
    "adapted for", "abridged", "illustrated edition",
]
_DERIVATIVE_AUTHOR_PATTERNS = [
    "short summary", "speedreader", "quickread", "instaread",
    "book summary", "readtrepreneur",
]


def _smart_title_case(s: str) -> str:
    """Title-case that handles apostrophes correctly (Man's not Man'S)."""
    return re.sub(
        r"[A-Za-z]+('[A-Za-z]+)?",
        lambda m: m.group(0).capitalize(),
        s,
    )


def _clean_title(title: str, raw_title: str = "") -> str:
    """Clean Open Library title quirks: strip ' / Author' suffixes, fix lowercase casing,
    and trim long subtitles when the user's raw title clearly matches the main part."""
    if " / " in title:
        title = title.split(" / ")[0].strip()
    raw_norm = re.sub(r"[^a-z0-9 ]", "", raw_title.lower()).strip()
    title_norm = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    if raw_norm and title_norm.startswith(raw_norm) and len(title_norm) > len(raw_norm) + 5:
        alnum_count = 0
        end_pos = 0
        for i, ch in enumerate(title.lower()):
            if re.match(r"[a-z0-9 ]", ch):
                alnum_count += 1
                if alnum_count >= len(raw_norm):
                    end_pos = i + 1
                    break
        if 0 < end_pos < len(title) - 2:
            title = title[:end_pos].strip().rstrip(":;,")
    if title == title.lower() and len(title) > 1:
        title = _smart_title_case(title)
    return title


def _isbn10_to_13(isbn10: str) -> str:
    """Convert an ISBN-10 to ISBN-13 by prepending 978 and recalculating the check digit."""
    base = "978" + isbn10[:9]
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(base))
    check = (10 - total % 10) % 10
    return base + str(check)


_BAD_AUTHOR_PATTERNS = [
    "publishing", "press", "books llc", "books ltd", "publications",
    "editions", "publishers", "verlag", "editora", "éditions",
    "house", "media", "group", "corporation", "company",
]


_OL_FIELDS = "title,author_name,language,isbn,first_publish_year,subject,cover_i,key"


def _query_openlibrary_structured(raw_title: str, raw_author: str | None = None) -> list[dict]:
    """DISABLED — restore urllib GET to OPENLIBRARY_SEARCH_URL (structured params) when re-enabling OL."""
    return []


def _query_openlibrary_freetext(raw_title: str, raw_author: str | None = None) -> list[dict]:
    """DISABLED — restore urllib GET with q= freetext when re-enabling OL."""
    return []


def _author_prefix_match(raw_author: str, candidate_authors: list[str], min_prefix: int = 4) -> bool:
    """Check if any word in raw_author shares a prefix (4+ chars) with any word in candidate authors."""
    raw_words = re.sub(r"[^a-z ]", "", raw_author.lower()).split()
    cand_words = re.sub(r"[^a-z ]", "", " ".join(candidate_authors).lower()).split()
    for rw in raw_words:
        if len(rw) < min_prefix:
            continue
        prefix = rw[:min_prefix]
        for cw in cand_words:
            if cw.startswith(prefix):
                return True
    return False


def _pick_clean_author(authors: list[str], raw_author: str | None = None) -> str:
    """Return the first author that isn't a publisher/corporate name."""
    for a in authors:
        a_lower = a.strip().lower()
        if any(p in a_lower for p in _BAD_AUTHOR_PATTERNS):
            continue
        return a.strip()
    return (raw_author or "")


def _score_docs(docs: list[dict], raw_title: str, raw_author: str | None = None) -> list[tuple[int, dict]]:
    """Score and filter Open Library search results, including author similarity."""
    raw_norm = re.sub(r"[^a-z0-9 ]", "", raw_title.lower()).strip()
    scored: list[tuple[int, dict]] = []
    for doc in docs:
        title = (doc.get("title") or "").strip()
        title_lower = title.lower()
        authors = doc.get("author_name") or []
        author_str_lower = " ".join(authors).lower()

        if any(p in title_lower for p in _DERIVATIVE_TITLE_PATTERNS):
            continue
        if any(p in author_str_lower for p in _DERIVATIVE_AUTHOR_PATTERNS):
            continue

        score = 0
        langs = doc.get("language") or []
        if "eng" in langs:
            score += 10
        if " / " not in title:
            score += 5
        title_norm = re.sub(r"[^a-z0-9 ]", "", title_lower).strip()
        if raw_norm in title_norm or title_norm in raw_norm:
            score += 8
        if title_norm.startswith(raw_norm):
            score += 5
        len_diff = abs(len(title_norm) - len(raw_norm))
        score -= min(len_diff, 20)

        if raw_author and authors:
            if _author_prefix_match(raw_author, authors):
                score += 20
            else:
                score -= 5
                if any(p in author_str_lower for p in _BAD_AUTHOR_PATTERNS):
                    score -= 15

        scored.append((score, doc))
    return scored


def _resolve_book_openlibrary(raw_title: str, raw_author: str | None = None) -> dict:
    """DISABLED — was: full Open Library resolution. Restore implementation from git when re-enabling OL."""
    return {
        "title": raw_title or "",
        "author": raw_author or "",
        "isbn": "",
        "publish_year": "",
        "openlibrary_key": "",
        "cover_url": "",
        "subjects": "",
    }


_BOOK_VALIDATION_MODEL = os.getenv("OPENROUTER_BOOK_VALIDATION_MODEL", "openai/gpt-4.1-mini")

_BOOK_TOOL_NORMALIZE_PROMPT = """The user mentioned these books in conversation. For each entry, output a clean standard English title and the primary author (fix typos and casing for well-known books).
Return ONLY a JSON array — same length and order as the input. Each element: {"title":"...","author":"..."}.
Use "" for author only if unknown. No markdown, no explanation.

Input:
"""


def resolve_books_via_openlibrary(
    books: list[dict],
) -> list[dict]:
    """
    Normalize title/author via OpenRouter (OPENROUTER_BOOK_VALIDATION_MODEL), then save_resolved_books.
    Open Library HTTP is stubbed; re-enable by restoring _query_openlibrary_* and _resolve_book_openlibrary bodies.
    """
    resolved: list[dict] = []
    for raw in books:
        if not isinstance(raw, dict):
            continue
        raw_title = (raw.get("raw_title") or "").strip()
        raw_author = (raw.get("raw_author") or "").strip() or None
        if not raw_title:
            continue
        resolved.append({
            "type": "book",
            "title": raw_title,
            "author": raw_author,
            "raw_title": raw_title,
            "raw_author": raw_author,
            "liked": raw.get("liked", True),
            "note": (raw.get("note") or "").strip() or None,
            "isbn": "",
            "publish_year": "",
            "openlibrary_key": "",
            "cover_url": "",
            "subjects": "",
        })

    if not resolved or not openrouter_api_configured():
        return resolved

    batch_size = 12
    for start in range(0, len(resolved), batch_size):
        batch = resolved[start : start + batch_size]
        payload = [
            {
                "raw_title": b.get("raw_title", ""),
                "raw_author": str(b.get("raw_author") or ""),
            }
            for b in batch
        ]
        prompt = _BOOK_TOOL_NORMALIZE_PROMPT + json.dumps(payload, ensure_ascii=False)
        try:
            raw = _openrouter_chat_completion(
                prompt,
                model=_BOOK_VALIDATION_MODEL,
                temperature=0.1,
                timeout_sec=45,
                max_tokens=2048,
            )
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            corrections = json.loads(raw)
            if not isinstance(corrections, list) or len(corrections) != len(batch):
                print(
                    f"[backend] Book normalize batch: expected {len(batch)} items, got "
                    f"{len(corrections) if isinstance(corrections, list) else 'non-list'} — skip"
                )
                continue
            for k, correction in enumerate(corrections):
                if not isinstance(correction, dict):
                    continue
                idx = start + k
                new_title = (correction.get("title") or "").strip()
                new_author = (correction.get("author") or "").strip()
                if new_title:
                    resolved[idx]["title"] = new_title
                if new_author:
                    resolved[idx]["author"] = new_author
                elif correction.get("author") == "":
                    resolved[idx]["author"] = None
        except Exception as e:
            print(f"[backend] Book normalize LLM error (non-fatal): {e}")

    return resolved


# Kept for when Open Library + second LLM pass are restored alongside catalog results.
_BOOK_VALIDATION_PROMPT = """You validate book metadata from a catalog search. Some entries have wrong authors (publisher, adapter, or summary writer instead of the real author) or wrong titles (non-English, adaptation, or subtitle appended).

For each entry, output the correct canonical English title and the original author's full name.
Return ONLY a JSON array — same length, same order. Each element: {"title":"...","author":"..."}.
If already correct, return unchanged. No markdown, no explanation.

"""


def _validate_books_via_llm(resolved: list[dict]) -> list[dict]:
    """
    Second LLM pass when Open Library catalog results need cleanup (unused while OL is disabled).
    Processes in batches of 10 to stay within token limits.
    """
    if not resolved or not openrouter_api_configured():
        return resolved

    batch_size = 10
    for start in range(0, len(resolved), batch_size):
        batch = resolved[start : start + batch_size]
        validation_input = [
            {"title": b.get("title", ""), "author": b.get("author", ""),
             "raw_title": b.get("raw_title", ""), "raw_author": b.get("raw_author", "")}
            for b in batch
        ]
        prompt = _BOOK_VALIDATION_PROMPT + json.dumps(validation_input)
        try:
            raw = _openrouter_chat_completion(
                prompt,
                model=_BOOK_VALIDATION_MODEL,
                temperature=0.0,
                timeout_sec=20,
                max_tokens=2048,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            corrections = json.loads(raw)
            if not isinstance(corrections, list) or len(corrections) != len(batch):
                print(f"[backend] Book validation batch returned {len(corrections) if isinstance(corrections, list) else 'non-list'}, expected {len(batch)} — skipping batch")
                continue
            for i, correction in enumerate(corrections):
                if not isinstance(correction, dict):
                    continue
                idx = start + i
                new_title = (correction.get("title") or "").strip()
                new_author = (correction.get("author") or "").strip()
                if new_title and new_title != resolved[idx].get("title"):
                    print(f"[backend] LLM corrected title: '{resolved[idx]['title']}' -> '{new_title}'")
                    resolved[idx]["title"] = new_title
                if new_author and new_author != resolved[idx].get("author"):
                    print(f"[backend] LLM corrected author: '{resolved[idx].get('author')}' -> '{new_author}'")
                    resolved[idx]["author"] = new_author
        except Exception as e:
            print(f"[backend] Book validation LLM error batch {start}-{start+len(batch)} (non-fatal): {e}")
    return resolved


def save_resolved_books(resolved: list[dict], instance_id: str = "") -> tuple[int, list[str]]:
    """
    Save Open-Library-resolved books to the knowledge base, skipping duplicates.
    Dedup by ISBN first, then by normalized title+author.
    Returns (count_saved, labels).
    """
    import vec_store

    saved = 0
    skipped = 0
    labels: list[str] = []
    for book in resolved:
        title = book.get("title", "")
        author = book.get("author")
        isbn = book.get("isbn", "")
        if not title:
            continue
        if vec_store.consumed_book_exists(
            isbn=isbn, title=title, author=author or "", instance_id=instance_id,
        ):
            skipped += 1
            print(f"[backend] Skipping duplicate book: {title} (ISBN {isbn or 'n/a'})")
            continue
        try:
            add_consumed(
                "book",
                title,
                author=author or None,
                liked=bool(book.get("liked", True)),
                note=book.get("note"),
                instance_id=instance_id,
                isbn=isbn,
                publish_year=book.get("publish_year", ""),
                openlibrary_key=book.get("openlibrary_key", ""),
                cover_url=book.get("cover_url", ""),
                subjects=book.get("subjects", ""),
            )
            saved += 1
            label = f"book: {title[:80]}"
            if author:
                label += f" ({author[:60]})"
            labels.append(label)
        except Exception as e:
            print(f"[backend] save_resolved_books error for '{title}': {e}")
    if skipped:
        labels.append(f"({skipped} already in library)")
    return saved, labels


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
    out: dict[str, list[dict]] = {"books": [], "podcasts": [], "articles": [], "research": [], "news": []}
    try:
        rows = vec_store.list_consumed_rows(max_items=max_items, instance_id=instance_id)
        type_to_key = {
            "book": "books",
            "podcast": "podcasts",
            "article": "articles",
            "research": "research",
            "news": "news",
        }
        for r in rows:
            key = type_to_key.get(r["type"])
            if not key:
                continue
            if not (r.get("title") or "").strip():
                continue
            item = {
                "id": r.get("id", ""),
                "title": r.get("title", "?"),
                "author": r.get("author", ""),
                "date_completed": r.get("date_completed", ""),
                "note": r.get("note", ""),
            }
            if r.get("isbn"):
                item["isbn"] = r["isbn"]
            if r.get("publish_year"):
                item["publish_year"] = r["publish_year"]
            if r.get("openlibrary_key"):
                item["openlibrary_key"] = r["openlibrary_key"]
            if r.get("cover_url"):
                item["cover_url"] = r["cover_url"]
            if r.get("subjects"):
                item["subjects"] = r["subjects"]
            out[key].append(item)
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


def _normalize_transcript_text(text: str) -> str:
    return "\n".join(line.strip() for line in (text or "").splitlines()).strip()


def _content_hash_normalized(text: str) -> str:
    return hashlib.sha256(_normalize_transcript_text(text).encode("utf-8")).hexdigest()


def _chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 120,
) -> list[str]:
    """Split journal text into overlapping chunks for embedding."""
    t = _normalize_transcript_text(text)
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    chunks: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        end = min(n, i + max_chars)
        piece = t[i:end]
        if end < n:
            cut = piece.rfind("\n\n")
            if cut > max_chars // 2:
                piece = piece[:cut].strip()
                end = i + cut
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        if end >= n:
                        break
        i = max(i + 1, end - overlap)
    return chunks


def _extract_session_data(transcript: str) -> dict:
    """Structured memory extraction (see `extraction.run.extract_journal_transcript`)."""
    from extraction.run import extract_journal_transcript

    data, _raw = extract_journal_transcript(transcript)
    return data


def _promote_globals_to_media_profile(instance_id: str, structured_facts: list) -> None:
    """Append high-confidence global facts into user_media_profile (deduped, capped)."""
    import vec_store

    hi: list[str] = []
    for f in structured_facts or []:
        if not isinstance(f, dict):
            continue
        if (f.get("scope") or "entry") != "global":
            continue
        try:
            conf = float(f.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.85:
            continue
        t = (f.get("text") or "").strip()
        if t:
            hi.append(t[:500])
    if not hi:
        return
    prof = vec_store.user_media_profile_get(instance_id or "")
    gfs = prof.get("high_confidence_globals")
    if not isinstance(gfs, list):
        gfs = []
    seen = {str(x).strip().lower() for x in gfs if isinstance(x, str)}
    for t in hi:
        key = t.lower()
        if key not in seen:
            gfs.append(t)
            seen.add(key)
    gfs = gfs[-50:]
    vec_store.user_media_profile_merge_json(instance_id or "", {"high_confidence_globals": gfs})


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

    try:
        for item in vec_store.list_journal_entries_with_ids():
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

    try:
        emb = _embed_texts([person])[0]
        for ch in vec_store.query_journal_chunks(emb, "", k=12):
            t = (ch.get("chunk_text") or "").strip()
            if t:
                passages.append(t)
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


def _bump_ingest_and_maybe_rolling_summary(instance_id: str) -> None:
    """Increment ingest counter; every N successful ingests refresh rolling_user_summary in profile."""
    import vec_store

    inst = instance_id or ""
    prof = vec_store.user_media_profile_get(inst)
    n = int(prof.get("ingest_count_since_summary") or 0) + 1
    vec_store.user_media_profile_merge_json(inst, {"ingest_count_since_summary": n})
    threshold = int((os.getenv("ROLLING_SUMMARY_INGEST_THRESHOLD") or "5").strip() or "5")
    if n < threshold:
        return
    rows = vec_store.list_journal_entries_with_ids(inst)[:10]
    texts = [(r.get("document") or "").strip() for r in rows if r.get("document")]
    blob = "\n\n".join(f"- {t[:900]}" for t in texts)[:12_000]
    prof2 = vec_store.user_media_profile_get(inst)
    t0 = time.perf_counter()
    prompt = f"""From these recent journal summaries and the profile JSON, write 2–3 short paragraphs describing who this person seems to be right now: cares, stressors, focus, tentative patterns. Invitational tone. No diagnosis or identity absolutes.

Summaries:
{blob}

Profile (JSON):
{json.dumps(prof2, ensure_ascii=False)[:4000]}
"""
    summary_text = (_call_gemini(prompt) or "").strip()[:25_000]
    ms = int((time.perf_counter() - t0) * 1000)
    vec_store.user_media_profile_merge_json(
        inst,
        {"rolling_user_summary": summary_text, "ingest_count_since_summary": 0},
    )
    DecisionLogger.log_profile_update(
        instance_id=inst,
        input_summary="rolling_user_summary regenerated",
        llm_prompt_summary=prompt[:8000],
        llm_response=summary_text[:8000],
        final_output=summary_text[:2000],
        reasoning_notes=f"threshold={threshold} episodic_chunks={len(texts)}",
        duration_ms=ms,
        model_used=extraction_llm_backend(),
    )


def process_content_feedback(
    instance_id: str,
    *,
    content_title: str,
    content_type: str = "article",
    content_url: str | None = None,
    feedback: str = "liked",
    user_notes: str | None = None,
) -> dict:
    """LLM tags/reasoning, persist content_feedback, sync rec_feedback + library."""
    import vec_store

    inst = instance_id or ""
    _ensure_storage()
    jrows = vec_store.list_journal_entries_with_ids(inst)[:5]
    ej = [{"excerpt": (e.get("document") or "")[:500], "date": e.get("timestamp")} for e in jrows]
    title = (content_title or "").strip()[:500]
    notes = (user_notes or "").strip()[:4000]
    fb = (feedback or "liked").strip().lower()
    if fb not in ("liked", "disliked", "loved", "not_relevant"):
        fb = "liked"
    t0 = time.perf_counter()
    prompt = f"""The user gave feedback on content they consumed. Output ONLY valid JSON:
{{
  "tags": ["3-5 short topic tags"],
  "reasoning": "1-3 sentences: why they might feel this way, tentatively",
  "journal_themes": ["0-4 short theme hooks connecting to journal excerpts if plausible; else empty array"]
}}

Feedback: {fb}
Type: {content_type}
Title: {title}
Notes: {notes or "(none)"}

Recent journal excerpts (for connection only):
{json.dumps(ej, ensure_ascii=False)[:6000]}
"""
    raw_llm = (_call_gemini(prompt) or "").strip()
    tags_json: list = []
    reasoning = ""
    themes_json: list = []
    try:
        txt = raw_llm
        if txt.startswith("```"):
            parts = txt.split("```")
            txt = (parts[1] if len(parts) > 1 else txt).strip()
            if txt.startswith("json"):
                txt = txt[4:].lstrip()
        data = json.loads(txt) if txt else {}
        if isinstance(data.get("tags"), list):
            tags_json = [str(x).strip() for x in data["tags"] if x][:8]
        reasoning = (data.get("reasoning") or "").strip()[:8000]
        if isinstance(data.get("journal_themes"), list):
            themes_json = [str(x).strip() for x in data["journal_themes"] if x][:8]
    except Exception as e:
        print("[backend] process_content_feedback JSON:", e)
        tags_json = []
        reasoning = ""
        themes_json = []
    tags_str = json.dumps(tags_json, ensure_ascii=False)
    themes_str = json.dumps(themes_json, ensure_ascii=False)
    try:
        vec_store.content_feedback_insert(
            inst,
            content_url=(content_url or None),
            content_title=title,
            content_type=(content_type or "article")[:40],
            feedback=fb,
            user_notes=notes or None,
            extracted_tags=tags_str,
            extracted_reasoning=reasoning or None,
            connected_journal_themes=themes_str,
        )
    except Exception as e:
        print("[backend] content_feedback_insert:", e)
    try:
        action = "like" if fb in ("liked", "loved") else "dislike" if fb == "disliked" else "not_for_me"
        record_rec_feedback_for_recs(
            inst,
            action,
            content_type=content_type,
            item_title=title,
            topic_tags=", ".join(tags_json[:5]) if tags_json else title[:120],
        )
    except Exception as e:
        print("[backend] record_rec_feedback_for_recs from content feedback:", e)
    liked_bool = fb in ("liked", "loved")
    try:
        add_consumed(
            content_type or "article",
            title,
            author=None,
            url=(content_url or "").strip()[:2000] or None,
            liked=liked_bool,
            note=(user_notes or None),
            instance_id=inst,
        )
    except Exception as e:
        print("[backend] add_consumed from feedback:", e)
    ms = int((time.perf_counter() - t0) * 1000)
    DecisionLogger.log_feedback_processing(
        instance_id=inst,
        input_summary=f"title={title[:200]} feedback={fb}",
        llm_prompt_summary=prompt[:8000],
        llm_response=raw_llm[:8000],
        final_output=f"tags={tags_str[:500]}",
        reasoning_notes=f"journal_themes={themes_str[:500]}",
        duration_ms=ms,
        model_used=extraction_llm_backend(),
    )
    return {"ok": True, "tags": tags_json, "reasoning": reasoning, "journal_themes": themes_json}


def ingest_journal_entry(
    session_id: str,
    transcript: str,
    entry_date: str | None = None,
    instance_id: str = "",
    content_hash: str | None = None,
) -> dict:
    """
    Replace-by-session_id: delete prior rows for this journal session, chunk raw text, embed, store in vec_journal.
    No LLM extraction on ingest.
    """
    import vec_store

    _ = content_hash
    inst = instance_id or ""
    norm = _normalize_transcript_text(transcript)
    if not norm:
        return {
            "summary": "",
            "facts": [],
            "metadata": {},
            "structured_facts": [],
            "skipped": False,
            "chunks": 0,
            "entry_id": None,
        }

    _ensure_storage()
    vec_store.journal_delete_by_session(inst, session_id)

    ed = datetime.utcnow().strftime("%Y-%m-%d")
    if entry_date:
        try:
            s = entry_date.strip()[:26].replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            ed = dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    chunks = _chunk_text(norm)
    entry_id = vec_store.journal_entry_insert(
        instance_id=inst,
        session_id=session_id or "",
        entry_date=ed,
        raw_text=norm,
    )
    embs: list[list[float]] = []
    if chunks:
        embs = _embed_texts(chunks)
    n = 0
    for i, ch in enumerate(chunks):
        emb = embs[i] if i < len(embs) else []
        vec_store.journal_chunk_insert(
            entry_id,
            instance_id=inst,
            chunk_index=i,
            chunk_text=ch,
            entry_date=ed,
            embedding=emb,
        )
        n += 1
    try:
        DecisionLogger._write(
            instance_id=inst,
            session_id=session_id,
            action_type="ingest",
            input_summary=f"ingest journal session_id={session_id} chunks={n} entry_date={ed}",
            llm_response=None,
            final_output=f"entry_id={entry_id} char_count={len(norm)}",
            reasoning_notes=json.dumps(
                {"chunk_count": n, "chunk_previews": [c[:120] for c in chunks[:5]]},
                ensure_ascii=False,
            )[:8000],
            duration_ms=None,
            model_used=None,
        )
    except Exception:
        pass
    try:
        _bump_ingest_and_maybe_rolling_summary(inst)
    except Exception as e:
        print("[backend] _bump_ingest_and_maybe_rolling_summary:", e)

    return {
        "summary": "",
        "facts": [],
        "metadata": {},
        "structured_facts": [],
        "skipped": False,
        "chunks": n,
        "entry_id": entry_id,
    }


def save_session_data(
    session_id: str,
    transcript: str,
    entry_date: str | None = None,
    instance_id: str = "",
    content_hash: str | None = None,
) -> dict:
    """Backward-compatible name for ingest_journal_entry."""
    return ingest_journal_entry(
        session_id, transcript, entry_date, instance_id, content_hash
    )


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


def _rerank_with_recency_dist(
    items: list[tuple[str, str, float]], k: int, now: datetime
) -> list[tuple[str, str, float, float]]:
    """Rerank (doc, ts, dist) by similarity=(1-dist) * exponential recency. Returns (doc, ts, dist, score)."""
    scored: list[tuple[float, str, str, float]] = []
    for doc, ts, dist in items:
        days_ago = 999.0
        dt = _parse_iso_date(ts)
        if dt:
            try:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_ago = max(0.0, (now - dt).total_seconds() / 86400)
            except Exception:
                pass
        sim = max(0.0, 1.0 - float(dist))
        rec_w = math.exp(-0.02 * days_ago)
        score = sim * rec_w
        scored.append((score, doc, ts, dist))
    scored.sort(key=lambda x: -x[0])
    return [(d, t, di, sc) for sc, d, t, di in scored[:k]]


def _build_processed_context_block(instance_id: str) -> str:
    """Rolling summary, traits, preferences, feedback themes, optional on-this-day."""
    import vec_store

    inst = instance_id or ""
    d = vec_store.user_media_profile_get(inst)
    parts: list[str] = []
    rus = (d.get("rolling_user_summary") or "").strip()
    if rus:
        parts.append(rus)
    gfs = d.get("high_confidence_globals")
    if isinstance(gfs, list) and gfs:
        lines = [f"- {x}" for x in gfs[:25] if x]
        if lines:
            parts.append("Key interests / traits:\n" + "\n".join(lines))
    cp = d.get("content_preferences") if isinstance(d.get("content_preferences"), dict) else {}
    if cp:
        slines: list[str] = []
        subs = cp.get("subscriptions") or []
        if subs:
            slines.append("Subscriptions (paywall exceptions): " + ", ".join(str(s) for s in subs[:15]))
        pol = (cp.get("paywall_policy") or "").strip()
        if pol:
            slines.append(f"Paywall policy: {pol}")
        ptypes = cp.get("preferred_types") or []
        if ptypes:
            slines.append("Preferred types: " + ", ".join(str(x) for x in ptypes))
        av = cp.get("avoid_types") or []
        if av:
            slines.append("Avoid types: " + ", ".join(str(x) for x in av))
        if slines:
            parts.append("Content preferences:\n" + "\n".join(slines))
    ft = d.get("feedback_themes")
    if isinstance(ft, list) and ft:
        parts.append("Recent feedback themes:\n" + "\n".join(f"- {x}" for x in ft[:15] if x))
    try:
        otd = vec_store.query_this_day_in_history(inst)
        if len(otd) >= 1:
            o_lines = []
            for row in otd[:6]:
                doc = (row.get("document") or "").strip()[:300]
                ts = (row.get("timestamp") or "")[:10]
                if doc:
                    o_lines.append(f"- [{ts}] {doc}")
            if o_lines:
                parts.append("On this day (prior years):\n" + "\n".join(o_lines))
    except Exception:
        pass
    return "\n\n".join(parts).strip()


def get_relevant_context_dual(
    query: str,
    top_k_gist: int = 8,
    top_k_episodic: int = 5,
    instance_id: str = "",
    *,
    session_id: str | None = None,
    log: bool = True,
) -> tuple[str, str]:
    """
    (processed_block, raw_block): prefs/on-this-day block + vector-retrieved journal chunk excerpts.
    top_k_gist / top_k_episodic are summed for total chunk budget (backward-compatible kwargs).
    """
    import vec_store

    t0 = time.perf_counter()
    processed = _build_processed_context_block(instance_id)
    if not query or not query.strip():
        raw = "None."
        if log:
            DecisionLogger.log_context_retrieval(
                instance_id=instance_id or "",
                session_id=session_id,
                query="",
                retrieved_items=[],
                final_output=(processed + "\n\n" + raw).strip() if processed else raw,
                reasoning_notes="empty query; raw block None",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        return processed, raw

    _ensure_storage()
    query_emb = _embed_texts([query.strip()])[0]
    k = max(4, min(top_k_gist + top_k_episodic, 24))
    retrieved_log: list[dict] = []
    parts: list[str] = []
    used_sql_fallback = False
    try:
        rows = vec_store.query_journal_chunks(
            query_emb, instance_id or "", k=k
        )
        lines: list[str] = []
        for rank, ch in enumerate(rows):
            txt = (ch.get("chunk_text") or "").strip()
            ed = (ch.get("entry_date") or "").strip()
            dist = float(ch.get("distance") or 0.0)
            sim = max(0.0, 1.0 - dist)
            retrieved_log.append(
                {
                    "content": txt[:2000],
                    "score": round(sim, 5),
                    "similarity": round(sim, 5),
                    "source": "journal_chunk",
                    "chunk_id": ch.get("chunk_id"),
                    "entry_id": ch.get("entry_id"),
                    "timestamp": ed,
                    "rerank_order": rank,
                }
            )
            if txt:
                lines.append(f"[{ed}] {txt}" if ed else txt)
        if lines:
            parts.append(
                "Relevant excerpts from the user's journals:\n" + "\n".join(f"- {ln}" for ln in lines)
            )
    except Exception as e:
        print("[backend] get_relevant_context_dual journal chunks:", e)
    if not parts:
        try:
            recent = vec_store.list_journal_entries_with_ids(instance_id or "")
            lines_fb: list[str] = []
            for row in recent[:8]:
                doc = (row.get("document") or "").strip()
                ts = (row.get("timestamp") or "").strip()
                if not doc:
                    continue
                cap = 1500
                snippet = doc[:cap] + ("…" if len(doc) > cap else "")
                lines_fb.append(f"[{ts}] {snippet}" if ts else snippet)
                retrieved_log.append(
                    {
                        "content": snippet[:2000],
                        "score": 0.0,
                        "similarity": 0.0,
                        "source": "journal_entry_fallback",
                        "entry_id": row.get("id"),
                        "timestamp": ts,
                    }
                )
            if lines_fb:
                used_sql_fallback = True
                parts.append(
                    "Recent journal entries (recency fallback when vector hits were empty; excerpts may be truncated):\n"
                    + "\n".join(f"- {ln}" for ln in lines_fb)
                )
        except Exception as e:
            print("[backend] get_relevant_context_dual journal SQL fallback:", e)
    raw = "\n\n".join(parts) if parts else "None."
    ms = int((time.perf_counter() - t0) * 1000)
    if log:
        notes = "sqlite-vec journal chunks (cosine distance)"
        if used_sql_fallback:
            notes += "; recent-entry SQL fallback"
        DecisionLogger.log_context_retrieval(
            instance_id=instance_id or "",
            session_id=session_id,
            query=query.strip(),
            retrieved_items=retrieved_log,
            final_output=((processed + "\n\n" + raw).strip() if processed else raw),
            reasoning_notes=notes,
            duration_ms=ms,
        )
    return processed, raw


def get_relevant_context(query: str, top_k_gist: int = 8, top_k_episodic: int = 5, instance_id: str = "") -> str:
    """Backward-compatible: processed profile block + raw vector hits."""
    processed, raw = get_relevant_context_dual(
        query, top_k_gist, top_k_episodic, instance_id, session_id=None, log=True
    )
    if (not processed.strip()) and ((not raw.strip()) or raw == "None."):
        return "None."
    if not processed.strip():
        return raw
    if not raw.strip() or raw == "None.":
        return processed
    return processed + "\n\n" + raw


def get_memory_for_visualization(instance_id: str = "") -> tuple[list[str], list[str]]:
    """Return (journal_entry_bodies, []) for diagram generation."""
    import vec_store

    _ensure_storage()
    try:
        rows = vec_store.list_journal_entries_with_ids(instance_id)
        docs = [(r.get("document") or "").strip() for r in rows if r.get("document")]
        return (docs, [])
    except Exception:
        return ([], [])


def get_memory_for_date(date_iso: str, instance_id: str = "") -> tuple[list[dict], list[dict]]:
    """Return (journal entries for date, []) for the given date (YYYY-MM-DD)."""
    import vec_store

    _ensure_storage()
    if not date_iso or len(date_iso) < 10:
        return ([], [])
    d = date_iso[:10]
    try:
        episodic = vec_store.journal_entries_for_date_range(
            instance_id, d, d, limit=80
        )
    except Exception as e:
        print("[backend] journal_entries_for_date_range error:", e)
        episodic = []
    return (episodic, [])


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
    """Return journal entries for Memory UI (legacy route name)."""
    import vec_store

    _ensure_storage()
    try:
        rows = vec_store.list_journal_entries_with_ids(instance_id=instance_id)
        return [
            {
                **r,
                "metadata_json": None,
            }
            for r in rows
        ]
    except Exception as e:
        print("[backend] list_memory_facts error:", e)
        return []


def list_memory_summaries(instance_id: str = "") -> list[dict]:
    """Legacy episodic route; journal system stores a single entry stream — return []."""
    return []


def get_person_events(person_name: str) -> list[dict]:
    """Best-effort: journal rows that mention the person in the raw text."""
    items = list_memory_facts()
    if not person_name or not person_name.strip():
        return []
    target_l = person_name.strip().lower()
    results: list[dict] = []
    for item in items:
        doc = (item.get("document") or "").strip()
        if not doc or target_l not in doc.lower():
            continue
        results.append(
            {
                "summary_id": item.get("id"),
                "timestamp": item.get("timestamp") or "",
                "events": [doc[:500]],
            }
        )
    return results


def _update_journal_entry_by_id(entry_id: int, document: str, instance_id: str = "") -> bool:
    import vec_store

    row = vec_store.journal_entry_get(entry_id, instance_id)
    if not row:
        return False
    sid = row.get("session_id") or ""
    ingest_journal_entry(sid, document, row.get("timestamp"), instance_id)
    return True


def update_memory_fact(fact_id: int, document: str) -> bool:
    """Update journal entry by id (re-chunk + re-embed)."""
    _ensure_storage()
    if not document or not document.strip():
        return False
    return _update_journal_entry_by_id(fact_id, document, "")


def update_memory_summary(
    summary_id: int, document: str, metadata: dict | None = None
) -> bool:
    _ = metadata
    return update_memory_fact(summary_id, document)


def delete_memory_fact(fact_id: int) -> bool:
    import vec_store

    _ensure_storage()
    return vec_store.journal_entry_delete_cascade(fact_id, "")


def delete_memory_summary(summary_id: int) -> bool:
    return delete_memory_fact(summary_id)


def add_memory_fact(document: str, session_id: str | None = None, instance_id: str = "") -> int | None:
    """Add a user note as a new journal entry; returns entry id."""
    import uuid

    _ensure_storage()
    if not document or not document.strip():
        return None
    sid = session_id or f"user-{uuid.uuid4().hex[:12]}"
    out = ingest_journal_entry(sid, document, None, instance_id)
    eid = out.get("entry_id")
    return int(eid) if eid else None


def add_memory_summary(document: str, session_id: str | None = None, instance_id: str = "") -> int | None:
    return add_memory_fact(document, session_id, instance_id)


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
    _rec_search_log_append(
        "perplexity_search",
        query.strip(),
        len(out),
        [x.get("url") or "" for x in out],
    )
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


def _subscription_hosts_normalized(subscriptions: list[str] | None) -> set[str]:
    if not subscriptions:
        return set()
    out: set[str] = set()
    for s in subscriptions:
        t = (s or "").lower().strip()
        if t.startswith("www."):
            t = t[4:]
        if t:
            out.add(t)
    return out


def _paywall_exclude_domains(subscriptions: list[str] | None) -> list[str]:
    """Domains to exclude from search APIs; user subscriptions are removed from the blocklist."""
    if not subscriptions:
        return list(PAYWALLED_ARTICLE_DOMAINS)
    subs = _subscription_hosts_normalized(subscriptions)
    return [d for d in PAYWALLED_ARTICLE_DOMAINS if d.lower() not in subs]


def _filter_hits_basic(
    hits: list[dict], consumed_lower: str, subscriptions: list[str] | None = None
) -> list[dict]:
    """Drop paywalled domains, video/replay junk, and obvious consumed URL/title overlaps."""
    out: list[dict] = []
    for h in hits:
        url = h.get("url") or ""
        title = h.get("title") or ""
        if _is_paywalled_domain(url, subscriptions):
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


def _books_agent(
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
    recent_summaries_blob: str = "",
    subscriptions: list[str] | None = None,
) -> list:
    """Book recommendations via Perplexity Search + Gemini structuring (no duplicate titles vs consumed)."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are a book curator. Based on this person's journal-derived memory and consumed library, suggest 3–5 books (title + author + one-sentence reason). No URLs.

JOURNAL CONTEXT:
{facts_blob or "(none yet)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title": "...", "author": "...", "reason": "..."}}, ...]"""
        return _parse_recommendation_json(_call_library_rec_web(prompt), [])

    q_prompt = f"""Suggest 3 short **web search** queries to discover **nonfiction or fiction books** (lists, reviews, "best books on…") for this person. Queries must help find real book titles and authors on the open web.
Respect **did not enjoy** rows in CONSUMED — steer away from similar themes or authors they rejected.

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY a JSON array of 3 strings. Example: ["best psychology books for anxiety 2024", "literary fiction grief healing"]"""
    queries = _parse_json_string_list(_call_library_rec_fast(q_prompt))
    if len(queries) < 2:
        queries = ["best nonfiction books personal growth highly rated", "literary fiction book recommendations deep characters"]
    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        queries[:4],
        max_per_query=5,
        search_recency_filter="year",
        max_total_hits=22,
    )
    hits = _filter_hits_basic(hits, consumed_lower, subscriptions)
    blob = _search_hits_context_lines(hits)
    if not blob.strip():
        prompt = f"""Suggest 3–5 books for this person (title, author, reason). {recent_section}{consumed[:4000]}
facts: {facts_blob[:2000]}
summaries: {summaries_blob[:2000]}
Return ONLY JSON array: [{{"title","author","reason","url"}}] url optional."""
        return _parse_recommendation_json(_call_library_rec_web(prompt), [])

    curate = f"""You are a book curator. Pick **3–5 distinct books** for this user using ONLY information grounded in SEARCH_RESULTS (real books mentioned there). Map each to canonical **title** and **author** as in the source; **url** must be copied exactly from SEARCH_RESULTS when present (review, publisher, Goodreads, etc.) or "".
Tie each **reason** to the user's themes in CONTEXT; keep reasons one sentence, honest.

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:5000]}

SEARCH_RESULTS:
{blob}

Rules: Do NOT output books that appear in CONSUMED as already read. Avoid books/authors clearly similar to **did not enjoy** notes.
Return ONLY JSON: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    out = _parse_recommendation_json(_call_library_rec_fast(curate), [])
    # Fallback if Gemini returned empty
    if not out:
        return _parse_recommendation_json(_call_library_rec_web(curate), [])
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
    _rec_search_log_append("listen_notes", query, len(out), [x.get("url") or "" for x in out])
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


def _podcasts_agent(
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
    recent_summaries_blob: str = "",
    subscriptions: list[str] | None = None,
) -> list:
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


def _is_paywalled_domain(url: str, subscriptions: list[str] | None = None) -> bool:
    """True if the URL's host is paywalled, unless the user subscribes to that source."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return False
        subs = _subscription_hosts_normalized(subscriptions)
        for sub in subs:
            if host == sub or host.endswith("." + sub):
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
    subscriptions: list[str] | None = None,
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
    exclude_dom = _paywall_exclude_domains(subscriptions)
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
                exclude_domains=exclude_dom,
            )
        except Exception as e:
            print("[backend] Tavily search error:", e)
            continue
        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", []) or []
        print(f"[backend] Tavily Search API: got {len(results)} results for {query_str!r}", flush=True)
        q_urls: list[str] = []
        for r in results:
            url = (r.get("url", "") if isinstance(r, dict) else getattr(r, "url", None)) or ""
            if url:
                q_urls.append(url)
            if not url or url in seen_urls or _is_paywalled_domain(url, subscriptions):
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
        _rec_search_log_append("tavily", query_str, len(results), q_urls[:30])
    if out:
        print(f"[backend] Tavily Search API: returning {len(out)} articles (after dedupe and paywall filter)")
    return out


def _articles_agent(
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
    recent_summaries_blob: str = "",
    subscriptions: list[str] | None = None,
) -> list:
    """Informational / thought-provoking articles via Perplexity Search + personalized reasons."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are an article curator. Suggest 3–5 **informational** articles or essays (helpful, thought-provoking explainers—not clickbait). Real URLs only.

JOURNAL CONTEXT:
{facts_blob or "(none yet)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title","author","reason","url"}}, ...]"""
        return _parse_recommendation_json(_call_library_rec_web(prompt), [])

    q_prompt = f"""Suggest 2–3 short **web search** queries for **in-depth readable articles**: explainers, essays, analysis, studies written for educated readers—not breaking headline chyrons.
Honor CONSUMED: respect **did not enjoy** and negative notes.

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY JSON array of 2–3 strings. Example: ["long read climate adaptation solutions", "cognitive science of habit formation explainer"]"""
    queries = _parse_json_string_list(_call_library_rec_fast(q_prompt))
    if len(queries) < 2:
        queries = ["thought-provoking long read science society", "deep dive psychology well-being evidence"]
    consumed_lower = (consumed or "").lower()
    hits = _merge_perplexity_queries(
        queries[:3],
        max_per_query=6,
        search_recency_filter="month",
        max_total_hits=18,
    )
    hits = _filter_hits_basic(hits, consumed_lower, subscriptions)
    out: list[dict] = []
    for h in hits[:8]:
        url = h.get("url") or ""
        author = ""
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
    return _parse_recommendation_json(_call_library_rec_web(
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
        ss_urls: list[str] = []
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
            if url_val:
                ss_urls.append(url_val)
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
        _rec_search_log_append(
            "semantic_scholar",
            query_str,
            len(results) if isinstance(results, list) else 0,
            ss_urls[:25],
        )
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


def _research_agent(
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
    recent_summaries_blob: str = "",
    subscriptions: list[str] | None = None,
) -> list:
    """Research papers via Perplexity Search (scholarly URLs preferred) + grounded reasons."""
    recent_section = _recent_journals_block(recent_summaries_blob)
    if not _pplx_search_api_key():
        prompt = f"""You are a research curator. Suggest 3–5 peer-reviewed or preprint papers with **verified** URLs (doi.org, PubMed, arXiv, Semantic Scholar only if certain).

JOURNAL CONTEXT:
{facts_blob or "(none yet)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Return ONLY JSON: [{{"title","author","reason","url"}}, ...]"""
        return _parse_recommendation_json(_call_library_rec_web(prompt), [])

    q_prompt = f"""Suggest 2–3 short **web search** queries to find **peer-reviewed research papers, systematic reviews, or reputable preprints** (PubMed, arXiv, journal DOI pages). Be concrete; include methodology terms when useful (RCT, meta-analysis, cohort).
Honor CONSUMED — avoid subfields or angles the user disliked.

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:6000]}

Return ONLY JSON array of 2–3 strings."""
    queries = _parse_json_string_list(_call_library_rec_fast(q_prompt))
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
    hits = _prefer_scholarly_hits(_filter_hits_basic(hits, consumed_lower, subscriptions))
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
JOURNAL CONTEXT: {facts_blob[:2000]}
RECENT SESSION SUMMARIES: {summaries_blob[:2000]}
Return ONLY JSON array."""
    return _parse_recommendation_json(_call_library_rec_web(prompt), [])


def _news_agent(
    facts_blob: str,
    summaries_blob: str,
    consumed: str,
    recent_summaries_blob: str = "",
    subscriptions: list[str] | None = None,
) -> list:
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

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
{summaries_blob or "(none)"}
{recent_section}{consumed[:5500]}

Return ONLY a JSON array of 1–2 strings."""

        extra = _parse_json_string_list(_call_gemini(gq))
        for q in extra[:2]:
            if q and q not in queries:
                queries.append(q)

    if not _pplx_search_api_key():
        prompt = f"""Curate 3–6 **positive or solution-focused** news stories with verifiable URLs for this reader.

JOURNAL CONTEXT:
{facts_blob or "(none)"}

RECENT SESSION SUMMARIES:
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
    hits = _filter_hits_basic(hits, consumed_lower, subscriptions)
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


def _infer_moment_intent(instance_id: str, latest_summary: str, profile: dict) -> str:
    latest = (latest_summary or "").strip()
    if not latest and not profile:
        return ""
    _ = instance_id
    prompt = f"""In one short line (max 30 words), what kind of media or support might fit this person right now?
Use tentative wording. Do not diagnose medical or psychiatric conditions.
Examples: "low-pressure comfort and calm" or "substantive nonfiction, intellectually engaging".

Latest journal themes:
{latest[:1500]}

Structured profile notes (JSON):
{json.dumps(profile, ensure_ascii=False)[:1800]}

Output plain text only, one line."""
    try:
        return (_call_gemini(prompt) or "").strip().split("\n")[0][:400]
    except Exception as e:
        print("[backend] _infer_moment_intent:", e)
        return ""


def _profile_intent_snippets_block(profile_blob: str, intent_line: str, memory_snippets: str) -> str:
    parts = []
    if profile_blob and profile_blob.strip():
        parts.append("STABLE PROFILE (long-term tastes):\n" + profile_blob.strip()[:4000])
    if intent_line and intent_line.strip():
        parts.append("CURRENT MOMENT (soft signal):\n" + intent_line.strip()[:500])
    if memory_snippets and memory_snippets.strip():
        parts.append("RETRIEVED MEMORY SNIPPETS:\n" + memory_snippets.strip()[:3500])
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def _subscriptions_from_profile_dict(profile: dict) -> list[str]:
    cp = profile.get("content_preferences") if isinstance(profile.get("content_preferences"), dict) else {}
    subs = cp.get("subscriptions") or []
    return [str(x).strip() for x in subs if x]


def generate_recommendations_simple_llm(instance_id: str = "") -> dict:
    """Flagged path: LLM proposes search queries only; APIs return URLs."""
    import vec_store

    inst = instance_id or ""
    _ensure_storage()
    _rec_search_log_begin()
    t0 = time.perf_counter()
    feedback_rows = vec_store.content_feedback_list_recent(inst, 28)
    profile = vec_store.user_media_profile_get(inst)
    journal_docs, _ = get_memory_for_visualization(instance_id=inst)
    consumed = get_consumed_context(instance_id=inst)
    consumed_lower = (consumed or "").lower()
    subs = _subscriptions_from_profile_dict(profile)
    fb_blob = json.dumps(feedback_rows, ensure_ascii=False, default=str)[:8000]
    summ_blob = "\n".join(f"- {s}" for s in (journal_docs or [])[-12:])
    prof_blob = json.dumps(profile, ensure_ascii=False)[:4000]
    prompt = f"""You suggest media for one journaler. Output ONLY valid JSON:
{{"suggestions": [
  {{"type": "article", "search_query": "string", "why": "one short sentence"}},
  ...
]}}
Rules: 3-5 items. type is one of: article, podcast, book, research, news.
search_query must be a real web-search style query (no URLs, no invented links).
why ties to their feedback and journals when possible.

Recent content_feedback JSON:
{fb_blob}

Journal summaries:
{summ_blob}

Profile JSON:
{prof_blob}

Library/consumed excerpt:
{consumed[:2800]}
"""
    raw = (_call_library_rec_fast(prompt) or "").strip()
    suggestions: list = []
    try:
        txt = raw
        if txt.startswith("```"):
            parts = txt.split("```")
            txt = (parts[1] if len(parts) > 1 else txt).strip()
            if txt.startswith("json"):
                txt = txt[4:].lstrip()
        data = json.loads(txt) if txt else {}
        suggestions = data.get("suggestions") if isinstance(data.get("suggestions"), list) else []
    except Exception as e:
        print("[backend] simple rec JSON:", e)
        suggestions = []

    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []
    news_list: list = []
    pplx_key = _pplx_search_api_key()
    ln_key = (os.getenv("LISTENNOTES_API_KEY") or "").strip()

    for s in suggestions[:10]:
        if not isinstance(s, dict):
            continue
        typ = str(s.get("type") or "article").lower().strip()
        q = str(s.get("search_query") or "").strip()
        why = str(s.get("why") or "").strip()[:500]
        if not q:
            continue
        if typ == "podcast" and ln_key and not PODCAST_API_PAUSED:
            for item in _listen_notes_search_episodes(q, ln_key, max_results=2):
                item = dict(item)
                item["reason"] = why or item.get("reason", "")
                podcasts_list.append(item)
            continue
        if not pplx_key:
            continue
        recency = "week" if typ == "news" else "year" if typ in ("book", "research") else "month"
        hits = _merge_perplexity_queries(
            [q],
            max_per_query=5,
            search_recency_filter=recency,
            max_total_hits=8,
        )
        hits = _filter_hits_basic(hits, consumed_lower, subs)
        for h in hits[:2]:
            url = h.get("url") or ""
            title = (h.get("title") or "")[:500]
            snip = (h.get("snippet") or "")[:220]
            try:
                parsed = urllib.parse.urlparse(url)
                author = (parsed.netloc or "").lower()
                if author.startswith("www."):
                    author = author[4:]
            except Exception:
                author = ""
            row = {
                "title": title or "Suggestion",
                "author": author[:200],
                "reason": why or (snip + "…") if len(h.get("snippet") or "") > 220 else (h.get("snippet") or ""),
                "url": url,
            }
            if typ == "book":
                books_list.append({"title": title, "author": author[:200] or "Unknown", "reason": row["reason"], "url": url})
            elif typ == "research":
                research_list.append(row)
            elif typ == "news":
                news_list.append(row)
            else:
                articles_list.append(row)

    out = {
        "books": books_list[:8],
        "podcasts": podcasts_list[:8],
        "articles": articles_list[:8],
        "research": research_list[:8],
        "news": news_list[:10],
    }
    ms = int((time.perf_counter() - t0) * 1000)
    DecisionLogger.log_recommendation(
        instance_id=inst,
        input_summary=f"simple_llm path feedback_rows={len(feedback_rows)}",
        llm_prompt_summary=prompt[:8000],
        llm_response=raw[:8000],
        final_output=json.dumps(out, ensure_ascii=False)[:12000],
        reasoning_notes=f"subscription_domains={subs[:2]}",
        duration_ms=ms,
        search_api_calls=_rec_search_log_snapshot(),
        model_used=extraction_llm_backend(),
    )
    return out


_RECOMMENDATION_CATEGORY_KEYS = frozenset({"books", "podcasts", "articles", "research", "news"})


def _recommendations_agent_context(
    instance_id: str,
) -> tuple[str, str, str, str, list[str] | None, str, str]:
    """
    Build shared inputs for _books_agent / _podcasts_agent / etc.
    Returns (facts_blob, summaries_blob, consumed, recent_summaries_blob, subs, intent_line, processed_block).
    facts_blob holds merged JOURNAL CONTEXT; summaries_blob is kept for prompt compatibility (often empty).
    """
    import vec_store

    _ensure_storage()
    gist_docs, _episodic_docs = get_memory_for_visualization(instance_id=instance_id)
    consumed = get_consumed_context(instance_id=instance_id)
    profile = vec_store.user_media_profile_get(instance_id)
    profile_blob = json.dumps(profile, ensure_ascii=False, indent=2) if profile else ""
    subs = _subscriptions_from_profile_dict(profile)

    episodic_list = gist_docs or []
    latest_summary = episodic_list[-1] if episodic_list else ""
    intent_line = _infer_moment_intent(instance_id, latest_summary, profile)
    q = "\n".join(x for x in (latest_summary, intent_line) if x).strip()[:3000] or "journal themes"
    processed_block, raw_block = get_relevant_context_dual(
        q, top_k_gist=5, top_k_episodic=4, instance_id=instance_id, log=False
    )
    memory_snippets = raw_block if raw_block != "None." else ""
    processed_header = ""
    if (processed_block or "").strip():
        processed_header = "## Who this person is (processed understanding)\n" + processed_block.strip()[:4000] + "\n\n"
    enrich = _profile_intent_snippets_block(profile_blob, intent_line, memory_snippets)
    if processed_header:
        enrich = processed_header + enrich

    likes = profile.get("feedback_likes") if isinstance(profile.get("feedback_likes"), list) else []
    dislikes = profile.get("feedback_dislikes") if isinstance(profile.get("feedback_dislikes"), list) else []
    fb_lines = ""
    if likes:
        fb_lines += "\nUser feedback — leaned positive toward: " + ", ".join(str(x) for x in likes[-14:] if x)
    if dislikes:
        fb_lines += "\nUser feedback — tended to reject or hide: " + ", ".join(str(x) for x in dislikes[-14:] if x)

    entry_lines = "\n".join(f"- {f}" for f in (gist_docs or [])[:55])
    jc_parts = [enrich, fb_lines, "RAW JOURNAL ENTRY SNIPPETS (recent):", entry_lines, memory_snippets]
    facts_blob = "\n\n".join(p.strip() for p in jc_parts if (p or "").strip()).strip()

    summaries_blob = "\n".join(f"- {s}" for s in episodic_list[-22:])
    n_recent = max(8, int(len(episodic_list) * 0.25)) if episodic_list else 0
    recent_episodic = episodic_list[-n_recent:] if n_recent else []
    recent_summaries_blob = "\n".join(f"- {s}" for s in recent_episodic) if recent_episodic else ""

    return facts_blob, summaries_blob, consumed, recent_summaries_blob, subs, intent_line, processed_block or ""


def generate_recommendations_category(instance_id: str, category: str) -> dict:
    """
    Regenerate one recommendation column (single agent). Other keys are empty lists.
    When RECOMMENDATIONS_SIMPLE_LLM is on, runs the full simple path once and returns only the requested slice.
    """
    cat = (category or "").strip().lower()
    if cat not in _RECOMMENDATION_CATEGORY_KEYS:
        raise ValueError(f"invalid category {category!r}")

    if (os.getenv("RECOMMENDATIONS_SIMPLE_LLM") or "").strip().lower() in ("1", "true", "yes", "on"):
        full = generate_recommendations_simple_llm(instance_id)
        return {c: (list(full.get(c) or []) if c == cat else []) for c in _RECOMMENDATION_CATEGORY_KEYS}

    facts_blob, summaries_blob, consumed, recent_summaries_blob, subs, intent_line, processed_block = (
        _recommendations_agent_context(instance_id)
    )
    _rec_search_log_begin()
    t0 = time.perf_counter()
    agent_timeout = float(os.getenv("RECOMMENDATIONS_AGENT_TIMEOUT_SEC", "68"))
    list_out: list = []
    try:
        if cat == "books":
            list_out = _books_agent(facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        elif cat == "podcasts":
            list_out = _podcasts_agent(facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        elif cat == "articles":
            list_out = _articles_agent(facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        elif cat == "research":
            list_out = _research_agent(facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        else:
            list_out = _news_agent(facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
    except Exception as e:
        print(
            f"[backend] recommendations category={cat} error:",
            type(e).__name__,
            str(e) or "(no message)",
        )
    if not isinstance(list_out, list):
        list_out = []
    out = {c: [] for c in _RECOMMENDATION_CATEGORY_KEYS}
    out[cat] = list_out
    total_ms = int((time.perf_counter() - t0) * 1000)
    n_cons = len((consumed or "").splitlines())
    DecisionLogger.log_recommendation(
        instance_id=instance_id or "",
        input_summary=(
            f"category_refresh={cat} facts_blob_chars={len(facts_blob)} "
            f"consumed_lines~{n_cons} intent={intent_line[:120]!r}"
        ),
        retrieved_items=[
            {"kind": "processed_context", "preview": (processed_block or "")[:500]},
            {"kind": "intent", "text": intent_line},
        ],
        llm_prompt_summary=f"single_agent={cat} facts_len={len(facts_blob)}",
        final_output=json.dumps(out, ensure_ascii=False, default=str)[:12000],
        reasoning_notes=json.dumps({f"{cat}_ms": total_ms}),
        duration_ms=total_ms,
        model_used=extraction_llm_backend(),
        search_api_calls=_rec_search_log_snapshot(),
    )
    return out


def generate_recommendations(instance_id: str = "") -> dict:
    """
    Profile- and intent-aware bundle: structured user_media_profile, inferred moment intent,
    recency-weighted retrieval snippets, then parallel media agents. Feedback likes/dislikes
    in profile nudge prompts; include occasional breadth where prompts already allow exploration.
    """
    if (os.getenv("RECOMMENDATIONS_SIMPLE_LLM") or "").strip().lower() in ("1", "true", "yes", "on"):
        return generate_recommendations_simple_llm(instance_id)

    facts_blob, summaries_blob, consumed, recent_summaries_blob, subs, intent_line, processed_block = (
        _recommendations_agent_context(instance_id)
    )

    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []
    news_list: list = []

    _rec_search_log_begin()
    t0 = time.perf_counter()
    agent_timeout = float(os.getenv("RECOMMENDATIONS_AGENT_TIMEOUT_SEC", "68"))
    per_agent: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_books = executor.submit(_books_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        future_podcasts = executor.submit(_podcasts_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        future_articles = executor.submit(_articles_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        future_research = executor.submit(_research_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        future_news = executor.submit(_news_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob, subs)
        try:
            t_a = time.perf_counter()
            books_list = future_books.result(timeout=agent_timeout)
            per_agent["books_ms"] = (time.perf_counter() - t_a) * 1000
        except Exception as e:
            print(
                "[backend] recommendations books_agent error:",
                type(e).__name__,
                str(e) or "(no message)",
            )
        try:
            t_a = time.perf_counter()
            podcasts_list = future_podcasts.result(timeout=agent_timeout)
            per_agent["podcasts_ms"] = (time.perf_counter() - t_a) * 1000
        except Exception as e:
            print(
                "[backend] recommendations podcasts_agent error:",
                type(e).__name__,
                str(e) or "(no message)",
            )
        try:
            t_a = time.perf_counter()
            articles_list = future_articles.result(timeout=agent_timeout)
            per_agent["articles_ms"] = (time.perf_counter() - t_a) * 1000
        except Exception as e:
            print(
                "[backend] recommendations articles_agent error:",
                type(e).__name__,
                str(e) or "(no message)",
            )
        try:
            t_a = time.perf_counter()
            research_list = future_research.result(timeout=agent_timeout)
            per_agent["research_ms"] = (time.perf_counter() - t_a) * 1000
        except Exception as e:
            print(
                "[backend] recommendations research_agent error:",
                type(e).__name__,
                str(e) or "(no message)",
            )
        try:
            t_a = time.perf_counter()
            news_list = future_news.result(timeout=agent_timeout)
            per_agent["news_ms"] = (time.perf_counter() - t_a) * 1000
        except Exception as e:
            print(
                "[backend] recommendations news_agent error:",
                type(e).__name__,
                str(e) or "(no message)",
            )

    out = {
        "books": books_list if isinstance(books_list, list) else [],
        "podcasts": podcasts_list if isinstance(podcasts_list, list) else [],
        "articles": articles_list if isinstance(articles_list, list) else [],
        "research": research_list if isinstance(research_list, list) else [],
        "news": news_list if isinstance(news_list, list) else [],
    }
    total_ms = int((time.perf_counter() - t0) * 1000)
    n_cons = len((consumed or "").splitlines())
    DecisionLogger.log_recommendation(
        instance_id=instance_id or "",
        input_summary=(
            f"facts_blob_chars={len(facts_blob)} summaries_blob_chars={len(summaries_blob)} "
            f"consumed_lines~{n_cons} intent={intent_line[:120]!r}"
        ),
        retrieved_items=[
            {"kind": "processed_context", "preview": (processed_block or "")[:500]},
            {"kind": "intent", "text": intent_line},
        ],
        llm_prompt_summary=f"parallel agents facts+enrich len={len(facts_blob)}",
        final_output=json.dumps(out, ensure_ascii=False, default=str)[:12000],
        reasoning_notes=json.dumps(per_agent),
        duration_ms=total_ms,
        model_used=extraction_llm_backend(),
        search_api_calls=_rec_search_log_snapshot(),
    )
    return out


def get_writing_loop_hints(draft_text: str, instance_id: str = "") -> dict:
    """Similar past episodic lines plus active insights/patterns for the journal composer."""
    import vec_store

    t0 = time.perf_counter()
    _ensure_storage()
    inst = instance_id or ""
    draft = (draft_text or "").strip()
    rows: list[tuple[str, str, float]] = []
    if draft:
        try:
            emb = _embed_texts([draft[:2000]])[0]
            for ch in vec_store.query_journal_chunks(emb, inst, k=12):
                rows.append(
                    (
                        (ch.get("chunk_text") or "").strip(),
                        (ch.get("entry_date") or "").strip(),
                        float(ch.get("distance") or 0.0),
                    )
                )
        except Exception as e:
            print("[backend] get_writing_loop_hints embed/query:", e)
    now = datetime.now(timezone.utc)
    scored: list[tuple[float, str, str, float]] = []
    for doc, ts, dist in rows:
        days_ago = 999.0
        ts_norm = (ts + "T12:00:00Z") if (ts and len(ts) <= 10) else ts
        dt = _parse_iso_date(ts_norm)
        if dt:
            try:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_ago = max(0.0, (now - dt).total_seconds() / 86400)
            except Exception:
                pass
        sim = max(0.0, 1.0 - float(dist))
        sc = sim * math.exp(-0.02 * days_ago)
        scored.append((sc, doc, ts, sim))
    scored.sort(key=lambda x: -x[0])
    similar = []
    retrieved_items: list[dict] = []
    for rank, (sc, doc, ts, sim) in enumerate(scored[:8]):
        if doc:
            retrieved_items.append(
                {
                    "content": doc[:1800],
                    "score": round(float(sc), 5),
                    "similarity": round(float(sim), 5),
                    "source": "journal_chunk",
                    "timestamp": ts,
                    "rerank_order": rank,
                }
            )
        if len(similar) < 4 and doc:
            similar.append({"excerpt": doc[:450], "date": ts})
    on_this_day_nudge = ""
    try:
        otd = vec_store.query_this_day_in_history(inst)
        if otd:
            snippet = (otd[0].get("document") or "").strip()[:220]
            y = (otd[0].get("timestamp") or "")[:4]
            if snippet:
                on_this_day_nudge = (
                    f"Years ago ({y}) you wrote about something like: {snippet}… "
                    "How does that land for you now?"
                )
    except Exception:
        on_this_day_nudge = ""
    result = {
        "similar_past_entries": similar,
        "insights": vec_store.derived_insights_list_active(inst, limit=6),
        "patterns": vec_store.pattern_memory_recent(inst, limit=3),
        "on_this_day_nudge": on_this_day_nudge or None,
    }
    DecisionLogger.log_writing_hint(
        instance_id=inst,
        input_summary=f"draft_chars={len(draft)}",
        retrieved_items=retrieved_items,
        final_output=json.dumps(
            {"similar_count": len(similar), "on_this_day": bool(on_this_day_nudge)},
            ensure_ascii=False,
        ),
        reasoning_notes="journal chunk similarity * recency decay",
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
    return result


def refresh_pattern_memory(instance_id: str = "") -> dict:
    import vec_store

    _ensure_storage()
    inst = instance_id or ""
    rows = vec_store.list_journal_entries_with_ids(inst)[:40]
    texts = [(r.get("document") or "").strip() for r in rows if r.get("document")]
    if len(texts) < 3:
        return {"ok": False, "reason": "not_enough_episodic"}
    blob = "\n".join(f"- {t[:500]}" for t in texts[:35])[:14_000]
    prompt = f"""Read these journal session summaries (recent first). Note tentative patterns across time — recurring topics, mood shifts, or behaviors.
Do NOT diagnose illness. Use invitational language.

Return ONLY valid JSON: {{"summary": "2-5 sentences", "tags": ["short-tag", ...]}}

Summaries:
{blob}
"""
    raw = (_call_gemini(prompt) or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = (parts[1] if len(parts) > 1 else raw).strip()
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    json_str = raw
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    json_str = raw[start : i + 1]
                    break
    try:
        data = json.loads(json_str)
    except Exception as e:
        print("[backend] refresh_pattern_memory JSON:", e)
        return {"ok": False, "reason": "parse_error"}
    summary = (data.get("summary") or "").strip()
    tags = data.get("tags") or []
    if not summary:
        return {"ok": False, "reason": "empty_summary"}
    tid = vec_store.pattern_memory_add(
        inst,
        "recent_sessions",
        summary[:8000],
        json.dumps(tags)[:2000] if isinstance(tags, list) else None,
    )
    return {"ok": True, "pattern_id": tid}


def generate_derived_insights(instance_id: str = "") -> dict:
    import vec_store

    _ensure_storage()
    inst = instance_id or ""
    prof = vec_store.user_media_profile_get(inst)
    patterns = vec_store.pattern_memory_recent(inst, limit=4)
    episodic = vec_store.list_journal_entries_with_ids(inst)[:6]
    excerpts = [((e.get("document") or "")[:400], e.get("timestamp") or "") for e in episodic]
    prompt = f"""You support reflective journaling. Output ONLY valid JSON: {{"items": [{{"text": "...", "kind": "pattern"}}, ...]}}
kind must be one of: pattern, reflection, tension, nudge.
Rules:
- 2-4 items. Tentative phrasing ("you've sometimes...", "you might notice...").
- No clinical diagnoses or clinical labels. No "you are X" identity claims.
- One item may gently suggest variety or exploration.

Profile: {json.dumps(prof)[:2500]}
Patterns: {json.dumps(patterns)[:2500]}
Recent excerpts: {json.dumps(excerpts)[:3500]}
"""
    raw = (_call_gemini(prompt) or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = (parts[1] if len(parts) > 1 else raw).strip()
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    json_str = raw
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    json_str = raw[start : i + 1]
                    break
    try:
        data = json.loads(json_str)
    except Exception as e:
        print("[backend] generate_derived_insights JSON:", e)
        return {"ok": False, "added": 0}
    items = data.get("items") or []
    if not isinstance(items, list):
        return {"ok": False, "added": 0}
    banned = ("depress", "bipolar", "adhd diagnosis", "ptsd diagnosis", "ocd diagnosis", "you are a")
    added = 0
    for it in items[:6]:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        kind = (it.get("kind") or "reflection").strip()[:40]
        low = text.lower()
        if not text or any(b in low for b in banned):
            continue
        vec_store.derived_insight_add(inst, text[:4000], kind, None)
        added += 1
    return {"ok": True, "added": added}


def record_rec_feedback_for_recs(
    instance_id: str,
    action: str,
    *,
    content_type: str | None = None,
    topic_tags: str | None = None,
    intent_context: str | None = None,
    item_title: str | None = None,
) -> None:
    import vec_store

    vec_store.rec_feedback_record(
        instance_id or "",
        action,
        content_type=content_type,
        topic_tags=topic_tags,
        intent_context=intent_context,
        item_title=item_title,
    )
    prof = vec_store.user_media_profile_get(instance_id or "")
    likes = prof.get("feedback_likes") if isinstance(prof.get("feedback_likes"), list) else []
    dislikes = prof.get("feedback_dislikes") if isinstance(prof.get("feedback_dislikes"), list) else []
    tag = (topic_tags or "").strip() or (item_title or "").strip()[:120]
    action_l = (action or "").strip().lower()
    if action_l in ("like", "loved", "click") and tag:
        if tag not in likes:
            likes = list(likes) + [tag]
        vec_store.user_media_profile_merge_json(instance_id or "", {"feedback_likes": likes[-30:]})
    elif action_l in ("dislike", "not_for_me", "hide") and tag:
        if tag not in dislikes:
            dislikes = list(dislikes) + [tag]
        vec_store.user_media_profile_merge_json(instance_id or "", {"feedback_dislikes": dislikes[-30:]})
