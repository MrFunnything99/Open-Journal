"""
Journal and memory helpers for Selfmeridian: gist_facts (semantic) and episodic_log (episodic)
via SQLite + sqlite-vec.

Embeddings: Perplexity (`_embed_texts` → PERPLEXITY_API_KEY); without a key, placeholder vectors
match EMBEDDING_DIM so memory rows still persist (semantic retrieval is degraded).

Extraction / helpers: OpenRouter chat completions (`OPENROUTER_API_KEY`; model
`OPENROUTER_EXTRACTION_MODEL`, with legacy fallback `OPENROUTER_GEMINI_MODEL`).
Tune `OPENROUTER_EXTRACTION_MAX_TOKENS` (default 8192) so OpenRouter does not reserve huge
output budgets per request.
"""
from __future__ import annotations

import base64
import hashlib
import ssl
import time
import json
import math
import os
import re
import struct
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import certifi
    ssl._create_default_https_context = lambda purpose=ssl.Purpose.SERVER_AUTH, cafile=certifi.where(): (
        ssl.create_default_context(purpose=purpose, cafile=cafile)
    )
except ImportError:
    pass

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from decision_logger import DecisionLogger

_PERPLEXITY_EMBED_FALLBACK_WARNED = False

PPLX_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/embeddings"
PPLX_CONTEXTUAL_EMBEDDINGS_URL = "https://api.perplexity.ai/v1/contextualizedembeddings"
# Context model: use contextualized endpoint (one chunk per pseudo-document for unrelated texts).
PPLX_EMBED_BATCH_DOCS = 480
DEFAULT_PERPLEXITY_EMBEDDING_MODEL = "pplx-embed-context-v1-4b"
PPLX_SEARCH_URL = "https://api.perplexity.ai/search"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_EXTRACTION_MODEL = "google/gemini-3-pro-preview"


def openrouter_api_configured() -> bool:
    return bool((os.getenv("OPENROUTER_API_KEY") or "").strip())


def openrouter_extraction_model() -> str:
    return (
        (os.getenv("OPENROUTER_EXTRACTION_MODEL") or os.getenv("OPENROUTER_GEMINI_MODEL") or DEFAULT_OPENROUTER_EXTRACTION_MODEL)
        .strip()
    )


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
                "(memory rows still save; semantic retrieval is degraded)."
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
    """Legacy alias: identical to `_call_gemini` (OpenRouter chat completions)."""
    return _call_gemini(prompt)


def wipe_memory() -> None:
    """Clear gist and episodic memory."""
    import vec_store

    _ensure_storage()
    vec_store.wipe_memory()


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


def ingest_journal_entry(
    session_id: str,
    transcript: str,
    entry_date: str | None = None,
    instance_id: str = "",
    content_hash: str | None = None,
    entry_source: str | None = None,
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
        entry_source=entry_source,
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
    entry_source: str | None = None,
) -> dict:
    """Backward-compatible name for ingest_journal_entry."""
    return ingest_journal_entry(
        session_id, transcript, entry_date, instance_id, content_hash, entry_source=entry_source
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


def effective_journal_entry_kind(*, entry_source: str | None, document: str) -> str:
    """
    Classify a stored journal row as manual (solo writing) vs assisted (chat transcript).
    Persisted entry_source wins; legacy rows infer assisted from User/Assistant transcript shape.
    """
    es = (entry_source or "").strip().lower()
    if es == "assisted":
        return "assisted"
    if es == "manual":
        return "manual"
    doc = document or ""
    if "Assistant:" in doc and ("User:" in doc or doc.lstrip().startswith("User:")):
        return "assisted"
    return "manual"


def pick_latest_journal_entries_balanced(
    instance_id: str,
    *,
    per_source: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Newest-first up to `per_source` rows per kind (manual vs assisted), independent quotas."""
    import vec_store

    cap = max(1, min(int(per_source), 50))
    # One bounded scan (newest-first) — avoids loading every journal row on large accounts.
    window = min(500, max(120, cap * 40))
    rows = vec_store.list_journal_entries_recent(instance_id or "", limit=window)
    manual: list[dict] = []
    assisted: list[dict] = []
    for row in rows:
        k = effective_journal_entry_kind(
            entry_source=row.get("entry_source"),
            document=(row.get("document") or ""),
        )
        if k == "manual" and len(manual) < cap:
            manual.append(row)
        elif k == "assisted" and len(assisted) < cap:
            assisted.append(row)
        if len(manual) >= cap and len(assisted) >= cap:
            break
    return manual, assisted


def get_relevant_context_dual(
    query: str,
    top_k_gist: int = 8,
    top_k_episodic: int = 5,
    instance_id: str = "",
    *,
    session_id: str | None = None,
    log: bool = True,
    balance_journal_sources: bool = False,
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

    def _append_chunk_lines(rows_in: list[dict], start_rank: int) -> tuple[list[str], int]:
        lines_out: list[str] = []
        rank = start_rank
        for ch in rows_in:
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
            rank += 1
            if txt:
                lines_out.append(f"[{ed}] {txt}" if ed else txt)
        return lines_out, rank

    try:
        if balance_journal_sources:
            # Enough candidates to split manual vs assisted without oversized vec scans (latency).
            fetch_k = min(max(k * 5, 24), 56)
            rows_wide = vec_store.query_journal_chunks(
                query_emb, instance_id or "", k=fetch_k
            )
            eids = sorted(
                {int(x) for x in (ch.get("entry_id") for ch in rows_wide) if x is not None}
            )
            meta = vec_store.journal_entry_meta_batch(instance_id or "", eids)

            def _kind_for_chunk(ch: dict) -> str:
                eid = int(ch.get("entry_id") or 0)
                m = meta.get(eid) or {}
                return effective_journal_entry_kind(
                    entry_source=m.get("entry_source"),
                    document=(m.get("document") or ""),
                )

            manual_chunks: list[dict] = []
            assisted_chunks: list[dict] = []
            for ch in rows_wide:
                if _kind_for_chunk(ch) == "assisted":
                    assisted_chunks.append(ch)
                else:
                    manual_chunks.append(ch)
            n_m = k // 2
            n_a = k - n_m
            picked_m = manual_chunks[:n_m]
            picked_a = assisted_chunks[:n_a]
            used_cids = {c.get("chunk_id") for c in picked_m + picked_a}
            rem = k - len(picked_m) - len(picked_a)
            for ch in rows_wide:
                if rem <= 0:
                    break
                cid = ch.get("chunk_id")
                if cid in used_cids:
                    continue
                used_cids.add(cid)
                if _kind_for_chunk(ch) == "assisted":
                    picked_a.append(ch)
                else:
                    picked_m.append(ch)
                rem -= 1
            rnk = 0
            lines_m, rnk = _append_chunk_lines(picked_m, rnk)
            lines_a, _ = _append_chunk_lines(picked_a, rnk)
            subparts: list[str] = []
            if lines_m:
                subparts.append(
                    "Relevant excerpts — solo / manual journals (balanced retrieval, equal budget to assisted):\n"
                    + "\n".join(f"- {ln}" for ln in lines_m)
                )
            if lines_a:
                subparts.append(
                    "Relevant excerpts — AI-assisted journal sessions (balanced retrieval, equal budget to manual):\n"
                    + "\n".join(f"- {ln}" for ln in lines_a)
                )
            if subparts:
                parts.append("\n\n".join(subparts))
        else:
            rows = vec_store.query_journal_chunks(query_emb, instance_id or "", k=k)
            lines, _ = _append_chunk_lines(rows, 0)
            if lines:
                parts.append(
                    "Relevant excerpts from the user's journals:\n" + "\n".join(f"- {ln}" for ln in lines)
                )
    except Exception as e:
        print("[backend] get_relevant_context_dual journal chunks:", e)
    if not parts:
        try:
            if balance_journal_sources:
                man_rows, asst_rows = pick_latest_journal_entries_balanced(
                    instance_id or "", per_source=3
                )
                lines_fb_m: list[str] = []
                lines_fb_a: list[str] = []
                for row in man_rows:
                    doc = (row.get("document") or "").strip()
                    ts = (row.get("timestamp") or "").strip()
                    if not doc:
                        continue
                    cap = 1500
                    snippet = doc[:cap] + ("…" if len(doc) > cap else "")
                    lines_fb_m.append(f"[{ts}] {snippet}" if ts else snippet)
                    retrieved_log.append(
                        {
                            "content": snippet[:2000],
                            "score": 0.0,
                            "similarity": 0.0,
                            "source": "journal_entry_fallback_manual",
                            "entry_id": row.get("id"),
                            "timestamp": ts,
                        }
                    )
                for row in asst_rows:
                    doc = (row.get("document") or "").strip()
                    ts = (row.get("timestamp") or "").strip()
                    if not doc:
                        continue
                    cap = 1500
                    snippet = doc[:cap] + ("…" if len(doc) > cap else "")
                    lines_fb_a.append(f"[{ts}] {snippet}" if ts else snippet)
                    retrieved_log.append(
                        {
                            "content": snippet[:2000],
                            "score": 0.0,
                            "similarity": 0.0,
                            "source": "journal_entry_fallback_assisted",
                            "entry_id": row.get("id"),
                            "timestamp": ts,
                        }
                    )
                fb_sections: list[str] = []
                if lines_fb_m:
                    fb_sections.append(
                        "Recent solo / manual journal entries (balanced fallback; truncated):\n"
                        + "\n".join(f"- {ln}" for ln in lines_fb_m)
                    )
                if lines_fb_a:
                    fb_sections.append(
                        "Recent AI-assisted journal sessions (balanced fallback; truncated):\n"
                        + "\n".join(f"- {ln}" for ln in lines_fb_a)
                    )
                if fb_sections:
                    used_sql_fallback = True
                    parts.append("\n\n".join(fb_sections))
            else:
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
        if balance_journal_sources:
            notes += "; manual+assisted balanced"
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


def get_assisted_journal_continuity_block(
    instance_id: str,
    *,
    max_per_source: int = 3,
    max_chars_per_entry: int = 3200,
) -> str:
    """
    Latest manual (solo) and AI-assisted journal bodies with independent quotas so chat volume
    does not crowd out manual writing. Used for Assisted Journal continuity openings.

    The LATEST manual journal entry is separated with an explicit highest-priority label so the
    LLM always starts from what the user most recently wrote by hand.
    """
    try:
        manual_rows, assisted_rows = pick_latest_journal_entries_balanced(
            instance_id or "", per_source=max_per_source
        )
    except Exception as e:
        print("[backend] get_assisted_journal_continuity_block:", e)
        return ""

    cap = max(0, int(max_chars_per_entry))

    def _fmt(row: dict) -> str:
        doc = (row.get("document") or "").strip()
        ts = (row.get("timestamp") or "").strip()
        if not doc:
            return ""
        snippet = doc[:cap] + ("…" if len(doc) > cap else "")
        return f"[entry_date {ts}]\n{snippet}" if ts else snippet

    sections: list[str] = []

    if manual_rows:
        latest = _fmt(manual_rows[0])
        if latest:
            sections.append(
                "### LATEST manual journal entry (HIGHEST PRIORITY for openers — start here)\n"
                + latest
            )
        older = [_fmt(r) for r in manual_rows[1:] if _fmt(r)]
        if older:
            sections.append(
                "### Older manual journals (recent, for recurring-theme context)\n"
                + "\n\n---\n\n".join(older)
            )

    if assisted_rows:
        assisted_chunks = [_fmt(r) for r in assisted_rows if _fmt(r)]
        if assisted_chunks:
            sections.append(
                "### AI-assisted journal sessions (newest up to "
                + str(len(assisted_chunks))
                + " — secondary to latest manual entry for openers)\n"
                + "\n\n---\n\n".join(assisted_chunks)
            )

    if not sections:
        return ""
    return (
        "RECENCY-FIRST RULE FOR OPENERS: The user's latest manual journal entry is the primary anchor for opening turns. "
        "Your opening MUST reference 1-2 specific concrete things from that latest manual entry before anything else. "
        "Older entries and assisted sessions are supporting context only — use them to notice recurring themes, "
        "but do NOT lead with older emotionally salient material unless the latest entry is empty or explicitly references it. "
        "Never tell the user whether a detail came from manual writing or assisted chat.\n\n"
        + "\n\n".join(sections)
    )


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


def list_consumed_media(instance_id: str = "", category: str | None = None) -> list[dict]:
    import vec_store

    _ensure_storage()
    return vec_store.consumed_media_list(instance_id, category)


def add_consumed_media(
    *,
    instance_id: str = "",
    category: str,
    title: str,
    creator_or_source: str | None = None,
    notes: str | None = None,
    consumed_on: str | None = None,
) -> int | None:
    import vec_store

    _ensure_storage()
    return vec_store.consumed_media_create(
        instance_id=instance_id,
        category=category,
        title=title,
        creator_or_source=creator_or_source,
        notes=notes,
        consumed_on=consumed_on,
    )


def update_consumed_media(
    *,
    instance_id: str = "",
    item_id: int,
    category: str,
    title: str,
    creator_or_source: str | None = None,
    notes: str | None = None,
    consumed_on: str | None = None,
) -> bool:
    import vec_store

    _ensure_storage()
    return vec_store.consumed_media_update(
        instance_id=instance_id,
        item_id=item_id,
        category=category,
        title=title,
        creator_or_source=creator_or_source,
        notes=notes,
        consumed_on=consumed_on,
    )


def delete_consumed_media(*, instance_id: str = "", item_id: int) -> bool:
    import vec_store

    _ensure_storage()
    return vec_store.consumed_media_delete(instance_id=instance_id, item_id=item_id)


def _compose_consumed_notes(
    *,
    note: str | None,
    url: str | None,
    liked: bool,
) -> str | None:
    parts: list[str] = []
    if note and str(note).strip():
        parts.append(str(note).strip())
    if url and str(url).strip():
        parts.append(f"URL: {str(url).strip()}")
    if not liked:
        parts.append("Did not enjoy.")
    if not parts:
        return None
    return "\n".join(parts)


def _consumed_media_near_duplicate(
    instance_id: str, category: str, title: str, creator: str | None
) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    t_key = norm(title)
    if not t_key:
        return False
    a_key = norm(creator or "")
    try:
        rows = list_consumed_media(instance_id, category)
    except Exception:
        rows = []
    for r in rows:
        if norm(r.get("title") or "") != t_key:
            continue
        ra = norm(r.get("creator_or_source") or "")
        if not a_key or not ra or a_key == ra:
            return True
    return False


def apply_library_tool_items(items: object, instance_id: str = "") -> tuple[int, list[str]]:
    """
    Validate structured items from the chat agent (tool calls) and insert into Semantic Memory (consumed_media).
    Returns (count_added, short labels for confirmation).
    """
    if not isinstance(items, list):
        return 0, []
    added = 0
    labels: list[str] = []
    type_to_category = {
        "book": "book",
        "podcast": "podcast",
        "article": "research_article",
        "research": "research_article",
    }
    for raw in items:
        if not isinstance(raw, dict):
            continue
        ctype = str(raw.get("type", "")).lower().strip()
        category = type_to_category.get(ctype)
        if not category:
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
        notes = _compose_consumed_notes(note=note, url=url, liked=bool(liked))
        if _consumed_media_near_duplicate(instance_id, category, title, author):
            continue
        try:
            row_id = add_consumed_media(
                instance_id=instance_id,
                category=category,
                title=title,
                creator_or_source=author,
                notes=notes,
            )
        except Exception as e:
            print("[backend] apply_library_tool_items add error:", e)
            continue
        if row_id:
            added += 1
            labels.append(
                f"{ctype}: {title[:80]}" + (f" ({author[:60]})" if author else "")
            )
    return added, labels


_BOOK_VALIDATION_MODEL = os.getenv("OPENROUTER_BOOK_VALIDATION_MODEL", "openai/gpt-4.1-mini")

_BOOK_TOOL_NORMALIZE_PROMPT = """The user mentioned these books in conversation. For each entry, output a clean standard English title and the primary author (fix typos and casing for well-known books).
Return ONLY a JSON array — same length and order as the input. Each element: {"title":"...","author":"..."}.
Use "" for author only if unknown. No markdown, no explanation.

Input:
"""


def resolve_books_via_openlibrary(books: list[dict]) -> list[dict]:
    """
    Normalize title/author via OpenRouter (OPENROUTER_BOOK_VALIDATION_MODEL), then save_resolved_books.
    """
    resolved: list[dict] = []
    for raw in books:
        if not isinstance(raw, dict):
            continue
        raw_title = (raw.get("raw_title") or "").strip()
        raw_author = (raw.get("raw_author") or "").strip() or None
        if not raw_title:
            continue
        resolved.append(
            {
                "type": "book",
                "title": raw_title,
                "author": raw_author,
                "raw_title": raw_title,
                "raw_author": raw_author,
                "liked": raw.get("liked", True),
                "note": (raw.get("note") or "").strip() or None,
            }
        )

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


def save_resolved_books(resolved: list[dict], instance_id: str = "") -> tuple[int, list[str]]:
    """Persist normalized books to consumed_media; skip near-duplicates. Returns (count_saved, labels)."""
    saved = 0
    skipped = 0
    labels: list[str] = []
    for book in resolved:
        title = (book.get("title") or "").strip()
        if not title:
            continue
        author = book.get("author")
        author_s = (author or "").strip() if author else ""
        liked = bool(book.get("liked", True))
        note = (book.get("note") or "").strip() or None
        notes = _compose_consumed_notes(note=note, url=None, liked=liked)
        if _consumed_media_near_duplicate(instance_id, "book", title, author_s or None):
            skipped += 1
            continue
        try:
            row_id = add_consumed_media(
                instance_id=instance_id,
                category="book",
                title=title,
                creator_or_source=author_s or None,
                notes=notes,
            )
        except Exception as e:
            print(f"[backend] save_resolved_books error for '{title}': {e}")
            continue
        if row_id:
            saved += 1
            label = f"book: {title[:80]}"
            if author_s:
                label += f" ({author_s[:60]})"
            labels.append(label)
    if skipped:
        labels.append(f"({skipped} already in library)")
    return saved, labels


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
    ingest_journal_entry(
        sid,
        document,
        row.get("timestamp"),
        instance_id,
        entry_source=row.get("entry_source"),
    )
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
    out = ingest_journal_entry(sid, document, None, instance_id, entry_source="manual")
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
