"""
Library for Open-Journal: gist_facts (semantic) and episodic_log (episodic) memory,
and consumed_content (library). Uses SQLite + sqlite-vec for vector storage.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Clients (lazy init)
_embeddings_client: genai.Client | None = None


def _ensure_storage() -> None:
    """Ensure SQLite + sqlite-vec DB is initialized."""
    import vec_store

    vec_store.ensure_db()


def _get_embeddings_client() -> genai.Client:
    global _embeddings_client
    if _embeddings_client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is required for embeddings. See https://ai.google.dev/gemini-api/docs/get-started"
            )
        _embeddings_client = genai.Client(api_key=key)
    return _embeddings_client


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Gemini Embedding 2.
    Returns a list of embedding vectors (one per text).
    """
    if not texts:
        return []
    client = _get_embeddings_client()
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
    result = client.models.embed_content(model=model, contents=texts)
    return [emb.values for emb in result.embeddings]


def _call_gemini(prompt: str) -> str:
    """
    Call the primary Gemini chat model (Flash 3.1 by default) with a single text prompt.
    Returns the response text (empty string on failure).
    """
    try:
        client = _get_embeddings_client()
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash")
        result = client.models.generate_content(model=model, contents=prompt)
        text = getattr(result, "text", "") or ""
        return text.strip()
    except Exception as e:
        print("[backend] _call_gemini error:", e)
        return ""


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
    uid = f"consumed_{ts_safe}_{hash(title) % 10**8}"
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
    )


def process_library_note(text: str, type_filter: str | None = None) -> int:
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
- Each line or item in the text should become one object. "title" is the work's title; "author" if obvious.
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
            add_consumed(ctype, title, author=author, url=url, liked=liked, note=note)
            added += 1
        except Exception as e:
            print("[backend] process_library_note add_consumed error:", e)
            continue
    return added


def get_consumed_context(max_items: int = 80) -> str:
    """
    Return a single string describing what the user has consumed and liked,
    for injection into the recommendation agent prompt.
    """
    import vec_store

    _ensure_storage()
    try:
        rows = vec_store.get_consumed_context_rows(max_items=max_items)
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
            line += " — liked" if liked else " — read/listened"
            if note:
                line += f". User reflection: {note}"
            lines.append(line)
        return "What the user has already read or listened to (do not recommend these again; use their tastes and reflections below to suggest similar things):\n" + "\n".join(lines)
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


def list_consumed(max_items: int = 200) -> dict[str, list[dict]]:
    """
    Return consumed items grouped by type for the Library UI, sorted by date_completed (newest first).
    Each item: { "id", "title", "author", "date_completed", "note" }.
    """
    import vec_store

    _ensure_storage()
    out: dict[str, list[dict]] = {"books": [], "podcasts": [], "articles": [], "research": []}
    try:
        rows = vec_store.list_consumed_rows(max_items=max_items)
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


def update_consumed(item_id: str, *, date_completed: str | None = None, note: str | None = None) -> bool:
    """Update date_completed and/or note for a consumed item. Returns True if updated."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.update_consumed(item_id, date_completed=date_completed, note=note)
    except Exception as e:
        print("[backend] update_consumed error:", e)
        return False


def delete_consumed(item_id: str) -> bool:
    """Remove a consumed item by id. Returns True if deleted."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.delete_consumed(item_id)
    except Exception as e:
        print("[backend] delete_consumed error:", e)
        return False


def _extract_session_data(transcript: str) -> dict:
    """Use Gemini LLM to extract summary, facts, and structured metadata from transcript."""
    prompt = f"""You are a journal analyst. Extract structured data from this journal session transcript.

Transcript:
---
{transcript}
---

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "summary": "A 3-sentence summary of the session: what was discussed, key themes, and emotions felt.",
  "facts": ["Fact 1 about the user", "Fact 2 about the user", ...],
  "metadata": {{
    "people": ["Name1", "Name2"],
    "topics": ["topic1", "topic2"],
    "mood": -2,
    "energy": 3,
    "activities": ["activity1", "activity2"]
  }}
}}

Rules:
- summary: exactly 3 sentences, capture emotions and themes
- facts: list of hard, verifiable facts about the user (job, relationships, preferences, life events). Empty list if none.
- metadata: for pattern analysis. people: list of people mentioned. topics: major discussion topics. mood: integer from -5 (very negative) to +5 (very positive). energy: integer from 1 (low) to 5 (high). activities: list of actions or activities described. Omit any field if unknown; use empty lists for people/topics/activities and null for mood/energy when unclear.
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
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print("[backend] _extract_session_data JSON error:", e)
        return {"summary": "", "facts": [], "metadata": {}}
    if not isinstance(data, dict):
        return {"summary": "", "facts": [], "metadata": {}}
    data.setdefault("summary", "")
    data.setdefault("facts", [])
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("people", [])
    meta.setdefault("topics", [])
    meta.setdefault("mood", None)
    meta.setdefault("energy", None)
    meta.setdefault("activities", [])
    if not isinstance(meta["people"], list):
        meta["people"] = []
    if not isinstance(meta["topics"], list):
        meta["topics"] = []
    if not isinstance(meta["activities"], list):
        meta["activities"] = []
    data["metadata"] = meta
    return data


def save_session_data(session_id: str, transcript: str) -> dict:
    """
    Extract summary + facts + metadata from transcript via Gemini, embed via Gemini, save to SQLite+sqlite-vec.
    Returns {"summary": str, "facts": list[str]} for callers (e.g. LightRAG feed).
    Episodic row also stores metadata_json for future time-series / pattern analysis; metadata is not embedded.
    """
    import vec_store

    _ensure_storage()
    data = _extract_session_data(transcript)
    summary = data.get("summary", "")
    facts = data.get("facts", [])
    metadata = data.get("metadata") or {}

    ts = datetime.utcnow().isoformat() + "Z"

    if summary:
        summary_emb = _embed_texts([summary])[0]
        metadata_json = json.dumps(metadata) if metadata else None
        vec_store.add_episodic(session_id, ts, summary, summary_emb, metadata_json=metadata_json)

    # Deduplicate facts: if a new fact is very similar to an existing one, update it instead of inserting
    GIST_SIMILARITY_THRESHOLD = 0.85  # cosine similarity above this => update existing fact
    if facts:
        fact_embs = _embed_texts(facts)
        n_existing = vec_store.gist_count()
        for fact in facts:
            emb = fact_embs[facts.index(fact)]
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
            vec_store.add_gist(session_id, ts, fact, emb)
            n_existing += 1

    return {"summary": summary, "facts": facts}


def get_relevant_context(query: str, top_k_gist: int = 8, top_k_episodic: int = 5) -> str:
    """
    Embed the query, retrieve relevant gist facts and episodic summaries from SQLite+sqlite-vec,
    and return a single string for injection into the interviewer's context.
    """
    import vec_store

    if not query or not query.strip():
        return "None."
    _ensure_storage()
    query_emb = _embed_texts([query.strip()])[0]

    parts = []
    try:
        if vec_store.gist_count() > 0:
            docs = vec_store.query_gist(query_emb, min(top_k_gist, vec_store.gist_count()))
            if docs:
                parts.append("Facts and details from the user's life and journals:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        pass

    try:
        if vec_store.episodic_count() > 0:
            docs = vec_store.query_episodic(query_emb, min(top_k_episodic, vec_store.episodic_count()))
            if docs:
                parts.append("Relevant journal summaries:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        pass

    if not parts:
        return "None."
    return "\n\n".join(parts)


def get_memory_for_visualization() -> tuple[list[str], list[str]]:
    """
    Return (gist_facts, episodic_summaries) as lists of document strings for diagram generation.
    """
    import vec_store

    _ensure_storage()
    gist_docs: list[str] = []
    episodic_docs: list[str] = []

    try:
        gist_docs = vec_store.get_all_gist()
    except Exception:
        pass

    try:
        episodic_docs = vec_store.get_all_episodic()
    except Exception:
        pass

    return (gist_docs, episodic_docs)


def list_memory_facts() -> list[dict]:
    """Return all gist facts with id, document, session_id, timestamp for Memory UI."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.list_gist_with_ids()
    except Exception as e:
        print("[backend] list_memory_facts error:", e)
        return []


def list_memory_summaries() -> list[dict]:
    """Return all episodic summaries with id, document, session_id, timestamp for Memory UI."""
    import vec_store

    _ensure_storage()
    try:
        return vec_store.list_episodic_with_ids()
    except Exception as e:
        print("[backend] list_memory_summaries error:", e)
        return []


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


def add_memory_fact(document: str, session_id: str | None = None) -> int | None:
    """Add a single user-created fact; returns new id or None on failure."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return None
    doc = document.strip()
    sid = session_id or "user"
    ts = datetime.utcnow().isoformat() + "Z"
    emb = _embed_texts([doc])[0]
    return vec_store.add_gist(sid, ts, doc, emb)


def add_memory_summary(document: str, session_id: str | None = None) -> int | None:
    """Add a single user-created summary; returns new id or None on failure."""
    import vec_store

    _ensure_storage()
    if not document or not document.strip():
        return None
    doc = document.strip()
    sid = session_id or "user"
    ts = datetime.utcnow().isoformat() + "Z"
    emb = _embed_texts([doc])[0]
    return vec_store.add_episodic(sid, ts, doc, emb)


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


def _books_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for book recommendations only."""
    prompt = f"""You are a book curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 books they might find helpful or comforting.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Do NOT suggest books they have already consumed. For each book give title, author, and a short "reason" (one sentence) tied to their life or journal themes. No URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


LISTEN_NOTES_BASE = "https://listen-api.listennotes.com/api/v2"


def _listen_notes_search_episodes(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Call Listen Notes API search (type=episode). Returns list of {title, author, url, reason}."""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"q": query, "type": "episode", "only_one_episode_per_podcast": 1})
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
        if title and ln_url:
            out.append({
                "title": title[:500],
                "author": author[:300],
                "reason": f"Suggested based on your interests: {query[:80]}.",
                "url": ln_url,
            })
        if len(out) >= max_results:
            break
    return out


def _podcasts_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated podcast agent: uses Listen Notes API for real episode links when key is set."""
    api_key = (os.getenv("LISTENNOTES_API_KEY") or "").strip()
    if api_key:
        prompt = f"""Based on this person's journal-derived memory and what they have already listened to, suggest 2–3 short search queries (topics or themes) to find relevant podcast episodes. Examples: "mindfulness sleep", "anxiety therapy", "Huberman Lab sleep".

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

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
        out = []
        per_query = 2
        for q in queries[:3]:
            if not isinstance(q, str) or not q.strip():
                continue
            for item in _listen_notes_search_episodes(q.strip(), api_key, max_results=per_query):
                if item["url"] in seen_urls:
                    continue
                if item["title"].lower() in consumed_lower or (item["author"] and item["author"].lower() in consumed_lower):
                    continue
                seen_urls.add(item["url"])
                out.append(item)
                if len(out) >= 5:
                    break
            if len(out) >= 5:
                break
        if out:
            return out
    prompt = f"""You are a podcast curator. Based on this person's journal-derived memory and what they have already listened to, suggest 3–5 specific podcast episodes (show + episode).

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Use "author" for the show name and "title" for the episode. Provide direct "url" (Spotify or Apple Podcasts episode link) when you know it; otherwise leave "url" empty. Do NOT suggest items they have already consumed. For each give a short "reason" (one sentence).
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def _articles_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for article recommendations only. Article URLs must work (non-negotiable)."""
    prompt = f"""You are an article curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 news articles or long-reads.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

CRITICAL: Every article MUST have a "url" that is a real, working link to the actual article page—no 404s. Use only URLs you are certain exist (exact paths from nytimes.com, theatlantic.com, healthline.com, bbc.com, nature.com, apa.org, etc.). Do NOT guess or construct URLs. If you cannot provide a verified working URL for an article, do NOT include it. For each give title, author/source, reason, and url.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def _research_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for research paper recommendations only."""
    prompt = f"""You are a research curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 academic or scientific research papers (peer-reviewed articles, studies, or review papers) they might find relevant or helpful.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Do NOT suggest papers they have already consumed. For each paper give: "title" (paper title), "author" (lead author or author list, or journal name and year), "reason" (one sentence on why it fits their interests), and "url" (working link to the paper—DOI link like https://doi.org/10.1234/... or publisher link to the abstract/full text). Only include papers where you can provide a real, working url (doi.org, PubMed, PMC, journal websites). Do not guess URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def generate_recommendations() -> dict:
    """
    Run four dedicated agents (books, podcasts, articles, research) in parallel, each with
    its own prompt and specialization. Combines results into one response.
    """
    gist_docs, episodic_docs = get_memory_for_visualization()
    consumed = get_consumed_context()
    facts_blob = "\n".join(f"- {f}" for f in (gist_docs or [])[:60])
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_docs or [])[:40])

    from concurrent.futures import ThreadPoolExecutor
    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []

    agent_timeout = 90  # seconds per agent so one hang doesn't block forever
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_books = executor.submit(_books_agent, facts_blob, summaries_blob, consumed)
        future_podcasts = executor.submit(_podcasts_agent, facts_blob, summaries_blob, consumed)
        future_articles = executor.submit(_articles_agent, facts_blob, summaries_blob, consumed)
        future_research = executor.submit(_research_agent, facts_blob, summaries_blob, consumed)
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

    return {
        "books": books_list if isinstance(books_list, list) else [],
        "podcasts": podcasts_list if isinstance(podcasts_list, list) else [],
        "articles": articles_list if isinstance(articles_list, list) else [],
        "research": research_list if isinstance(research_list, list) else [],
    }
