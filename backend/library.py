"""
Library for Open-Journal: gist_facts (semantic) and episodic_log (episodic) memory,
and consumed_content (library). Uses SQLite + sqlite-vec for vector storage.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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
        return (
            "What the user has already consumed (books, podcasts, articles, research) and their reflections. "
            "Do not recommend these same items again. Use their tastes and reflections from ALL categories to inform your suggestions—e.g. their notes on a research paper or podcast can shape book recommendations, and vice versa.\n"
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
    try:
        data = json.loads(text)
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


def save_session_data(session_id: str, transcript: str, entry_date: str | None = None) -> dict:
    """
    Extract summary + facts + metadata from transcript via Gemini, embed via Gemini, save to SQLite+sqlite-vec.
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


def get_relevant_context(query: str, top_k_gist: int = 8, top_k_episodic: int = 5) -> str:
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
        if vec_store.gist_count() > 0:
            count = vec_store.gist_count()
            fetch_k = min(max(top_k_gist * 2, 10), count)
            raw = vec_store.query_gist_with_timestamp(query_emb, fetch_k)
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
            docs = vec_store.query_gist(query_emb, min(top_k_gist, vec_store.gist_count()))
            if docs:
                parts.append("Facts and details from the user's life and journals:\n" + "\n".join(f"- {d}" for d in docs))
        except Exception:
            pass

    try:
        if vec_store.episodic_count() > 0:
            count = vec_store.episodic_count()
            fetch_k = min(max(top_k_episodic * 2, 8), count)
            raw = vec_store.query_episodic_with_timestamp(query_emb, fetch_k)
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
            elif not items and vec_store.episodic_count() > 0:
                docs = vec_store.query_episodic(query_emb, min(top_k_episodic, vec_store.episodic_count()))
                if docs:
                    parts.append("Relevant journal summaries:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        try:
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


def get_memory_for_date(date_iso: str) -> tuple[list[dict], list[dict]]:
    """Return (episodic_summaries, gist_facts) for the given date (YYYY-MM-DD)."""
    import vec_store

    _ensure_storage()
    episodic: list[dict] = []
    gist: list[dict] = []
    try:
        episodic = vec_store.get_episodic_for_date(date_iso)
    except Exception as e:
        print("[backend] get_episodic_for_date error:", e)
    try:
        gist = vec_store.get_gist_for_date(date_iso)
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


def _books_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Dedicated agent for book recommendations only."""
    recent_section = ""
    if recent_summaries_blob:
        recent_section = f"""
MOST RECENT JOURNAL ENTRIES (give ~20% more weight to themes and needs that appear here—what's most present for them right now):
{recent_summaries_blob}

"""
    prompt = f"""You are a book curator. Based on this person's journal-derived memory and what they have consumed (books, podcasts, articles, research) and their reflections on any of them, suggest 3–5 books they might find helpful or comforting.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES (all themes over time):
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Rules: Do NOT suggest books they have already consumed. Use their reflections and tastes from podcasts, articles, and research too—e.g. if they liked a paper or podcast on a topic, that can inform book picks. For each book give title, author, and a short "reason" (one sentence) tied to their life or journal themes. No URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


LISTEN_NOTES_BASE = "https://listen-api.listennotes.com/api/v2"
# Set to True to skip calling the Listen Notes API (use LLM-only fallback for podcast suggestions).
PODCAST_API_PAUSED = True


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
    recent_section = ""
    if recent_summaries_blob:
        recent_section = f"""
MOST RECENT JOURNAL ENTRIES (weight these ~20% more when picking topics):
{recent_summaries_blob}

"""
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
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def _articles_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Dedicated agent for article recommendations only. Article URLs must work (non-negotiable)."""
    recent_section = ""
    if recent_summaries_blob:
        recent_section = f"""
MOST RECENT JOURNAL ENTRIES (weight these ~20% more when matching interests):
{recent_summaries_blob}

"""
    prompt = f"""You are an article curator. Based on this person's journal-derived memory and what they have consumed (books, podcasts, articles, research) and their reflections on any of them, suggest 3–5 news articles or long-reads. Use their tastes from books, podcasts, and research too—e.g. themes from a book or paper they liked can inform article picks.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES (all themes):
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

CRITICAL: Every article MUST have a "url" that is a real, working link to the actual article page—no 404s. Use only URLs you are certain exist (exact paths from nytimes.com, theatlantic.com, healthline.com, bbc.com, nature.com, apa.org, etc.). Do NOT guess or construct URLs. If you cannot provide a verified working URL for an article, do NOT include it. For each give title, author/source, reason, and url.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def _research_agent(facts_blob: str, summaries_blob: str, consumed: str, recent_summaries_blob: str = "") -> list:
    """Dedicated agent for research paper recommendations only."""
    recent_section = ""
    if recent_summaries_blob:
        recent_section = f"""
MOST RECENT JOURNAL ENTRIES (weight these ~20% more when matching interests):
{recent_summaries_blob}

"""
    prompt = f"""You are a research curator. Based on this person's journal-derived memory and what they have consumed (books, podcasts, articles, research) and their reflections on any of them, suggest 3–5 academic or scientific research papers (peer-reviewed articles, studies, or review papers) they might find relevant or helpful. Use their tastes from books, podcasts, and articles too—e.g. themes from a book or podcast they liked can inform paper picks.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES (all themes):
{summaries_blob or "(none yet)"}
{recent_section}{consumed}

Rules: Do NOT suggest papers they have already consumed. For each paper give: "title" (paper title), "author" (lead author or author list, or journal name and year), "reason" (one sentence on why it fits their interests), and "url" (working link to the paper—DOI link like https://doi.org/10.1234/... or publisher link to the abstract/full text). Only include papers where you can provide a real, working url (doi.org, PubMed, PMC, journal websites). Do not guess URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    text = _call_gemini(prompt)
    return _parse_recommendation_json(text, [])


def generate_recommendations() -> dict:
    """
    Run four dedicated agents (books, podcasts, articles, research) in parallel, each with
    its own prompt and specialization. Combines results into one response.
    Uses full journal history for themes but gives ~20% more weight to the most recent entries.
    """
    gist_docs, episodic_docs = get_memory_for_visualization()
    consumed = get_consumed_context()
    facts_blob = "\n".join(f"- {f}" for f in (gist_docs or [])[:60])
    # Episodic from get_memory_for_visualization is oldest-first (ORDER BY id). Use all for themes.
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_docs or [])[:40])
    # Most recent ~20% of journal entries (by count) for extra weight—what's most present right now.
    episodic_list = episodic_docs or []
    n_recent = max(5, int(len(episodic_list) * 0.2)) if episodic_list else 0
    recent_episodic = episodic_list[-n_recent:] if n_recent else []
    recent_summaries_blob = "\n".join(f"- {s}" for s in recent_episodic) if recent_episodic else ""

    from concurrent.futures import ThreadPoolExecutor
    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []

    agent_timeout = 90  # seconds per agent so one hang doesn't block forever
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_books = executor.submit(_books_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_podcasts = executor.submit(_podcasts_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_articles = executor.submit(_articles_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
        future_research = executor.submit(_research_agent, facts_blob, summaries_blob, consumed, recent_summaries_blob)
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
