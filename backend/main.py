"""
FastAPI backend for Open-Journal: /chat and /end-session.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env from project root first (parent of backend/), then cwd
_root = Path(__file__).resolve().parent.parent
for name in (".env", ".env.local"):
    load_dotenv(_root / name)
    load_dotenv(Path.cwd() / name)
    load_dotenv(Path.cwd().parent / name)

from graph import build_graph, build_librarian_graph, JournalState
from langchain_core.messages import HumanMessage
from library import (
    add_consumed,
    add_memory_fact,
    add_memory_summary,
    delete_consumed,
    delete_memory_fact,
    delete_memory_summary,
    generate_day_summary,
    generate_memory_mermaid,
    generate_recommendations,
    get_memory_for_date,
    get_memory_for_visualization,
    get_person_events,
    list_consumed,
    list_memory_facts,
    list_memory_summaries,
    run_library_interview,
    run_person_facts_agent,
    run_relationship_summary_agent,
    run_people_grouping_agent,
    save_session_data,
    update_consumed,
    update_memory_fact,
    update_memory_summary,
    wipe_memory,
)
from google import genai

# In-memory session store (minimal for 1hr sprint; replace with Redis/DB later)
sessions: dict[str, list] = {}
library_interview_sessions: dict[str, list] = {}  # session_id -> list of {role, content}

CHAT_INVOKE_TIMEOUT_SEC = 60
LIBRARY_INTERVIEW_TIMEOUT_SEC = 45


def get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    return session_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Avoid proxy 403s: send API calls (Gemini, Listen Notes) direct, not via system proxy
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "*"
    # Startup: log whether env is loaded (no secrets)
    gemini_key = os.getenv("GEMINI_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    if gemini_key and gemini_key.strip():
        print("[backend] GEMINI_API_KEY is set (embeddings, memory, etc.)")
    else:
        print("[backend] WARNING: GEMINI_API_KEY is missing. Set it in .env for embeddings and memory.")
    if xai_key and xai_key.strip():
        print("[backend] XAI_API_KEY is set (Grok 4.20 reasoning for interviewer)")
    else:
        print("[backend] WARNING: XAI_API_KEY is missing. Set it in .env for /chat interviewer.")
    yield
    # Cleanup if needed
    pass


app = FastAPI(title="Open-Journal Backend", lifespan=lifespan)
# Explicit origins so CORS works with credentials; "*" cannot be used when credentials=True
_extra_origins = [s.strip() for s in os.getenv("CORS_ORIGINS", "").split(",") if s.strip()]
_cors_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "https://open-journal-pearl.vercel.app",
    *_extra_origins,
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

chat_graph = build_graph()
librarian_graph = build_librarian_graph()


class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    personalization: Optional[float] = None
    intrusiveness: Optional[float] = None
    mode: Optional[str] = None  # "journal" (default) | "recommendations"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    retrieval_log: Optional[str] = None
    notes_saved: Optional[List[Dict[str, str]]] = None  # [{"item_id": "...", "note": "..."}]


class EndSessionRequest(BaseModel):
    session_id: str


class EndSessionResponse(BaseModel):
    ok: bool
    session_id: str


class LibraryInterviewItem(BaseModel):
    id: str
    title: str
    author: Optional[str] = None
    note: Optional[str] = None


class LibraryInterviewRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    library_snapshot: Optional[List[LibraryInterviewItem]] = None


class LibraryInterviewResponse(BaseModel):
    response: str
    session_id: str
    notes_saved: Optional[List[Dict[str, str]]] = None  # [{"item_id": "...", "note": "..."}]


class IngestHistoryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    entry_date: Optional[str] = None  # ISO date/datetime when the journal was written; used for vector DB timestamp


class IngestHistoryResponse(BaseModel):
    ok: bool
    session_id: str


class InferEntryDateRequest(BaseModel):
    text: str
    filename: Optional[str] = None  # optional; used only when text doesn't provide a date


class InferEntryDateResponse(BaseModel):
    date: Optional[str] = None  # ISO 8601 or null if unclear


class MemoryItem(BaseModel):
    id: int
    document: str
    session_id: str = ""
    timestamp: str = ""
    metadata_json: Optional[str] = None  # episodic only; JSON string for people, topics, mood, energy, activities


class MemoryFactUpdate(BaseModel):
    document: str


class MemorySummaryUpdate(BaseModel):
    document: str
    metadata: Optional[dict] = None  # optional; updates metadata_json (people, topics, mood, energy, activities)


class MemoryFactCreate(BaseModel):
    document: str


class MemorySummaryCreate(BaseModel):
    document: str


class MemoryStats(BaseModel):
    gist_facts_count: int
    episodic_log_count: int
    episodic_metadata_count: int = 0  # summaries with metadata (people, topics, etc.)


class BrainPersonNode(BaseModel):
    id: str
    label: str
    type: str  # "person" or "group"


class BrainPersonLink(BaseModel):
    source: str
    target: str


class BrainPeopleGraphResponse(BaseModel):
    nodes: List[BrainPersonNode]
    links: List[BrainPersonLink]


class BrainPersonSummary(BaseModel):
    relationship_summary: str = ""
    relationship_type: str = ""
    closeness_label: str = ""


class BrainRelationshipState(BaseModel):
    user_summary: str = ""
    relationship_type: str = ""
    closeness_level: str = ""
    updated_at: str = ""


class BrainPersonFact(BaseModel):
    id: int
    fact_text: str
    confidence: Optional[float] = None
    source_journal_id: str = ""
    created_at: str = ""


class BrainPersonDetail(BaseModel):
    id: int
    name: str
    relationship_state: BrainRelationshipState
    ai_relationship_summary: str = ""
    groups: List[str]
    events: List[Dict]
    facts: List[BrainPersonFact]
    thoughts: List[Dict]


class BrainPersonCreateRequest(BaseModel):
    name: str


class BrainPersonRenameRequest(BaseModel):
    name: str


class BrainPersonProfileUpdate(BaseModel):
    relationship_summary: str = ""
    relationship_type: Optional[str] = None
    closeness_label: Optional[str] = None


class BrainPersonGroupsUpdate(BaseModel):
    groups: List[str]


class BrainPersonThoughtCreate(BaseModel):
    date: Optional[str] = None
    thought_text: str


class BrainPersonThoughtUpdate(BaseModel):
    date: Optional[str] = None
    thought_text: Optional[str] = None


class RecommendationItem(BaseModel):
    title: str
    author: str = ""
    reason: str = ""
    url: str = ""


class RecommendationsResponse(BaseModel):
    books: list[RecommendationItem]
    podcasts: list[RecommendationItem]
    articles: list[RecommendationItem]
    research: list[RecommendationItem]


class ConsumedRequest(BaseModel):
    type: str  # "book" | "podcast" | "article"
    title: str
    author: Optional[str] = None
    url: Optional[str] = None
    liked: bool = True


class ConsumedResponse(BaseModel):
    ok: bool


class LibraryNoteRequest(BaseModel):
    text: str
    type: Optional[str] = None  # "book" | "podcast" | "article" | "research" to restrict to one category


class LibraryNoteResponse(BaseModel):
    ok: bool
    items_added: int


class LibraryItemUpdate(BaseModel):
    date_completed: Optional[str] = None
    note: Optional[str] = None


class MemoryDiagramResponse(BaseModel):
    mermaid: str


@app.get("/memory-diagram", response_model=MemoryDiagramResponse)
async def memory_diagram():
    """
    Legacy endpoint for memory diagram; retained for compatibility but not used by Brain UI.
    """
    gist_facts, episodic_summaries = get_memory_for_visualization()
    mermaid_code = generate_memory_mermaid(gist_facts, episodic_summaries)
    return MemoryDiagramResponse(mermaid=mermaid_code)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """User sends text; Interviewer responds. State updated in memory. mode=recommendations uses library interview (books, notes)."""
    session_id = get_or_create_session(req.session_id)
    mode = (req.mode or "journal").strip().lower()
    if mode not in ("journal", "recommendations", "extreme", "therapy"):
        mode = "journal"

    if mode == "recommendations":
        # Library interview: ask about books, save short notes. Uses library_interview_sessions.
        _get_or_create_library_interview_session(session_id)
        messages = library_interview_sessions[session_id]
        try:
            library_data = await asyncio.to_thread(list_consumed)
            books = library_data.get("books") or []
            library_items = [
                {"id": b.get("id", ""), "title": b.get("title", "?"), "author": b.get("author") or "", "note": (b.get("note") or "").strip() or None}
                for b in books
            ]
        except Exception as e:
            print("[backend] /chat recommendations list_consumed error:", e)
            library_items = []
        user_message = (req.text or "").strip()
        if not user_message:
            user_message = "Start"
        try:
            reply, notes_saved = await asyncio.wait_for(
                asyncio.to_thread(run_library_interview, list(messages), library_items, user_message),
                timeout=LIBRARY_INTERVIEW_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                response="That took a bit long—try again in a moment.",
                session_id=session_id,
            )
        except Exception as e:
            print("[backend] /chat recommendations error:", e)
            return ChatResponse(
                response="Something went wrong. Please try again.",
                session_id=session_id,
            )
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
        library_interview_sessions[session_id] = messages
        return ChatResponse(
            response=reply,
            session_id=session_id,
            notes_saved=notes_saved if notes_saved else None,
        )

    # Default: journal interview (existing flow)
    messages = sessions[session_id]
    personalization = req.personalization if req.personalization is not None else 1.0
    try:
        personalization = float(personalization)
    except (TypeError, ValueError):
        personalization = 1.0
    personalization = max(0.0, min(1.0, personalization))
    intrusiveness = req.intrusiveness if req.intrusiveness is not None else 0.5
    try:
        intrusiveness = float(intrusiveness)
    except (TypeError, ValueError):
        intrusiveness = 0.5
    intrusiveness = max(0.0, min(1.0, intrusiveness))
    messages.append(HumanMessage(content=req.text))
    state: JournalState = {
        "messages": list(messages),
        "session_id": session_id,
        "personalization": personalization,
        "intrusiveness": intrusiveness,
        "mode": mode,
    }
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(chat_graph.invoke, state),
            timeout=CHAT_INVOKE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        messages.pop()
        print("[backend] /chat timed out after %ss" % CHAT_INVOKE_TIMEOUT_SEC)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=504,
            detail=f"Chat took longer than {CHAT_INVOKE_TIMEOUT_SEC}s (LLM or memory slow). Try again or lower personalization.",
        )
    sessions[session_id] = list(result["messages"])
    last = result["messages"][-1]
    response_text = getattr(last, "content", str(last))
    if isinstance(response_text, list):
        response_text = " ".join(
            c.get("text", str(c)) for c in response_text if isinstance(c, dict)
        )
    retrieval_log = result.get("retrieval_log")
    return ChatResponse(response=response_text, session_id=session_id, retrieval_log=retrieval_log)


def _get_or_create_library_interview_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in library_interview_sessions:
        library_interview_sessions[session_id] = []
    return session_id


@app.post("/library-interview", response_model=LibraryInterviewResponse)
async def library_interview(req: LibraryInterviewRequest):
    """
    One turn of the library interview: user message + optional library snapshot.
    Agent asks about books and may save short notes; notes_saved lists any updates.
    """
    session_id = _get_or_create_library_interview_session(req.session_id)
    messages = library_interview_sessions[session_id]

    library_items = []
    if req.library_snapshot:
        for it in req.library_snapshot:
            library_items.append({
                "id": it.id,
                "title": it.title or "?",
                "author": (it.author or "").strip() or None,
                "note": (it.note or "").strip() or None,
            })

    user_message = (req.message or "").strip()
    if not user_message and not library_items:
        return LibraryInterviewResponse(
            response="Add some books to your library first, then we can chat about them.",
            session_id=session_id,
        )
    if not user_message:
        user_message = "Start"

    try:
        reply, notes_saved = await asyncio.wait_for(
            asyncio.to_thread(
                run_library_interview,
                list(messages),
                library_items,
                user_message,
            ),
            timeout=LIBRARY_INTERVIEW_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        return LibraryInterviewResponse(
            response="That took a bit long—try again in a moment.",
            session_id=session_id,
        )
    except Exception as e:
        print("[backend] /library-interview error:", e)
        return LibraryInterviewResponse(
            response="Something went wrong. Please try again.",
            session_id=session_id,
        )

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": reply})
    library_interview_sessions[session_id] = messages

    return LibraryInterviewResponse(
        response=reply,
        session_id=session_id,
        notes_saved=notes_saved if notes_saved else None,
    )


@app.post("/end-session", response_model=EndSessionResponse)
async def end_session(req: EndSessionRequest):
    """Trigger Librarian: extract, embed, save to SQLite+sqlite-vec (and LightRAG when enabled)."""
    session_id = get_or_create_session(req.session_id)
    messages = sessions.get(session_id, [])

    state: JournalState = {
        "messages": messages,
        "session_id": session_id,
        "personalization": 1.0,
    }
    state = await asyncio.to_thread(librarian_graph.invoke, state)
    # Feed LightRAG in background so response returns fast
    summary = state.get("last_summary") or ""
    facts = state.get("last_facts") or []
    if summary or facts:
        parts = []
        if summary:
            parts.append(f"Summary: {summary}")
        if facts:
            parts.append("Facts: " + "; ".join(facts))
        doc = "\n\n".join(parts)

        async def _bg_lightrag():
            try:
                from lightrag_bridge import insert_text as lightrag_insert
                await lightrag_insert(doc)
            except Exception as e:
                print("[backend] LightRAG insert after end_session:", e)

        asyncio.create_task(_bg_lightrag())

    return EndSessionResponse(ok=True, session_id=session_id)


@app.post("/ingest-history", response_model=IngestHistoryResponse)
async def ingest_history(req: IngestHistoryRequest):
    """
    Ingest a prior journal text into SQLite+sqlite-vec and LightRAG.
    Treats `text` as a single-session transcript. LightRAG gets same summary+facts as vec_store.
    Returns 200 always (so CORS headers are sent); ok=False on failure.
    """
    session_id = req.session_id or f"import-{uuid.uuid4()}"
    try:
        text = (req.text or "").strip()
        if not text:
            return IngestHistoryResponse(ok=True, session_id=session_id)
        entry_date = (req.entry_date or "").strip() or None
        extracted = await asyncio.to_thread(save_session_data, session_id, text, entry_date=entry_date)
        summary = extracted.get("summary") or ""
        facts = extracted.get("facts") or []
        if summary or facts:
            parts = []
            if summary:
                parts.append(f"Summary: {summary}")
            if facts:
                parts.append("Facts: " + "; ".join(facts))
            doc = "\n\n".join(parts)

            async def _bg_lightrag():
                try:
                    from lightrag_bridge import insert_text as lightrag_insert
                    await lightrag_insert(doc)
                except Exception as e:
                    print("[backend] LightRAG insert after ingest:", e)

            asyncio.create_task(_bg_lightrag())

        return IngestHistoryResponse(ok=True, session_id=session_id)
    except Exception as e:
        import traceback
        print("[backend] ingest_history error:", e)
        traceback.print_exc()
        return IngestHistoryResponse(ok=False, session_id=session_id)


def _parse_date_from_entry_text(text: str) -> Optional[str]:
    """Extract an explicit date from the entry text (absolute precedence). Prefer start of text (when the entry was written)."""
    if not text or not text.strip():
        return None
    # Look in first 500 chars for a clear written date (header/timestamp)
    head = text.strip()[:500]
    # MM/DD/YYYY or M/D/YYYY (e.g. "12/18/2025 at 5:47am" or "12/18/2025:")
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", head)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    # YYYY-MM-DD
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", head)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    # MM-DD-YYYY
    m = re.search(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b", head)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    return None


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Try to extract a date from a filename (e.g. 2025-03-01.txt, journal_mar5_2024). Returns ISO 8601 or None."""
    base = Path(filename).stem
    # YYYY-MM-DD or YYYY_MM_DD
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", base)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    # MM-DD-YYYY or MM_DD_YYYY (US style)
    m = re.search(r"(\d{1,2})[-_](\d{1,2})[-_](\d{4})", base)
    if m:
        mo, d, y = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    # YYYYMMDD
    m = re.search(r"(\d{4})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT12:00:00.000Z")
        except ValueError:
            pass
    return None


INFER_DATE_SYSTEM = """You are a date extractor. Determine the single date when this journal entry was written.

CRITICAL: What is written in the entry has absolute precedence. Use the exact day, month, and year as written in the entry (e.g. if it says 12/18/2025, output 2025-12-18—do not change the year). The filename must be ignored whenever the entry text contains any date or time; only use the filename when the entry text contains no date at all.

Reply with ONLY a single line: either an ISO 8601 date in UTC (e.g. 2025-12-18T12:00:00.000Z), or the word NONE if you cannot determine a date from the entry. No other text."""


@app.post("/infer-entry-date", response_model=InferEntryDateResponse)
async def infer_entry_date(req: InferEntryDateRequest):
    """
    Use an LLM to infer the best-guess date/time when a journal entry was written.
    Returns ISO 8601 string or null if unclear.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        return InferEntryDateResponse(date=None)
    try:
        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_INFER_ENTRY_DATE_MODEL", "gemini-3.1-flash-lite-preview")
        # Truncate very long text to avoid token limits
        text = (req.text or "")[:8000].strip()
        if not text:
            return InferEntryDateResponse(date=None)
        # Written text has absolute precedence: if the entry contains an explicit date, use it and do not use filename or LLM
        from_entry = _parse_date_from_entry_text(text)
        if from_entry:
            return InferEntryDateResponse(date=from_entry)
        filename = (req.filename or "").strip() or None
        prompt = INFER_DATE_SYSTEM + "\n\nEntry:\n" + text
        if filename:
            prompt += f'\n\nOnly if the entry above has no date: file name is "{filename}". Otherwise ignore the filename.'

        def _call():
            return client.models.generate_content(model=model, contents=prompt)

        result = await asyncio.to_thread(_call)
        raw = getattr(result, "text", "")
        if isinstance(raw, list):
            raw = " ".join(c.get("text", str(c)) for c in raw if isinstance(c, dict))
        raw = (raw or "").strip()
        if not raw or raw.upper() == "NONE":
            # Fallback: try to parse date from filename when text gave no date
            if filename:
                parsed = _parse_date_from_filename(filename)
                if parsed:
                    return InferEntryDateResponse(date=parsed)
            return InferEntryDateResponse(date=None)
        # Try to parse as ISO; accept if it looks like ISO date
        iso = raw.split()[0].strip()
        if not iso:
            return InferEntryDateResponse(date=None)
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return InferEntryDateResponse(date=dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        except Exception:
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(iso[:26], fmt)
                return InferEntryDateResponse(date=dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
            except Exception:
                continue
        return InferEntryDateResponse(date=None)
    except Exception as e:
        print("[backend] /infer-entry-date error:", e)
        return InferEntryDateResponse(date=None)


@app.get("/memory-stats", response_model=MemoryStats)
async def memory_stats():
    """
    Lightweight stats endpoint to verify vector store is populated.
    """
    import vec_store

    vec_store.ensure_db()
    return MemoryStats(
        gist_facts_count=vec_store.gist_count(),
        episodic_log_count=vec_store.episodic_count(),
        episodic_metadata_count=vec_store.episodic_metadata_count(),
    )


@app.get("/memory/facts")
async def get_memory_facts():
    """List all gist facts with ids for Memory UI (view/edit/delete)."""
    try:
        items = await asyncio.to_thread(list_memory_facts)
        return {"facts": [MemoryItem(**x) for x in items]}
    except Exception as e:
        print("[backend] GET /memory/facts error:", e)
        return {"facts": []}


@app.get("/memory/summaries")
async def get_memory_summaries():
    """List all episodic summaries with ids for Memory UI."""
    try:
        items = await asyncio.to_thread(list_memory_summaries)
        return {"summaries": [MemoryItem(**x) for x in items]}
    except Exception as e:
        print("[backend] GET /memory/summaries error:", e)
        return {"summaries": []}


@app.patch("/memory/facts/{fact_id}")
async def update_memory_fact_route(fact_id: int, req: MemoryFactUpdate):
    """Update a gist fact by id; re-embeds and updates store."""
    try:
        ok = await asyncio.to_thread(update_memory_fact, fact_id, req.document)
        return {"ok": ok}
    except Exception as e:
        print("[backend] PATCH /memory/facts error:", e)
        return {"ok": False}


@app.patch("/memory/summaries/{summary_id}")
async def update_memory_summary_route(summary_id: int, req: MemorySummaryUpdate):
    """Update an episodic summary by id; re-embeds and updates store. Optionally update metadata_json."""
    try:
        ok = await asyncio.to_thread(
            update_memory_summary, summary_id, req.document, req.metadata
        )
        return {"ok": ok}
    except Exception as e:
        print("[backend] PATCH /memory/summaries error:", e)
        return {"ok": False}


@app.delete("/memory/facts/{fact_id}")
async def delete_memory_fact_route(fact_id: int):
    """Delete a gist fact by id."""
    try:
        ok = await asyncio.to_thread(delete_memory_fact, fact_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /memory/facts error:", e)
        return {"ok": False}


@app.delete("/memory/summaries/{summary_id}")
async def delete_memory_summary_route(summary_id: int):
    """Delete an episodic summary by id."""
    try:
        ok = await asyncio.to_thread(delete_memory_summary, summary_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /memory/summaries error:", e)
        return {"ok": False}


@app.post("/memory/facts")
async def create_memory_fact(req: MemoryFactCreate):
    """Add a user-created fact; returns new id."""
    try:
        fid = await asyncio.to_thread(add_memory_fact, req.document)
        return {"ok": fid is not None, "id": fid}
    except Exception as e:
        print("[backend] POST /memory/facts error:", e)
        return {"ok": False, "id": None}


@app.post("/memory/summaries")
async def create_memory_summary(req: MemorySummaryCreate):
    """Add a user-created summary; returns new id."""
    try:
        sid = await asyncio.to_thread(add_memory_summary, req.document)
        return {"ok": sid is not None, "id": sid}
    except Exception as e:
        print("[backend] POST /memory/summaries error:", e)
        return {"ok": False, "id": None}


@app.get("/brain/people-graph", response_model=BrainPeopleGraphResponse)
async def brain_people_graph():
    """
    Return nodes and links for Brain -> People force-directed graph.
    Nodes: group and person; links: membership person -> group.
    """
    import vec_store
    import library

    vec_store.ensure_db()

    # Auto-populate people table from episodic metadata (people list) so the graph is never empty
    try:
        summaries = library.list_memory_summaries()
        seen_names: set[str] = set()
        for item in summaries:
            meta_json = item.get("metadata_json")
            if not meta_json:
                continue
            try:
                meta = json.loads(meta_json)
            except Exception:
                continue
            people = meta.get("people") or []
            if not isinstance(people, list):
                continue
            for name in people:
                if not isinstance(name, (str, int, float)):
                    continue
                s = str(name).strip()
                if not s or s in seen_names:
                    continue
                seen_names.add(s)
                vec_store.create_person(s)
    except Exception:
        # Graph should still render even if this seeding pass fails
        pass
    people = vec_store.list_people_with_groups()
    # Default groups to always show, even if empty
    default_groups = ["Family", "Friends", "School", "Work", "Healthcare"]
    group_set = set(default_groups)
    for p in people:
        for g in p.get("groups", []):
            group_set.add(g)

    nodes: list[BrainPersonNode] = []
    links: list[BrainPersonLink] = []

    # Group nodes
    for g in sorted(group_set):
        nodes.append(BrainPersonNode(id=f"group:{g}", label=g, type="group"))

    # Person nodes and links
    for p in people:
        pid = p["id"]
        name = p["name"]
        nodes.append(BrainPersonNode(id=f"person:{pid}", label=name, type="person"))
        for g in p.get("groups", []):
            links.append(
                BrainPersonLink(
                    source=f"group:{g}",
                    target=f"person:{pid}",
                )
            )

    return BrainPeopleGraphResponse(nodes=nodes, links=links)


@app.post("/brain/people/auto-groups", response_model=Dict)
async def brain_people_auto_groups():
    """
    Use an LLM to propose social groups (UNC Charlotte, CPCC, Mentors, etc.)
    and assign people to them. This updates person_groups and then the
    /brain/people-graph endpoint will reflect the new clusters.
    """
    await asyncio.to_thread(run_people_grouping_agent)
    return {"ok": True}


@app.get("/brain/people", response_model=List[Dict])
async def brain_people_list():
    """Return raw people list with groups for Brain UI."""
    import vec_store

    vec_store.ensure_db()
    return vec_store.list_people_with_groups()


@app.post("/brain/people", response_model=Dict)
async def brain_person_create(req: BrainPersonCreateRequest):
    """Create a new person node for Brain -> People graph."""
    import vec_store

    vec_store.ensure_db()
    pid = vec_store.create_person(req.name)
    return {"id": pid, "name": req.name}


@app.patch("/brain/people/{person_id}", response_model=Dict)
async def brain_person_rename(person_id: int, req: BrainPersonRenameRequest):
    """Rename an existing person."""
    import vec_store

    vec_store.ensure_db()
    ok = vec_store.update_person(person_id, req.name)
    return {"ok": ok}


@app.get("/brain/people/{person_id}", response_model=BrainPersonDetail)
async def brain_person_detail(person_id: int):
    """
    Return detail for a person: profile, groups, episodic events, and thoughts.
    Events are derived from episodic metadata where this person's name appears.
    """
    import vec_store

    vec_store.ensure_db()
    # Basic person record
    all_people = vec_store.list_people_with_groups()
    person_row = next((p for p in all_people if p["id"] == person_id), None)
    if not person_row:
        raise HTTPException(status_code=404, detail="Person not found")

    name = person_row["name"]
    # User-authored relationship state (Panel 1)
    profile_data = vec_store.get_person_profile(person_id) or {}
    relationship_state = BrainRelationshipState(
        user_summary=profile_data.get("relationship_summary", ""),
        relationship_type=profile_data.get("relationship_type", ""),
        closeness_level=profile_data.get("closeness_label", ""),
        updated_at="",  # can be extended later
    )
    groups = vec_store.get_person_groups(person_id)
    # Episodic events with this person (Panel 3)
    events = get_person_events(name)
    # AI relationship summary + factual profile (Panels 2 and 4)
    ai_summary = await asyncio.to_thread(run_relationship_summary_agent, person_id, name)
    facts = await asyncio.to_thread(run_person_facts_agent, person_id, name)
    thoughts = vec_store.list_person_thoughts(person_id)

    return BrainPersonDetail(
        id=person_id,
        name=name,
        relationship_state=relationship_state,
        ai_relationship_summary=ai_summary,
        groups=groups,
        events=events,
        facts=[BrainPersonFact(**f) for f in facts],
        thoughts=thoughts,
    )


@app.put("/brain/people/{person_id}/profile", response_model=Dict)
async def brain_person_profile_update(person_id: int, req: BrainPersonProfileUpdate):
    """Update relationship summary and structured relationship fields for a person."""
    import vec_store

    vec_store.ensure_db()
    summary = req.relationship_summary or ""
    rel_type = req.relationship_type or ""
    closeness = req.closeness_label or ""
    vec_store.upsert_person_profile(person_id, summary, rel_type, closeness)
    return {"ok": True}


@app.put("/brain/people/{person_id}/groups", response_model=Dict)
async def brain_person_groups_update(person_id: int, req: BrainPersonGroupsUpdate):
    """Replace groups for a person."""
    import vec_store

    vec_store.ensure_db()
    vec_store.set_person_groups(person_id, req.groups)
    return {"ok": True}


@app.post("/brain/people/{person_id}/thoughts", response_model=Dict)
async def brain_person_thought_create(person_id: int, req: BrainPersonThoughtCreate):
    """Add a new reflection/thought about a person."""
    import vec_store

    vec_store.ensure_db()
    tid = vec_store.add_person_thought(person_id, req.date, req.thought_text)
    return {"ok": tid is not None, "id": tid}


@app.patch("/brain/people/{person_id}/thoughts/{thought_id}", response_model=Dict)
async def brain_person_thought_update(person_id: int, thought_id: int, req: BrainPersonThoughtUpdate):
    """Edit an existing thought about a person."""
    import vec_store

    vec_store.ensure_db()
    # Fetch existing to preserve fields not provided
    existing = vec_store.list_person_thoughts(person_id)
    row = next((t for t in existing if t["id"] == thought_id), None)
    if not row:
        raise HTTPException(status_code=404, detail="Thought not found")
    new_date = req.date if req.date is not None else row.get("date") or None
    new_text = req.thought_text if req.thought_text is not None else row.get("thought_text", "")
    ok = vec_store.update_person_thought(thought_id, new_date, new_text)
    return {"ok": ok}


@app.delete("/brain/people/{person_id}/thoughts/{thought_id}", response_model=Dict)
async def brain_person_thought_delete(person_id: int, thought_id: int):
    """Delete a thought about a person."""
    import vec_store

    vec_store.ensure_db()
    ok = vec_store.delete_person_thought(thought_id)
    return {"ok": ok}


RECOMMENDATIONS_TIMEOUT_SEC = 120

@app.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations():
    """
    Generate personalized book, podcast, and article recommendations from journal memory
    and what the user has already consumed/liked. May take 30–90s; runs with a timeout to avoid connection resets.
    """
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(generate_recommendations),
            timeout=RECOMMENDATIONS_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print("[backend] /recommendations timed out after", RECOMMENDATIONS_TIMEOUT_SEC, "s")
        return RecommendationsResponse(books=[], podcasts=[], articles=[], research=[])
    except Exception as e:
        print("[backend] /recommendations error:", e)
        return RecommendationsResponse(books=[], podcasts=[], articles=[], research=[])
    return RecommendationsResponse(
        books=[RecommendationItem(**x) for x in data.get("books", [])],
        podcasts=[RecommendationItem(**x) for x in data.get("podcasts", [])],
        articles=[RecommendationItem(**x) for x in data.get("articles", [])],
        research=[RecommendationItem(**x) for x in data.get("research", [])],
    )


class CalendarDayRequest(BaseModel):
    date: str  # YYYY-MM-DD
    raw_transcript: Optional[str] = None


class CalendarDayResponse(BaseModel):
    summary: str
    has_journal: bool


@app.post("/calendar-day-summary", response_model=CalendarDayResponse)
async def calendar_day_summary(req: CalendarDayRequest):
    """
    For a given date, combine raw journal transcript (if any) with DB memory for that day
    and return an AI-generated day summary/highlights.
    """
    date_iso = (req.date or "").strip()[:10]
    if not date_iso or len(date_iso) < 10:
        return CalendarDayResponse(summary="Please provide a valid date (YYYY-MM-DD).", has_journal=False)
    try:
        episodic, gist = await asyncio.to_thread(get_memory_for_date, date_iso)
        summary = await asyncio.to_thread(
            generate_day_summary,
            date_iso,
            (req.raw_transcript or "").strip() or None,
            episodic,
            gist,
        )
        return CalendarDayResponse(
            summary=summary,
            has_journal=bool((req.raw_transcript or "").strip()),
        )
    except Exception as e:
        print("[backend] /calendar-day-summary error:", e)
        return CalendarDayResponse(
            summary="Could not generate summary for this day.",
            has_journal=bool((req.raw_transcript or "").strip()),
        )


@app.post("/recommendations/consumed", response_model=ConsumedResponse)
async def mark_consumed(req: ConsumedRequest):
    """
    Record that the user has read/listened to a recommendation. Stored in the vector store
    so future recommendations avoid repeats and better match their tastes.
    """
    content_type = (req.type or "article").lower()
    if content_type not in ("book", "podcast", "article", "research"):
        content_type = "article"
    try:
        add_consumed(
            content_type=content_type,
            title=req.title,
            author=req.author,
            url=req.url,
            liked=req.liked,
        )
        return ConsumedResponse(ok=True)
    except Exception as e:
        print("[backend] /recommendations/consumed error:", e)
        return ConsumedResponse(ok=False)


@app.get("/library")
async def get_library():
    """
    Return consumed items grouped by type for the Library UI.
    """
    try:
        data = await asyncio.to_thread(list_consumed)
        return data
    except Exception as e:
        print("[backend] GET /library error:", e)
        return {"books": [], "podcasts": [], "articles": [], "research": []}


@app.patch("/library/{item_id}")
async def update_library_item(item_id: str, req: LibraryItemUpdate):
    """
    Update date_completed and/or note for a library item by id.
    """
    try:
        ok = await asyncio.to_thread(
            update_consumed,
            item_id,
            date_completed=req.date_completed,
            note=req.note,
        )
        return {"ok": ok}
    except Exception as e:
        print("[backend] PATCH /library error:", e)
        return {"ok": False}


@app.delete("/library/{item_id}")
async def delete_library_item(item_id: str):
    """
    Remove a library item from the consumed collection.
    """
    try:
        ok = await asyncio.to_thread(delete_consumed, item_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /library error:", e)
        return {"ok": False}


@app.post("/library-notes", response_model=LibraryNoteResponse)
async def library_notes(req: LibraryNoteRequest):
    """
    Library helper endpoint: user can paste titles or notes about books, podcasts,
    articles, or research they've read. An agent organizes this into structured
    consumed items to improve future recommendations.
    """
    try:
        from library import process_library_note

        count = await asyncio.to_thread(process_library_note, req.text, req.type)
        return LibraryNoteResponse(ok=count > 0, items_added=count)
    except Exception as e:
        print("[backend] /library-notes error:", e)
        return LibraryNoteResponse(ok=False, items_added=0)


@app.get("/lightrag-context")
async def lightrag_context(q: str = "", mode: str = "hybrid"):
    """
    Optional RAG context from LightRAG (knowledge-graph + vector). Use when LightRAG is enabled.
    Query param: q=... (required), mode=local|global|hybrid|naive|mix (default hybrid).
    """
    if not (q or "").strip():
        return {"context": ""}
    try:
        from lightrag_bridge import query_for_context
        context = await query_for_context(q.strip(), mode=mode)
        return {"context": context}
    except Exception as e:
        print("[backend] /lightrag-context error:", e)
        return {"context": ""}


@app.post("/memory-wipe")
async def memory_wipe():
    """
    Wipe gist and episodic memory (SQLite+sqlite-vec). Consumed library is kept.
    """
    wipe_memory()
    return {"ok": True, "message": "Memory wiped."}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
