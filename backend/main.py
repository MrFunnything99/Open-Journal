"""
FastAPI backend for Open-Journal: /chat and /end-session.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
    generate_memory_mermaid,
    generate_recommendations,
    get_memory_for_visualization,
    list_consumed,
    list_memory_facts,
    list_memory_summaries,
    save_session_data,
    update_consumed,
    update_memory_fact,
    update_memory_summary,
    wipe_memory,
)
from google import genai

# In-memory session store (minimal for 1hr sprint; replace with Redis/DB later)
sessions: dict[str, list] = {}

CHAT_INVOKE_TIMEOUT_SEC = 60


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
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip():
        print("[backend] GEMINI_API_KEY is set (ready for /chat and LLM ops)")
    else:
        print("[backend] WARNING: GEMINI_API_KEY is missing. Set it in .env at project root and restart.")
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


class ChatResponse(BaseModel):
    response: str
    session_id: str
    retrieval_log: Optional[str] = None


class EndSessionRequest(BaseModel):
    session_id: str


class EndSessionResponse(BaseModel):
    ok: bool
    session_id: str


class IngestHistoryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class IngestHistoryResponse(BaseModel):
    ok: bool
    session_id: str


class InferEntryDateRequest(BaseModel):
    text: str


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
    Return a Mermaid diagram string generated by LLM from vector DB content (gist facts + episodic summaries).
    """
    gist_facts, episodic_summaries = get_memory_for_visualization()
    mermaid_code = generate_memory_mermaid(gist_facts, episodic_summaries)
    return MemoryDiagramResponse(mermaid=mermaid_code)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """User sends text; Interviewer responds. State updated in memory."""
    session_id = get_or_create_session(req.session_id)
    messages = sessions[session_id]

    # Clamp personalization between 0 and 1.0, default to 1.0 if missing.
    personalization = req.personalization if req.personalization is not None else 1.0
    try:
        personalization = float(personalization)
    except (TypeError, ValueError):
        personalization = 1.0
    personalization = max(0.0, min(1.0, personalization))

    # Clamp intrusiveness between 0 and 1.0, default to 0.5 if missing.
    intrusiveness = req.intrusiveness if req.intrusiveness is not None else 0.5
    try:
        intrusiveness = float(intrusiveness)
    except (TypeError, ValueError):
        intrusiveness = 0.5
    intrusiveness = max(0.0, min(1.0, intrusiveness))

    # Append user message
    messages.append(HumanMessage(content=req.text))

    # Run Interviewer in a thread so we don't block the event loop; timeout to avoid hanging
    state: JournalState = {
        "messages": list(messages),
        "session_id": session_id,
        "personalization": personalization,
        "intrusiveness": intrusiveness,
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

    # Persist updated messages
    sessions[session_id] = list(result["messages"])

    # Last message is the assistant reply
    last = result["messages"][-1]
    response_text = getattr(last, "content", str(last))
    if isinstance(response_text, list):
        response_text = " ".join(
            c.get("text", str(c)) for c in response_text if isinstance(c, dict)
        )
    retrieval_log = result.get("retrieval_log")

    return ChatResponse(response=response_text, session_id=session_id, retrieval_log=retrieval_log)


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
        extracted = await asyncio.to_thread(save_session_data, session_id, text)
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


INFER_DATE_SYSTEM = """You are a date extractor. Given journal text, determine the single date and time when this journal entry was written.
The text may mention multiple dates (e.g. past events, "yesterday", "last week"). Choose the date that best represents when the author wrote this entry (e.g. "It is currently January 5, 2026, 1:59 a.m.").
Reply with ONLY a single line: either an ISO 8601 date-time in UTC (e.g. 2026-01-05T01:59:00.000Z), or the word NONE if you cannot determine it. No other text."""


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
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash")
        # Truncate very long text to avoid token limits
        text = (req.text or "")[:8000].strip()
        if not text:
            return InferEntryDateResponse(date=None)
        prompt = INFER_DATE_SYSTEM + "\n\nEntry:\n" + text

        def _call():
            return client.models.generate_content(model=model, contents=prompt)

        result = await asyncio.to_thread(_call)
        raw = getattr(result, "text", "")
        if isinstance(raw, list):
            raw = " ".join(c.get("text", str(c)) for c in raw if isinstance(c, dict))
        raw = (raw or "").strip()
        if not raw or raw.upper() == "NONE":
            return InferEntryDateResponse(date=None)
        # Try to parse as ISO; accept if it looks like ISO date
        from datetime import datetime
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
