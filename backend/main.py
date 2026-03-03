"""
FastAPI backend for Open-Journal: /chat and /end-session.
"""
from __future__ import annotations

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
    COLLECTION_EPISODIC,
    COLLECTION_GIST,
    _get_chroma,
    save_session_data,
)

# In-memory session store (minimal for 1hr sprint; replace with Redis/DB later)
sessions: dict[str, list] = {}


def get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    return session_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: log whether env is loaded (no secrets)
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        print("[backend] OPENROUTER_API_KEY is set (ready for /chat)")
    else:
        print("[backend] WARNING: OPENROUTER_API_KEY is missing. Set it in .env at project root and restart.")
    yield
    # Cleanup if needed
    pass


app = FastAPI(title="Open-Journal Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_graph = build_graph()
librarian_graph = build_librarian_graph()


class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


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


class MemoryStats(BaseModel):
    gist_facts_count: int
    episodic_log_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """User sends text; Interviewer responds. State updated in memory."""
    session_id = get_or_create_session(req.session_id)
    messages = sessions[session_id]

    # Append user message
    messages.append(HumanMessage(content=req.text))

    # Run Interviewer
    state: JournalState = {"messages": messages, "session_id": session_id}
    result = chat_graph.invoke(state)

    # Persist updated messages
    sessions[session_id] = list(result["messages"])

    # Last message is the assistant reply
    last = result["messages"][-1]
    response_text = getattr(last, "content", str(last))
    if isinstance(response_text, list):
        response_text = " ".join(
            c.get("text", str(c)) for c in response_text if isinstance(c, dict)
        )

    return ChatResponse(response=response_text, session_id=session_id)


@app.post("/end-session", response_model=EndSessionResponse)
async def end_session(req: EndSessionRequest):
    """Trigger Librarian: extract, embed, save to ChromaDB."""
    session_id = get_or_create_session(req.session_id)
    messages = sessions.get(session_id, [])

    state: JournalState = {"messages": messages, "session_id": session_id}
    librarian_graph.invoke(state)

    return EndSessionResponse(ok=True, session_id=session_id)


@app.post("/ingest-history", response_model=IngestHistoryResponse)
async def ingest_history(req: IngestHistoryRequest):
    """
    Ingest a prior journal text directly into Chroma memory.
    Treats `text` as a single-session transcript.
    """
    session_id = req.session_id or f"import-{uuid.uuid4()}"
    save_session_data(session_id, req.text)
    return IngestHistoryResponse(ok=True, session_id=session_id)


@app.get("/memory-stats", response_model=MemoryStats)
async def memory_stats():
    """
    Lightweight stats endpoint to verify Chroma is populated.
    """
    client = _get_chroma()
    gist = client.get_collection(COLLECTION_GIST)
    episodic = client.get_collection(COLLECTION_EPISODIC)
    return MemoryStats(
        gist_facts_count=gist.count(),
        episodic_log_count=episodic.count(),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
