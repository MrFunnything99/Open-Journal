"""
FastAPI backend for Selfmeridian: /chat and /end-session.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRouter
from pydantic import BaseModel

# Load .env from project root first (parent of backend/), then cwd
_root = Path(__file__).resolve().parent.parent
for name in (".env", ".env.local"):
    load_dotenv(_root / name)
    load_dotenv(Path.cwd() / name)
    load_dotenv(Path.cwd().parent / name)

import vec_store
from lightrag_bridge import query_for_context, schedule_lightrag_index_after_ingest
from graph import build_graph, build_librarian_graph, JournalState
from langchain_core.messages import AIMessage, HumanMessage
from library import (
    add_consumed,
    add_memory_fact,
    add_memory_summary,
    delete_consumed,
    delete_memory_fact,
    delete_memory_summary,
    DEFAULT_PERPLEXITY_EMBEDDING_MODEL,
    generate_day_summary,
    generate_memory_mermaid,
    _embed_texts,
    generate_recommendations,
    generate_recommendations_category,
    generate_derived_insights,
    get_memory_for_date,
    get_memory_for_visualization,
    get_person_events,
    get_writing_loop_hints,
    list_consumed,
    list_memory_facts,
    list_memory_summaries,
    run_library_interview,
    run_person_facts_agent,
    run_relationship_summary_agent,
    run_people_grouping_agent,
    process_content_feedback,
    record_rec_feedback_for_recs,
    refresh_pattern_memory,
    save_session_data,
    extraction_llm_backend,
    library_recommendations_llm_label,
    _openrouter_chat_completion,
    update_consumed,
    update_memory_fact,
    update_memory_summary,
    wipe_memory,
)

def _instance_id(request: Request) -> str:
    """Per-browser instance id for scoping memory (sent as X-Instance-ID)."""
    return (request.headers.get("X-Instance-ID") or "").strip()


# In-memory session store (minimal for 1hr sprint; replace with Redis/DB later)
sessions: dict[str, list] = {}
library_interview_sessions: dict[str, list] = {}  # session_id -> list of {role, content}

# Chat may run OpenRouter + tool round-trips + external API lookups (e.g. Open Library); keep headroom.
CHAT_INVOKE_TIMEOUT_SEC = 180
LIBRARY_INTERVIEW_TIMEOUT_SEC = 45


def get_or_create_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    return session_id


def _session_messages_for_client(session_id: str) -> List[Dict[str, str]]:
    """Serialize in-memory LangChain transcript for the chat UI (user/assistant turns only)."""
    out: List[Dict[str, str]] = []
    for m in sessions.get(session_id) or []:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            continue
        c = getattr(m, "content", "")
        if isinstance(c, list):
            c = " ".join(
                part.get("text", str(part)) for part in c if isinstance(part, dict)
            )
        text = (str(c) if c is not None else "").strip()
        if text:
            out.append({"role": role, "content": text})
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Avoid proxy 403s: send API calls (OpenRouter, Listen Notes) direct, not via system proxy
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(v, None)
    os.environ["NO_PROXY"] = "*"
    # Startup: pipeline roles (no secrets)
    emb_model = (os.getenv("PERPLEXITY_EMBEDDING_MODEL") or DEFAULT_PERPLEXITY_EMBEDDING_MODEL).strip()
    pplx_key = (os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY") or "").strip()
    print(f"[backend] Embeddings: Perplexity ({emb_model})")
    if not pplx_key:
        print("[backend] WARNING: PERPLEXITY_API_KEY missing — vector ingest and retrieval will fail until set.")

    print(f"[backend] Extraction / library LLM: {extraction_llm_backend()}")
    print(f"[backend] Library recommendations (books, articles, research): {library_recommendations_llm_label()}")
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    chat_model = (os.getenv("OPENROUTER_CHAT_MODEL") or "openai/gpt-4.1-mini").strip()
    chat_fallback = (os.getenv("OPENROUTER_CHAT_FALLBACK_MODEL") or "openai/gpt-5.4").strip()
    convo_model = (os.getenv("OPENROUTER_CONVERSATION_MODEL") or "x-ai/grok-4.1-fast").strip()
    if or_key:
        print(
            f"[backend] OPENROUTER_API_KEY is set — /chat journal ({chat_model}; fallback {chat_fallback}), "
            f"conversation ({convo_model} + reasoning), journal validation, "
            "voice-memo polish, date inference, and library extraction helpers"
        )
        print(
            f"[backend] Speech-to-text: OpenRouter ({(os.getenv('OPENROUTER_TRANSCRIPTION_MODEL') or 'openai/gpt-audio-mini').strip()})"
        )
    elif (os.getenv("OPENAI_API_KEY") or "").strip():
        print("[backend] Speech-to-text: OpenAI direct (OPENAI_TRANSCRIPTION_MODEL / gpt-4o-mini-transcribe)")
    elif (os.getenv("ELEVENLABS_API_KEY") or "").strip():
        print("[backend] Speech-to-text: ElevenLabs scribe_v2")
    else:
        print("[backend] WARNING: Set OPENROUTER_API_KEY (recommended for STT), or OPENAI_API_KEY, or ELEVENLABS_API_KEY — transcription routes need one of these.")
    if not or_key:
        print(
            "[backend] WARNING: OPENROUTER_API_KEY missing — /chat interviewer and journal validation will not work until set."
        )
    vec_store.ensure_db()
    try:
        vec_store.decision_log_rotate()
    except Exception as e:
        print("[backend] decision_log_rotate startup:", e)
    yield
    # Cleanup if needed
    pass


app = FastAPI(title="Selfmeridian Backend", lifespan=lifespan)
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

api_router = APIRouter()
chat_graph = build_graph()
librarian_graph = build_librarian_graph()


class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    personalization: Optional[float] = None  # ignored for graph /chat; server uses 1.0 (full memory)
    intrusiveness: Optional[float] = None
    mode: Optional[str] = None  # "journal" | "conversation" | "autobiography" (Assisted Journal) | "recommendations" | "learning"
    # Allowlisted OpenRouter id for conversation + autobiography only (see graph.USER_SELECTABLE_CHAT_MODELS)
    openrouter_model: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    retrieval_log: Optional[str] = None
    notes_saved: Optional[List[Dict[str, str]]] = None  # [{"item_id": "...", "note": "..."}]
    library_items_added: Optional[int] = None  # journal mode: agent saved N items to Library
    agent_steps: Optional[List[Dict[str, Any]]] = None  # retrieval + tool summaries for UI
    # Allowlisted UI actions from journal chat agent (e.g. navigate). Frontend must ignore unknown types.
    actions: Optional[List[Dict[str, Any]]] = None


class ChatSessionHistoryMessage(BaseModel):
    role: str
    content: str


class ChatSessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[ChatSessionHistoryMessage]


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
    content_hash: Optional[str] = None  # optional client hint; server uses canonical normalized SHA-256 for skip gate


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
    gist_facts_count: int  # journal entry count (legacy field name for API clients)
    episodic_log_count: int  # journal chunk count (legacy field name)
    episodic_metadata_count: int = 0  # unused; journal chunks have no episodic metadata
    journal_entry_count: int = 0
    journal_chunk_count: int = 0


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
    news: list[RecommendationItem] = []


class ConsumedRequest(BaseModel):
    type: str  # "book" | "podcast" | "article"
    title: str
    author: Optional[str] = None
    url: Optional[str] = None
    liked: bool = True


class ConsumedResponse(BaseModel):
    ok: bool


class WritingHintsRequest(BaseModel):
    draft: str = ""


class RecFeedbackRequest(BaseModel):
    action: str
    content_type: Optional[str] = None
    topic_tags: Optional[str] = None
    intent_context: Optional[str] = None
    item_title: Optional[str] = None


class RecFeedbackResponse(BaseModel):
    ok: bool


class ContentFeedbackRequest(BaseModel):
    content_title: str
    content_type: str = "article"
    content_url: Optional[str] = None
    feedback: str = "liked"
    user_notes: Optional[str] = None


class LibraryNoteRequest(BaseModel):
    text: str
    type: Optional[str] = None  # "book" | "podcast" | "article" | "research" to restrict to one category


class LibraryNoteResponse(BaseModel):
    ok: bool
    items_added: int


class LibraryBulkImportItem(BaseModel):
    id: str
    type: str  # book | podcast | article | research
    title: str
    author: Optional[str] = None
    note: Optional[str] = None
    date_completed: Optional[str] = None
    url: Optional[str] = None
    liked: bool = True


class LibraryBulkImportRequest(BaseModel):
    items: List[LibraryBulkImportItem]


class LibraryBulkImportResponse(BaseModel):
    ok: bool
    count: int = 0


class LibraryItemUpdate(BaseModel):
    date_completed: Optional[str] = None
    note: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None


class MemoryDiagramResponse(BaseModel):
    mermaid: str


# --- ElevenLabs proxy: TTS and voices (no auth required; used by session UI) ---
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_SCRIBE_TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
FALLBACK_VOICES = [
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
    {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
    {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
    {"voice_id": "onwK4e9ZLuTAKqWW03F9", "name": "Domi"},
    {"voice_id": "N2lVS1w4EtoT3dr4eOWO", "name": "Sam"},
]

_MISTRAL_VOICE_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)
# Preset voices from Mistral (voice_id can be a slug, not only a saved UUID).
_MISTRAL_PRESET_VOICE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,62}$", re.I)

# Catalog voice slug from GET /v1/audio/voices (e.g. "Paul - Neutral" → en_paul_neutral).
DEFAULT_MISTRAL_TTS_VOICE_ID = "en_paul_neutral"


def _mistral_voice_id_valid(raw: str) -> bool:
    t = raw.strip()
    if not t:
        return False
    return bool(_MISTRAL_VOICE_UUID_RE.match(t) or _MISTRAL_PRESET_VOICE_RE.match(t))


def _resolve_mistral_voice_id(req_voice_id: Optional[str]) -> str:
    """
    Voxtral voice_id: saved voice UUID or catalog slug from /v1/audio/voices (e.g. en_paul_neutral).
    https://docs.mistral.ai/capabilities/audio/text_to_speech
    """
    env_vid = os.getenv("MISTRAL_TTS_VOICE_ID", "").strip()
    if env_vid:
        return env_vid
    if req_voice_id and _mistral_voice_id_valid(req_voice_id):
        return req_voice_id.strip()
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if api_key:
        import urllib.request

        try:
            request = urllib.request.Request(
                "https://api.mistral.ai/v1/audio/voices?limit=30&offset=0",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(request, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items") or []
            # Prefer the default catalog neutral voice if Mistral lists it (slug or UUID).
            want = DEFAULT_MISTRAL_TTS_VOICE_ID.lower()
            for it in items:
                if not isinstance(it, dict):
                    continue
                slug = ((it.get("slug") or "") or "").lower()
                vid = (it.get("id") or "").strip()
                if slug == want or vid.lower() == want:
                    return (it.get("slug") or it.get("id") or "").strip() or vid
            if len(items) == 1 and isinstance(items[0], dict):
                vid = (items[0].get("id") or "").strip()
                if vid:
                    return vid
        except Exception as e:
            print("[backend] Mistral voices list:", e)
    return DEFAULT_MISTRAL_TTS_VOICE_ID


def _mistral_tts_playback_rate() -> float:
    """Client hint for HTMLAudioElement.playbackRate (API has no speed field in OpenAPI)."""
    raw = (os.getenv("MISTRAL_TTS_PLAYBACK_RATE") or "1.05").strip()
    try:
        r = float(raw)
    except (TypeError, ValueError):
        return 1.05
    if r < 0.25 or r > 4.0:
        return 1.05
    return r


def _mistral_tts_response_format() -> str:
    """Smaller/faster-to-encode opus vs mp3 for chat read-aloud; override with MISTRAL_TTS_RESPONSE_FORMAT."""
    raw = (os.getenv("MISTRAL_TTS_RESPONSE_FORMAT") or "opus").strip().lower()
    if raw in ("opus", "mp3", "wav", "flac"):
        return raw
    return "opus"


def _mistral_tts_speech(text: str, voice_id: str, response_format: str) -> bytes:
    """https://docs.mistral.ai/capabilities/audio/text_to_speech/speech"""
    import base64 as b64mod
    import urllib.request
    import urllib.error

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key or not voice_id:
        raise ValueError("Mistral TTS not configured (API key or voice id missing).")
    model = (os.getenv("MISTRAL_TTS_MODEL") or "voxtral-mini-tts-2603").strip()
    inp = text.strip()
    if len(inp) > 12_000:
        inp = inp[:12_000] + "…"
    fmt = (response_format or "opus").strip().lower()
    if fmt not in ("opus", "mp3", "wav", "flac", "pcm"):
        fmt = "opus"
    body = json.dumps({
        "model": model,
        "input": inp,
        "voice_id": voice_id,
        "response_format": fmt,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.mistral.ai/v1/audio/speech",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body) if err_body.strip() else {}
            detail = err_json.get("detail") or err_json.get("message")
            if isinstance(detail, list) and detail:
                detail = detail[0] if isinstance(detail[0], str) else detail[0].get("msg")
            if isinstance(detail, dict):
                detail = detail.get("message") or str(detail)
            msg = detail if isinstance(detail, str) else (err_body[:400] if err_body else str(e))
        except Exception:
            msg = str(e)
        print(f"[backend] Mistral TTS HTTP {e.code}: {msg}")
        raise ValueError(msg or f"Mistral TTS failed (HTTP {e.code})") from e
    if not isinstance(payload, dict):
        raise ValueError("Invalid Mistral TTS response")
    audio_b64 = payload.get("audio_data")
    if not audio_b64 or not isinstance(audio_b64, str):
        raise ValueError("Mistral TTS returned no audio_data")
    return b64mod.b64decode(audio_b64)


class VoiceRequest(BaseModel):
    text: str
    voiceId: Optional[str] = None
    voice_settings: Optional[Dict[str, float]] = None


class TranscribeRequest(BaseModel):
    """Batch STT for voice-memo flow (iOS Safari, etc.). Same contract as api/transcribe.ts."""
    audio: str
    format: Optional[str] = None


class VoiceMemoRequest(BaseModel):
    """Voice Memo tab: base64 audio → OpenRouter gpt-audio-mini (preferred) or OpenAI / ElevenLabs → optional OpenRouter text polish."""
    audio: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    journal_mode: bool = False


class JournalValidateRequest(BaseModel):
    text: str
    model: Optional[str] = None


class JournalValidateResponse(BaseModel):
    reformatted_journal: str
    feedback: str
    validation_notes: list[str] = []
    model_used: Optional[str] = None


VOICE_MEMO_POLISH_INSTRUCTION = """You are editing a voice memo transcript. Clean it into clear, readable prose suitable for a personal journal.
- Fix punctuation, capitalization, and obvious speech-to-text errors.
- Remove filler words (um, uh, like, you know), false starts, and repeated words.
- Preserve meaning, factual content, and the author's wording as much as possible; do not invent events.
- Use first person when the speaker is reflecting on themselves.
- Do NOT rewrite the text as a diary entry or narrative; keep the author's sentence structure.
- Do NOT add new content, headings, sign-offs, or commentary.
Output ONLY the cleaned text with no title or preamble."""

JOURNAL_VALIDATE_INSTRUCTION = """You are a supportive assistant helping someone turn a voice transcript into readable journal text.

Your first job is reformatting only: make the passage coherent and pleasant to read. Smooth out filler and repeated words (e.g. "um", stutters, accidental duplicates), fix obvious transcription errors and misspellings, and resolve small incoherencies—without changing what the author meant. Preserve roughly 98% of their wording; only light edits. Do not invent facts or add new events.

Your second job is short, friendly feedback or light reflection. Stay warm and uplifting toward the writer. Never attack or belittle them. You may name difficult realities in a protective, caring way (e.g. a situation sounds unsafe, or someone treated them poorly).

Do not ask questions or prompt for replies; the writer will respond to feedback on their own.

Respond in this exact format:
===REFORMATTED===
<journal text>
===FEEDBACK===
<feedback text>
===VALIDATION===
- <0–3 short optional lines: only gentle or protective observations; if none, write a single line: - (none)>"""


def _guess_audio_filename(filename: Optional[str], mime_type: Optional[str]) -> str:
    fn = (filename or "").strip()
    if fn and "." in fn:
        base = fn.replace("\\", "/").split("/")[-1]
        if len(base) <= 120:
            return base
    mt = (mime_type or "").lower()
    if "mp3" in mt or mt == "audio/mpeg":
        return "audio.mp3"
    if "mp4" in mt or "m4a" in mt or "aac" in mt:
        return "audio.m4a"
    if "wav" in mt:
        return "audio.wav"
    if "ogg" in mt or "opus" in mt:
        return "audio.ogg"
    if "flac" in mt:
        return "audio.flac"
    return "recording.webm"


def _openrouter_transcription_model() -> str:
    return (os.getenv("OPENROUTER_TRANSCRIPTION_MODEL") or "openai/gpt-audio-mini").strip() or "openai/gpt-audio-mini"


def _openrouter_audio_format(filename: str, mime_type: Optional[str], format_hint: Optional[str] = None) -> str:
    """Format string for OpenRouter input_audio (see https://openrouter.ai/docs/guides/overview/multimodal/audio)."""
    h = (format_hint or "").strip().lower().lstrip(".")
    allowed = {"wav", "mp3", "aac", "ogg", "flac", "m4a", "aiff", "pcm16", "pcm24"}
    if h in allowed:
        return h
    fn = (filename or "").lower()
    ext = fn.rsplit(".", 1)[-1] if "." in fn else ""
    if ext in allowed:
        return ext
    if ext in ("mp4", "mpeg"):
        return "m4a"
    if ext == "webm":
        return "webm"
    if ext == "wav":
        return "wav"
    mt = (mime_type or "").lower()
    if "webm" in mt:
        return "webm"
    if "mp4" in mt or "m4a" in mt or "aac" in mt:
        return "m4a"
    if "mpeg" in mt or "mp3" in mt:
        return "mp3"
    if "wav" in mt:
        return "wav"
    if "ogg" in mt or "opus" in mt:
        return "ogg"
    if "flac" in mt:
        return "flac"
    return "wav"


def _openrouter_completion_assistant_text(data: dict) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return ""


def _openrouter_speech_to_text(
    audio_bytes: bytes,
    filename: str,
    mime_type: Optional[str] = None,
    format_hint: Optional[str] = None,
    model_override: Optional[str] = None,
) -> str:
    import base64 as b64
    import urllib.error
    import urllib.request

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not configured")
    model = model_override or _openrouter_transcription_model()
    fmt = _openrouter_audio_format(filename, mime_type, format_hint)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe this audio verbatim. Reply with the transcript only — no preamble or quotes.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64.b64encode(audio_bytes).decode("ascii"),
                            "format": fmt,
                        },
                    },
                ],
            }
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw_text = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            raw_text = e.read().decode("utf-8")
        except Exception:
            raw_text = ""
    except Exception as e:
        raise ValueError(f"OpenRouter transcription request failed: {e}") from e
    if status < 200 or status >= 300:
        try:
            err_j = json.loads(raw_text)
            err_obj = err_j.get("error")
            if isinstance(err_obj, dict):
                detail = err_obj.get("message") or str(err_obj)
            else:
                detail = err_obj or err_j.get("message")
        except Exception:
            detail = raw_text[:500] if raw_text else None
        raise ValueError(detail or f"OpenRouter transcription error ({status})")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return (raw_text or "").strip()
    return _openrouter_completion_assistant_text(data)


def _transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    mime_type: Optional[str] = None,
    format_hint: Optional[str] = None,
) -> tuple[str, str]:
    """Speech-to-text via OpenRouter with model fallback: primary → xiaomi/mimo-v2-omni."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ValueError("Configure OPENROUTER_API_KEY for transcription.")
    primary = _openrouter_transcription_model()
    fallback = (os.getenv("OPENROUTER_TRANSCRIPTION_FALLBACK_MODEL") or "xiaomi/mimo-v2-omni").strip()
    for model in (primary, fallback):
        try:
            text = _openrouter_speech_to_text(audio_bytes, filename, mime_type, format_hint, model_override=model)
            return (text, f"openrouter/{model}")
        except Exception as e:
            print(f"[backend] STT {model} failed, trying next: {e}")
    raise ValueError(f"All OpenRouter transcription models failed ({primary}, {fallback})")


def _openai_transcription_model() -> str:
    return (os.getenv("OPENAI_TRANSCRIPTION_MODEL") or "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"


def _build_openai_transcription_multipart(audio_bytes: bytes, filename: str, model: str) -> tuple[str, bytes]:
    boundary = f"----SelfMeridianSTT{uuid.uuid4().hex[:20]}"
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    buf.write(model.encode("utf-8"))
    buf.write(b"\r\n")
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
    buf.write(audio_bytes)
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    return boundary, buf.getvalue()


def _openai_speech_to_text(audio_bytes: bytes, filename: str) -> str:
    import urllib.error
    import urllib.request

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is not configured")
    model = _openai_transcription_model()
    boundary, body = _build_openai_transcription_multipart(audio_bytes, filename, model)
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_text = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            raw_text = e.read().decode("utf-8")
        except Exception:
            raw_text = ""
    except Exception as e:
        raise ValueError(f"OpenAI transcription request failed: {e}") from e
    if status < 200 or status >= 300:
        try:
            err_j = json.loads(raw_text)
            detail = err_j.get("error", {}).get("message") if isinstance(err_j.get("error"), dict) else err_j.get("error")
        except Exception:
            detail = raw_text[:400] if raw_text else None
        raise ValueError(detail or f"OpenAI transcription error ({status})")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text.strip()
    text = data.get("text")
    return str(text).strip() if text is not None else ""


def _elevenlabs_transcribe_bytes(audio_bytes: bytes) -> str:
    """Same wire format as /api/transcribe (ElevenLabs scribe_v2)."""
    import urllib.error
    import urllib.request

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not configured")
    model_id = "scribe_v2"
    boundary = f"----SelfmeridianBoundary{uuid.uuid4().hex[:24]}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model_id"\r\n\r\n'
        f"{model_id}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + audio_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_type = f"multipart/form-data; boundary={boundary}"
    request = urllib.request.Request(
        ELEVENLABS_STT_URL,
        data=body,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            raw_text = resp.read().decode("utf-8")
            status_code = resp.status
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            raw_text = e.read().decode("utf-8")
        except Exception:
            raw_text = ""
    except Exception as e:
        raise ValueError(f"ElevenLabs STT failed: {e}") from e
    data: dict = {}
    if raw_text.strip():
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            pass
    if status_code < 200 or status_code >= 300:
        err_detail = data.get("detail")
        if isinstance(err_detail, dict):
            msg = err_detail.get("message") or str(err_detail)
        elif isinstance(err_detail, str):
            msg = err_detail
        else:
            msg = data.get("message") or (raw_text.strip()[:300] if raw_text.strip() else None) or f"ElevenLabs STT failed ({status_code})"
        raise ValueError(str(msg))
    text = data.get("text")
    return str(text).strip() if text is not None else ""


async def _polish_voice_memo_openrouter(raw: str) -> str:
    """Single LLM call: clean up filler words, fix STT errors, polish into readable prose."""
    text = (raw or "").strip()
    if not text or not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        return text
    model = (os.getenv("OPENROUTER_VOICE_MEMO_POLISH_MODEL") or "openai/gpt-5.4").strip()
    prompt = VOICE_MEMO_POLISH_INSTRUCTION + "\n\n--- Transcript ---\n" + text[:48000]

    def _call():
        try:
            return _openrouter_chat_completion(prompt, model=model, temperature=0.2, timeout_sec=90.0)
        except Exception as e:
            print("[backend] voice-memo polish error:", e)
            return ""

    out = (await asyncio.to_thread(_call) or "").strip()
    return out if out else text


async def _validate_journal_openrouter(raw: str, model_override: Optional[str] = None) -> tuple[str, str, list[str], Optional[str]]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    text = (raw or "").strip()
    if not text:
        return "", "", [], None
    if not key:
        return (
            text,
            "Add OPENROUTER_API_KEY to the project root .env (uncomment and paste your key), save the file, and restart the Python API server (uvicorn / backend on port 8000) so it reloads environment variables.",
            [],
            None,
        )
    try:
        import urllib.error
        import urllib.request

        model = (model_override or os.getenv("OPENROUTER_JOURNAL_VALIDATE_MODEL", "")).strip() or "openai/gpt-5.4"
        prompt = JOURNAL_VALIDATE_INSTRUCTION + "\n\n--- Transcript ---\n" + text[:48000]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://selfmeridian.local"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "SelfMeridian"),
            },
        )

        def _call() -> str:
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw_text = resp.read().decode("utf-8")
                    status = resp.status
            except urllib.error.HTTPError as e:
                status = e.code
                try:
                    raw_text = e.read().decode("utf-8")
                except Exception:
                    raw_text = ""
            if status < 200 or status >= 300:
                try:
                    err_j = json.loads(raw_text)
                    detail = err_j.get("error", {}).get("message") if isinstance(err_j.get("error"), dict) else err_j.get("error")
                except Exception:
                    detail = raw_text[:300] if raw_text else None
                raise ValueError(detail or f"OpenRouter error ({status})")
            data = json.loads(raw_text) if raw_text else {}
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                content = msg.get("content") if isinstance(msg, dict) else ""
                return str(content or "").strip()
            return ""

        out = await asyncio.to_thread(_call)
        if not out:
            return text, "AI returned empty output.", [], model

        def _between(src: str, start: str, end: Optional[str]) -> str:
            i = src.find(start)
            if i == -1:
                return ""
            i += len(start)
            if end is None:
                return src[i:].strip()
            j = src.find(end, i)
            return src[i:j].strip() if j != -1 else src[i:].strip()

        reformatted = _between(out, "===REFORMATTED===", "===FEEDBACK===")
        feedback = _between(out, "===FEEDBACK===", "===VALIDATION===")
        validation_block = _between(out, "===VALIDATION===", None)

        notes: list[str] = []
        for line in validation_block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            if line:
                notes.append(line)

        if not reformatted:
            reformatted = text
        if not feedback:
            feedback = "No major issues found."
        return reformatted, feedback, notes[:8], model
    except Exception as e:
        print("[backend] journal validate openrouter error:", e)
        return text, "Validation failed; using original transcript.", [str(e)[:180]], None


@api_router.post("/voice")
async def api_voice(req: VoiceRequest):
    """
    Text-to-speech: Mistral Voxtral when MISTRAL_API_KEY is set, else ElevenLabs.
    Returns { audio: base64, format: \"opus\" | \"mp3\" | ... }.
    Voxtral: https://docs.mistral.ai/capabilities/audio/text_to_speech
    """
    import base64 as b64
    import urllib.request
    import urllib.error

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        vid = _resolve_mistral_voice_id(req.voiceId)
        tts_fmt = _mistral_tts_response_format()
        try:
            audio_bytes = await asyncio.to_thread(_mistral_tts_speech, text, vid, tts_fmt)
        except ValueError as e:
            print("[backend] Mistral TTS:", e)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            print("[backend] Mistral TTS error:", e)
            raise HTTPException(status_code=500, detail=str(e))
        rate = _mistral_tts_playback_rate()
        return {
            "audio": b64.b64encode(audio_bytes).decode("ascii"),
            "format": tts_fmt,
            "provider": "mistral",
            "playback_rate": rate,
        }

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No TTS configured: set MISTRAL_API_KEY for Voxtral (default voice en_paul_neutral), or ELEVENLABS_API_KEY.",
        )
    voice_id = req.voiceId or DEFAULT_VOICE_ID
    raw = req.voice_settings or {}
    stability = max(0, min(1, raw.get("stability", 0.5) if isinstance(raw.get("stability"), (int, float)) else 0.5))
    similarity_boost = max(0, min(1, raw.get("similarity_boost", 0.75) if isinstance(raw.get("similarity_boost"), (int, float)) else 0.75))
    style = max(0, min(1, raw.get("style", 0) if isinstance(raw.get("style"), (int, float)) else 0))
    speed = max(0.5, min(2, raw.get("speed", 1) if isinstance(raw.get("speed"), (int, float)) else 1))
    body = json.dumps({
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_128",
        "voice_settings": {"stability": stability, "similarity_boost": similarity_boost, "style": style, "speed": speed},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{ELEVENLABS_TTS_URL}/{voice_id}",
        data=body,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            audio_bytes = resp.read()
        b64_audio = b64.b64encode(audio_bytes).decode("ascii")
        return {"audio": b64_audio, "format": "mp3"}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body) if err_body.strip() else {}
            msg = (err_json.get("detail") or {}).get("message") if isinstance(err_json.get("detail"), dict) else err_json.get("message") or err_body[:200]
        except Exception:
            msg = str(e)
        raise HTTPException(status_code=500, detail=msg or f"ElevenLabs TTS failed ({e.code})")
    except Exception as e:
        print("[backend] /api/voice error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/voices")
async def api_voices():
    """Voices for TTS UI: Mistral when MISTRAL_API_KEY is set, else ElevenLabs. Returns { voices: [{ voice_id, name }], provider? }."""
    import urllib.request
    import urllib.error

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        try:
            request = urllib.request.Request(
                "https://api.mistral.ai/v1/audio/voices?limit=50&offset=0",
                headers={"Authorization": f"Bearer {mistral_key}"},
            )
            with urllib.request.urlopen(request, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("items") or []
            voices = [
                {"voice_id": (v.get("id") or ""), "name": (v.get("name") or "Voice")}
                for v in items
                if isinstance(v, dict) and v.get("id")
            ]
            if voices:
                return {"voices": voices, "provider": "mistral"}
        except Exception as e:
            print("[backend] Mistral /voices list:", e)

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return {"voices": FALLBACK_VOICES, "provider": "fallback"}
    request = urllib.request.Request(
        ELEVENLABS_VOICES_URL,
        headers={"xi-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return {"voices": FALLBACK_VOICES}
    raw = data.get("voices") or FALLBACK_VOICES
    voices = [
        {"voice_id": (v.get("voice_id") or v.get("id") or ""), "name": v.get("name", "?")}
        for v in raw
    ]
    voices = [v for v in voices if v["voice_id"]]
    if not any(v["voice_id"] == DEFAULT_VOICE_ID for v in voices):
        voices.insert(0, {"voice_id": DEFAULT_VOICE_ID, "name": "Rachel"})
    return {"voices": voices, "provider": "elevenlabs"}


@api_router.get("/scribe-token")
async def api_scribe_token():
    """Return a single-use token for ElevenLabs Realtime Speech-to-Text WebSocket."""
    import urllib.request
    import urllib.error

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is not configured")
    request = urllib.request.Request(
        ELEVENLABS_SCRIBE_TOKEN_URL,
        data=b"{}",
        method="POST",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body) if err_body.strip() else {}
            msg = (err_json.get("detail") or {}).get("message") if isinstance(err_json.get("detail"), dict) else err_json.get("message") or err_body[:200]
        except Exception:
            msg = str(e)
        raise HTTPException(status_code=500, detail=msg or f"ElevenLabs token failed ({e.code})")
    token = data.get("token")
    if not token:
        raise HTTPException(status_code=500, detail="No token in response")
    return {"token": token}


@api_router.get("/memory-diagram", response_model=MemoryDiagramResponse)
async def memory_diagram(request: Request):
    """
    Legacy endpoint for memory diagram; retained for compatibility but not used by Brain UI.
    """
    instance_id = _instance_id(request)
    gist_facts, episodic_summaries = get_memory_for_visualization(instance_id=instance_id)
    mermaid_code = generate_memory_mermaid(gist_facts, episodic_summaries)
    return MemoryDiagramResponse(mermaid=mermaid_code)


@api_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """User sends text; Interviewer responds. State updated in memory. mode=recommendations uses library interview (books, notes)."""
    instance_id = _instance_id(request)
    session_id = get_or_create_session(req.session_id)
    mode = (req.mode or "journal").strip().lower()
    if mode not in ("journal", "conversation", "autobiography", "recommendations", "learning"):
        mode = "journal"

    if mode == "recommendations":
        # Library interview: ask about books, save short notes. Uses library_interview_sessions.
        _get_or_create_library_interview_session(session_id)
        messages = library_interview_sessions[session_id]
        try:
            library_data = await asyncio.to_thread(list_consumed, 200, instance_id)
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
    # Full memory retrieval (vec store + gist) for all graph-backed modes; not client-tunable.
    personalization = 1.0
    intrusiveness = req.intrusiveness if req.intrusiveness is not None else 0.5
    try:
        intrusiveness = float(intrusiveness)
    except (TypeError, ValueError):
        intrusiveness = 0.5
    intrusiveness = max(0.0, min(1.0, intrusiveness))
    messages.append(HumanMessage(content=req.text))
    _orm = (req.openrouter_model or "").strip() or None
    state: JournalState = {
        "messages": list(messages),
        "session_id": session_id,
        "personalization": personalization,
        "intrusiveness": intrusiveness,
        "mode": mode,
        "instance_id": instance_id,
        **({"openrouter_model": _orm} if _orm else {}),
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
    lib_n = result.get("library_items_added")
    lib_opt = int(lib_n) if isinstance(lib_n, int) and lib_n > 0 else None
    raw_steps = result.get("agent_steps")
    agent_steps: Optional[List[Dict[str, Any]]] = None
    if isinstance(raw_steps, list) and raw_steps:
        agent_steps = [s for s in raw_steps if isinstance(s, dict)]
        if not agent_steps:
            agent_steps = None
    raw_actions = result.get("client_actions")
    actions: Optional[List[Dict[str, Any]]] = None
    if isinstance(raw_actions, list) and raw_actions:
        actions = [a for a in raw_actions if isinstance(a, dict) and a.get("type") == "navigate"]
        if not actions:
            actions = None
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        retrieval_log=retrieval_log,
        library_items_added=lib_opt,
        agent_steps=agent_steps,
        actions=actions,
    )


@api_router.get("/chat-session/{session_id}", response_model=ChatSessionMessagesResponse)
async def get_chat_session_history(session_id: str):
    """Return transcript for a session (in-memory). Empty if unknown or after server restart."""
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    if sid not in sessions:
        return ChatSessionMessagesResponse(session_id=sid, messages=[])
    rows = _session_messages_for_client(sid)
    return ChatSessionMessagesResponse(
        session_id=sid,
        messages=[ChatSessionHistoryMessage(role=r["role"], content=r["content"]) for r in rows],
    )


def _get_or_create_library_interview_session(session_id: Optional[str]) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in library_interview_sessions:
        library_interview_sessions[session_id] = []
    return session_id


@api_router.post("/library-interview", response_model=LibraryInterviewResponse)
async def library_interview(req: LibraryInterviewRequest, request: Request):
    """
    One turn of the library interview: user message + optional library snapshot.
    Agent asks about books and may save short notes; notes_saved lists any updates.
    """
    instance_id = _instance_id(request)
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


@api_router.post("/end-session", response_model=EndSessionResponse)
async def end_session(req: EndSessionRequest, request: Request):
    """Trigger Librarian: extract, embed, save to SQLite+sqlite-vec. Optional LightRAG indexing via bridge when enabled."""
    instance_id = _instance_id(request)
    session_id = get_or_create_session(req.session_id)
    messages = sessions.get(session_id, [])

    state: JournalState = {
        "messages": messages,
        "session_id": session_id,
        "personalization": 1.0,
        "instance_id": instance_id,
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
        asyncio.create_task(schedule_lightrag_index_after_ingest(doc))

    return EndSessionResponse(ok=True, session_id=session_id)


# Stay under common proxy timeouts (e.g. Cloudflare 100–120s) to avoid 524
INGEST_HISTORY_TIMEOUT_SEC = 55

@api_router.post("/ingest-history", response_model=IngestHistoryResponse)
async def ingest_history(req: IngestHistoryRequest, request: Request):
    """
    Ingest a prior journal text into SQLite+sqlite-vec.
    Treats `text` as a single-session transcript. Optional LightRAG indexing when LIGHTRAG_ENABLED=true.
    Returns 200 always (so CORS headers are sent); ok=False on failure or timeout.
    """
    instance_id = _instance_id(request)
    session_id = req.session_id or f"import-{uuid.uuid4()}"
    try:
        text = (req.text or "").strip()
        if not text:
            return IngestHistoryResponse(ok=True, session_id=session_id)
        entry_date = (req.entry_date or "").strip() or None
        extracted = await asyncio.wait_for(
            asyncio.to_thread(
                save_session_data,
                session_id,
                text,
                entry_date,
                instance_id,
                req.content_hash,
            ),
            timeout=INGEST_HISTORY_TIMEOUT_SEC,
        )
        if extracted.get("skipped"):
            return IngestHistoryResponse(ok=True, session_id=session_id)
        summary = extracted.get("summary") or ""
        facts = extracted.get("facts") or []
        if summary or facts:
            parts = []
            if summary:
                parts.append(f"Summary: {summary}")
            if facts:
                parts.append("Facts: " + "; ".join(facts))
            doc = "\n\n".join(parts)
            asyncio.create_task(schedule_lightrag_index_after_ingest(doc))

        return IngestHistoryResponse(ok=True, session_id=session_id)
    except asyncio.TimeoutError:
        print("[backend] ingest_history timed out after", INGEST_HISTORY_TIMEOUT_SEC, "s")
        return IngestHistoryResponse(ok=False, session_id=session_id)
    except Exception as e:
        import traceback
        print("[backend] ingest_history error:", e)
        traceback.print_exc()
        return IngestHistoryResponse(ok=False, session_id=session_id)


@api_router.post("/memory/writing-hints")
async def memory_writing_hints(req: WritingHintsRequest, request: Request):
    """Continuity + similarity for the journal composer (episodic retrieval + active insights)."""
    instance_id = _instance_id(request)
    return await asyncio.to_thread(get_writing_loop_hints, req.draft, instance_id)


@api_router.get("/memory/insights")
async def memory_insights_list(request: Request):
    instance_id = _instance_id(request)
    rows = await asyncio.to_thread(vec_store.derived_insights_list_active, instance_id, 20)
    return {"insights": rows}


@api_router.post("/memory/insights/refresh")
async def memory_insights_refresh(request: Request):
    instance_id = _instance_id(request)
    return await asyncio.to_thread(generate_derived_insights, instance_id)


@api_router.post("/memory/patterns/refresh")
async def memory_patterns_refresh(request: Request):
    instance_id = _instance_id(request)
    return await asyncio.to_thread(refresh_pattern_memory, instance_id)


@api_router.post("/recommendations/feedback", response_model=RecFeedbackResponse)
async def recommendations_feedback(req: RecFeedbackRequest, request: Request):
    instance_id = _instance_id(request)

    def _run() -> None:
        record_rec_feedback_for_recs(
            instance_id,
            req.action,
            content_type=req.content_type,
            topic_tags=req.topic_tags,
            intent_context=req.intent_context,
            item_title=req.item_title,
        )

    await asyncio.to_thread(_run)
    return RecFeedbackResponse(ok=True)


@api_router.post("/feedback")
async def submit_content_feedback(req: ContentFeedbackRequest, request: Request):
    instance_id = _instance_id(request)

    def _run():
        return process_content_feedback(
            instance_id,
            content_title=req.content_title,
            content_type=req.content_type,
            content_url=req.content_url,
            feedback=req.feedback,
            user_notes=req.user_notes,
        )

    return await asyncio.to_thread(_run)


@api_router.get("/profile")
async def get_media_profile(request: Request):
    instance_id = _instance_id(request)
    prof, updated_at = vec_store.user_media_profile_get_with_meta(instance_id)
    return {"profile": prof, "updated_at": updated_at}


@api_router.get("/logs/decisions")
async def list_decision_logs(
    request: Request,
    action_type: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
):
    instance_id = _instance_id(request)
    rows = vec_store.decision_log_list(
        instance_id,
        action_type=action_type,
        session_id=session_id,
        limit=limit,
    )
    return {"logs": rows}


@api_router.get("/logs/decisions/{log_id}")
async def get_decision_log_detail(log_id: int, request: Request):
    instance_id = _instance_id(request)
    row = vec_store.decision_log_get(log_id, instance_id)
    if not row:
        raise HTTPException(status_code=404, detail="Log not found")
    return row


@api_router.get("/memory/date-range")
async def memory_date_range(start: str, end: str, request: Request, limit: int = 20):
    instance_id = _instance_id(request)
    rows = vec_store.query_episodic_by_date_range(instance_id, start, end, limit=limit)
    return {"entries": rows}


@api_router.get("/memory/on-this-day")
async def memory_on_this_day(request: Request, month_day: Optional[str] = None, years_back: int = 5):
    instance_id = _instance_id(request)
    rows = vec_store.query_this_day_in_history(instance_id, month_day=month_day, years_back=years_back)
    return {"entries": rows}


@api_router.get("/memory/timeline")
async def memory_timeline(topic: str, request: Request, limit: int = 30):
    instance_id = _instance_id(request)
    if not topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")

    def _run():
        emb = _embed_texts([topic.strip()[:2000]])[0]
        return vec_store.query_episodic_timeline_by_topic(emb, instance_id, limit=limit)

    rows = await asyncio.to_thread(_run)
    return {"entries": rows}


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


@api_router.post("/infer-entry-date", response_model=InferEntryDateResponse)
async def infer_entry_date(req: InferEntryDateRequest):
    """
    Use an LLM to infer the best-guess date/time when a journal entry was written.
    Returns ISO 8601 string or null if unclear.
    """
    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not or_key:
        return InferEntryDateResponse(date=None)
    try:
        model = (os.getenv("OPENROUTER_INFER_ENTRY_DATE_MODEL") or "openai/gpt-4.1-mini").strip()
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
            return _openrouter_chat_completion(
                prompt, model=model, temperature=0.0, timeout_sec=60.0, max_tokens=256
            )

        raw = (await asyncio.to_thread(_call) or "").strip()
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


@api_router.get("/memory-stats", response_model=MemoryStats)
async def memory_stats(request: Request):
    """
    Lightweight stats endpoint; scoped to current user or anonymous instance.
    """
    import vec_store

    vec_store.ensure_db()
    instance_id = _instance_id(request)
    je = vec_store.journal_entry_count(instance_id)
    jc = vec_store.journal_chunk_count(instance_id)
    return MemoryStats(
        gist_facts_count=je,
        episodic_log_count=jc,
        episodic_metadata_count=vec_store.episodic_metadata_count(),
        journal_entry_count=je,
        journal_chunk_count=jc,
    )


@api_router.get("/memory/facts")
async def get_memory_facts(request: Request):
    """List all gist facts with ids for Memory UI (view/edit/delete)."""
    instance_id = _instance_id(request)
    try:
        items = await asyncio.to_thread(list_memory_facts, instance_id)
        return {"facts": [MemoryItem(**x) for x in items]}
    except Exception as e:
        print("[backend] GET /memory/facts error:", e)
        return {"facts": []}


@api_router.get("/memory/summaries")
async def get_memory_summaries(request: Request):
    """List all episodic summaries with ids for Memory UI."""
    instance_id = _instance_id(request)
    try:
        items = await asyncio.to_thread(list_memory_summaries, instance_id)
        return {"summaries": [MemoryItem(**x) for x in items]}
    except Exception as e:
        print("[backend] GET /memory/summaries error:", e)
        return {"summaries": []}


@api_router.patch("/memory/facts/{fact_id}")
async def update_memory_fact_route(fact_id: int, req: MemoryFactUpdate):
    """Update a gist fact by id; re-embeds and updates store."""
    try:
        ok = await asyncio.to_thread(update_memory_fact, fact_id, req.document)
        return {"ok": ok}
    except Exception as e:
        print("[backend] PATCH /memory/facts error:", e)
        return {"ok": False}


@api_router.patch("/memory/summaries/{summary_id}")
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


@api_router.delete("/memory/facts/{fact_id}")
async def delete_memory_fact_route(fact_id: int):
    """Delete a gist fact by id."""
    try:
        ok = await asyncio.to_thread(delete_memory_fact, fact_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /memory/facts error:", e)
        return {"ok": False}


@api_router.delete("/memory/summaries/{summary_id}")
async def delete_memory_summary_route(summary_id: int):
    """Delete an episodic summary by id."""
    try:
        ok = await asyncio.to_thread(delete_memory_summary, summary_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /memory/summaries error:", e)
        return {"ok": False}


@api_router.post("/memory/facts")
async def create_memory_fact(req: MemoryFactCreate, request: Request):
    """Add a user-created fact; returns new id."""
    instance_id = _instance_id(request)
    try:
        fid = await asyncio.to_thread(add_memory_fact, req.document, None, instance_id)
        return {"ok": fid is not None, "id": fid}
    except Exception as e:
        print("[backend] POST /memory/facts error:", e)
        return {"ok": False, "id": None}


@api_router.post("/memory/summaries")
async def create_memory_summary(req: MemorySummaryCreate, request: Request):
    """Add a user-created summary; returns new id."""
    instance_id = _instance_id(request)
    try:
        sid = await asyncio.to_thread(add_memory_summary, req.document, None, instance_id)
        return {"ok": sid is not None, "id": sid}
    except Exception as e:
        print("[backend] POST /memory/summaries error:", e)
        return {"ok": False, "id": None}


@api_router.get("/brain/people-graph", response_model=BrainPeopleGraphResponse)
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
        summaries = library.list_memory_facts()
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


@api_router.post("/brain/people/auto-groups", response_model=Dict)
async def brain_people_auto_groups():
    """
    Use an LLM to propose social groups (UNC Charlotte, CPCC, Mentors, etc.)
    and assign people to them. This updates person_groups and then the
    /brain/people-graph endpoint will reflect the new clusters.
    """
    await asyncio.to_thread(run_people_grouping_agent)
    return {"ok": True}


@api_router.get("/brain/people", response_model=List[Dict])
async def brain_people_list():
    """Return raw people list with groups for Brain UI."""
    import vec_store

    vec_store.ensure_db()
    return vec_store.list_people_with_groups()


@api_router.post("/brain/people", response_model=Dict)
async def brain_person_create(req: BrainPersonCreateRequest):
    """Create a new person node for Brain -> People graph."""
    import vec_store

    vec_store.ensure_db()
    pid = vec_store.create_person(req.name)
    return {"id": pid, "name": req.name}


@api_router.patch("/brain/people/{person_id}", response_model=Dict)
async def brain_person_rename(person_id: int, req: BrainPersonRenameRequest):
    """Rename an existing person."""
    import vec_store

    vec_store.ensure_db()
    ok = vec_store.update_person(person_id, req.name)
    return {"ok": ok}


@api_router.get("/brain/people/{person_id}", response_model=BrainPersonDetail)
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


@api_router.put("/brain/people/{person_id}/profile", response_model=Dict)
async def brain_person_profile_update(person_id: int, req: BrainPersonProfileUpdate):
    """Update relationship summary and structured relationship fields for a person."""
    import vec_store

    vec_store.ensure_db()
    summary = req.relationship_summary or ""
    rel_type = req.relationship_type or ""
    closeness = req.closeness_label or ""
    vec_store.upsert_person_profile(person_id, summary, rel_type, closeness)
    return {"ok": True}


@api_router.put("/brain/people/{person_id}/groups", response_model=Dict)
async def brain_person_groups_update(person_id: int, req: BrainPersonGroupsUpdate):
    """Replace groups for a person."""
    import vec_store

    vec_store.ensure_db()
    vec_store.set_person_groups(person_id, req.groups)
    return {"ok": True}


@api_router.post("/brain/people/{person_id}/thoughts", response_model=Dict)
async def brain_person_thought_create(person_id: int, req: BrainPersonThoughtCreate):
    """Add a new reflection/thought about a person."""
    import vec_store

    vec_store.ensure_db()
    tid = vec_store.add_person_thought(person_id, req.date, req.thought_text)
    return {"ok": tid is not None, "id": tid}


@api_router.patch("/brain/people/{person_id}/thoughts/{thought_id}", response_model=Dict)
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


@api_router.delete("/brain/people/{person_id}/thoughts/{thought_id}", response_model=Dict)
async def brain_person_thought_delete(person_id: int, thought_id: int):
    """Delete a thought about a person."""
    import vec_store

    vec_store.ensure_db()
    ok = vec_store.delete_person_thought(thought_id)
    return {"ok": ok}


# Keep under typical proxy origin timeout (e.g. Cloudflare 100s) to avoid 524.
# Slightly above RECOMMENDATIONS_AGENT_TIMEOUT_SEC so parallel agents can finish first.
RECOMMENDATIONS_TIMEOUT_SEC = float(os.getenv("RECOMMENDATIONS_HTTP_TIMEOUT_SEC", "78"))

@api_router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    request: Request,
    category: Optional[str] = Query(
        None,
        description="If set (books, podcasts, articles, research, news), only that column is regenerated; other lists are empty.",
    ),
):
    """
    Personalized books, podcasts, articles, research, and news from journal memory and consumed library.
    May take ~35–80s depending on Perplexity + LLM latency; override RECOMMENDATIONS_HTTP_TIMEOUT_SEC if needed.
    With ?category=books (etc.), only that agent runs — faster for per-column refresh; merge client-side with cache.
    """
    instance_id = _instance_id(request)
    cat = (category or "").strip().lower()
    allowed = ("books", "podcasts", "articles", "research", "news")
    if cat and cat not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Use one of: {', '.join(allowed)}",
        )
    gen_fn = generate_recommendations_category if cat else generate_recommendations
    gen_args = (instance_id, cat) if cat else (instance_id,)
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(gen_fn, *gen_args),
            timeout=RECOMMENDATIONS_TIMEOUT_SEC,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        print("[backend] /recommendations timed out after", RECOMMENDATIONS_TIMEOUT_SEC, "s")
        return RecommendationsResponse(books=[], podcasts=[], articles=[], research=[], news=[])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        print("[backend] /recommendations error:", e)
        return RecommendationsResponse(books=[], podcasts=[], articles=[], research=[], news=[])
    return RecommendationsResponse(
        books=[RecommendationItem(**x) for x in data.get("books", [])],
        podcasts=[RecommendationItem(**x) for x in data.get("podcasts", [])],
        articles=[RecommendationItem(**x) for x in data.get("articles", [])],
        research=[RecommendationItem(**x) for x in data.get("research", [])],
        news=[RecommendationItem(**x) for x in data.get("news", [])],
    )


class CalendarDayRequest(BaseModel):
    date: str  # YYYY-MM-DD
    raw_transcript: Optional[str] = None


class CalendarDayResponse(BaseModel):
    summary: str
    has_journal: bool


@api_router.post("/calendar-day-summary", response_model=CalendarDayResponse)
async def calendar_day_summary(req: CalendarDayRequest, request: Request):
    """
    For a given date, combine raw journal transcript (if any) with DB memory for that day
    and return an AI-generated day summary/highlights.
    """
    instance_id = _instance_id(request)
    date_iso = (req.date or "").strip()[:10]
    if not date_iso or len(date_iso) < 10:
        return CalendarDayResponse(summary="Please provide a valid date (YYYY-MM-DD).", has_journal=False)
    try:
        episodic, gist = await asyncio.to_thread(get_memory_for_date, date_iso, instance_id)
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


@api_router.post("/recommendations/consumed", response_model=ConsumedResponse)
async def mark_consumed(req: ConsumedRequest, request: Request):
    """
    Record that the user has read/listened to a recommendation. Stored in the vector store
    so future recommendations avoid repeats and better match their tastes.
    """
    instance_id = _instance_id(request)
    content_type = (req.type or "article").lower()
    if content_type not in ("book", "podcast", "article", "research", "news"):
        content_type = "article"
    try:
        add_consumed(
            content_type=content_type,
            title=req.title,
            author=req.author,
            url=req.url,
            liked=req.liked,
            instance_id=instance_id,
        )
        return ConsumedResponse(ok=True)
    except Exception as e:
        print("[backend] /recommendations/consumed error:", e)
        return ConsumedResponse(ok=False)


@api_router.get("/library")
async def get_library(request: Request):
    """
    Return consumed items grouped by type for the Library UI.
    """
    instance_id = _instance_id(request)
    try:
        data = await asyncio.to_thread(list_consumed, 200, instance_id)
        return data
    except Exception as e:
        print("[backend] GET /library error:", e)
        return {"books": [], "podcasts": [], "articles": [], "research": [], "news": []}


@api_router.patch("/library/{item_id}")
async def update_library_item(item_id: str, req: LibraryItemUpdate, request: Request):
    """
    Update library item metadata by id.
    """
    instance_id = _instance_id(request)
    try:
        def _wrap():
            return update_consumed(
                item_id,
                date_completed=req.date_completed,
                note=req.note,
                title=req.title,
                author=req.author,
                url=req.url,
                instance_id=instance_id,
            )
        ok = await asyncio.to_thread(_wrap)
        return {"ok": ok}
    except Exception as e:
        print("[backend] PATCH /library error:", e)
        return {"ok": False}


@api_router.delete("/library/{item_id}")
async def delete_library_item(item_id: str, request: Request):
    """
    Remove a library item from the consumed collection.
    """
    instance_id = _instance_id(request)
    try:
        ok = await asyncio.to_thread(delete_consumed, item_id, instance_id)
        return {"ok": ok}
    except Exception as e:
        print("[backend] DELETE /library error:", e)
        return {"ok": False}


@api_router.post("/library-notes", response_model=LibraryNoteResponse)
async def library_notes(req: LibraryNoteRequest, request: Request):
    """
    Library helper endpoint: user can paste titles or notes about books, podcasts,
    articles, or research they've read. An agent organizes this into structured
    consumed items to improve future recommendations.
    """
    instance_id = _instance_id(request)
    try:
        from library import process_library_note

        count = await asyncio.to_thread(process_library_note, req.text, req.type, instance_id)
        return LibraryNoteResponse(ok=count > 0, items_added=count)
    except Exception as e:
        print("[backend] /library-notes error:", e)
        return LibraryNoteResponse(ok=False, items_added=0)


@api_router.get("/lightrag-context")
async def lightrag_context(q: str = "", mode: str = "hybrid"):
    """
    Optional LightRAG-only RAG context (disabled by default). Primary retrieval is sqlite-vec via /chat.
    Query param: q=... (required), mode=local|global|hybrid|naive|mix (default hybrid).
    """
    if not (q or "").strip():
        return {"context": ""}
    try:
        context = await query_for_context(q.strip(), mode=mode)
        return {"context": context}
    except Exception as e:
        print("[backend] /lightrag-context error:", e)
        return {"context": ""}


@api_router.post("/memory-wipe")
async def memory_wipe(request: Request):
    """
    Wipe gist and episodic memory for the current user/instance. Consumed library is kept.
    """
    instance_id = _instance_id(request)
    if instance_id:
        vec_store.wipe_memory_for_instance(instance_id)
    else:
        wipe_memory()
    return {"ok": True, "message": "Memory wiped."}


@api_router.post("/memory-reset-knowledge-base-import")
async def memory_reset_knowledge_base_import(request: Request):
    """
    Wipe all vector-backed memory for this instance: journal gist/episodic embeddings and the
    consumed library index. Used before a full knowledge-base folder re-import so embeddings
    match the uploaded export only.
    """
    instance_id = _instance_id(request)
    try:
        await asyncio.to_thread(vec_store.wipe_all_vector_memory_for_instance, instance_id)
        return {"ok": True}
    except Exception as e:
        print("[backend] /memory-reset-knowledge-base-import error:", e)
        return {"ok": False, "detail": str(e)}


@api_router.post("/library/bulk-import", response_model=LibraryBulkImportResponse)
async def library_bulk_import(req: LibraryBulkImportRequest, request: Request):
    """
    Rebuild consumed library rows with embeddings (after KB import). Items use client ids as stable keys.
    """
    instance_id = _instance_id(request)
    n = 0
    for it in req.items:
        ct = (it.type or "article").lower()
        if ct not in ("book", "podcast", "article", "research"):
            ct = "article"
        if not (it.id or "").strip() or not (it.title or "").strip():
            continue
        try:
            await asyncio.to_thread(
                add_consumed,
                ct,
                it.title.strip(),
                author=it.author,
                url=it.url,
                liked=it.liked,
                note=it.note,
                date_completed=it.date_completed,
                instance_id=instance_id,
                id_override=it.id.strip(),
            )
            n += 1
        except Exception as e:
            print("[backend] /library/bulk-import item error:", e)
    return LibraryBulkImportResponse(ok=True, count=n)


@api_router.get("/health")
async def health():
    return {"status": "ok"}


# Mount API under /api
app.include_router(api_router, prefix="/api")

_LOGS_HTML = Path(__file__).resolve().parent / "static" / "logs.html"


@app.get("/logs", include_in_schema=False)
def serve_decision_logs_html():
    if not _LOGS_HTML.is_file():
        raise HTTPException(status_code=404, detail="logs UI missing (backend/static/logs.html)")
    return FileResponse(_LOGS_HTML)


@app.post("/api/transcribe", include_in_schema=False)
async def api_transcribe(req: TranscribeRequest):
    """
    Speech-to-text: OpenRouter `openai/gpt-audio-mini` when OPENROUTER_API_KEY is set, else OpenAI, else ElevenLabs.
    Registered on the main app (not only APIRouter) so POST is never shadowed by the SPA
    catch-all GET /{full_path} (which would otherwise yield 405 Method Not Allowed).
    """
    import base64 as b64

    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    el_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not or_key and not openai_key and not el_key:
        raise HTTPException(
            status_code=500,
            detail="Configure OPENROUTER_API_KEY for transcription (default openai/gpt-audio-mini), or OPENAI_API_KEY, or ELEVENLABS_API_KEY.",
        )
    audio_b64 = (req.audio or "").strip()
    if not audio_b64:
        raise HTTPException(status_code=400, detail="audio (base64) is required")
    try:
        raw = b64.b64decode(audio_b64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")
    if len(raw) < 100:
        raise HTTPException(status_code=400, detail="Audio too short to transcribe")

    fmt_hint = (req.format or "").strip() or None
    ext = fmt_hint.lstrip(".").lower() if fmt_hint else "wav"
    fname = f"audio.{ext}" if ext else "audio.wav"
    try:
        transcript, _engine = await asyncio.to_thread(_transcribe_audio_bytes, raw, fname, None, fmt_hint)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": transcript or ""}


@app.post("/api/voice-memo", include_in_schema=False)
async def api_voice_memo(req: VoiceMemoRequest):
    """
    Voice Memo tab: transcribe via OpenRouter openai/gpt-audio-mini when OPENROUTER_API_KEY is set, else OpenAI, else ElevenLabs.
    Optionally polish transcript via OpenRouter (OPENROUTER_VOICE_MEMO_POLISH_MODEL, default openai/gpt-4.1-mini).
    """
    import base64 as b64

    audio_b64 = (req.audio or "").strip()
    if not audio_b64:
        raise HTTPException(status_code=400, detail="audio (base64) is required")
    try:
        raw_audio = b64.b64decode(audio_b64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")
    if len(raw_audio) < 100:
        raise HTTPException(status_code=400, detail="Audio too short to transcribe")

    import time as _time
    _t0 = _time.monotonic()

    fname = _guess_audio_filename(req.filename, req.mime_type)
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    el_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not or_key and not openai_key and not el_key:
        raise HTTPException(
            status_code=500,
            detail="Configure OPENROUTER_API_KEY for transcription (default openai/gpt-audio-mini), or OPENAI_API_KEY, or ELEVENLABS_API_KEY.",
        )
    audio_kb = len(raw_audio) / 1024
    print(f"[backend] voice-memo: transcribing {audio_kb:.0f} KB audio ({fname})")
    try:
        raw_transcript, transcribe_engine = await asyncio.to_thread(
            _transcribe_audio_bytes,
            raw_audio,
            fname,
            req.mime_type,
            None,
        )
    except (ValueError, Exception) as e:
        print(f"[backend] /api/voice-memo transcribe error ({type(e).__name__}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    _t1 = _time.monotonic()
    print(f"[backend] voice-memo: transcribe done in {_t1 - _t0:.1f}s ({transcribe_engine})")

    polished = await _polish_voice_memo_openrouter(raw_transcript)
    _t2 = _time.monotonic()
    print(f"[backend] voice-memo: polish done in {_t2 - _t1:.1f}s (total {_t2 - _t0:.1f}s)")
    did_polish = bool(
        or_key and (raw_transcript or "").strip() and (polished or "").strip() and (polished or "").strip() != (raw_transcript or "").strip()
    )
    cleaned = polished if did_polish else ""

    return {
        "raw_transcript": raw_transcript or "",
        "polished_text": polished or raw_transcript or "",
        "cleaned_transcript": cleaned,
        "transcribe_engine": transcribe_engine,
        "polished_by_llm": did_polish,
    }


@app.post("/api/journal-validate", response_model=JournalValidateResponse, include_in_schema=False)
async def api_journal_validate(req: JournalValidateRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please provide a longer transcript before validation")
    reformatted, feedback, notes, model_used = await _validate_journal_openrouter(text, req.model)
    return JournalValidateResponse(
        reformatted_journal=reformatted,
        feedback=feedback,
        validation_notes=notes,
        model_used=model_used,
    )


# ---------------------------------------------------------------------------
#  Learning Tab endpoints
# ---------------------------------------------------------------------------

from learning import generate_daily_article


class LearningStatusRequest(BaseModel):
    status: str  # "read" | "skipped"


@app.get("/api/learning/today", include_in_schema=False)
async def api_learning_today(request: Request):
    instance_id = _instance_id(request)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_daily_article, instance_id, False),
            timeout=CHAT_INVOKE_TIMEOUT_SEC,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Article generation timed out")
    except Exception as e:
        print(f"[backend] Learning today error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learning/regenerate", include_in_schema=False)
async def api_learning_regenerate(request: Request):
    instance_id = _instance_id(request)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_daily_article, instance_id, True),
            timeout=CHAT_INVOKE_TIMEOUT_SEC,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Article regeneration timed out")
    except Exception as e:
        print(f"[backend] Learning regenerate error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learning/status", include_in_schema=False)
async def api_learning_status(request: Request, req: LearningStatusRequest):
    instance_id = _instance_id(request)
    from datetime import date
    today = date.today().isoformat()
    updated = vec_store.daily_article_update_status(instance_id, today, req.status)
    return {"ok": updated}


# Serve built frontend (monolith: when built assets exist, e.g. Docker build).
# Do not mount /assets when only auxiliary files (e.g. logs.html) live under static/.
STATIC_DIR = Path(__file__).resolve().parent / "static"
_ASSETS_DIR = STATIC_DIR / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="static_assets")

if STATIC_DIR.exists() and (STATIC_DIR / "index.html").is_file():

    @app.get("/", include_in_schema=False)
    def _index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str):
        # Don't serve index.html for /api/* — let those 404 so API routes are the only handlers
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
