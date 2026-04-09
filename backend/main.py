"""
FastAPI backend for Selfmeridian: /chat and /end-session.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import ssl
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict

# Ensure urllib uses certifi CA bundle on macOS (python.org builds ship without
# system root certs; the "Install Certificates.command" symlinks them, but if
# that hasn't been run yet we fall back to certifi directly).
try:
    import certifi
    ssl._create_default_https_context = lambda purpose=ssl.Purpose.SERVER_AUTH, cafile=certifi.where(): (
        ssl.create_default_context(purpose=purpose, cafile=cafile)
    )
except ImportError:
    pass

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
    add_memory_fact,
    add_memory_summary,
    delete_memory_fact,
    delete_memory_summary,
    DEFAULT_PERPLEXITY_EMBEDDING_MODEL,
    generate_memory_mermaid,
    _embed_texts,
    generate_derived_insights,
    get_memory_for_visualization,
    get_person_events,
    get_writing_loop_hints,
    list_memory_facts,
    list_memory_summaries,
    run_person_facts_agent,
    run_relationship_summary_agent,
    run_people_grouping_agent,
    refresh_pattern_memory,
    save_session_data,
    extraction_llm_backend,
    _openrouter_chat_completion,
    update_memory_fact,
    update_memory_summary,
    wipe_memory,
)

def _instance_id(request: Request) -> str:
    """Per-browser instance id for scoping memory (sent as X-Instance-ID)."""
    return (request.headers.get("X-Instance-ID") or "").strip()


# In-memory session store (minimal for 1hr sprint; replace with Redis/DB later)
sessions: dict[str, list] = {}

# Chat may run OpenRouter + tool round-trips; keep headroom.
CHAT_INVOKE_TIMEOUT_SEC = 180


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

    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    chat_model = (os.getenv("OPENROUTER_CHAT_MODEL") or "openai/gpt-4.1-mini").strip()
    chat_fallback = (os.getenv("OPENROUTER_CHAT_FALLBACK_MODEL") or "openai/gpt-5.4").strip()
    convo_model = (os.getenv("OPENROUTER_CONVERSATION_MODEL") or "x-ai/grok-4.1-fast").strip()
    if or_key:
        print(
            f"[backend] OPENROUTER_API_KEY is set — /chat journal ({chat_model}; fallback {chat_fallback}), "
            f"conversation ({convo_model} + reasoning), journal validation, "
            "voice-memo polish, and date inference"
        )
        print(
            f"[backend] Speech-to-text: OpenRouter ({(os.getenv('OPENROUTER_TRANSCRIPTION_MODEL') or 'openai/gpt-audio-mini').strip()})"
        )
    elif (os.getenv("OPENAI_API_KEY") or "").strip():
        print("[backend] Speech-to-text: OpenAI direct (OPENAI_TRANSCRIPTION_MODEL / gpt-4o-mini-transcribe)")
    else:
        print("[backend] WARNING: Set OPENROUTER_API_KEY (recommended for STT) or OPENAI_API_KEY — transcription routes need one of these.")
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
    mode: Optional[str] = None  # "journal" | "conversation" | "autobiography"
    # Allowlisted OpenRouter id for conversation + autobiography only (see graph.USER_SELECTABLE_CHAT_MODELS)
    openrouter_model: Optional[str] = None
    # Optional: client-formatted local time + daypart string for Assisted Journal (autobiography) check-ins
    client_time_context: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    retrieval_log: Optional[str] = None
    notes_saved: Optional[List[Dict[str, str]]] = None  # [{"item_id": "...", "note": "..."}]
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



class IngestHistoryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    entry_date: Optional[str] = None  # ISO date/datetime when the journal was written; used for vector DB timestamp
    content_hash: Optional[str] = None  # optional client hint; server uses canonical normalized SHA-256 for skip gate
    # "manual" = solo journal text; "assisted" = saved AI-assisted session transcript (balanced retrieval in Assisted Journal)
    entry_source: Optional[str] = None


class IngestHistoryResponse(BaseModel):
    ok: bool
    session_id: str


class InferEntryDateRequest(BaseModel):
    text: str
    filename: Optional[str] = None  # optional; used only when text doesn't provide a date


class InferEntryDateResponse(BaseModel):
    date: Optional[str] = None  # ISO 8601 or null if unclear


class InferJournalFilenameDatesRequest(BaseModel):
    paths: List[str]  # file names or relative paths only — no file contents


class InferJournalFilenameDatesResponse(BaseModel):
    dates: List[Optional[str]]  # YYYY-MM-DD or null per path, same order


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



class WritingHintsRequest(BaseModel):
    draft: str = ""



class MemoryDiagramResponse(BaseModel):
    mermaid: str


# --- TTS voice list fallback when Mistral list API fails (catalog slugs) ---
MISTRAL_TTS_FALLBACK_VOICES = [
    {"voice_id": "en_paul_neutral", "name": "Paul (neutral)"},
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
    """Voice Memo tab: base64 audio → OpenRouter gpt-audio-mini → optional OpenRouter text polish."""
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


class JournalCleanupRequest(BaseModel):
    text: str
    model: Optional[str] = None


class JournalCleanupResponse(BaseModel):
    cleaned_text: str


VOICE_MEMO_POLISH_INSTRUCTION = """You are editing a voice memo transcript. Clean it into clear, readable prose suitable for a personal journal.
- Fix punctuation, capitalization, and obvious speech-to-text errors.
- Remove filler words (um, uh, like, you know), false starts, and repeated words.
- Preserve meaning, factual content, and the author's wording as much as possible; do not invent events.
- Use first person when the speaker is reflecting on themselves.
- Do NOT rewrite the text as a diary entry or narrative; keep the author's sentence structure.
- Do NOT add new content, headings, sign-offs, or commentary.
Output ONLY the cleaned text with no title or preamble."""

JOURNAL_FEEDBACK_INSTRUCTION = """You are a warm, perceptive companion reading someone's personal journal. Your job is to give thoughtful, substantive feedback that helps the writer understand themselves better.

You have been given:
1. The journal entry the writer just composed.
2. Their most recent previous entries (if any) for continuity.
3. Semantically related passages from their journal history found by vector search — these reveal recurring themes, patterns, and echoes across time.

How to respond:
- Start by genuinely engaging with what they wrote today. Reflect back what you notice — the emotions, the undercurrents, the things said between the lines.
- Then weave in connections to their past writing. Name patterns you see: recurring themes, evolving perspectives, unresolved tensions, or growth arcs. Quote or paraphrase specific past passages when relevant.
- Offer gentle observations or reframings that might help them see their situation from a new angle. You may name difficult realities in a protective, caring way (e.g. a situation sounds unsafe, or someone is treating them poorly).
- Be substantial — aim for several rich paragraphs, not a few sentences. The writer values depth.
- Stay warm, honest, and supportive. Never condescending or preachy. Write like a trusted friend who has been reading their journal for months and genuinely cares.
- Do not ask questions or prompt for replies; the writer will respond on their own.
- Do not reformat or rewrite their entry. Focus entirely on feedback and reflection.

Respond in this exact format:
===FEEDBACK===
<your multi-paragraph feedback>
===OBSERVATIONS===
- <2–5 bullet points: specific patterns, themes, or growth you noticed across entries; reference dates or content where possible>"""


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


async def _polish_voice_memo_openrouter(raw: str, model_override: Optional[str] = None) -> str:
    """Single LLM call: clean up filler words, fix STT errors, polish into readable prose."""
    text = (raw or "").strip()
    if not text or not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        return text
    model = (
        (model_override or "").strip()
        or (os.getenv("OPENROUTER_VOICE_MEMO_POLISH_MODEL") or "openai/gpt-5.4").strip()
    )
    prompt = VOICE_MEMO_POLISH_INSTRUCTION + "\n\n--- Transcript ---\n" + text[:48000]

    def _call():
        try:
            return _openrouter_chat_completion(prompt, model=model, temperature=0.2, timeout_sec=90.0)
        except Exception as e:
            print("[backend] voice-memo polish error:", e)
            return ""

    out = (await asyncio.to_thread(_call) or "").strip()
    return out if out else text


async def _validate_journal_openrouter(
    raw: str,
    model_override: Optional[str] = None,
    instance_id: str = "",
) -> tuple[str, str, list[str], Optional[str]]:
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
        import time as _time

        t0 = _time.perf_counter()
        model = (model_override or os.getenv("OPENROUTER_JOURNAL_VALIDATE_MODEL", "")).strip() or "openai/gpt-5.4"

        # --- 1. Fetch the last 2 saved journal entries for continuity ---
        recent_entries: list[dict] = []
        try:
            recent_entries = await asyncio.to_thread(
                vec_store.list_journal_entries_recent, instance_id or "", limit=2
            )
        except Exception as e:
            print(f"[backend] journal-feedback: recent entries fetch failed: {e}")

        # --- 2. Vector search for thematic echoes across journal history ---
        vec_chunks: list[dict] = []
        try:
            embs = await asyncio.to_thread(_embed_texts, [text[:2000]])
            if embs and embs[0]:
                vec_chunks = await asyncio.to_thread(
                    vec_store.query_journal_chunks, embs[0], instance_id or "", k=12
                )
        except Exception as e:
            print(f"[backend] journal-feedback: vector search failed: {e}")

        t_ctx = _time.perf_counter()
        print(f"[backend] journal-feedback: context retrieval in {t_ctx - t0:.1f}s "
              f"({len(recent_entries)} recent, {len(vec_chunks)} vec chunks)")

        # --- 3. Assemble context block for the prompt ---
        context_parts: list[str] = []

        if recent_entries:
            for i, entry in enumerate(recent_entries):
                doc = (entry.get("document") or "").strip()
                date = (entry.get("timestamp") or entry.get("created_at") or "").strip()
                if doc:
                    label = f"Previous entry #{i + 1}" + (f" ({date})" if date else "")
                    context_parts.append(f"{label}:\n{doc[:4000]}")

        if vec_chunks:
            theme_lines: list[str] = []
            seen: set[str] = set()
            for ch in vec_chunks:
                txt = (ch.get("chunk_text") or "").strip()
                ed = (ch.get("entry_date") or "").strip()
                if txt and txt not in seen:
                    seen.add(txt)
                    theme_lines.append(f"[{ed}] {txt}" if ed else txt)
            if theme_lines:
                context_parts.append(
                    "Related passages from journal history (vector search — recurring themes, "
                    "echoes, patterns):\n" + "\n---\n".join(theme_lines)
                )

        context_block = ""
        if context_parts:
            context_block = (
                "\n\n=== CONTEXT: RECENT ENTRIES & RELATED HISTORY ===\n"
                + "\n\n".join(context_parts)
                + "\n=== END CONTEXT ===\n"
            )

        # --- 4. Build final prompt ---
        prompt = (
            JOURNAL_FEEDBACK_INSTRUCTION
            + context_block
            + "\n\n--- Today's Journal Entry ---\n"
            + text[:48000]
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.45,
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
                with urllib.request.urlopen(req, timeout=180) as resp:
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
        t_llm = _time.perf_counter()
        print(f"[backend] journal-feedback: LLM response in {t_llm - t_ctx:.1f}s (total {t_llm - t0:.1f}s)")

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

        feedback = _between(out, "===FEEDBACK===", "===OBSERVATIONS===")
        observations_block = _between(out, "===OBSERVATIONS===", None)

        # If the model didn't use the delimiters, treat the whole output as feedback
        if not feedback:
            feedback = out.strip()

        notes: list[str] = []
        for line in observations_block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            if line:
                notes.append(line)

        return text, feedback, notes[:8], model
    except Exception as e:
        print("[backend] journal feedback openrouter error:", e)
        import traceback; traceback.print_exc()
        return text, "Feedback failed; please try again.", [str(e)[:180]], None


@api_router.post("/voice")
async def api_voice(req: VoiceRequest):
    """
    Text-to-speech: Mistral Voxtral (requires MISTRAL_API_KEY).
    Returns { audio: base64, format: \"opus\" | \"mp3\" | ... }.
    """
    import base64 as b64

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not mistral_key:
        raise HTTPException(
            status_code=500,
            detail="TTS requires MISTRAL_API_KEY (Mistral Voxtral).",
        )
    vid = _resolve_mistral_voice_id(req.voiceId)
    tts_fmt = _mistral_tts_response_format()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            audio_bytes = await asyncio.to_thread(_mistral_tts_speech, text, vid, tts_fmt)
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                print(f"[backend] Mistral TTS attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                print(f"[backend] Mistral TTS failed after 3 attempts: {e}")
    if last_err is not None:
        raise HTTPException(status_code=500, detail=str(last_err))
    rate = _mistral_tts_playback_rate()
    return {
        "audio": b64.b64encode(audio_bytes).decode("ascii"),
        "format": tts_fmt,
        "provider": "mistral",
        "playback_rate": rate,
    }


@api_router.get("/voices")
async def api_voices():
    """Voices for TTS UI (Mistral Voxtral catalog). Returns { voices: [{ voice_id, name }], provider? }."""
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

    return {"voices": MISTRAL_TTS_FALLBACK_VOICES, "provider": "fallback"}


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
    """User sends text; Interviewer responds. State updated in memory."""
    instance_id = _instance_id(request)
    session_id = get_or_create_session(req.session_id)
    mode = (req.mode or "journal").strip().lower()
    if mode not in ("journal", "conversation", "autobiography"):
        mode = "journal"

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
    _ctc = (req.client_time_context or "").strip() or None
    state: JournalState = {
        "messages": list(messages),
        "session_id": session_id,
        "personalization": personalization,
        "intrusiveness": intrusiveness,
        "mode": mode,
        "instance_id": instance_id,
        **({"openrouter_model": _orm} if _orm else {}),
        **({"client_time_context": _ctc} if _ctc else {}),
    }
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(chat_graph.invoke, state),
            timeout=CHAT_INVOKE_TIMEOUT_SEC,
        )
    except asyncio.CancelledError:
        try:
            messages.pop()
        except IndexError:
            pass
        # Avoid ASGI "500" noise when the client disconnects or the server is stopping mid-request.
        raise HTTPException(status_code=499, detail="Request cancelled")
    except asyncio.TimeoutError:
        messages.pop()
        print("[backend] /chat timed out after %ss" % CHAT_INVOKE_TIMEOUT_SEC)
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
        _es = (req.entry_source or "").strip().lower()
        entry_source = _es if _es in ("manual", "assisted") else None
        extracted = await asyncio.wait_for(
            asyncio.to_thread(
                save_session_data,
                session_id,
                text,
                entry_date,
                instance_id,
                req.content_hash,
                entry_source,
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


JOURNAL_FILENAME_DATE_SYSTEM = """You assign journal entries to a filing date from paths/filenames only. You ONLY see file names or relative paths — never file contents.

For each numbered item, infer the single calendar day the user meant (from the name/path: e.g. 1-30-2026.md, 1-30-26.md, 2026-02-03.md, journals/2026/January/2026-01-30.md). Two-digit years 00–69 mean 2000–2069; 70–99 mean 1970–1999. Use US month-day-year when a short date is ambiguous (first number = month). If you cannot infer a day, use null for that index.

Reply with ONLY valid JSON: an object {"dates": [...]} where dates is an array of exactly N entries in order, each either a string "YYYY-MM-DD" or null. Use two-digit month and day (e.g. 2026-01-30). No markdown, no commentary. N is given in the user message."""


def _normalize_filing_yyyy_mm_dd(raw: str) -> Optional[str]:
    """Accept YYYY-MM-DD, YYYY-M-D, ISO datetime prefix, etc.; return canonical YYYY-MM-DD."""
    s = (raw or "").strip().strip('"').strip("'")
    if not s or s.lower() == "null":
        return None
    s = s.split()[0]
    if "T" in s:
        s = s.split("T", 1)[0].strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        datetime(y, mo, d)
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except ValueError:
        return None


def _year_from_filename_token(tok: str) -> Optional[int]:
    """4-digit year, or 2-digit (00–69 → 20xx, 70–99 → 19xx)."""
    t = (tok or "").strip()
    if not t.isdigit():
        return None
    if len(t) == 4:
        y = int(t)
        return y if 1900 <= y <= 2100 else None
    if len(t) == 2:
        yy = int(t)
        return 1900 + yy if yy >= 70 else 2000 + yy
    return None


def _journal_path_date_fallback(path: str) -> Optional[str]:
    """Path/filename-only calendar day when the LLM returns null (no file contents)."""
    if not (path or "").strip():
        return None
    normalized = path.replace("\\", "/").strip()
    base = normalized.split("/")[-1] or normalized
    stem = Path(base).stem if base else ""
    if not stem:
        return None

    def _ok(y: int, mo: int, d: int) -> Optional[str]:
        try:
            datetime(y, mo, d)
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            return None

    # US: M-D-YYYY or M-D-YY (e.g. 1-30-26.md)
    m = re.match(r"^(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})$", stem)
    if m:
        mo, da = int(m.group(1)), int(m.group(2))
        y = _year_from_filename_token(m.group(3))
        if y is not None:
            hit = _ok(y, mo, da)
            if hit:
                return hit
    # ISO-ish: YYYY-M-D or YY-M-D
    m = re.match(r"^(\d{4}|\d{2})[-_.](\d{1,2})[-_.](\d{1,2})$", stem)
    if m:
        y = _year_from_filename_token(m.group(1))
        if y is not None:
            hit = _ok(y, int(m.group(2)), int(m.group(3)))
            if hit:
                return hit
    # Compact YYYYMMDD or YYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", stem)
    if m:
        hit = _ok(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if hit:
            return hit
    m = re.match(r"^(\d{2})(\d{2})(\d{2})$", stem)
    if m:
        y = _year_from_filename_token(m.group(1))
        if y is not None:
            hit = _ok(y, int(m.group(2)), int(m.group(3)))
            if hit:
                return hit
    m = re.search(r"(?:^|[^\d])(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})(?:[^\d]|$)", base)
    if m:
        y = _year_from_filename_token(m.group(3))
        if y is not None:
            hit = _ok(y, int(m.group(1)), int(m.group(2)))
            if hit:
                return hit
    return None


def _parse_journal_filename_dates_llm_json(text: str, expected_n: int) -> List[Optional[str]]:
    out: List[Optional[str]] = [None] * expected_n
    if expected_n <= 0 or not (text or "").strip():
        return out
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.I)
        if m:
            t = m.group(1).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return out
    arr = data.get("dates") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return out
    for i in range(min(expected_n, len(arr))):
        el = arr[i]
        if el is None:
            continue
        if isinstance(el, str):
            norm = _normalize_filing_yyyy_mm_dd(el)
            if norm:
                out[i] = norm
    return out


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


@api_router.post("/infer-journal-filename-dates", response_model=InferJournalFilenameDatesResponse)
async def infer_journal_filename_dates(req: InferJournalFilenameDatesRequest):
    """
    OpenRouter (default MiniMax M2.7): infer filing calendar day from paths/filenames only.
    Returns YYYY-MM-DD per path; null when unclear. No file contents are sent or read.
    """
    paths = [(p or "").strip() for p in req.paths]
    paths = [p for p in paths if p]
    n = len(paths)
    if n == 0:
        return InferJournalFilenameDatesResponse(dates=[])
    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not or_key:
        fb_only = [_journal_path_date_fallback(paths[i]) for i in range(n)]
        return InferJournalFilenameDatesResponse(dates=fb_only)
    if n > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 paths per request.")
    model = (os.getenv("OPENROUTER_JOURNAL_FILENAME_DATE_MODEL") or "minimax/minimax-m2.7").strip()
    numbered = "\n".join(f"{i + 1}. {paths[i]}" for i in range(n))
    user_msg = f"N={n}. Return JSON with key \"dates\" (array of length {n}).\n\n{numbered}"
    prompt = JOURNAL_FILENAME_DATE_SYSTEM + "\n\n" + user_msg
    try:

        def _call():
            return _openrouter_chat_completion(
                prompt,
                model=model,
                temperature=0.0,
                timeout_sec=120.0,
                max_tokens=min(256 + n * 32, 4096),
            )

        raw = (await asyncio.to_thread(_call) or "").strip()
        parsed = _parse_journal_filename_dates_llm_json(raw, n)
        for i in range(n):
            if parsed[i] is None:
                fb = _journal_path_date_fallback(paths[i])
                if fb:
                    parsed[i] = fb
        return InferJournalFilenameDatesResponse(dates=parsed)
    except Exception as e:
        print("[backend] /infer-journal-filename-dates error:", e)
        fb_only = [_journal_path_date_fallback(paths[i]) for i in range(n)]
        return InferJournalFilenameDatesResponse(dates=fb_only)


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
    Wipe memory for the current user.
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
    Wipe all vector-backed memory for this instance: journal gist/episodic embeddings.
    Used before a full knowledge-base folder re-import so embeddings match the uploaded export only.
    """
    instance_id = _instance_id(request)
    try:
        await asyncio.to_thread(vec_store.wipe_all_vector_memory_for_instance, instance_id)
        return {"ok": True}
    except Exception as e:
        print("[backend] /memory-reset-knowledge-base-import error:", e)
        return {"ok": False, "detail": str(e)}



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
    Speech-to-text: OpenRouter `openai/gpt-audio-mini` (requires OPENROUTER_API_KEY).
    Registered on the main app (not only APIRouter) so POST is never shadowed by the SPA
    catch-all GET /{full_path} (which would otherwise yield 405 Method Not Allowed).
    """
    import base64 as b64

    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        raise HTTPException(
            status_code=500,
            detail="Configure OPENROUTER_API_KEY for transcription (default openai/gpt-audio-mini).",
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
    Voice Memo tab: transcribe via OpenRouter openai/gpt-audio-mini (requires OPENROUTER_API_KEY).
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
    if not or_key:
        raise HTTPException(
            status_code=500,
            detail="Configure OPENROUTER_API_KEY for transcription (default openai/gpt-audio-mini).",
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

    if req.journal_mode:
        # Journal Mode on Home: transcribe only; polish runs when the user taps AI Spelling Correction/Reformatting.
        polished = raw_transcript or ""
        did_polish = False
        cleaned = ""
        _t2 = _time.monotonic()
        print(f"[backend] voice-memo: journal_mode polish skipped in {_t2 - _t1:.1f}s (total {_t2 - _t0:.1f}s)")
    else:
        polished = await _polish_voice_memo_openrouter(raw_transcript)
        _t2 = _time.monotonic()
        print(f"[backend] voice-memo: polish done in {_t2 - _t1:.1f}s (total {_t2 - _t0:.1f}s)")
        did_polish = bool(
            or_key
            and (raw_transcript or "").strip()
            and (polished or "").strip()
            and (polished or "").strip() != (raw_transcript or "").strip()
        )
        cleaned = polished if did_polish else ""

    return {
        "raw_transcript": raw_transcript or "",
        "polished_text": polished or raw_transcript or "",
        "cleaned_transcript": cleaned,
        "transcribe_engine": transcribe_engine,
        "polished_by_llm": did_polish,
    }


@app.post("/api/journal-cleanup", response_model=JournalCleanupResponse, include_in_schema=False)
async def api_journal_cleanup(req: JournalCleanupRequest):
    """Journal Mode: optional LLM pass (same instruction as voice-memo polish) to clean journal text."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text too short to clean up")
    cleaned = await _polish_voice_memo_openrouter(text, req.model)
    return JournalCleanupResponse(cleaned_text=cleaned)


@app.post("/api/journal-validate", response_model=JournalValidateResponse, include_in_schema=False)
async def api_journal_validate(req: JournalValidateRequest, request: Request):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please provide a longer transcript before validation")
    instance_id = _instance_id(request)
    reformatted, feedback, notes, model_used = await _validate_journal_openrouter(
        text, req.model, instance_id=instance_id
    )
    return JournalValidateResponse(
        reformatted_journal=reformatted,
        feedback=feedback,
        validation_notes=notes,
        model_used=model_used,
    )


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
