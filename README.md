# SelfMeridian

SelfMeridian is a private journaling and memory app with a React/Vite frontend and FastAPI backend. The backend uses LangGraph for the journaling chat loop, SQLite + sqlite-vec for local memory, and Tinfoil for all model-backed work.

## Provider Policy

All inference, embeddings, and speech-to-text are routed through Tinfoil. There are no direct calls to other model providers in the app.

Current defaults:

| Capability | Tinfoil model | Notes |
| --- | --- | --- |
| Chat / assisted journal / helper inference | `kimi-k2-6` | Fallback: `deepseek-v4-pro` |
| Manual journal feedback | `kimi-k2-6` or `deepseek-v4-pro` | User-selectable in the UI |
| Manual journal spelling/correction | `gemma4-31b` | Can be overridden with `TINFOIL_VOICE_MEMO_POLISH_MODEL` |
| Speech-to-text | `whisper-large-v3-turbo` | Tinfoil documents mp3 + wav support; browser recordings are converted to wav before upload |
| Embeddings | `nomic-embed-text` | 768 dimensions |

Text-to-speech/read-aloud is disabled for now because Tinfoil does not currently document a TTS model endpoint suitable for this app.

## External APIs Kept

- OpenLibrary remains enabled for book title/author lookup and does not require an API key.
- `LISTENNOTES_API_KEY` is preserved in configuration for future podcast search work, but it is currently unused.

## Setup

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Configure environment

Copy the example and add your Tinfoil key:

```bash
cp .env.example .env
```

Required:

```bash
TINFOIL_API_KEY=...
```

Optional reserved key:

```bash
LISTENNOTES_API_KEY=...
```

### 3. Python backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --reload --port 8000
```

### 4. Frontend

In another terminal from the repository root:

```bash
npm run dev:vite
```

Frontend: `http://localhost:5173`
Backend: `http://localhost:8000`

## Common Overrides

Model defaults live in code and can be overridden through `.env`:

```bash
TINFOIL_CHAT_MODEL=kimi-k2-6
TINFOIL_CHAT_FALLBACK_MODEL=deepseek-v4-pro
TINFOIL_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
TINFOIL_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

Changing `EMBEDDING_DIM` or embedding model requires rebuilding vector memory. The current Tinfoil embedding model uses 768 dimensions; the backend will recreate vector tables when dimensions change.

## Project Layout

```text
src/                 React + Vite frontend
backend/             FastAPI backend, LangGraph chat flow, memory storage
api/                 Optional Node dev handlers mirroring selected API routes
scripts/api-server.ts Local Node API server for dev-only routes
```

## Verification

Useful checks during development:

```bash
python3 -m py_compile backend/main.py backend/graph.py backend/library.py backend/tinfoil_client.py
npm run build
```
