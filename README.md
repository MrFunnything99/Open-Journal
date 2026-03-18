# Selfmeridian

A lightweight AI voice journaling application. Speak naturally to an AI assistant that guides self-reflection through probing questions. Includes memory (facts & summaries), a people graph, recommendations (books, podcasts, articles), and calendar day summaries. Voice uses ElevenLabs; the journaling agent and memory use a Python backend (Gemini, Grok, SQLite + vector store).

**By:** John S., Sherelle M., Aniyah T., Dominique S., Andy C., Jackeline G.

---

## Features

- **Voice journaling** — Real-time speech-to-text (ElevenLabs Scribe v2) and text-to-speech. Say "Start" or "Hi" to begin; the agent asks reflective questions and remembers context.
- **Session modes** — Journal (default), Recommendations (talk about books/media and save notes), Extreme (more probing), Therapy.
- **Memory** — Facts and session summaries stored in a vector DB; editable in the Memory tab. Optional LightRAG knowledge graph.
- **People graph** — View and edit people and relationships extracted from journal entries; auto-grouping and thoughts.
- **Recommendations** — AI-suggested books, podcasts, articles, and research; mark as consumed and add notes. Podcast links via Listen Notes (optional).
- **Library** — Your saved books/media; library interview to populate, and notes tied to recommendations.
- **Calendar** — Browse by month; get an AI-generated day summary for a selected date.
- **Journal history & ingest** — Paste or upload past journal entries; optional date inference and ingest into history/memory.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 18, Vite, Tailwind CSS, React Router |
| **Voice** | ElevenLabs TTS + Realtime Speech-to-Text (Scribe v2) |
| **Node API** (voice/interview/reformat) | TypeScript serverless-style routes (`api/*`), run locally via `scripts/api-server.ts` or Vercel serverless |
| **Backend** | Python 3, FastAPI, LangGraph, SQLite + sqlite-vec, LightRAG (optional), Gemini (embeddings, memory, date inference), Grok/XAI (chat) |

---

## Project Structure

```
├── api/                    # Node API routes (Vercel serverless or dev proxy)
│   ├── interviewer.ts      # OpenRouter LLM for interviewer flow
│   ├── voice.ts            # ElevenLabs TTS
│   ├── voices.ts           # List voices
│   ├── scribe-token.ts     # ElevenLabs Scribe token for real-time STT
│   ├── transcribe.ts       # Transcription
│   └── reformat.ts         # Journal reformat
├── backend/                # Python FastAPI backend (chat, memory, recommendations, etc.)
│   ├── main.py             # Routes and app
│   ├── graph.py            # LangGraph chat/librarian
│   ├── library.py          # Memory, recommendations, library, LightRAG
│   ├── vec_store.py        # SQLite + sqlite-vec
│   ├── lightrag_bridge.py  # LightRAG integration
│   ├── requirements.txt
│   ├── Dockerfile
│   └── fly.toml            # Fly.io deployment
├── scripts/
│   └── api-server.ts       # Local dev server for /api/* (interviewer, voice, etc.)
├── src/
│   ├── app.tsx
│   ├── pages/
│   │   └── Personaplex/     # Main UI (orb, settings, session, history, memory, recommendations, calendar)
│   └── ...
├── public/
├── .env.example
├── package.json
├── vite.config.ts
└── vercel.json             # Frontend deploy (Vercel)
```

---

## Quick Start

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Copy `.env.example` to `.env` in the project root and set:

**Required for voice and Node API (interviewer/reformat):**

- `OPENROUTER_API_KEY` — [OpenRouter](https://openrouter.ai/keys) (for interviewer/reformat if used via Node)
- `ELEVENLABS_API_KEY` — [ElevenLabs](https://elevenlabs.io/app/settings/api-keys) (TTS + Scribe)

**Required for Python backend (chat, memory, recommendations):**

- `GEMINI_API_KEY` — Embeddings, memory extraction, date inference
- `XAI_API_KEY` — Grok for `/chat` interviewer

**Optional:**

- `VITE_BACKEND_URL` — Python backend URL (default `http://localhost:8000`). Set to your Fly backend URL in production.
- `API_PORT`, `VITE_API_URL` — If port 3001 is in use (e.g. `API_PORT=3002`, `VITE_API_URL=http://localhost:3002`)
- `LISTENNOTES_API_KEY` — Podcast episode links in recommendations
- `CORS_ORIGINS` — Comma-separated origins for the Python backend (production); default list includes common dev and one Vercel URL

### 3. Run the Python backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: on macOS if sqlite-vec fails, use: pip install pysqlite3
.venv/bin/python -m uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  

### 4. Run the frontend and Node API (dev)

In a second terminal from the project root:

```bash
npm run dev
```

This starts:

- **Vite** on port 5173 (proxies `/api` to the Node API server)
- **Node API server** on port 3001 (serves `/api/interviewer`, `/api/voice`, `/api/voices`, `/api/scribe-token`, `/api/transcribe`, `/api/reformat`)

Open http://localhost:5173. The app will use:

- `http://localhost:8000` for chat, memory, recommendations, library, calendar, ingest (Python backend)
- `http://localhost:3001/api/*` for voice and interviewer (via Vite proxy)

### 5. Build for production

```bash
npm run build
```

Output is in `dist/`. Preview with `npm run preview`.

---

## Environment Variables Reference

| Variable | Where | Purpose |
|----------|--------|--------|
| `OPENROUTER_API_KEY` | Root `.env` | OpenRouter for Node interviewer/reformat |
| `ELEVENLABS_API_KEY` | Root `.env` | ElevenLabs TTS + Scribe |
| `XAI_API_KEY` | Root `.env` | Grok for Python `/chat` |
| `GEMINI_API_KEY` | Root `.env` (or backend) | Gemini for embeddings, memory, date inference |
| `VITE_BACKEND_URL` | Root `.env` (build-time) | Python backend URL (e.g. Fly); default `http://localhost:8000` |
| `API_PORT` | Root `.env` | Node API server port (default 3001) |
| `VITE_API_URL` | Root `.env` | URL for Vite proxy to Node API (default `http://localhost:3001`) |
| `LISTENNOTES_API_KEY` | Root `.env` | Optional; podcast links in recommendations |
| `CORS_ORIGINS` | Backend (e.g. Fly secrets) | Allowed CORS origins for backend (comma-separated) |
| `GEMINI_CHAT_MODEL` | Backend | Optional; default used for some backend tasks |
| `GEMINI_INFER_ENTRY_DATE_MODEL` | Backend | Optional; model for `/infer-entry-date` (default `gemini-3.1-flash-lite-preview`) |

---

## Deployment

### Frontend (Vercel)

- **Root directory:** `.` (project root)
- **Build:** `npm run build`  
- **Output:** `dist`
- **Environment variables:** Set `VITE_BACKEND_URL` to your Python backend URL (e.g. `https://your-app.fly.dev`). Also set `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY` if your Vercel app serves the `/api` routes (interviewer, voice, etc.).
- **Rewrites:** `vercel.json` sends non-`/api` traffic to `index.html` for SPA routing.

### Backend (Fly.io)

- From `backend/`: `fly launch` (if not already), then `fly deploy`.
- Set secrets, e.g.:  
  `fly secrets set GEMINI_API_KEY=... XAI_API_KEY=...`  
  Optionally: `fly secrets set CORS_ORIGINS="https://your-vercel-app.vercel.app"`
- The app listens on port 8080 (see `fly.toml`). Docs: `https://your-app.fly.dev/docs`.

---

## Backend API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send user text; get interviewer response (Grok) with memory context |
| POST | `/end-session` | End session; librarian extracts and saves to memory |
| POST | `/ingest-history` | Ingest past journal text (optional date inference) |
| POST | `/infer-entry-date` | Infer date from entry text (and optional filename) |
| GET | `/memory-stats` | Vector store stats |
| GET/POST/DELETE | `/memory/facts`, `/memory/summaries` | List, create, delete memory facts/summaries |
| GET | `/memory-diagram` | Mermaid diagram of memory |
| POST | `/memory-wipe` | Wipe vector memory |
| GET | `/brain/people-graph`, `/brain/people`, `/brain/people/{id}` | People graph and CRUD |
| POST | `/brain/people/auto-groups` | Auto-group people |
| GET | `/recommendations` | Books, podcasts, articles, research suggestions |
| POST | `/recommendations/consumed` | Mark items consumed, add notes |
| GET/POST/DELETE | `/library` | Library items |
| POST | `/library-notes` | Save notes for library/recommendations |
| POST | `/library-interview` | Conversational library interview |
| POST | `/calendar-day-summary` | Day summary for a given date |
| GET | `/lightrag-context` | Optional RAG context (LightRAG) |
| GET | `/health` | Health check |

---

## License

MIT (see `LICENSE-MIT`).
