# SelfMeridian

SelfMeridian is a local-first journaling app: type, dictate, or import entries, and chat with an AI journaling companion. **This build is centered on the journaling experience itself**—not on semantic search over your past entries.

This document is the **single source of truth** for setup, API keys, environment variables, and deployment. The Python backend loads `.env` from the **project root** (parent of `backend/`).

---

## Current functionality (what actually matters right now)

| Area | Behavior in this build |
|------|-------------------------|
| **AI-Assisted Journal** (default home experience) | The model replies using **only the current thread** (and system instructions). It does **not** run **vector / semantic search** over older journals to retrieve context. |
| **Manual journal** | Write and save in the editor; dictation and cleanup helpers use the LLM/STT stack as implemented in the UI. |
| **Vector search / RAG** | **Not used for AI-Assisted Journal** (`autobiography`, the default home mode): replies use the **current thread only**. The backend may still expose **journal** / **conversation** `/chat` paths that *can* pull vector context when enabled; this README describes the **assisted-journal–first** product. Storage and embedding code remains for **future** work. |
| **OpenRouter** | Required for chat, transcription, and most model-backed helpers. |
| **Perplexity** | Optional here: only needed if you turn on workflows that **embed or query** the vector store (not required to use assisted or manual journaling as described above). |
| **Mistral** | Optional: text-to-speech (read-aloud) only. |

---

## Table of contents

1. [Current functionality](#current-functionality-what-actually-matters-right-now)
2. [What you need](#what-you-need)
3. [Getting API keys](#getting-api-keys)
4. [Quick start (local)](#quick-start-local)
5. [Environment variables](#environment-variables)
6. [How the app is structured](#how-the-app-is-structured)
7. [Architecture](#architecture)
8. [Deployment (Fly.io)](#deployment-flyio)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Node.js 18+** | For Vite and the dev toolchain |
| **Python 3.11+** | Recommended for the FastAPI backend |
| **OpenRouter account** | Powers chat, speech-to-text (via multimodal models), transcript polish, and most LLM-backed helpers |
| **Perplexity API key** | **Optional in this build**—only for embedding/indexing flows if you use them; **not** required for assisted-journal chat (no vector retrieval there) |
| **Mistral API key** | Optional; required only for **text-to-speech** (read-aloud / voice playback) via Mistral Voxtral |

You do **not** need separate API keys for OpenAI, Anthropic, or Google **if** you run those models **through OpenRouter** (one key, many providers).

---

## Getting API keys

### 1. OpenRouter (primary)

OpenRouter is an API gateway: one key can call models from OpenAI, Anthropic, Google, xAI, Meta, and others—billed through OpenRouter.

**Steps:**

1. Open **[openrouter.ai](https://openrouter.ai/)** and sign up or log in.
2. Go to **[openrouter.ai/keys](https://openrouter.ai/keys)** (or **Settings → Keys**).
3. Click **Create Key**, name it (e.g. `selfmeridian-dev`), and copy the secret. **Store it only in `.env`**—never commit it.
4. Add **credits** under **Credits** / billing if prompted (many models require a positive balance).
5. Browse **[openrouter.ai/models](https://openrouter.ai/models)** to see model IDs (e.g. `openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4.6`). The app uses these strings in `OPENROUTER_*` overrides (see [Environment variables](#environment-variables)).

**Docs:** [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)

**Optional headers** (some setups use them for rankings/analytics on OpenRouter’s side):

- `OPENROUTER_REFERER` (default in code: `https://selfmeridian.local`)
- `OPENROUTER_TITLE` (default: `SelfMeridian`)

### 2. Frontier providers (direct sites)—when you might use them

| Provider | Where to get a key | Typical use with SelfMeridian |
|----------|--------------------|--------------------------------|
| **OpenAI** | [platform.openai.com](https://platform.openai.com/) → API keys | Optional: **fallback** STT via `OPENAI_API_KEY` in some dev/Node transcribe paths—not the main backend path |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) | Not required if you use Anthropic models **via OpenRouter** |
| **Google (Gemini)** | [aistudio.google.com](https://aistudio.google.com/) | Not required for this repo’s default path; extraction/helpers use **OpenRouter** (`OPENROUTER_EXTRACTION_MODEL`, etc.) |
| **xAI (Grok)** | [console.x.ai](https://console.x.ai/) | Optional; Grok is often used **through OpenRouter** (e.g. `x-ai/grok-4.1-fast`) |

**Practical rule:** start with **only `OPENROUTER_API_KEY`**. Add **Mistral** if you want TTS. Add **Perplexity** only when you need embedding-backed features (not required for the current no–vector-search journaling path).

### 3. Perplexity (optional—embeddings / future memory pipelines)

Skip this section if you are only testing **assisted** and **manual** journaling: those flows do **not** depend on Perplexity today.

When you need it:

1. Open **[perplexity.ai](https://www.perplexity.ai/)** and sign in.
2. Open API settings: **[docs.perplexity.ai](https://docs.perplexity.ai/docs/getting-started/quickstart)**.
3. Create an API key and set `PERPLEXITY_API_KEY` in `.env`.

Used for contextual embeddings when those code paths are active (`PERPLEXITY_EMBEDDING_MODEL`, `EMBEDDING_DIM`, etc.).

### 4. Mistral (TTS only)

1. Open **[console.mistral.ai](https://console.mistral.ai/)** (or **[docs.mistral.ai](https://docs.mistral.ai/)** → API keys).
2. Create a key and set `MISTRAL_API_KEY` in `.env`.

Without it, chat and journaling still work; **read-aloud / spoken replies** will not.

### 5. Optional: login (`JWT_SECRET`)

For register/login/refresh flows, set a long random string:

```bash
openssl rand -hex 32
```

Put the result in `JWT_SECRET`. For production cookies, set `ENVIRONMENT=production` (see `.env.example`).

---

## Quick start (local)

### 1. Clone and install frontend

```bash
cd Selfmeridian
npm install
```

### 2. Environment file

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

Add `MISTRAL_API_KEY` if you want read-aloud TTS. Add `PERPLEXITY_API_KEY` only if you are using embedding-heavy features (not needed for the default journaling experience without vector retrieval).

### 3. Python backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`

### 4. Frontend

In a **second** terminal, from the **project root**:

```bash
npm run dev
```

Open **`http://localhost:5173`**.

By default, Vite proxies **`/api`** to **`http://localhost:8000`** (`VITE_API_URL` in `.env` overrides this). Ensure the FastAPI server is running before using the app.

### 5. Smoke test

- Open the app, send a short message in **AI-Assisted Journal** or use **Manual Journal** as you prefer.
- Confirm the backend logs show successful `/chat` or voice-memo requests and no missing **`OPENROUTER_API_KEY`** errors.

---

## Environment variables

### Essential for the current journaling UX

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Chat (`/chat`), dictation STT, voice-memo transcription, polish/validation helpers |

### Optional

| Variable | Purpose |
|----------|---------|
| `MISTRAL_API_KEY` | `/api/voice` TTS (read-aloud, voice playback) |
| `PERPLEXITY_API_KEY` | Embeddings when ingest/vector workflows are used—not required for assisted-journal chat without retrieval |

### Auth (optional)

| Variable | Purpose |
|----------|---------|
| `JWT_SECRET` | Enables `/api/register`, `/api/login`, `/api/refresh` |
| `ENVIRONMENT` | Set to `production` for secure refresh cookies in production |

### OpenRouter model overrides (optional)

Defaults are set in code (`backend/graph.py`, `backend/main.py`, etc.). Override only when you need a different model or timeout:

| Variable | Typical role |
|----------|----------------|
| `OPENROUTER_CHAT_MODEL` | Journal / interviewer chat |
| `OPENROUTER_CHAT_FALLBACK_MODEL` | Fallback when primary errors |
| `OPENROUTER_CONVERSATION_MODEL` | Conversation-style subgraph |
| `OPENROUTER_ASSISTED_JOURNAL_MODEL` | AI-assisted journal mode |
| `OPENROUTER_TRANSCRIPTION_MODEL` | STT (default `openai/gpt-audio-mini`) |
| `OPENROUTER_VOICE_MEMO_POLISH_MODEL` | Optional polish after STT |
| `OPENROUTER_EXTRACTION_MODEL` | Library / extraction LLM (alias: `OPENROUTER_GEMINI_MODEL`) |

See **`.env.example`** for additional optional keys (Tavily, Semantic Scholar, Listen Notes, timeouts, etc.).

### Local dev / proxy

| Variable | Purpose |
|----------|---------|
| `VITE_API_URL` | Vite dev proxy target for `/api` (default `http://localhost:8000`) |
| `API_PORT` | Port for the optional Node `scripts/api-server.ts` (default `3001`) |

---

## How the app is structured

```text
.
├── src/                      # React + Vite frontend
│   └── pages/Personaplex/    # Main UI, chat context, journal history
├── backend/                  # FastAPI + LangGraph
│   ├── main.py               # Routes, app entry
│   ├── graph.py              # Chat graph (journal vs assisted journal, tools)
│   ├── library.py            # Ingest helpers and related logic (vector search not used for assisted journal replies)
│   └── vec_store.py          # SQLite + sqlite-vec (used when embedding pipelines run)
├── api/                      # Node route handlers (used by api-server / serverless paths)
├── scripts/api-server.ts     # Local Node server (optional; see Vite proxy)
├── .env.example
├── fly.toml
└── README.md
```

More backend-only operational notes: **`backend/STORAGE.md`**. Short backend runbook: **`backend/README.md`**.

---

## Architecture

```mermaid
flowchart LR
  U[User] --> FE[React + Vite]
  FE --> API[FastAPI backend]
  API --> OR[OpenRouter chat + STT]
  API --> MIST[Mistral TTS optional]
  API -.-> DB[(SQLite / vectors optional)]
  API -.-> PPLX[Perplexity optional]
```

- **Assisted journal `/chat`**: model sees **this session’s messages** only—**no vector retrieval** step in that path.
- **History** (browser) holds what the user saves; backend may still persist or index data for **future** features—see code and `.env.example` if you enable those paths.
- **LightRAG** (`LIGHTRAG_ENABLED`) and **Perplexity** are optional add-ons, not part of the default journaling loop described above.

---

## Deployment (Fly.io)

Single app: FastAPI serves the API and the built SPA.

```bash
fly deploy
```

**Persistent data:** Fly machines are ephemeral. If you rely on a local SQLite file (any feature that writes to disk), create a volume and set `VECTOR_DB_PATH` (or your DB path) on that volume (see `.env.example` and Fly docs).

```bash
fly volumes create data --size 1 --region iad
fly secrets set VECTOR_DB_PATH=/data/open_journal.db
fly deploy
```

Mount the volume in `fly.toml` under `[mounts]` if not already configured.

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Chat or STT fails immediately | `OPENROUTER_API_KEY` in **project root** `.env`; restart `uvicorn` after edits |
| Embedding or ingest features fail | `PERPLEXITY_API_KEY` and matching `EMBEDDING_DIM`; persistent `VECTOR_DB_PATH` on Fly if using disk-backed stores |
| No read-aloud / TTS | `MISTRAL_API_KEY` on the server that handles `/api/voice` |
| Frontend 404 on `/api/*` | FastAPI running on the host/port `VITE_API_URL` points to |
| CORS / wrong API in prod | Production is same-origin; avoid pointing the built app at the wrong backend URL |

---

## License

MIT — see `LICENSE-MIT`.
