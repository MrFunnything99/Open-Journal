# Selfmeridian Python Backend

Minimal FastAPI + LangGraph backend for stateful multi-agent journaling with **SQLite + sqlite-vec** for vector memory and **LightRAG** for knowledge-graph RAG. The main app uses **Gemini** (embeddings, memory extraction) and **Grok/XAI** (chat); OpenRouter is optional and used only in some Node-side routes if you enable it.

## Setup (1-time)

```bash
# From project root
cd backend
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or as a one-liner from project root:

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Environment

Copy `.env.example` to `.env` in the project root and set:

- `GEMINI_API_KEY` – for embeddings, memory extraction, date inference (required)
- `XAI_API_KEY` – for interviewer chat (Grok 4.20 reasoning, required for /chat)
- `GEMINI_CHAT_MODEL` – e.g. `gemini-3-flash-preview` (optional, for date inference etc.; interviewer uses Grok)
- `GEMINI_EMBEDDING_MODEL` – e.g. `gemini-embedding-2-preview` (optional)
- `EMBEDDING_DIM` – vector size, default `768` (optional)
- `LIGHTRAG_ENABLED` – set to `false` to disable LightRAG indexing (optional)
- `VECTOR_DB_PATH` – full path to SQLite DB file (optional; for production so the vector DB persists)
- `TAVILY_API_KEY` – optional; when set, News & article recommendations use [Tavily Search](https://docs.tavily.com/welcome) (topic=news) for real article URLs
- `SEMANTIC_SCHOLAR_API_KEY` – **currently unused**; Semantic Scholar + PubMed are commented out. Research (and all recommendation categories) use Gemini with Google Search grounding for now.
- `JWT_SECRET` – optional; when set, simple login (username + password, no email) is enabled. Logged-in users get persistent data; anonymous users can still use the app but their data is forgotten after 1 hour (client-side ephemeral instance). Use a long random string.

On macOS, the system Python SQLite may not support extensions; install `pysqlite3` so sqlite-vec works (`pip install pysqlite3`).

## Run

```bash
# From backend/ with venv activated
# From backend directory. Use the venv so uvicorn is available:
.venv/bin/python -m uvicorn main:app --reload --port 8000

# Or activate the venv first, then:
# source .venv/bin/activate   # on macOS/Linux
# uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send user text, get Interviewer response |
| POST | `/end-session` | Trigger Librarian to extract & save to SQLite+sqlite-vec and LightRAG |
| GET | `/lightrag-context?q=...` | Optional RAG context from LightRAG (hybrid/local/global) |
| GET | `/health` | Health check |

## Fly.io: persistent vector DB

**Deploy from repo root** (not from `backend/`):

```bash
cd /path/to/Selfmeridian
fly deploy
```

The `fly.toml` is at the repo root; it builds the monolith (frontend + backend) via `backend/Dockerfile`.

On Fly the app filesystem is **ephemeral**: the SQLite DB is lost on every deploy or machine restart, so memory stats stay at 0. To persist the vector DB:

1. **Create a volume** (one-time, same region as your app, e.g. `iad`):

   ```bash
   fly volumes create data --size 1 --region iad
   ```

2. **Mount it** in `fly.toml`:

   ```toml
   [mounts]
     source = "data"
     destination = "/data"
   ```

3. **Point the app at it** with a secret:

   ```bash
   fly secrets set VECTOR_DB_PATH=/data/open_journal.db
   ```

4. **Redeploy** so the app uses the volume. The DB file will be created under `/data` and will persist across deploys.
