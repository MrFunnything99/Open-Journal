# Selfmeridian Python Backend

Minimal FastAPI + LangGraph backend for stateful multi-agent journaling with **SQLite + sqlite-vec** as the primary vector memory. **LightRAG** is optional and **off by default** (see `lightrag_bridge.py`). The main app uses **Perplexity** (vector embeddings), **Gemini or OpenRouter** (memory extraction and helpers), and **OpenRouter** (journal `/chat` interviewer, default `openai/gpt-5.4`).

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

- `PERPLEXITY_API_KEY` – for vector embeddings (gist / episodic / library search); required for memory retrieval indexing
- `GEMINI_API_KEY` – for Gemini `generate_content` only: extraction, recommendations helpers, date inference (not embeddings)
- `OPENROUTER_API_KEY` – required for `/chat` (journal interviewer) and journal validation; optional model via `OPENROUTER_CHAT_MODEL` (default `openai/gpt-5.4`)
- `GEMINI_CHAT_MODEL` – e.g. `gemini-3-flash-preview` when using direct Google SDK for extraction (not used by `/chat`)
- `PERPLEXITY_EMBEDDING_MODEL` – default `pplx-embed-context-v1-4b` (contextualized API; use `pplx-embed-v1-4b` for standard `/v1/embeddings` if you switch)
- `EMBEDDING_DIM` – must match model output (default `2560` for full-size 4B Perplexity embeddings; use Matryoshka `dimensions` in API only if you align this env)
- `LIGHTRAG_ENABLED` – default `false`; set to `true` to enable optional LightRAG indexing (`lightrag_bridge.py`)
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
| GET | `/lightrag-context?q=...` | LightRAG-only context (empty when LightRAG disabled; primary RAG is sqlite-vec in `/chat`) |
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
