# Open-Journal Python Backend

Minimal FastAPI + LangGraph backend for stateful multi-agent journaling with **SQLite + sqlite-vec** for vector memory and **LightRAG** for knowledge-graph RAG.

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

- `GEMINI_API_KEY` – for embeddings and chat (required)
- `GEMINI_CHAT_MODEL` – e.g. `gemini-1.5-flash` (optional)
- `GEMINI_EMBEDDING_MODEL` – e.g. `gemini-embedding-2-preview` (optional)
- `EMBEDDING_DIM` – vector size, default `768` (optional)
- `LIGHTRAG_ENABLED` – set to `false` to disable LightRAG indexing (optional)

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
