# SelfMeridian Python backend

FastAPI application plus LangGraph chat flow, SQLite + **sqlite-vec** for memory, and optional **LightRAG** (`LIGHTRAG_ENABLED`, off by default).

**Environment:** copy `.env.example` to `.env` in the **project root** (not only `backend/`). See the main **[README.md](../README.md)** for OpenRouter, Perplexity, and Mistral setup.

## One-time setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, if system SQLite lacks extension support, you may need `pysqlite3` for sqlite-vec (see `requirements.txt` / install notes in the main README).

## Run

```bash
# From backend/ with venv active
.venv/bin/python -m uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

## Fly.io

Deploy from the **repository root** (`fly deploy`). Persist the vector DB with a volume and `VECTOR_DB_PATH` — details in the main **[README.md](../README.md#deployment-flyio)** and **[STORAGE.md](./STORAGE.md)**.
