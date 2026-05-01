# SelfMeridian Python backend

FastAPI application plus LangGraph chat flow and SQLite + **sqlite-vec** memory. All inference, embeddings, and speech-to-text route through Tinfoil.

**Environment:** copy `.env.example` to `.env` in the project root (not only `backend/`). Required: `TINFOIL_API_KEY`. Reserved: `LISTENNOTES_API_KEY`. OpenLibrary lookup is used without an API key for book normalization.

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

Deploy from the repository root (`fly deploy`). Persist the vector DB with a volume and `VECTOR_DB_PATH`; details live in the main README and `STORAGE.md`.
