# Open-Journal Python Backend

Minimal FastAPI + LangGraph backend for stateful multi-agent journaling with ChromaDB memory.

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

Copy `.env.example` to `.env` in the project root and add:

- `OPENROUTER_API_KEY` – from https://openrouter.ai/keys
- `VOYAGE_API_KEY` – from https://dash.voyageai.com/

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
| POST | `/end-session` | Trigger Librarian to extract & save to ChromaDB |
| GET | `/health` | Health check |
