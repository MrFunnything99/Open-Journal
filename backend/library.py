"""
ChromaDB library for Open-Journal: gist_facts (semantic) and episodic_log (episodic) memory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import chromadb

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langchain_openai import ChatOpenAI
from voyageai import Client as VoyageClient

# Paths
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"
COLLECTION_GIST = "gist_facts"
COLLECTION_EPISODIC = "episodic_log"

# Clients (lazy init)
_client: chromadb.PersistentClient | None = None
_voyage: VoyageClient | None = None
_llm: ChatOpenAI | None = None


def _get_chroma() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_voyage() -> VoyageClient:
    global _voyage
    if _voyage is None:
        key = os.getenv("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY is required. Get one at https://dash.voyageai.com/")
        _voyage = VoyageClient(api_key=key)
    return _voyage


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required. Get one at https://openrouter.ai/keys")
        _llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
        )
    return _llm


def _ensure_collections():
    """Create gist_facts and episodic_log collections with cosine similarity."""
    client = _get_chroma()
    for name in [COLLECTION_GIST, COLLECTION_EPISODIC]:
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(name, metadata={"hnsw:space": "cosine"})


def _extract_session_data(transcript: str) -> dict:
    """Use OpenRouter LLM to extract summary and facts from transcript."""
    llm = _get_llm()
    prompt = f"""You are a journal analyst. Extract structured data from this journal session transcript.

Transcript:
---
{transcript}
---

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "summary": "A 3-sentence summary of the session: what was discussed, key themes, and emotions felt.",
  "facts": ["Fact 1 about the user", "Fact 2 about the user", ...]
}}

Rules:
- summary: exactly 3 sentences, capture emotions and themes
- facts: list of hard, verifiable facts about the user (job, relationships, preferences, life events). Empty list if none.
"""
    response = llm.invoke(prompt)
    text = response.content.strip()
    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def save_session_data(session_id: str, transcript: str) -> None:
    """
    Extract summary + facts from transcript via OpenRouter, embed via Voyage, save to ChromaDB.
    """
    _ensure_collections()
    client = _get_chroma()
    voyage = _get_voyage()

    data = _extract_session_data(transcript)
    summary = data.get("summary", "")
    facts = data.get("facts", [])

    ts = datetime.utcnow().isoformat() + "Z"

    # Episodic: add summary with metadata
    if summary:
        ep_col = client.get_collection(COLLECTION_EPISODIC)
        summary_emb = voyage.embed(
            [summary],
            model=os.getenv("VOYAGE_MODEL", "voyage-3"),
            input_type="document",
        ).embeddings[0]
        ep_id = f"{session_id}_ep_{ts.replace(':', '-').replace('.', '-')}"
        ep_col.add(
            ids=[ep_id],
            embeddings=[summary_emb],
            documents=[summary],
            metadatas=[{"session_id": session_id, "timestamp": ts}],
        )

    # Gist: add each fact
    if facts:
        gist_col = client.get_collection(COLLECTION_GIST)
        fact_embs = voyage.embed(
            facts,
            model=os.getenv("VOYAGE_MODEL", "voyage-3"),
            input_type="document",
        ).embeddings
        ts_safe = ts.replace(":", "-").replace(".", "-")
        gist_col.add(
            ids=[f"{session_id}_gist_{i}_{ts_safe}" for i in range(len(facts))],
            embeddings=fact_embs,
            documents=facts,
            metadatas=[{"session_id": session_id, "timestamp": ts} for _ in facts],
        )
