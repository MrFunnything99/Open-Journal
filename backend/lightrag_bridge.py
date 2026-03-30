"""
LightRAG bridge (optional, **off by default**).

Selfmeridian’s primary memory path is SQLite + sqlite-vec (`vec_store` / `library.get_relevant_context`).
This module is isolated here so it can be re-enabled with `LIGHTRAG_ENABLED=true` without touching
the main retrieval pipeline. When disabled, public helpers return immediately and never import `lightrag`.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

WORKING_DIR = Path(__file__).resolve().parent.parent / "data" / "lightrag"
LIGHTRAG_ENABLED = os.getenv("LIGHTRAG_ENABLED", "false").lower() in ("true", "1", "yes")

_rag = None
_rag_ready = False
_lightrag_runtime_disabled = False


def _disable_lightrag_runtime(reason: str) -> None:
    """Disable LightRAG for this process after a hard runtime incompatibility."""
    global _lightrag_runtime_disabled
    if not _lightrag_runtime_disabled:
        print("[backend] LightRAG disabled for this process:", reason)
    _lightrag_runtime_disabled = True


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Call library embeddings (sync)."""
    from library import _embed_texts
    return _embed_texts(texts)


def _call_extraction_llm_sync(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
    """Call library OpenRouter-backed extraction LLM. Not used for embeddings."""
    from library import _call_gemini
    return _call_gemini(prompt)


async def _embed_async(texts: list[str]):
    import numpy as np
    loop = asyncio.get_event_loop()
    embs = await loop.run_in_executor(None, _embed_texts_sync, texts)
    return np.array(embs, dtype=np.float32)


async def _llm_async(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_gemini_sync, prompt)


async def _get_rag():
    """Create and initialize LightRAG once (async)."""
    global _rag, _rag_ready
    if _rag_ready and _rag is not None:
        return _rag
    if not LIGHTRAG_ENABLED or _lightrag_runtime_disabled:
        return None
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        return None
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    # LightRAG needs EmbeddingFunc with embedding_dim and max_token_size
    embed_dim = int(os.getenv("EMBEDDING_DIM", "2560"))
    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=8192,
        func=_embed_async,
    )
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        embedding_func=embedding_func,
        llm_model_func=_llm_async,
        llm_model_name=(
            (os.getenv("OPENROUTER_EXTRACTION_MODEL") or os.getenv("OPENROUTER_GEMINI_MODEL") or "google/gemini-3-pro-preview")
            .strip()
        ),
    )
    try:
        await rag.initialize_storages()
    except Exception as e:
        # Known compatibility breakage in some lightrag-hku versions.
        if "DocProcessingStatus.__init__()" in str(e) and "error_msg" in str(e):
            _disable_lightrag_runtime("DocProcessingStatus schema mismatch")
            return None
        raise
    _rag = rag
    _rag_ready = True
    return rag


async def schedule_lightrag_index_after_ingest(doc: str) -> None:
    """
    Background hook after session ingest: index summary+facts into LightRAG only when enabled.
    No-op when `LIGHTRAG_ENABLED` is false (default). Safe to `asyncio.create_task(...)` from FastAPI.
    """
    if not LIGHTRAG_ENABLED or not (doc or "").strip():
        return
    try:
        await insert_text(doc.strip())
    except Exception as e:
        if "GenericAlias" not in str(e) and "NoneType" not in str(e):
            print("[backend] LightRAG schedule_lightrag_index_after_ingest:", e)


async def insert_text(text: str) -> bool:
    """Index text into LightRAG (e.g. session summary or transcript). Returns True if inserted."""
    if not LIGHTRAG_ENABLED or not text or not text.strip():
        return False
    rag = await _get_rag()
    if rag is None:
        return False
    try:
        # Keep this strictly best-effort so background indexing cannot degrade UX.
        await asyncio.wait_for(rag.ainsert(text.strip()), timeout=20)
        return True
    except Exception as e:
        if "DocProcessingStatus.__init__()" in str(e) and "error_msg" in str(e):
            _disable_lightrag_runtime("DocProcessingStatus schema mismatch")
            return False
        if "history_messages" not in str(e):
            print("[backend] LightRAG ainsert error:", e)
        return False


async def query_for_context(query: str, mode: str = "hybrid", top_k: int = 20) -> str:
    """
    Run LightRAG query with only_need_context=True and return the retrieved context string.
    mode: local | global | hybrid | naive | mix
    Returns "" when LightRAG is disabled or unavailable.
    """
    if not LIGHTRAG_ENABLED or not query or not query.strip():
        return ""
    rag = await _get_rag()
    if rag is None:
        return ""
    try:
        from lightrag import QueryParam
        result = await rag.aquery(
            query.strip(),
            param=QueryParam(mode=mode, only_need_context=True, top_k=top_k),
        )
        return (result or "").strip()
    except Exception as e:
        print("[backend] LightRAG aquery error:", e)
        return ""
