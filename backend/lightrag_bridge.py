"""
LightRAG bridge for Selfmeridian: index journal content and run RAG queries (hybrid/local/global).
Uses the same Gemini embedding and LLM as the rest of the app. Data is stored under data/lightrag/.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

WORKING_DIR = Path(__file__).resolve().parent.parent / "data" / "lightrag"
LIGHTRAG_ENABLED = os.getenv("LIGHTRAG_ENABLED", "true").lower() in ("true", "1", "yes")

_rag = None
_rag_ready = False


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Call library embeddings (sync)."""
    from library import _embed_texts
    return _embed_texts(texts)


def _call_gemini_sync(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
    """Call library LLM (sync). history_messages ignored for single-turn."""
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
    if not LIGHTRAG_ENABLED:
        return None
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
    except ImportError:
        return None
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    # LightRAG needs EmbeddingFunc with embedding_dim and max_token_size
    embed_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=8192,
        func=_embed_async,
    )
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        embedding_func=embedding_func,
        llm_model_func=_llm_async,
        llm_model_name=os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
    )
    await rag.initialize_storages()
    _rag = rag
    _rag_ready = True
    return rag


async def insert_text(text: str) -> bool:
    """Index text into LightRAG (e.g. session summary or transcript). Returns True if inserted."""
    if not text or not text.strip():
        return False
    rag = await _get_rag()
    if rag is None:
        return False
    try:
        await rag.ainsert(text.strip())
        return True
    except Exception as e:
        if "history_messages" not in str(e):
            print("[backend] LightRAG ainsert error:", e)
        return False


async def query_for_context(query: str, mode: str = "hybrid", top_k: int = 20) -> str:
    """
    Run LightRAG query with only_need_context=True and return the retrieved context string.
    mode: local | global | hybrid | naive | mix
    """
    if not query or not query.strip():
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
