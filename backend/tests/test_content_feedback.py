from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def isolated_vec_db(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_vec.db"
        monkeypatch.setenv("VECTOR_DB_PATH", str(p))
        import vec_store

        importlib.reload(vec_store)
        vec_store._conn = None  # type: ignore[attr-defined]
        vec_store.ensure_db()
        if "decision_logger" in sys.modules:
            importlib.reload(sys.modules["decision_logger"])
        if "library" in sys.modules:
            importlib.reload(sys.modules["library"])
        yield vec_store
        vec_store._conn = None  # type: ignore[attr-defined]


def test_content_feedback_process_mock_llm(isolated_vec_db, monkeypatch):
    import vec_store

    eid = vec_store.journal_entry_insert(
        instance_id="cf",
        session_id="e1",
        entry_date="2025-01-01",
        raw_text="journaled about stress",
    )
    emb = [0.01 * (i % 7) for i in range(vec_store.EMBEDDING_DIM)]
    vec_store.journal_chunk_insert(
        eid,
        instance_id="cf",
        chunk_index=0,
        chunk_text="journaled about stress",
        entry_date="2025-01-01",
        embedding=emb,
    )

    def fake_gemini(_prompt: str) -> str:
        return json.dumps(
            {
                "tags": ["stress", "resilience"],
                "reasoning": "They seem to care about coping.",
                "journal_themes": ["stress"],
            }
        )

    monkeypatch.setattr("library._call_gemini", fake_gemini)
    monkeypatch.setattr("library._embed_texts", lambda texts: [[0.0] * vec_store.EMBEDDING_DIM for _ in texts])

    from library import process_content_feedback

    out = process_content_feedback(
        "cf",
        content_title="Article on burnout",
        content_type="article",
        feedback="liked",
        user_notes="Helpful framing",
    )
    assert out.get("ok") is True
    assert "stress" in (out.get("tags") or [])
    recent = vec_store.content_feedback_list_recent("cf", 5)
    assert len(recent) >= 1
    assert recent[0]["content_title"] == "Article on burnout"
