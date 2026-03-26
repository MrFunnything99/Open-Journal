from __future__ import annotations

import importlib
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


def test_extraction_schema_coerces_mixed_facts():
    from extraction.schema import ExtractionV1

    m = ExtractionV1.model_validate(
        {
            "facts": ["hello", {"text": "stable", "scope": "global", "confidence": 0.92}],
        }
    )
    assert len(m.facts) == 2
    assert m.facts[0].text == "hello"
    assert m.facts[0].scope == "entry"
    assert m.facts[1].scope == "global"


def test_content_hash_stable_trailing_whitespace():
    from library import _content_hash_normalized, _normalize_transcript_text

    assert _normalize_transcript_text(" a \n\n b ") == "a\n\nb"
    assert _content_hash_normalized("hello") == _content_hash_normalized("hello\n")


def test_journal_delete_by_session_clears_entry(isolated_vec_db):
    vec_store = isolated_vec_db
    inst = "inst-a"
    sid = "journal-entry-1"
    emb = [0.001 * (i % 17) for i in range(vec_store.EMBEDDING_DIM)]
    eid = vec_store.journal_entry_insert(
        instance_id=inst,
        session_id=sid,
        entry_date="2025-03-01",
        raw_text="entry-only detail about today",
    )
    vec_store.journal_chunk_insert(
        eid,
        instance_id=inst,
        chunk_index=0,
        chunk_text="entry-only detail about today",
        entry_date="2025-03-01",
        embedding=emb,
    )
    assert vec_store.journal_entry_count(inst) == 1
    vec_store.journal_delete_by_session(inst, sid)
    assert vec_store.journal_entry_count(inst) == 0
    assert vec_store.journal_chunk_count(inst) == 0


def test_ingest_journal_replaces_same_session(isolated_vec_db, monkeypatch):
    vec_store = isolated_vec_db
    from library import ingest_journal_entry

    monkeypatch.setattr(
        "library._embed_texts",
        lambda texts: [[0.0] * vec_store.EMBEDDING_DIM for _ in texts],
    )

    sid = "je-1"
    inst = "inst-replace"
    ingest_journal_entry(sid, "Day one: walked the dog.", None, inst)
    ingest_journal_entry(sid, "Day one revised: ran instead.", None, inst)
    rows = vec_store.list_journal_entries_with_ids(inst)
    assert len(rows) == 1
    assert "revised" in (rows[0].get("document") or "")


def test_writing_hints_empty_draft_returns_structure(isolated_vec_db):
    _vs = isolated_vec_db
    from library import get_writing_loop_hints

    out = get_writing_loop_hints("", "x-instance")
    assert isinstance(out.get("similar_past_entries"), list)
    assert isinstance(out.get("insights"), list)
    assert isinstance(out.get("patterns"), list)


def test_ingest_writes_decision_log_ingest(isolated_vec_db, monkeypatch):
    vec_store = isolated_vec_db
    monkeypatch.setattr(
        "library._embed_texts",
        lambda texts: [[0.0] * vec_store.EMBEDDING_DIM for _ in texts],
    )

    from library import ingest_journal_entry

    inst = "inst-e2e-log"
    sid = "session-e2e-log"
    ingest_journal_entry(sid, "Day one: walked the dog and reflected.", None, inst)
    rows = vec_store.decision_log_list(inst, action_type="ingest", limit=10)
    assert rows, "expected ingest decision_log row"
    assert any(sid in (r.get("input_summary") or "") for r in rows)
    assert rows[0].get("action_type") == "ingest"
