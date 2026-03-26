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
        yield vec_store
        vec_store._conn = None  # type: ignore[attr-defined]


def _seed_journal(vs, session_id: str, entry_date: str, text: str, inst: str) -> None:
    eid = vs.journal_entry_insert(
        instance_id=inst, session_id=session_id, entry_date=entry_date, raw_text=text
    )
    emb = [0.001 * (i % 11) for i in range(vs.EMBEDDING_DIM)]
    vs.journal_chunk_insert(
        eid,
        instance_id=inst,
        chunk_index=0,
        chunk_text=text,
        entry_date=entry_date[:10],
        embedding=emb,
    )


def test_query_episodic_by_date_range(isolated_vec_db):
    vs = isolated_vec_db
    inst = "u1"
    _seed_journal(vs, "s1", "2024-06-10", "alpha day", inst)
    _seed_journal(vs, "s2", "2024-06-15", "beta day", inst)
    rows = vs.query_episodic_by_date_range(inst, "2024-06-01", "2024-06-20", limit=10)
    docs = [r["document"] for r in rows]
    assert "alpha day" in docs and "beta day" in docs


def test_query_this_day_in_history_year_filter(isolated_vec_db):
    vs = isolated_vec_db
    inst = "u2"
    _seed_journal(vs, "a", "2020-03-15", "old ides", inst)
    _seed_journal(vs, "b", "2026-03-15", "current year same md", inst)
    rows = vs.query_this_day_in_history(inst, month_day="03-15", years_back=10)
    assert any("old ides" in (r.get("document") or "") for r in rows)
    assert not any("current year" in (r.get("document") or "") for r in rows)
