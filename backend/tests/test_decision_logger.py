from __future__ import annotations

import json
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


def test_decision_log_insert_list_get(isolated_vec_db):
    vs = isolated_vec_db
    rid = vs.decision_log_insert(
        instance_id="i1",
        session_id="s1",
        action_type="context_retrieval",
        input_summary="hello",
        retrieved_items=json.dumps([{"content": "a", "score": 0.9}]),
        duration_ms=12,
        model_used="test-model",
    )
    assert rid is not None
    rows = vs.decision_log_list("i1", limit=10)
    assert len(rows) == 1
    assert rows[0]["action_type"] == "context_retrieval"
    one = vs.decision_log_get(int(rows[0]["id"]), "i1")
    assert one is not None
    assert one["input_summary"] == "hello"


def test_decision_logger_context_retrieval(isolated_vec_db):
    import vec_store
    from decision_logger import DecisionLogger

    DecisionLogger.log_context_retrieval(
        instance_id="z",
        query="q",
        retrieved_items=[{"content": "x", "score": 1.0, "source": "gist"}],
        final_output="out",
        duration_ms=5,
    )
    rows = vec_store.decision_log_list("z", action_type="context_retrieval", limit=5)
    assert len(rows) >= 1
    assert rows[0]["final_output"] == "out"
