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


def test_profile_merge_and_get_meta(isolated_vec_db):
    vs = isolated_vec_db
    vs.user_media_profile_merge_json(
        "p1",
        {
            "content_preferences": {"subscriptions": ["nytimes.com"], "paywall_policy": "only_subscribed"},
            "high_confidence_globals": ["reads widely"],
        },
    )
    d, upd = vs.user_media_profile_get_with_meta("p1")
    assert d.get("high_confidence_globals") == ["reads widely"]
    assert "nytimes.com" in (d.get("content_preferences") or {}).get("subscriptions", [])
    assert upd is not None
