"""
SQLite + sqlite-vec storage for Open-Journal: gist_facts, episodic_log, consumed_content.
Replaces ChromaDB with a single SQLite DB and vec0 virtual tables for vector search.
"""
from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path

from dotenv import load_dotenv

# Prefer pysqlite3 so enable_load_extension works (required on macOS with system Python)
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass
import sqlite3

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None  # type: ignore

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Embedding dimension must match Gemini embedding model output (gemini-embedding-2* often returns 3072)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "open_journal.db"

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    if sqlite_vec is None:
        raise RuntimeError(
            "sqlite-vec is required. Install with: pip install sqlite-vec"
        )
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    _conn = conn
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create vec0 tables and sequence table if they don't exist. Recreates vec tables if embedding_dim changed."""
    # Persist embedding dim so we can detect dimension changes (e.g. 768 -> 3072)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _vec_config (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    cur = conn.execute("SELECT value FROM _vec_config WHERE key = 'embedding_dim'")
    row = cur.fetchone()
    vec_gist_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_gist'"
    ).fetchone() is not None
    need_recreate = (row is not None and row[0] != EMBEDDING_DIM) or (
        row is None and vec_gist_exists
    )
    if need_recreate:
        # Dimension changed or old DB without _vec_config: drop vec and memory tables
        for tbl in ("vec_gist", "vec_episodic", "vec_consumed", "memory_facts", "memory_episodic"):
            try:
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            except Exception:
                pass
        conn.execute("DELETE FROM _vec_seq WHERE name IN ('gist', 'episodic')")
    conn.execute(
        "INSERT OR REPLACE INTO _vec_config (key, value) VALUES ('embedding_dim', ?)",
        (EMBEDDING_DIM,),
    )

    # Sequence for gist/episodic integer ids
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _vec_seq (
            name TEXT PRIMARY KEY,
            val INTEGER NOT NULL DEFAULT 0
        )
    """)
    for name in ("gist", "episodic"):
        conn.execute(
            "INSERT OR IGNORE INTO _vec_seq (name, val) VALUES (?, 0)",
            (name,),
        )

    # Gist facts: id, embedding, session_id, timestamp, document
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_gist USING vec0(
            doc_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine,
            session_id TEXT,
            timestamp TEXT,
            +document TEXT
        )
    """)

    # Episodic log: same shape
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodic USING vec0(
            doc_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine,
            session_id TEXT,
            timestamp TEXT,
            +document TEXT
        )
    """)

    # Consumed content: id_original for lookup, embedding, metadata, document
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_consumed USING vec0(
            row_id INTEGER PRIMARY KEY,
            id_original TEXT,
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine,
            type TEXT,
            title TEXT,
            author TEXT,
            url TEXT,
            liked BOOLEAN,
            timestamp TEXT,
            note TEXT,
            date_completed TEXT,
            +document TEXT
        )
    """)

    # Source of truth for consumed listing/context (avoids full-scan of vec0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consumed_meta (
            id_original TEXT PRIMARY KEY,
            type TEXT,
            title TEXT,
            author TEXT,
            url TEXT,
            liked INTEGER,
            timestamp TEXT,
            date_completed TEXT,
            note TEXT
        )
    """)

    # Canonical list of gist/episodic for UI and edits (id = doc_id used in vec tables)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY,
            document TEXT NOT NULL,
            session_id TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_episodic (
            id INTEGER PRIMARY KEY,
            document TEXT NOT NULL,
            session_id TEXT,
            timestamp TEXT
        )
    """)
    # Optional structured metadata for time-series / pattern analysis (backwards compatible)
    try:
        conn.execute("ALTER TABLE memory_episodic ADD COLUMN metadata_json TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    # People / social graph tables for Brain -> People view
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_profiles (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            relationship_summary TEXT,
            relationship_type TEXT,
            closeness_label TEXT,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_groups (
            person_id INTEGER NOT NULL,
            group_name TEXT NOT NULL,
            PRIMARY KEY (person_id, group_name),
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_thoughts (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            date TEXT,
            thought_text TEXT,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_facts (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            fact_text TEXT NOT NULL,
            confidence REAL,
            source_journal_id TEXT,
            created_at TEXT,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_ai_summaries (
            person_id INTEGER PRIMARY KEY,
            summary TEXT,
            updated_at TEXT,
            FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
        )
    """)

    conn.commit()


def _next_id(conn: sqlite3.Connection, name: str) -> int:
    conn.execute(
        "UPDATE _vec_seq SET val = val + 1 WHERE name = ?",
        (name,),
    )
    row = conn.execute(
        "SELECT val FROM _vec_seq WHERE name = ?",
        (name,),
    ).fetchone()
    return row[0] if row else 0


def _blob_from_floats(emb: list[float]) -> bytes:
    """Serialize float list to vec0 BLOB (float32)."""
    import struct
    return struct.pack(f"{len(emb)}f", *emb)


def ensure_db() -> None:
    """Ensure DB and tables exist (idempotent)."""
    _get_conn()


def add_gist(session_id: str, timestamp: str, document: str, embedding: list[float]) -> int:
    """Insert gist fact; return stable id (doc_id) for later update/delete."""
    conn = _get_conn()
    doc_id = _next_id(conn, "gist")
    blob = _blob_from_floats(embedding)
    conn.execute(
        """
        INSERT INTO vec_gist (doc_id, embedding, session_id, timestamp, document)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, blob, session_id, timestamp, document),
    )
    conn.execute(
        "INSERT INTO memory_facts (id, document, session_id, timestamp) VALUES (?, ?, ?, ?)",
        (doc_id, document, session_id, timestamp),
    )
    conn.commit()
    return doc_id


def add_episodic(
    session_id: str,
    timestamp: str,
    document: str,
    embedding: list[float],
    metadata_json: str | None = None,
) -> int:
    """Insert episodic summary; return stable id (doc_id). metadata_json is optional structured JSON."""
    conn = _get_conn()
    doc_id = _next_id(conn, "episodic")
    blob = _blob_from_floats(embedding)
    conn.execute(
        """
        INSERT INTO vec_episodic (doc_id, embedding, session_id, timestamp, document)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_id, blob, session_id, timestamp, document),
    )
    conn.execute(
        "INSERT INTO memory_episodic (id, document, session_id, timestamp, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (doc_id, document, session_id, timestamp, metadata_json),
    )
    conn.commit()
    return doc_id


def query_gist(embedding: list[float], k: int) -> list[str]:
    """Return top-k gist document strings by cosine similarity."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    rows = conn.execute(
        """
        SELECT document
        FROM vec_gist
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, k),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def query_gist_nearest(
    embedding: list[float], k: int = 1
) -> list[tuple[int, str, float]]:
    """Return top-k gist rows as (doc_id, document, distance). Uses cosine distance (lower = more similar)."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    rows = conn.execute(
        """
        SELECT doc_id, document, distance
        FROM vec_gist
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, k),
    ).fetchall()
    return [(r[0], r[1] or "", r[2]) for r in rows]


def query_episodic(embedding: list[float], k: int) -> list[str]:
    """Return top-k episodic document strings by cosine similarity."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    rows = conn.execute(
        """
        SELECT document
        FROM vec_episodic
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, k),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def query_episodic_with_timestamp(embedding: list[float], k: int) -> list[tuple[str, str]]:
    """Return top-k (document, timestamp) by cosine similarity. Timestamp may be ''."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    rows = conn.execute(
        """
        SELECT document, timestamp
        FROM vec_episodic
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, k),
    ).fetchall()
    return [(r[0] or "", r[1] or "") for r in rows if r[0]]


def query_gist_with_timestamp(embedding: list[float], k: int) -> list[tuple[str, str]]:
    """Return top-k (document, timestamp) by cosine similarity. Timestamp may be ''."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    rows = conn.execute(
        """
        SELECT document, timestamp
        FROM vec_gist
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, k),
    ).fetchall()
    return [(r[0] or "", r[1] or "") for r in rows if r[0]]


def get_all_gist() -> list[str]:
    """Return all gist documents (for visualization); from canonical memory_facts."""
    conn = _get_conn()
    rows = conn.execute("SELECT document FROM memory_facts ORDER BY id").fetchall()
    return [r[0] for r in rows if r[0]]


def get_all_episodic() -> list[str]:
    """Return all episodic documents (for visualization); from canonical memory_episodic."""
    conn = _get_conn()
    rows = conn.execute("SELECT document FROM memory_episodic ORDER BY id").fetchall()
    return [r[0] for r in rows if r[0]]


def list_gist_with_ids() -> list[dict]:
    """Return all gist facts with id, document, session_id, timestamp for Memory UI."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, document, session_id, timestamp FROM memory_facts ORDER BY id DESC"
    ).fetchall()
    return [
        {"id": r[0], "document": r[1] or "", "session_id": r[2] or "", "timestamp": r[3] or ""}
        for r in rows
    ]


def list_episodic_with_ids() -> list[dict]:
    """Return all episodic summaries with id, document, session_id, timestamp, metadata_json for Memory UI."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, document, session_id, timestamp, metadata_json FROM memory_episodic ORDER BY id DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback if metadata_json column missing (very old DB)
        rows = conn.execute(
            "SELECT id, document, session_id, timestamp FROM memory_episodic ORDER BY id DESC"
        ).fetchall()
        rows = [(*r, None) for r in rows]
    return [
        {
            "id": r[0],
            "document": r[1] or "",
            "session_id": r[2] or "",
            "timestamp": r[3] or "",
            "metadata_json": r[4] if len(r) > 4 else None,
        }
        for r in rows
    ]


def get_episodic_for_date(date_iso: str) -> list[dict]:
    """Return episodic summaries whose timestamp falls on the given date (YYYY-MM-DD)."""
    if not date_iso or len(date_iso) < 10:
        return []
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, document, session_id, timestamp, metadata_json FROM memory_episodic WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (date_iso[:10],),
        ).fetchall()
    except sqlite3.OperationalError:
        all_rows = conn.execute(
            "SELECT id, document, session_id, timestamp FROM memory_episodic ORDER BY id DESC"
        ).fetchall()
        out = []
        for r in all_rows:
            ts = (r[3] or "")[:10]
            if ts == date_iso[:10]:
                out.append({"id": r[0], "document": r[1] or "", "session_id": r[2] or "", "timestamp": r[3] or "", "metadata_json": None})
        return out
    return [
        {"id": r[0], "document": r[1] or "", "session_id": r[2] or "", "timestamp": r[3] or "", "metadata_json": r[4] if len(r) > 4 else None}
        for r in rows
    ]


def get_gist_for_date(date_iso: str) -> list[dict]:
    """Return gist facts whose timestamp falls on the given date (YYYY-MM-DD)."""
    if not date_iso or len(date_iso) < 10:
        return []
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, document, session_id, timestamp FROM memory_facts WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (date_iso[:10],),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
        for r in conn.execute("SELECT id, document, session_id, timestamp FROM memory_facts ORDER BY id DESC").fetchall():
            ts = r[3] or ""
            if ts.startswith(date_iso[:10]):
                rows.append(r)
    return [{"id": r[0], "document": r[1] or "", "session_id": r[2] or "", "timestamp": r[3] or ""} for r in rows]


def update_gist(fact_id: int, document: str, embedding: list[float]) -> bool:
    """Update gist fact by id; rewrites vec_gist row with new embedding. Returns True if found."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM memory_facts WHERE id = ?", (fact_id,))
    if not cur.fetchone():
        return False
    blob = _blob_from_floats(embedding)
    conn.execute("UPDATE memory_facts SET document = ? WHERE id = ?", (document, fact_id))
    conn.execute("DELETE FROM vec_gist WHERE doc_id = ?", (fact_id,))
    conn.execute(
        """
        INSERT INTO vec_gist (doc_id, embedding, session_id, timestamp, document)
        SELECT ?, ?, session_id, timestamp, ? FROM memory_facts WHERE id = ?
        """,
        (fact_id, blob, document, fact_id),
    )
    conn.commit()
    return True


def update_episodic(
    summary_id: int,
    document: str,
    embedding: list[float],
    metadata_json: str | None = None,
) -> bool:
    """Update episodic summary by id; rewrites vec_episodic row. Optionally update metadata_json."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM memory_episodic WHERE id = ?", (summary_id,))
    if not cur.fetchone():
        return False
    blob = _blob_from_floats(embedding)
    conn.execute("UPDATE memory_episodic SET document = ? WHERE id = ?", (document, summary_id))
    if metadata_json is not None:
        conn.execute(
            "UPDATE memory_episodic SET metadata_json = ? WHERE id = ?",
            (metadata_json, summary_id),
        )
    conn.execute("DELETE FROM vec_episodic WHERE doc_id = ?", (summary_id,))
    conn.execute(
        """
        INSERT INTO vec_episodic (doc_id, embedding, session_id, timestamp, document)
        SELECT ?, ?, session_id, timestamp, ? FROM memory_episodic WHERE id = ?
        """,
        (summary_id, blob, document, summary_id),
    )
    conn.commit()
    return True


def delete_gist(fact_id: int) -> bool:
    """Remove gist fact by id. Returns True if found and deleted."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM memory_facts WHERE id = ?", (fact_id,))
    if not cur.fetchone():
        return False
    conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
    conn.execute("DELETE FROM vec_gist WHERE doc_id = ?", (fact_id,))
    conn.commit()
    return True


def delete_episodic(summary_id: int) -> bool:
    """Remove episodic summary by id. Returns True if found and deleted."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM memory_episodic WHERE id = ?", (summary_id,))
    if not cur.fetchone():
        return False
    conn.execute("DELETE FROM memory_episodic WHERE id = ?", (summary_id,))
    conn.execute("DELETE FROM vec_episodic WHERE doc_id = ?", (summary_id,))
    conn.commit()
    return True


def gist_count() -> int:
    conn = _get_conn()
    r = conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()
    return r[0] if r else 0


def episodic_count() -> int:
    conn = _get_conn()
    r = conn.execute("SELECT COUNT(*) FROM memory_episodic").fetchone()
    return r[0] if r else 0


def episodic_metadata_count() -> int:
    """Count episodic rows that have non-null, non-empty metadata_json."""
    conn = _get_conn()
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM memory_episodic WHERE metadata_json IS NOT NULL AND trim(metadata_json) != ''"
        ).fetchone()
        return r[0] if r else 0
    except sqlite3.OperationalError:
        return 0


def create_person(name: str) -> int:
    """Create a person row (if not exists) and return id."""
    conn = _get_conn()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        "INSERT OR IGNORE INTO people (name, created_at) VALUES (?, ?)",
        (name.strip(), now),
    )
    row = conn.execute("SELECT id FROM people WHERE name = ?", (name.strip(),)).fetchone()
    return row[0] if row else 0


def update_person(person_id: int, name: str) -> bool:
    """Rename a person."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM people WHERE id = ?", (person_id,))
    if not cur.fetchone():
        return False
    conn.execute("UPDATE people SET name = ? WHERE id = ?", (name.strip(), person_id))
    conn.commit()
    return True


def list_people_with_groups() -> list[dict]:
    """Return all people with their groups."""
    conn = _get_conn()
    rows = conn.execute("SELECT id, name, created_at FROM people ORDER BY name COLLATE NOCASE").fetchall()
    groups_map: dict[int, list[str]] = {}
    g_rows = conn.execute("SELECT person_id, group_name FROM person_groups").fetchall()
    for r in g_rows:
        groups_map.setdefault(r[0], []).append(r[1])
    result: list[dict] = []
    for r in rows:
        pid = r[0]
        result.append(
            {
                "id": pid,
                "name": r[1] or "",
                "created_at": r[2] or "",
                "groups": groups_map.get(pid, []),
            }
        )
    return result


def get_person_profile(person_id: int) -> dict | None:
    """Return profile for a person, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT relationship_summary, relationship_type, closeness_label FROM person_profiles WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "relationship_summary": row[0] or "",
        "relationship_type": row[1] or "",
        "closeness_label": row[2] or "",
    }


def upsert_person_profile(
    person_id: int,
    relationship_summary: str,
    relationship_type: str | None,
    closeness_label: str | None,
) -> None:
    """Insert or update a person's profile."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM person_profiles WHERE person_id = ?", (person_id,))
    if cur.fetchone():
        conn.execute(
            """
            UPDATE person_profiles
            SET relationship_summary = ?, relationship_type = ?, closeness_label = ?
            WHERE person_id = ?
            """,
            (relationship_summary, relationship_type, closeness_label, person_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO person_profiles (person_id, relationship_summary, relationship_type, closeness_label)
            VALUES (?, ?, ?, ?)
            """,
            (person_id, relationship_summary, relationship_type, closeness_label),
        )
    conn.commit()


def get_person_ai_summary(person_id: int) -> dict | None:
    """Return cached AI relationship summary for a person, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT summary, updated_at FROM person_ai_summaries WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if not row:
        return None
    return {"summary": row[0] or "", "updated_at": row[1] or ""}


def set_person_ai_summary(person_id: int, summary: str) -> None:
    """Insert or update cached AI relationship summary for a person."""
    conn = _get_conn()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO person_ai_summaries (person_id, summary, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(person_id) DO UPDATE SET
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (person_id, summary, now),
    )
    conn.commit()


def get_person_groups(person_id: int) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT group_name FROM person_groups WHERE person_id = ? ORDER BY group_name COLLATE NOCASE",
        (person_id,),
    ).fetchall()
    return [r[0] for r in rows]


def set_person_groups(person_id: int, groups: list[str]) -> None:
    """Replace a person's groups with the given list."""
    conn = _get_conn()
    conn.execute("DELETE FROM person_groups WHERE person_id = ?", (person_id,))
    cleaned = {g.strip() for g in groups if g and g.strip()}
    for g in cleaned:
        conn.execute(
            "INSERT INTO person_groups (person_id, group_name) VALUES (?, ?)",
            (person_id, g),
        )
    conn.commit()


def list_person_thoughts(person_id: int) -> list[dict]:
    """Return all thoughts for a person ordered by date descending."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, date, thought_text FROM person_thoughts WHERE person_id = ? ORDER BY COALESCE(date, '') DESC, id DESC",
        (person_id,),
    ).fetchall()
    return [{"id": r[0], "date": r[1] or "", "thought_text": r[2] or ""} for r in rows]


def add_person_thought(person_id: int, date: str | None, text: str) -> int:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO person_thoughts (person_id, date, thought_text) VALUES (?, ?, ?)",
        (person_id, date, text.strip()),
    )
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    conn.commit()
    return int(row[0]) if row else 0


def update_person_thought(thought_id: int, date: str | None, text: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM person_thoughts WHERE id = ?", (thought_id,))
    if not cur.fetchone():
        return False
    conn.execute(
        "UPDATE person_thoughts SET date = ?, thought_text = ? WHERE id = ?",
        (date, text.strip(), thought_id),
    )
    conn.commit()
    return True


def delete_person_thought(thought_id: int) -> bool:
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM person_thoughts WHERE id = ?", (thought_id,))
    if not cur.fetchone():
        return False
    conn.execute("DELETE FROM person_thoughts WHERE id = ?", (thought_id,))
    conn.commit()
    return True


def list_person_facts(person_id: int) -> list[dict]:
    """Return stored person facts for a person, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, fact_text, confidence, source_journal_id, created_at
        FROM person_facts
        WHERE person_id = ?
        ORDER BY COALESCE(created_at, '') DESC, id DESC
        """,
        (person_id,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "fact_text": r[1] or "",
            "confidence": r[2],
            "source_journal_id": r[3] or "",
            "created_at": r[4] or "",
        }
        for r in rows
    ]


def replace_person_facts(person_id: int, facts: list[dict]) -> None:
    """
    Replace all person_facts for a person with the provided list.
    Each fact dict should have keys: fact_text, confidence?, source_id?.
    """
    conn = _get_conn()
    conn.execute("DELETE FROM person_facts WHERE person_id = ?", (person_id,))
    now = datetime.datetime.utcnow().isoformat() + "Z"
    for f in facts:
        text = (f.get("fact_text") or "").strip()
        if not text:
            continue
        conf = f.get("confidence")
        src = f.get("source_id") or ""
        conn.execute(
            """
            INSERT INTO person_facts (person_id, fact_text, confidence, source_journal_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (person_id, text, conf, src, now),
        )
    conn.commit()

def add_consumed(
    item_id: str,
    document: str,
    embedding: list[float],
    *,
    type_: str,
    title: str,
    author: str = "",
    url: str = "",
    liked: bool = True,
    timestamp: str = "",
    note: str = "",
    date_completed: str = "",
) -> None:
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    # row_id: use a new sequence or hash; vec0 needs integer PK. Use max+1 for simplicity.
    r = conn.execute("SELECT COALESCE(MAX(row_id), 0) + 1 FROM vec_consumed").fetchone()
    row_id = r[0] if r else 1
    conn.execute(
        """
        INSERT INTO vec_consumed (
            row_id, id_original, embedding, type, title, author, url,
            liked, timestamp, note, date_completed, document
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_id, item_id, blob, type_, title[:500], author[:300], url[:500],
            1 if liked else 0, timestamp, note[:2000], date_completed[:50], document,
        ),
    )
    conn.execute(
        """INSERT OR REPLACE INTO consumed_meta (
            id_original, type, title, author, url, liked, timestamp, date_completed, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id, type_, title[:500], author[:300], url[:500],
            1 if liked else 0, timestamp, date_completed[:50], note[:2000],
        ),
    )
    conn.commit()


def list_consumed_rows(max_items: int = 200) -> list[dict]:
    """Return all consumed rows from consumed_meta (source of truth for listing)."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id_original, type, title, author, date_completed, note
        FROM consumed_meta
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (max_items,),
    ).fetchall()
    return [
        {
            "id": r[0] or "",
            "type": (r[1] or "article").lower(),
            "title": (r[2] or "?").strip(),
            "author": (r[3] or "").strip(),
            "date_completed": (r[4] or "").strip(),
            "note": (r[5] or "").strip(),
        }
        for r in rows
    ]


def update_consumed(item_id: str, *, date_completed: str | None = None, note: str | None = None) -> bool:
    """Update date_completed and/or note by id_original in consumed_meta."""
    conn = _get_conn()
    cur = conn.execute("SELECT 1 FROM consumed_meta WHERE id_original = ?", (item_id,))
    if not cur.fetchone():
        return False
    updates = []
    params = []
    if date_completed is not None:
        updates.append("date_completed = ?")
        params.append((date_completed or "")[:50])
    if note is not None:
        updates.append("note = ?")
        params.append((note or "")[:2000])
    if not updates:
        return True
    params.append(item_id)
    conn.execute(
        f"UPDATE consumed_meta SET {', '.join(updates)} WHERE id_original = ?",
        params,
    )
    conn.commit()
    return True


def delete_consumed(item_id: str) -> bool:
    """Remove consumed item by id_original from both vec_consumed and consumed_meta."""
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT row_id FROM vec_consumed WHERE id_original = ?", (item_id,))
        row = cur.fetchone()
        if row:
            conn.execute("DELETE FROM vec_consumed WHERE row_id = ?", (row[0],))
        conn.execute("DELETE FROM consumed_meta WHERE id_original = ?", (item_id,))
        conn.commit()
        return True
    except Exception:
        return False


def get_consumed_context_rows(max_items: int = 80) -> list[dict]:
    """Return consumed rows with type, title, author, liked, note from consumed_meta."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT type, title, author, liked, note
        FROM consumed_meta
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (max_items,),
    ).fetchall()
    return [
        {
            "type": (r[0] or "item").lower(),
            "title": (r[1] or "?").strip(),
            "author": (r[2] or "").strip(),
            "liked": bool(r[3]) if r[3] is not None else True,
            "note": (r[4] or "").strip(),
        }
        for r in rows
    ]


def wipe_memory() -> None:
    """Clear gist and episodic (memory_facts, memory_episodic, vec_*) only; keep consumed."""
    conn = _get_conn()
    conn.execute("DELETE FROM memory_facts")
    conn.execute("DELETE FROM memory_episodic")
    conn.execute("DELETE FROM vec_gist")
    conn.execute("DELETE FROM vec_episodic")
    conn.execute("UPDATE _vec_seq SET val = 0 WHERE name IN ('gist', 'episodic')")
    conn.commit()
