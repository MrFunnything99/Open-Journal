"""
SQLite + sqlite-vec storage for Open-Journal: gist_facts, episodic_log, consumed_content.
Replaces ChromaDB with a single SQLite DB and vec0 virtual tables for vector search.
"""
from __future__ import annotations

import os
import sys
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
