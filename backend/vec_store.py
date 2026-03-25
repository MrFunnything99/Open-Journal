"""
SQLite + sqlite-vec storage for Selfmeridian: gist_facts, episodic_log, consumed_content.
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

# Embedding dimension must match the active embedding model (Perplexity pplx-embed-context-v1-4b full = 2560)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2560"))
# Use VECTOR_DB_PATH in production (e.g. Fly volume) so data persists across deploys
_default_db = Path(__file__).resolve().parent.parent / "data" / "open_journal.db"
DB_PATH = Path(os.getenv("VECTOR_DB_PATH", str(_default_db))).resolve()

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
    # Persist embedding dim so we can detect dimension changes (e.g. after switching embedding models)
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
    # Optional instance isolation (X-Instance-ID): each instance only sees/writes its own rows
    for tbl, col in (("memory_facts", "instance_id"), ("memory_episodic", "instance_id"), ("consumed_meta", "instance_id")):
        try:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Simple login (no email): username + password_hash for persistent data (legacy)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Two-token auth: email + passlib hashed password
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

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


def auth_user_create(username: str, password_hash: str, salt: str) -> int | None:
    """Create a user; return user id or None if username taken."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO auth_users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), password_hash, salt, datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return row[0] if row else None
    except sqlite3.IntegrityError:
        return None


def auth_user_get_by_username(username: str) -> dict | None:
    """Return {id, username, password_hash, salt} or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, username, password_hash, salt FROM auth_users WHERE username = ?",
        (username.strip().lower(),),
    ).fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "salt": row[3]}


def user_create(email: str, hashed_password: str) -> int | None:
    """Create user (email + passlib hash). Return user id or None if email taken."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, hashed_password, created_at) VALUES (?, ?, ?)",
            (email.strip().lower(), hashed_password, datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return row[0] if row else None
    except sqlite3.IntegrityError:
        return None


def user_get_by_email(email: str) -> dict | None:
    """Return {id, email, hashed_password} or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, email, hashed_password FROM users WHERE email = ?",
        (email.strip().lower(),),
    ).fetchone()
    if not row:
        return None
    return {"id": row[0], "email": row[1], "hashed_password": row[2]}


# Username-only accounts stored as email = "username@anonymous.local"
ANONYMOUS_EMAIL_SUFFIX = "@anonymous.local"


def user_get_by_email_or_username(identifier: str) -> dict | None:
    """Look up by email or by username (tries identifier then identifier@anonymous.local). Return {id, email, hashed_password} or None."""
    ident = identifier.strip().lower()
    user = user_get_by_email(ident)
    if user:
        return user
    if "@" not in ident:
        user = user_get_by_email(ident + ANONYMOUS_EMAIL_SUFFIX)
    return user


def _instance_where(instance_id: str, table: str = "memory_facts") -> tuple[str, list]:
    """Return (WHERE clause fragment, params) for instance scoping. instance_id '' = legacy shared rows."""
    if not instance_id:
        return ("(instance_id = '' OR instance_id IS NULL)", [])
    return ("instance_id = ?", [instance_id])


def add_gist(session_id: str, timestamp: str, document: str, embedding: list[float], instance_id: str = "") -> int:
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
    try:
        conn.execute(
            "INSERT INTO memory_facts (id, document, session_id, timestamp, instance_id) VALUES (?, ?, ?, ?, ?)",
            (doc_id, document, session_id, timestamp, instance_id or ""),
        )
    except sqlite3.OperationalError:
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
    instance_id: str = "",
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
    try:
        conn.execute(
            "INSERT INTO memory_episodic (id, document, session_id, timestamp, metadata_json, instance_id) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, document, session_id, timestamp, metadata_json, instance_id or ""),
        )
    except sqlite3.OperationalError:
        conn.execute(
            "INSERT INTO memory_episodic (id, document, session_id, timestamp, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (doc_id, document, session_id, timestamp, metadata_json),
        )
    conn.commit()
    return doc_id


def query_gist(embedding: list[float], k: int, instance_id: str = "") -> list[str]:
    """Return top-k gist document strings by cosine similarity. Optional instance_id scopes to that instance."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = (k * 4 + 20) if instance_id else k
    rows = conn.execute(
        """
        SELECT doc_id, document
        FROM vec_gist
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    if not instance_id:
        return [r[1] for r in rows if r[1]][:k]
    allowed = set(
        r[0] for r in conn.execute(
            "SELECT id FROM memory_facts WHERE instance_id = ?", (instance_id,)
        ).fetchall()
    )
    out = []
    for doc_id, doc in rows:
        if doc_id in allowed and doc:
            out.append(doc)
            if len(out) >= k:
                break
    return out


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


def query_episodic(embedding: list[float], k: int, instance_id: str = "") -> list[str]:
    """Return top-k episodic document strings by cosine similarity. Optional instance_id scopes to that instance."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = (k * 4 + 20) if instance_id else k
    rows = conn.execute(
        """
        SELECT doc_id, document
        FROM vec_episodic
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    if not instance_id:
        return [r[1] for r in rows if r[1]][:k]
    allowed = set(
        r[0] for r in conn.execute(
            "SELECT id FROM memory_episodic WHERE instance_id = ?", (instance_id,)
        ).fetchall()
    )
    out = []
    for doc_id, doc in rows:
        if doc_id in allowed and doc:
            out.append(doc)
            if len(out) >= k:
                break
    return out


def query_episodic_with_timestamp(embedding: list[float], k: int, instance_id: str = "") -> list[tuple[str, str]]:
    """Return top-k (document, timestamp) by cosine similarity. Optional instance_id scopes results."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = (k * 4 + 20) if instance_id else k
    rows = conn.execute(
        """
        SELECT doc_id, document, timestamp
        FROM vec_episodic
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    if not instance_id:
        return [(r[1] or "", r[2] or "") for r in rows if r[1]][:k]
    allowed = set(r[0] for r in conn.execute("SELECT id FROM memory_episodic WHERE instance_id = ?", (instance_id,)).fetchall())
    out = []
    for doc_id, doc, ts in rows:
        if doc_id in allowed and doc:
            out.append((doc, ts or ""))
            if len(out) >= k:
                break
    return out


def query_gist_with_timestamp(embedding: list[float], k: int, instance_id: str = "") -> list[tuple[str, str]]:
    """Return top-k (document, timestamp) by cosine similarity. Optional instance_id scopes results."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = (k * 4 + 20) if instance_id else k
    rows = conn.execute(
        """
        SELECT doc_id, document, timestamp
        FROM vec_gist
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    if not instance_id:
        return [(r[1] or "", r[2] or "") for r in rows if r[1]][:k]
    allowed = set(r[0] for r in conn.execute("SELECT id FROM memory_facts WHERE instance_id = ?", (instance_id,)).fetchall())
    out = []
    for doc_id, doc, ts in rows:
        if doc_id in allowed and doc:
            out.append((doc, ts or ""))
            if len(out) >= k:
                break
    return out


def get_all_gist(instance_id: str = "") -> list[str]:
    """Return all gist documents (for visualization); from canonical memory_facts."""
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    rows = conn.execute(
        f"SELECT document FROM memory_facts WHERE {where} ORDER BY id",
        params,
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def get_all_episodic(instance_id: str = "") -> list[str]:
    """Return all episodic documents (for visualization); from canonical memory_episodic."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "memory_episodic")
    rows = conn.execute(
        f"SELECT document FROM memory_episodic WHERE {where} ORDER BY id",
        params,
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def list_gist_with_ids(instance_id: str = "") -> list[dict]:
    """Return all gist facts with id, document, session_id, timestamp for Memory UI."""
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    rows = conn.execute(
        f"SELECT id, document, session_id, timestamp FROM memory_facts WHERE {where} ORDER BY id DESC",
        params,
    ).fetchall()
    return [
        {"id": r[0], "document": r[1] or "", "session_id": r[2] or "", "timestamp": r[3] or ""}
        for r in rows
    ]


def list_episodic_with_ids(instance_id: str = "") -> list[dict]:
    """Return all episodic summaries with id, document, session_id, timestamp, metadata_json for Memory UI."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "memory_episodic")
    try:
        rows = conn.execute(
            f"SELECT id, document, session_id, timestamp, metadata_json FROM memory_episodic WHERE {where} ORDER BY id DESC",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(
            f"SELECT id, document, session_id, timestamp FROM memory_episodic WHERE {where} ORDER BY id DESC",
            params,
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


def get_episodic_for_date(date_iso: str, instance_id: str = "") -> list[dict]:
    """Return episodic summaries whose timestamp falls on the given date (YYYY-MM-DD)."""
    if not date_iso or len(date_iso) < 10:
        return []
    conn = _get_conn()
    where, params = _instance_where(instance_id, "memory_episodic")
    params = [date_iso[:10]] + params
    try:
        rows = conn.execute(
            f"SELECT id, document, session_id, timestamp, metadata_json FROM memory_episodic WHERE date(timestamp) = ? AND {where} ORDER BY timestamp DESC",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        where2, params2 = _instance_where(instance_id, "memory_episodic")
        all_rows = conn.execute(
            f"SELECT id, document, session_id, timestamp FROM memory_episodic WHERE {where2} ORDER BY id DESC",
            params2,
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


def get_gist_for_date(date_iso: str, instance_id: str = "") -> list[dict]:
    """Return gist facts whose timestamp falls on the given date (YYYY-MM-DD)."""
    if not date_iso or len(date_iso) < 10:
        return []
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    params = [date_iso[:10]] + params
    try:
        rows = conn.execute(
            f"SELECT id, document, session_id, timestamp FROM memory_facts WHERE date(timestamp) = ? AND {where} ORDER BY timestamp DESC",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
        for r in conn.execute(f"SELECT id, document, session_id, timestamp FROM memory_facts WHERE {where} ORDER BY id DESC", params).fetchall():
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
        INSERT OR REPLACE INTO vec_gist (doc_id, embedding, session_id, timestamp, document)
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
        INSERT OR REPLACE INTO vec_episodic (doc_id, embedding, session_id, timestamp, document)
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


def gist_count(instance_id: str = "") -> int:
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    try:
        r = conn.execute(f"SELECT COUNT(*) FROM memory_facts WHERE {where}", params).fetchone()
    except sqlite3.OperationalError:
        r = conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()
    return r[0] if r else 0


def episodic_count(instance_id: str = "") -> int:
    conn = _get_conn()
    where, params = _instance_where(instance_id, "memory_episodic")
    try:
        r = conn.execute(f"SELECT COUNT(*) FROM memory_episodic WHERE {where}", params).fetchone()
    except sqlite3.OperationalError:
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
    instance_id: str = "",
) -> None:
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
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
    try:
        conn.execute(
            """INSERT OR REPLACE INTO consumed_meta (
                id_original, type, title, author, url, liked, timestamp, date_completed, note, instance_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item_id, type_, title[:500], author[:300], url[:500],
                1 if liked else 0, timestamp, date_completed[:50], note[:2000], instance_id or "",
            ),
        )
    except sqlite3.OperationalError:
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


def list_consumed_rows(max_items: int = 200, instance_id: str = "") -> list[dict]:
    """Return all consumed rows from consumed_meta (source of truth for listing)."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "consumed_meta")
    try:
        rows = conn.execute(
            f"""
            SELECT id_original, type, title, author, date_completed, note
            FROM consumed_meta
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params + [max_items],
        ).fetchall()
    except sqlite3.OperationalError:
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


def update_consumed(
    item_id: str,
    *,
    date_completed: str | None = None,
    note: str | None = None,
    title: str | None = None,
    author: str | None = None,
    url: str | None = None,
    instance_id: str = "",
) -> bool:
    """Update consumed metadata by id_original. When instance_id is set, only update consumed_meta rows for that instance."""
    conn = _get_conn()
    if instance_id:
        cur = conn.execute("SELECT 1 FROM consumed_meta WHERE id_original = ? AND instance_id = ?", (item_id, instance_id))
    else:
        try:
            cur = conn.execute(
                "SELECT 1 FROM consumed_meta WHERE id_original = ? AND (instance_id = '' OR instance_id IS NULL)",
                (item_id,),
            )
        except sqlite3.OperationalError:
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
    if title is not None:
        updates.append("title = ?")
        params.append((title or "")[:500])
    if author is not None:
        updates.append("author = ?")
        params.append((author or "")[:300])
    if url is not None:
        updates.append("url = ?")
        params.append((url or "")[:500])
    if not updates:
        return True
    params.append(item_id)
    if instance_id:
        conn.execute(
            f"UPDATE consumed_meta SET {', '.join(updates)} WHERE id_original = ? AND instance_id = ?",
            params + [instance_id],
        )
    else:
        try:
            conn.execute(
                f"UPDATE consumed_meta SET {', '.join(updates)} WHERE id_original = ? AND (instance_id = '' OR instance_id IS NULL)",
                params,
            )
        except sqlite3.OperationalError:
            conn.execute(
                f"UPDATE consumed_meta SET {', '.join(updates)} WHERE id_original = ?",
                params,
            )
    # Keep vec_consumed metadata in sync (vec table has no instance_id column).
    # This does not re-embed; it only updates displayed metadata/document fields for consistency.
    vec_updates = []
    vec_params: list[Any] = []
    if title is not None:
        vec_updates.append("title = ?")
        vec_params.append((title or "")[:500])
    if author is not None:
        vec_updates.append("author = ?")
        vec_params.append((author or "")[:300])
    if url is not None:
        vec_updates.append("url = ?")
        vec_params.append((url or "")[:500])
    if note is not None:
        vec_updates.append("note = ?")
        vec_params.append((note or "")[:2000])
    if date_completed is not None:
        vec_updates.append("date_completed = ?")
        vec_params.append((date_completed or "")[:50])
    if vec_updates:
        conn.execute(
            f"UPDATE vec_consumed SET {', '.join(vec_updates)} WHERE id_original = ?",
            vec_params + [item_id],
        )
    conn.commit()
    return True


def delete_consumed(item_id: str, instance_id: str = "") -> bool:
    """Remove consumed item by id_original from both vec_consumed and consumed_meta. When instance_id is set, only delete rows for that instance."""
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT row_id FROM vec_consumed WHERE id_original = ?", (item_id,))
        row = cur.fetchone()
        if row:
            conn.execute("DELETE FROM vec_consumed WHERE row_id = ?", (row[0],))
        if instance_id:
            cur = conn.execute("DELETE FROM consumed_meta WHERE id_original = ? AND instance_id = ?", (item_id, instance_id))
        else:
            try:
                conn.execute("DELETE FROM consumed_meta WHERE id_original = ? AND (instance_id = '' OR instance_id IS NULL)", (item_id,))
            except sqlite3.OperationalError:
                conn.execute("DELETE FROM consumed_meta WHERE id_original = ?", (item_id,))
        conn.commit()
        return True
    except Exception:
        return False


def get_consumed_context_rows(max_items: int = 80, instance_id: str = "") -> list[dict]:
    """Return consumed rows with type, title, author, liked, note from consumed_meta."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "consumed_meta")
    try:
        rows = conn.execute(
            f"""
            SELECT type, title, author, liked, note
            FROM consumed_meta
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params + [max_items],
        ).fetchall()
    except sqlite3.OperationalError:
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


def wipe_memory_for_instance(instance_id: str) -> None:
    """Clear gist and episodic for one instance only; leave vec tables in sync by deleting by doc_id."""
    if not instance_id:
        return
    conn = _get_conn()
    for doc_id, in conn.execute("SELECT id FROM memory_facts WHERE instance_id = ?", (instance_id,)).fetchall():
        conn.execute("DELETE FROM vec_gist WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM memory_facts WHERE instance_id = ?", (instance_id,))
    for doc_id, in conn.execute("SELECT id FROM memory_episodic WHERE instance_id = ?", (instance_id,)).fetchall():
        conn.execute("DELETE FROM vec_episodic WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM memory_episodic WHERE instance_id = ?", (instance_id,))
    conn.commit()


def wipe_consumed_for_instance(instance_id: str) -> None:
    """Remove consumed library rows and embeddings for this instance (or legacy unscoped rows if instance_id is '')."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "consumed_meta")
    try:
        rows = conn.execute(
            f"SELECT id_original FROM consumed_meta WHERE {where}",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return
    for (oid,) in rows:
        if not oid:
            continue
        conn.execute("DELETE FROM vec_consumed WHERE id_original = ?", (oid,))
    try:
        conn.execute(f"DELETE FROM consumed_meta WHERE {where}", params)
    except sqlite3.OperationalError:
        pass
    conn.commit()


def wipe_all_vector_memory_for_instance(instance_id: str) -> None:
    """
    Clear gist, episodic, and consumed vectors for one instance (full knowledge-base reset before re-import).
    When instance_id is empty, clears global gist/episodic (legacy) and legacy consumed rows.
    """
    if instance_id:
        wipe_memory_for_instance(instance_id)
        wipe_consumed_for_instance(instance_id)
    else:
        wipe_memory()
        wipe_consumed_for_instance("")


def memory_count_for_instance(instance_id: str) -> tuple[int, int]:
    """Return (gist_count, episodic_count) for the given instance_id. Used to ask anonymous user if they want to sync."""
    if not instance_id or not instance_id.strip():
        return (0, 0)
    return (gist_count(instance_id), episodic_count(instance_id))


def merge_instance_memory(from_instance_id: str, to_instance_id: str) -> None:
    """Copy all memory (gist, episodic, consumed) from from_instance_id to to_instance_id. Used when anonymous user logs in and opts to sync."""
    if not from_instance_id or not to_instance_id or from_instance_id == to_instance_id:
        return
    conn = _get_conn()
    # Copy gist: memory_facts + vec_gist
    try:
        mrows = conn.execute(
            "SELECT id, document, session_id, timestamp FROM memory_facts WHERE instance_id = ?",
            (from_instance_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        mrows = []
    for (doc_id, document, session_id, timestamp) in mrows:
        row = conn.execute(
            "SELECT embedding FROM vec_gist WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            continue
        new_id = _next_id(conn, "gist")
        conn.execute(
            "INSERT INTO vec_gist (doc_id, embedding, session_id, timestamp, document) VALUES (?, ?, ?, ?, ?)",
            (new_id, row[0], session_id or "", timestamp or "", document or ""),
        )
        conn.execute(
            "INSERT INTO memory_facts (id, document, session_id, timestamp, instance_id) VALUES (?, ?, ?, ?, ?)",
            (new_id, document, session_id, timestamp, to_instance_id),
        )
    # Copy episodic: memory_episodic + vec_episodic
    try:
        erows = conn.execute(
            "SELECT id, document, session_id, timestamp, metadata_json FROM memory_episodic WHERE instance_id = ?",
            (from_instance_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        erows = []
    for row in erows:
        doc_id, document, session_id, timestamp, metadata_json = row[0], row[1], row[2], row[3], row[4] if len(row) > 4 else None
        vrow = conn.execute(
            "SELECT embedding FROM vec_episodic WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not vrow:
            continue
        new_id = _next_id(conn, "episodic")
        conn.execute(
            "INSERT INTO vec_episodic (doc_id, embedding, session_id, timestamp, document) VALUES (?, ?, ?, ?, ?)",
            (new_id, vrow[0], session_id or "", timestamp or "", document or ""),
        )
        try:
            conn.execute(
                "INSERT INTO memory_episodic (id, document, session_id, timestamp, metadata_json, instance_id) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, document, session_id, timestamp, metadata_json, to_instance_id),
            )
        except sqlite3.OperationalError:
            conn.execute(
                "INSERT INTO memory_episodic (id, document, session_id, timestamp, instance_id) VALUES (?, ?, ?, ?, ?)",
                (new_id, document, session_id, timestamp, to_instance_id),
            )
    # Copy consumed_meta + vec_consumed (by id_original from consumed_meta for from_id)
    try:
        crows = conn.execute(
            """SELECT id_original, type, title, author, url, liked, timestamp, date_completed, note
               FROM consumed_meta WHERE instance_id = ?""",
            (from_instance_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        crows = []
    for c in crows:
        id_orig, type_, title, author, url, liked, ts, date_completed, note = c
        vrow = conn.execute(
            "SELECT embedding, document FROM vec_consumed WHERE id_original = ? LIMIT 1",
            (id_orig,),
        ).fetchone()
        if not vrow:
            continue
        r = conn.execute("SELECT COALESCE(MAX(row_id), 0) + 1 FROM vec_consumed").fetchone()
        new_row_id = r[0] if r else 1
        conn.execute(
            """INSERT INTO vec_consumed (row_id, id_original, embedding, type, title, author, url, liked, timestamp, note, date_completed, document)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (new_row_id, id_orig, vrow[0], type_ or "article", (title or "?")[:500], (author or "")[:300], (url or "")[:500], liked or 1, ts or "", (note or "")[:2000], (date_completed or "")[:50], vrow[1] or ""),
        )
        try:
            conn.execute(
                """INSERT OR REPLACE INTO consumed_meta (id_original, type, title, author, url, liked, timestamp, date_completed, note, instance_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (id_orig, type_, title, author, url, liked, ts, date_completed, note, to_instance_id),
            )
        except sqlite3.OperationalError:
            conn.execute(
                """INSERT OR REPLACE INTO consumed_meta (id_original, type, title, author, url, liked, timestamp, date_completed, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (id_orig, type_, title, author, url, liked, ts, date_completed, note),
            )
    conn.commit()
