"""
SQLite + sqlite-vec storage for Selfmeridian: journal chunks (vec_journal), consumed_content, people, auth.
Journal memory is raw text chunked + embedded; no separate gist/episodic tables.
"""
from __future__ import annotations

import functools
import os
import sys
import datetime
import threading
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
# Single shared connection + check_same_thread=False: serialize all use (ingest, chat, etc.).
_SQLITE_CONN_LOCK = threading.RLock()


def _sqlite_serialized(fn):
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _SQLITE_CONN_LOCK:
            return fn(*args, **kwargs)

    return _wrapped


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    if sqlite_vec is None:
        raise RuntimeError(
            "sqlite-vec is required. Install with: pip install sqlite-vec"
        )
    with _SQLITE_CONN_LOCK:
        if _conn is not None:
            return _conn
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        _init_db(conn)
        _conn = conn
        return _conn


def _legacy_memory_drop(conn: sqlite3.Connection) -> None:
    """One-time migration: remove gist/episodic/profile/extraction tables."""
    for tbl in (
        "vec_gist",
        "vec_episodic",
        "memory_facts",
        "memory_episodic",
        "ingest_meta",
        "extraction_artifact",
        "user_media_profile",
        "pattern_memory",
        "derived_insights",
    ):
        try:
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        except Exception:
            pass
    try:
        conn.execute("DELETE FROM _vec_seq WHERE name IN ('gist', 'episodic')")
    except sqlite3.OperationalError:
        pass


def _init_db(conn: sqlite3.Connection) -> None:
    """Create vec0 tables and journal tables. Migrates away from legacy gist/episodic once."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _vec_config (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    cur = conn.execute("SELECT value FROM _vec_config WHERE key = 'embedding_dim'")
    row = cur.fetchone()
    vec_journal_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_journal'"
    ).fetchone() is not None
    need_recreate = (row is not None and row[0] != EMBEDDING_DIM) or (
        row is None and vec_journal_exists
    )
    if need_recreate:
        for tbl in ("vec_journal", "vec_consumed"):
            try:
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            except Exception:
                pass
    conn.execute(
        "INSERT OR REPLACE INTO _vec_config (key, value) VALUES ('embedding_dim', ?)",
        (EMBEDDING_DIM,),
    )

    ms_row = conn.execute(
        "SELECT value FROM _vec_config WHERE key = 'memory_schema_v2'"
    ).fetchone()
    if ms_row is None or int(ms_row[0]) != 1:
        _legacy_memory_drop(conn)
        conn.execute(
            "INSERT OR REPLACE INTO _vec_config (key, value) VALUES ('memory_schema_v2', 1)"
        )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _vec_seq (
            name TEXT PRIMARY KEY,
            val INTEGER NOT NULL DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            instance_id TEXT NOT NULL DEFAULT '',
            entry_date TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            char_count INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            entry_source TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE journal_entries ADD COLUMN entry_source TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_entries_instance ON journal_entries(instance_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_entries_session ON journal_entries(instance_id, session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_entries_date ON journal_entries(entry_date)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            instance_id TEXT NOT NULL DEFAULT '',
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (entry_id) REFERENCES journal_entries(id) ON DELETE CASCADE
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_chunks_instance ON journal_chunks(instance_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_chunks_entry ON journal_chunks(entry_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_chunks_date ON journal_chunks(entry_date)"
    )

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_journal USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine,
            instance_id TEXT,
            entry_date TEXT,
            +chunk_text TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS instance_settings (
            instance_id TEXT PRIMARY KEY,
            preferences_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
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

    for col_def in [
        "instance_id TEXT NOT NULL DEFAULT ''",
        "isbn TEXT DEFAULT ''",
        "publish_year TEXT DEFAULT ''",
        "openlibrary_key TEXT DEFAULT ''",
        "cover_url TEXT DEFAULT ''",
        "subjects TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(f"ALTER TABLE consumed_meta ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

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

    _ensure_extraction_and_ingest_schema(conn)
    conn.commit()


def _ensure_extraction_and_ingest_schema(conn: sqlite3.Connection) -> None:
    """Migrations: rec feedback, decision log, content feedback."""
    for tbl in (
        """
        CREATE TABLE IF NOT EXISTS rec_feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL,
            content_type TEXT,
            topic_tags TEXT,
            intent_context TEXT,
            item_title TEXT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            instance_id TEXT NOT NULL DEFAULT '',
            session_id TEXT,
            action_type TEXT NOT NULL,
            input_summary TEXT,
            retrieved_items TEXT,
            llm_prompt_summary TEXT,
            llm_response TEXT,
            final_output TEXT,
            reasoning_notes TEXT,
            duration_ms INTEGER,
            model_used TEXT,
            search_api_calls TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS content_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            instance_id TEXT NOT NULL DEFAULT '',
            content_url TEXT,
            content_title TEXT,
            content_type TEXT,
            feedback TEXT NOT NULL,
            user_notes TEXT,
            extracted_tags TEXT,
            extracted_reasoning TEXT,
            connected_journal_themes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL DEFAULT '',
            entry_date TEXT NOT NULL,
            article_title TEXT,
            article_url TEXT,
            article_snippet TEXT,
            hook TEXT,
            candidates_json TEXT,
            themes_json TEXT,
            search_queries_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(instance_id, entry_date)
        )
        """,
    ):
        conn.execute(tbl)
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decision_log_instance_ts ON decision_log(instance_id, timestamp)"
        )
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decision_log_action ON decision_log(action_type)")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_feedback_instance ON content_feedback(instance_id, timestamp)"
        )
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_articles_lookup ON daily_articles(instance_id, entry_date)"
        )
    except sqlite3.OperationalError:
        pass


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


@_sqlite_serialized
def ensure_db() -> None:
    """Ensure DB and tables exist (idempotent)."""
    _get_conn()


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
def user_get_by_email_or_username(identifier: str) -> dict | None:
    """Look up by email or by username (tries identifier then identifier@anonymous.local). Return {id, email, hashed_password} or None."""
    ident = identifier.strip().lower()
    user = user_get_by_email(ident)
    if user:
        return user
    if "@" not in ident:
        user = user_get_by_email(ident + ANONYMOUS_EMAIL_SUFFIX)
    return user


def _instance_where(instance_id: str, table: str = "journal_entries") -> tuple[str, list]:
    """Return (WHERE clause fragment, params) for instance scoping. instance_id '' = legacy shared rows."""
    if not instance_id:
        return ("(instance_id = '' OR instance_id IS NULL)", [])
    return ("instance_id = ?", [instance_id])


@_sqlite_serialized
def instance_settings_get(instance_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT preferences_json FROM instance_settings WHERE instance_id = ?",
        (instance_id or "",),
    ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        import json

        return json.loads(row[0]) if isinstance(row[0], str) else {}
    except Exception:
        return {}


@_sqlite_serialized
def instance_settings_merge_json(instance_id: str, patch: dict) -> None:
    import json

    conn = _get_conn()
    cur = instance_settings_get(instance_id)
    for k, v in patch.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    now = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO instance_settings (instance_id, preferences_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(instance_id) DO UPDATE SET
            preferences_json = excluded.preferences_json,
            updated_at = excluded.updated_at
        """,
        (instance_id or "", json.dumps(cur, ensure_ascii=False), now),
    )
    conn.commit()


@_sqlite_serialized
def instance_settings_get_with_meta(instance_id: str) -> tuple[dict, str | None]:
    """Return (parsed preferences dict, updated_at or None)."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT preferences_json, updated_at FROM instance_settings WHERE instance_id = ?",
        (instance_id or "",),
    ).fetchone()
    if not row:
        return {}, None
    try:
        import json

        d = json.loads(row[0]) if row[0] and isinstance(row[0], str) else {}
    except Exception:
        d = {}
    return d, (row[1] if len(row) > 1 else None)


@_sqlite_serialized
def user_media_profile_get(instance_id: str) -> dict:
    """Legacy alias for instance_settings JSON prefs."""
    return instance_settings_get(instance_id)


@_sqlite_serialized
def user_media_profile_merge_json(instance_id: str, patch: dict) -> None:
    """Legacy alias for instance_settings."""
    instance_settings_merge_json(instance_id, patch)


@_sqlite_serialized
def pattern_memory_add(
    instance_id: str,
    window_label: str,
    summary: str,
    structured_tags: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> int:
    return 0


@_sqlite_serialized
def pattern_memory_recent(instance_id: str, limit: int = 5) -> list[dict]:
    return []


@_sqlite_serialized
def derived_insights_list_active(instance_id: str, limit: int = 20) -> list[dict]:
    return []


@_sqlite_serialized
def derived_insight_add(
    instance_id: str, text: str, kind: str | None = None, pattern_ids: str | None = None
) -> int:
    return 0


@_sqlite_serialized
def derived_insight_dismiss(insight_id: int, instance_id: str) -> bool:
    return False


@_sqlite_serialized
def journal_delete_by_session(instance_id: str, session_id: str) -> None:
    """Remove journal entry + chunks + vectors for (instance_id, session_id)."""
    if not session_id:
        return
    conn = _get_conn()
    inst = instance_id or ""
    rows = conn.execute(
        """
        SELECT id FROM journal_entries
        WHERE session_id = ? AND (instance_id = ? OR (? = '' AND (instance_id = '' OR instance_id IS NULL)))
        """,
        (session_id, inst, inst),
    ).fetchall()
    for (eid,) in rows:
        chunk_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM journal_chunks WHERE entry_id = ?", (eid,)
            ).fetchall()
        ]
        for cid in chunk_ids:
            conn.execute("DELETE FROM vec_journal WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM journal_chunks WHERE entry_id = ?", (eid,))
    conn.execute(
        """
        DELETE FROM journal_entries
        WHERE session_id = ? AND (instance_id = ? OR (? = '' AND (instance_id = '' OR instance_id IS NULL)))
        """,
        (session_id, inst, inst),
    )
    conn.commit()


@_sqlite_serialized
def journal_entry_insert(
    *,
    instance_id: str,
    session_id: str,
    entry_date: str,
    raw_text: str,
    entry_source: str | None = None,
) -> int:
    """Insert journal entry row; returns id. Caller adds chunks + vec rows."""
    conn = _get_conn()
    ed = (entry_date or "")[:10]
    es = (entry_source or "").strip() or None
    if es is not None and es not in ("manual", "assisted"):
        es = None
    conn.execute(
        """
        INSERT INTO journal_entries (session_id, instance_id, entry_date, raw_text, char_count, entry_source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id or "",
            instance_id or "",
            ed,
            raw_text or "",
            len(raw_text or ""),
            es,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    return int(row[0]) if row and row[0] else 0


@_sqlite_serialized
def journal_chunk_insert(
    entry_id: int,
    *,
    instance_id: str,
    chunk_index: int,
    chunk_text: str,
    entry_date: str,
    embedding: list[float],
) -> int:
    """Insert chunk + vec_journal row; returns chunk id."""
    conn = _get_conn()
    ed = (entry_date or "")[:10]
    conn.execute(
        """
        INSERT INTO journal_chunks (entry_id, instance_id, chunk_index, chunk_text, entry_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (entry_id, instance_id or "", chunk_index, chunk_text or "", ed),
    )
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    cid = int(row[0]) if row and row[0] else 0
    if cid and embedding:
        blob = _blob_from_floats(embedding)
        conn.execute(
            """
            INSERT INTO vec_journal (chunk_id, embedding, instance_id, entry_date, chunk_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, blob, instance_id or "", ed, chunk_text or ""),
        )
        conn.commit()
    return cid


@_sqlite_serialized
def query_journal_chunks(
    embedding: list[float],
    instance_id: str,
    k: int = 8,
    *,
    max_distance: float | None = None,
) -> list[dict]:
    """Vector nearest journal chunks for instance. Each dict: chunk_text, entry_date, distance, chunk_id, entry_id."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = max(k * 10, 32)
    rows = conn.execute(
        """
        SELECT chunk_id, chunk_text, entry_date, distance
        FROM vec_journal
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    inst = instance_id or ""
    out: list[dict] = []
    for r in rows:
        cid = int(r[0])
        text = r[1] or ""
        ed = r[2] or ""
        dist = float(r[3])
        if max_distance is not None and dist > max_distance:
            continue
        row_inst = conn.execute(
            "SELECT instance_id, entry_id FROM journal_chunks WHERE id = ?", (cid,)
        ).fetchone()
        if not row_inst:
            continue
        row_i, eid = row_inst[0] or "", int(row_inst[1])
        if inst:
            if (row_i or "") != inst:
                continue
        elif row_i not in ("", None):
            continue
        out.append(
            {
                "chunk_id": cid,
                "entry_id": eid,
                "chunk_text": text,
                "entry_date": ed,
                "distance": dist,
            }
        )
        if len(out) >= k:
            break
    return out


@_sqlite_serialized
def journal_entries_for_date_range(
    instance_id: str, start_date: str, end_date: str, limit: int = 20
) -> list[dict]:
    """Journal entries with entry_date between start and end (YYYY-MM-DD), chronological."""
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    lim = max(1, min(limit, 200))
    try:
        rows = conn.execute(
            f"""
            SELECT id, raw_text, session_id, entry_date, created_at
            FROM journal_entries
            WHERE date(entry_date) >= date(?) AND date(entry_date) <= date(?) AND {where}
            ORDER BY entry_date ASC, id ASC
            LIMIT ?
            """,
            [start_date[:10], end_date[:10], *params, lim],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": r[0],
            "document": r[1] or "",
            "session_id": r[2] or "",
            "timestamp": r[3] or "",
            "metadata_json": None,
            "created_at": r[4] if len(r) > 4 else None,
        }
        for r in rows
    ]


@_sqlite_serialized
def journal_learning_bookends(instance_id: str) -> list[dict]:
    """Three most recent journal entries for learning theme synthesis.

    Returns a list of up to 3 dicts with id, document, session_id,
    timestamp (entry_date), created_at, ordered newest-first.
    Falls back across all instances if the scoped instance has no rows.
    """
    conn = _get_conn()

    def _row_dict(r: tuple) -> dict:
        return {
            "id": r[0],
            "document": r[1] or "",
            "session_id": r[2] or "",
            "timestamp": r[3] or "",
            "created_at": r[4] if len(r) > 4 else None,
        }

    def _last_three(where_sql: str, params: list) -> list[dict]:
        rows = conn.execute(
            f"""
            SELECT id, raw_text, session_id, entry_date, created_at
            FROM journal_entries
            WHERE {where_sql}
            ORDER BY entry_date DESC, id DESC
            LIMIT 3
            """,
            params,
        ).fetchall()
        return [_row_dict(r) for r in rows]

    where, params = _instance_where(instance_id)
    recent = _last_three(where, params)
    if not recent and instance_id:
        recent = _last_three("1=1", [])
    return recent


@_sqlite_serialized
def journal_this_day_in_history(
    instance_id: str, month_day: str | None = None, years_back: int = 5
) -> list[dict]:
    """Entries where strftime('%m-%d', entry_date) matches month_day; excludes current calendar year."""
    if not month_day or len(month_day) < 5:
        now = datetime.datetime.utcnow()
        month_day = now.strftime("%m-%d")
    md = month_day.strip()[:5]
    if len(md) == 4 and md[2] != "-":
        md = md[:2] + "-" + md[2:]
    cur_year = datetime.datetime.utcnow().year
    oldest_year = cur_year - max(1, min(years_back, 50))
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    try:
        rows = conn.execute(
            f"""
            SELECT id, raw_text, session_id, entry_date, created_at
            FROM journal_entries
            WHERE strftime('%m-%d', entry_date) = ?
            AND CAST(strftime('%Y', entry_date) AS INTEGER) < ?
            AND CAST(strftime('%Y', entry_date) AS INTEGER) >= ?
            AND {where}
            ORDER BY entry_date ASC
            """,
            [md, cur_year, oldest_year, *params],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": r[0],
            "document": r[1] or "",
            "session_id": r[2] or "",
            "timestamp": r[3] or "",
            "metadata_json": None,
        }
        for r in rows
    ]


@_sqlite_serialized
def query_journal_timeline_by_topic(
    embedding: list[float], instance_id: str, limit: int = 30
) -> list[dict]:
    """Vector-nearest journal chunks, resolved to entry rows, sorted chronologically by entry_date."""
    conn = _get_conn()
    blob = _blob_from_floats(embedding)
    fetch_k = max(limit * 10, 40)
    rows = conn.execute(
        """
        SELECT chunk_id, chunk_text, entry_date, distance
        FROM vec_journal
        WHERE embedding MATCH ? AND k = ?
        """,
        (blob, fetch_k),
    ).fetchall()
    inst = instance_id or ""
    by_chunk: dict[int, float] = {}
    for r in rows:
        cid = int(r[0])
        dist = float(r[3])
        row_inst = conn.execute(
            "SELECT instance_id FROM journal_chunks WHERE id = ?", (cid,)
        ).fetchone()
        if not row_inst:
            continue
        ri = row_inst[0] or ""
        if inst:
            if ri != inst:
                continue
        elif ri not in ("", None):
            continue
        if cid not in by_chunk or dist < by_chunk[cid]:
            by_chunk[cid] = dist
    lim = max(1, min(limit, 100))
    pairs = sorted(by_chunk.items(), key=lambda x: x[1])[: max(lim * 4, lim)]
    out: list[dict] = []
    seen_entry: set[int] = set()
    for cid, dist in pairs:
        row = conn.execute(
            "SELECT entry_id FROM journal_chunks WHERE id = ?", (cid,)
        ).fetchone()
        if not row:
            continue
        eid = int(row[0])
        if eid in seen_entry:
            continue
        seen_entry.add(eid)
        er = conn.execute(
            "SELECT id, raw_text, session_id, entry_date FROM journal_entries WHERE id = ?",
            (eid,),
        ).fetchone()
        if not er:
            continue
        out.append(
            {
                "id": er[0],
                "document": er[1] or "",
                "session_id": er[2] or "",
                "timestamp": er[3] or "",
                "metadata_json": None,
                "distance": dist,
            }
        )
        if len(out) >= lim:
            break
    out.sort(key=lambda x: x.get("timestamp") or "")
    return out[:lim]


def _journal_row_dict(r: tuple) -> dict:
    return {
        "id": r[0],
        "document": r[1] or "",
        "session_id": r[2] or "",
        "timestamp": r[3] or "",
        "created_at": r[4] if len(r) > 4 else None,
        "entry_source": r[5] if len(r) > 5 else None,
    }


@_sqlite_serialized
def list_journal_entries_with_ids(instance_id: str = "") -> list[dict]:
    """Journal entries for Memory UI (id, document=raw_text, session_id, timestamp=entry_date)."""
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    rows = conn.execute(
        f"""
        SELECT id, raw_text, session_id, entry_date, created_at, entry_source
        FROM journal_entries WHERE {where} ORDER BY id DESC
        """,
        params,
    ).fetchall()
    return [_journal_row_dict(r) for r in rows]


@_sqlite_serialized
def list_journal_entries_recent(instance_id: str = "", *, limit: int = 200) -> list[dict]:
    """Newest journal rows by id, capped—avoids loading entire history for balanced picks."""
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    lim = max(1, min(int(limit), 500))
    rows = conn.execute(
        f"""
        SELECT id, raw_text, session_id, entry_date, created_at, entry_source
        FROM journal_entries WHERE {where} ORDER BY id DESC LIMIT ?
        """,
        [*params, lim],
    ).fetchall()
    return [_journal_row_dict(r) for r in rows]


@_sqlite_serialized
def journal_entry_get(entry_id: int, instance_id: str) -> dict | None:
    conn = _get_conn()
    inst = instance_id or ""
    row = conn.execute(
        """
        SELECT id, raw_text, session_id, entry_date, created_at, entry_source
        FROM journal_entries WHERE id = ?
        AND (instance_id = ? OR (? = '' AND (instance_id = '' OR instance_id IS NULL)))
        """,
        (entry_id, inst, inst),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "document": row[1] or "",
        "session_id": row[2] or "",
        "timestamp": row[3] or "",
        "created_at": row[4] if len(row) > 4 else None,
        "entry_source": row[5] if len(row) > 5 else None,
    }


@_sqlite_serialized
def journal_entry_meta_batch(instance_id: str, entry_ids: list[int]) -> dict[int, dict]:
    """Map entry_id -> {entry_source, document} for classification (instance-scoped)."""
    if not entry_ids:
        return {}
    conn = _get_conn()
    inst = instance_id or ""
    uniq = sorted({int(x) for x in entry_ids if x})
    if not uniq:
        return {}
    where, ip = _instance_where(instance_id)
    out: dict[int, dict] = {}
    chunk = 200
    for i in range(0, len(uniq), chunk):
        part = uniq[i : i + chunk]
        ph = ",".join("?" * len(part))
        rows = conn.execute(
            f"""
            SELECT id, entry_source, raw_text
            FROM journal_entries WHERE id IN ({ph}) AND {where}
            """,
            [*part, *ip],
        ).fetchall()
        for r in rows:
            eid = int(r[0])
            out[eid] = {"entry_source": r[1], "document": r[2] or ""}
    return out


@_sqlite_serialized
def journal_entry_delete_cascade(entry_id: int, instance_id: str) -> bool:
    conn = _get_conn()
    inst = instance_id or ""
    row = conn.execute(
        "SELECT id FROM journal_entries WHERE id = ? AND (instance_id = ? OR (? = '' AND (instance_id = '' OR instance_id IS NULL)))",
        (entry_id, inst, inst),
    ).fetchone()
    if not row:
        return False
    chunk_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM journal_chunks WHERE entry_id = ?", (entry_id,)
        ).fetchall()
    ]
    for cid in chunk_ids:
        conn.execute("DELETE FROM vec_journal WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM journal_chunks WHERE entry_id = ?", (entry_id,))
    conn.execute("DELETE FROM journal_entries WHERE id = ?", (entry_id,))
    conn.commit()
    return True


@_sqlite_serialized
def journal_entry_count(instance_id: str = "") -> int:
    conn = _get_conn()
    where, params = _instance_where(instance_id)
    try:
        r = conn.execute(
            f"SELECT COUNT(*) FROM journal_entries WHERE {where}", params
        ).fetchone()
        return int(r[0]) if r and r[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


@_sqlite_serialized
def journal_chunk_count(instance_id: str = "") -> int:
    conn = _get_conn()
    where, params = _instance_where(instance_id, "journal_chunks")
    try:
        r = conn.execute(
            f"SELECT COUNT(*) FROM journal_chunks WHERE {where}", params
        ).fetchone()
        return int(r[0]) if r and r[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


@_sqlite_serialized
def wipe_journal_memory() -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM vec_journal")
    conn.execute("DELETE FROM journal_chunks")
    conn.execute("DELETE FROM journal_entries")
    conn.commit()


@_sqlite_serialized
def wipe_journal_memory_for_instance(instance_id: str) -> None:
    if not instance_id:
        return
    conn = _get_conn()
    eids = [
        int(r[0])
        for r in conn.execute(
            "SELECT id FROM journal_entries WHERE instance_id = ?", (instance_id,)
        ).fetchall()
    ]
    for eid in eids:
        chunk_ids = [
            int(r[0])
            for r in conn.execute(
                "SELECT id FROM journal_chunks WHERE entry_id = ?", (eid,)
            ).fetchall()
        ]
        for cid in chunk_ids:
            conn.execute("DELETE FROM vec_journal WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM journal_chunks WHERE entry_id = ?", (eid,))
    conn.execute("DELETE FROM journal_entries WHERE instance_id = ?", (instance_id,))
    conn.commit()


@_sqlite_serialized
def rec_feedback_record(
    instance_id: str,
    action: str,
    *,
    content_type: str | None = None,
    topic_tags: str | None = None,
    intent_context: str | None = None,
    item_title: str | None = None,
) -> None:
    conn = _get_conn()
    now = datetime.datetime.utcnow().isoformat() + "Z"
    conn.execute(
        """
        INSERT INTO rec_feedback_events (instance_id, action, content_type, topic_tags, intent_context, item_title, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instance_id or "",
            action,
            content_type,
            topic_tags,
            intent_context,
            item_title[:500] if item_title else None,
            now,
        ),
    )
    conn.commit()


_DECISION_LOG_INSERTS_SINCE_ROTATE = 0


@_sqlite_serialized
def decision_log_rotate(max_keep: int = 10_000) -> None:
    """Delete oldest decision_log rows, keeping the newest max_keep."""
    conn = _get_conn()
    try:
        conn.execute(
            f"""
            DELETE FROM decision_log WHERE id NOT IN (
                SELECT id FROM decision_log ORDER BY id DESC LIMIT ?
            )
            """,
            (max_keep,),
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


@_sqlite_serialized
def decision_log_insert(
    *,
    instance_id: str = "",
    session_id: str | None = None,
    action_type: str,
    input_summary: str | None = None,
    retrieved_items: str | None = None,
    llm_prompt_summary: str | None = None,
    llm_response: str | None = None,
    final_output: str | None = None,
    reasoning_notes: str | None = None,
    duration_ms: int | None = None,
    model_used: str | None = None,
    search_api_calls: str | None = None,
) -> int | None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO decision_log (
            instance_id, session_id, action_type, input_summary, retrieved_items,
            llm_prompt_summary, llm_response, final_output, reasoning_notes,
            duration_ms, model_used, search_api_calls
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instance_id or "",
            session_id,
            action_type,
            input_summary,
            retrieved_items,
            llm_prompt_summary,
            llm_response,
            final_output,
            reasoning_notes,
            duration_ms,
            model_used,
            search_api_calls,
        ),
    )
    conn.commit()
    global _DECISION_LOG_INSERTS_SINCE_ROTATE
    _DECISION_LOG_INSERTS_SINCE_ROTATE += 1
    if _DECISION_LOG_INSERTS_SINCE_ROTATE >= 100:
        _DECISION_LOG_INSERTS_SINCE_ROTATE = 0
        decision_log_rotate()
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    return int(row[0]) if row and row[0] else None


@_sqlite_serialized
def decision_log_list(
    instance_id: str,
    *,
    action_type: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _get_conn()
    inst = instance_id or ""
    clauses = ["instance_id = ?"]
    params: list = [inst]
    if action_type:
        clauses.append("action_type = ?")
        params.append(action_type)
    if session_id is not None and session_id != "":
        clauses.append("session_id = ?")
        params.append(session_id)
    where = " AND ".join(clauses)
    params.append(max(1, min(limit, 500)))
    rows = conn.execute(
        f"""
        SELECT id, timestamp, instance_id, session_id, action_type, input_summary,
               retrieved_items, llm_prompt_summary, llm_response, final_output,
               reasoning_notes, duration_ms, model_used, search_api_calls
        FROM decision_log WHERE {where} ORDER BY id DESC LIMIT ?
        """,
        params,
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "timestamp": r[1] or "",
                "instance_id": r[2] or "",
                "session_id": r[3],
                "action_type": r[4] or "",
                "input_summary": r[5],
                "retrieved_items": r[6],
                "llm_prompt_summary": r[7],
                "llm_response": r[8],
                "final_output": r[9],
                "reasoning_notes": r[10],
                "duration_ms": r[11],
                "model_used": r[12],
                "search_api_calls": r[13],
            }
        )
    return out


@_sqlite_serialized
def decision_log_get(log_id: int, instance_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT id, timestamp, instance_id, session_id, action_type, input_summary,
               retrieved_items, llm_prompt_summary, llm_response, final_output,
               reasoning_notes, duration_ms, model_used, search_api_calls
        FROM decision_log WHERE id = ? AND instance_id = ?
        """,
        (log_id, instance_id or ""),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "timestamp": row[1] or "",
        "instance_id": row[2] or "",
        "session_id": row[3],
        "action_type": row[4] or "",
        "input_summary": row[5],
        "retrieved_items": row[6],
        "llm_prompt_summary": row[7],
        "llm_response": row[8],
        "final_output": row[9],
        "reasoning_notes": row[10],
        "duration_ms": row[11],
        "model_used": row[12],
        "search_api_calls": row[13],
    }


@_sqlite_serialized
def content_feedback_list_recent(instance_id: str, limit: int = 30) -> list[dict]:
    import json as _json

    conn = _get_conn()
    lim = max(1, min(int(limit), 80))
    rows = conn.execute(
        """
        SELECT id, timestamp, content_url, content_title, content_type, feedback, user_notes,
               extracted_tags, extracted_reasoning, connected_journal_themes
        FROM content_feedback WHERE instance_id = ? ORDER BY id DESC LIMIT ?
        """,
        (instance_id or "", lim),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        tags = None
        themes = None
        try:
            if r[7]:
                tags = _json.loads(r[7])
        except Exception:
            pass
        try:
            if r[9]:
                themes = _json.loads(r[9])
        except Exception:
            pass
        out.append(
            {
                "id": r[0],
                "timestamp": r[1] or "",
                "content_url": r[2],
                "content_title": r[3] or "",
                "content_type": r[4],
                "feedback": r[5] or "",
                "user_notes": r[6],
                "extracted_tags": tags,
                "extracted_reasoning": r[8],
                "connected_journal_themes": themes,
            }
        )
    return out


@_sqlite_serialized
def content_feedback_insert(
    instance_id: str,
    *,
    content_url: str | None = None,
    content_title: str | None = None,
    content_type: str | None = None,
    feedback: str,
    user_notes: str | None = None,
    extracted_tags: str | None = None,
    extracted_reasoning: str | None = None,
    connected_journal_themes: str | None = None,
) -> int | None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO content_feedback (
            instance_id, content_url, content_title, content_type, feedback,
            user_notes, extracted_tags, extracted_reasoning, connected_journal_themes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instance_id or "",
            content_url,
            content_title,
            content_type,
            feedback,
            user_notes,
            extracted_tags,
            extracted_reasoning,
            connected_journal_themes,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    return int(row[0]) if row and row[0] else None


@_sqlite_serialized
def query_episodic_by_date_range(
    instance_id: str, start_date: str, end_date: str, limit: int = 20
) -> list[dict]:
    """Journal entries with entry_date between start and end (YYYY-MM-DD), chronological."""
    return journal_entries_for_date_range(instance_id, start_date, end_date, limit=limit)


@_sqlite_serialized
def query_this_day_in_history(
    instance_id: str, month_day: str | None = None, years_back: int = 5
) -> list[dict]:
    return journal_this_day_in_history(instance_id, month_day, years_back)


@_sqlite_serialized
def query_episodic_timeline_by_topic(
    embedding: list[float], instance_id: str, limit: int = 30
) -> list[dict]:
    return query_journal_timeline_by_topic(embedding, instance_id, limit=limit)


@_sqlite_serialized
def user_media_profile_get_with_meta(instance_id: str) -> tuple[dict, str | None]:
    """Alias for instance_settings (legacy name)."""
    return instance_settings_get_with_meta(instance_id)


@_sqlite_serialized
def gist_count(instance_id: str = "") -> int:
    """Deprecated alias: journal entry count."""
    return journal_entry_count(instance_id)


@_sqlite_serialized
def episodic_count(instance_id: str = "") -> int:
    """Deprecated alias: journal chunk count."""
    return journal_chunk_count(instance_id)


@_sqlite_serialized
def episodic_metadata_count() -> int:
    """Legacy field; journal chunks do not use episodic metadata."""
    return 0


@_sqlite_serialized
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


@_sqlite_serialized
def update_person(person_id: int, name: str) -> bool:
    """Rename a person."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM people WHERE id = ?", (person_id,))
    if not cur.fetchone():
        return False
    conn.execute("UPDATE people SET name = ? WHERE id = ?", (name.strip(), person_id))
    conn.commit()
    return True


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
def get_person_groups(person_id: int) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT group_name FROM person_groups WHERE person_id = ? ORDER BY group_name COLLATE NOCASE",
        (person_id,),
    ).fetchall()
    return [r[0] for r in rows]


@_sqlite_serialized
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


@_sqlite_serialized
def list_person_thoughts(person_id: int) -> list[dict]:
    """Return all thoughts for a person ordered by date descending."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, date, thought_text FROM person_thoughts WHERE person_id = ? ORDER BY COALESCE(date, '') DESC, id DESC",
        (person_id,),
    ).fetchall()
    return [{"id": r[0], "date": r[1] or "", "thought_text": r[2] or ""} for r in rows]


@_sqlite_serialized
def add_person_thought(person_id: int, date: str | None, text: str) -> int:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO person_thoughts (person_id, date, thought_text) VALUES (?, ?, ?)",
        (person_id, date, text.strip()),
    )
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    conn.commit()
    return int(row[0]) if row else 0


@_sqlite_serialized
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


@_sqlite_serialized
def delete_person_thought(thought_id: int) -> bool:
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM person_thoughts WHERE id = ?", (thought_id,))
    if not cur.fetchone():
        return False
    conn.execute("DELETE FROM person_thoughts WHERE id = ?", (thought_id,))
    conn.commit()
    return True


@_sqlite_serialized
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


@_sqlite_serialized
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

@_sqlite_serialized
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
    isbn: str = "",
    publish_year: str = "",
    openlibrary_key: str = "",
    cover_url: str = "",
    subjects: str = "",
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
                id_original, type, title, author, url, liked, timestamp, date_completed, note,
                instance_id, isbn, publish_year, openlibrary_key, cover_url, subjects
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item_id, type_, title[:500], author[:300], url[:500],
                1 if liked else 0, timestamp, date_completed[:50], note[:2000],
                instance_id or "", isbn[:20], publish_year[:10],
                openlibrary_key[:100], cover_url[:300], subjects[:1000],
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


@_sqlite_serialized
def consumed_book_exists(isbn: str = "", title: str = "", author: str = "", instance_id: str = "") -> bool:
    """Check if a book already exists in consumed_meta by ISBN (preferred) or normalized title+author."""
    conn = _get_conn()
    where_base, params_base = _instance_where(instance_id, "consumed_meta")
    if isbn:
        try:
            row = conn.execute(
                f"SELECT 1 FROM consumed_meta WHERE {where_base} AND type='book' AND isbn=? LIMIT 1",
                params_base + [isbn],
            ).fetchone()
            if row:
                return True
        except sqlite3.OperationalError:
            pass
    if title:
        title_norm = title.strip().lower()
        try:
            rows = conn.execute(
                f"SELECT title, author FROM consumed_meta WHERE {where_base} AND type='book'",
                params_base,
            ).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                "SELECT title, author FROM consumed_meta WHERE type='book'",
            ).fetchall()
        for r in rows:
            existing_title = (r[0] or "").strip().lower()
            existing_author = (r[1] or "").strip().lower()
            if existing_title == title_norm:
                if not author or existing_author == author.strip().lower():
                    return True
    return False


@_sqlite_serialized
def list_consumed_rows(max_items: int = 200, instance_id: str = "") -> list[dict]:
    """Return all consumed rows from consumed_meta (source of truth for listing)."""
    conn = _get_conn()
    where, params = _instance_where(instance_id, "consumed_meta")
    try:
        rows = conn.execute(
            f"""
            SELECT id_original, type, title, author, date_completed, note,
                   isbn, publish_year, openlibrary_key, cover_url, subjects
            FROM consumed_meta
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params + [max_items],
        ).fetchall()
        has_extra = True
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
        has_extra = False
    result = []
    for r in rows:
        item = {
            "id": r[0] or "",
            "type": (r[1] or "article").lower(),
            "title": (r[2] or "?").strip(),
            "author": (r[3] or "").strip(),
            "date_completed": (r[4] or "").strip(),
            "note": (r[5] or "").strip(),
        }
        if has_extra:
            item["isbn"] = (r[6] or "").strip()
            item["publish_year"] = (r[7] or "").strip()
            item["openlibrary_key"] = (r[8] or "").strip()
            item["cover_url"] = (r[9] or "").strip()
            item["subjects"] = (r[10] or "").strip()
        result.append(item)
    return result


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
def wipe_memory() -> None:
    """Clear journal memory, instance prefs, and rec feedback; keep consumed and people."""
    conn = _get_conn()
    conn.execute("DELETE FROM vec_journal")
    conn.execute("DELETE FROM journal_chunks")
    conn.execute("DELETE FROM journal_entries")
    conn.execute("DELETE FROM instance_settings")
    conn.execute("DELETE FROM rec_feedback_events")
    conn.commit()


@_sqlite_serialized
def wipe_memory_for_instance(instance_id: str) -> None:
    if not instance_id:
        return
    conn = _get_conn()
    wipe_journal_memory_for_instance(instance_id)
    conn.execute("DELETE FROM instance_settings WHERE instance_id = ?", (instance_id,))
    conn.execute("DELETE FROM rec_feedback_events WHERE instance_id = ?", (instance_id,))
    conn.commit()


@_sqlite_serialized
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


@_sqlite_serialized
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


@_sqlite_serialized
def memory_count_for_instance(instance_id: str) -> tuple[int, int]:
    """Return (gist_count, episodic_count) for the given instance_id. Used to ask anonymous user if they want to sync."""
    if not instance_id or not instance_id.strip():
        return (0, 0)
    return (gist_count(instance_id), episodic_count(instance_id))


@_sqlite_serialized
def merge_instance_memory(from_instance_id: str, to_instance_id: str) -> None:
    """Copy journal memory, instance prefs, and consumed from from_instance_id to to_instance_id."""
    if not from_instance_id or not to_instance_id or from_instance_id == to_instance_id:
        return
    conn = _get_conn()
    erows = conn.execute(
        """
        SELECT id, session_id, entry_date, raw_text, char_count, created_at
        FROM journal_entries WHERE instance_id = ?
        """,
        (from_instance_id,),
    ).fetchall()
    for eid, session_id, entry_date, raw_text, char_count, created_at in erows:
        conn.execute(
            """
            INSERT INTO journal_entries (session_id, instance_id, entry_date, raw_text, char_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id or "",
                to_instance_id,
                entry_date or "",
                raw_text or "",
                char_count,
                created_at
                or datetime.datetime.utcnow().isoformat() + "Z",
            ),
        )
        new_eid_row = conn.execute("SELECT last_insert_rowid()").fetchone()
        new_eid = int(new_eid_row[0]) if new_eid_row and new_eid_row[0] else 0
        if not new_eid:
            continue
        crows = conn.execute(
            """
            SELECT id, chunk_index, chunk_text, entry_date
            FROM journal_chunks WHERE entry_id = ?
            """,
            (eid,),
        ).fetchall()
        for old_cid, chunk_index, chunk_text, ed in crows:
            conn.execute(
                """
                INSERT INTO journal_chunks (entry_id, instance_id, chunk_index, chunk_text, entry_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_eid, to_instance_id, chunk_index, chunk_text or "", ed or ""),
            )
            nc_row = conn.execute("SELECT last_insert_rowid()").fetchone()
            new_cid = int(nc_row[0]) if nc_row and nc_row[0] else 0
            vrow = conn.execute(
                "SELECT embedding FROM vec_journal WHERE chunk_id = ?", (old_cid,)
            ).fetchone()
            if vrow and new_cid:
                conn.execute(
                    """
                    INSERT INTO vec_journal (chunk_id, embedding, instance_id, entry_date, chunk_text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        new_cid,
                        vrow[0],
                        to_instance_id,
                        (ed or "")[:10],
                        chunk_text or "",
                    ),
                )
    patch = instance_settings_get(from_instance_id)
    if patch:
        instance_settings_merge_json(to_instance_id, patch)
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


# ---------------------------------------------------------------------------
#  daily_articles helpers
# ---------------------------------------------------------------------------

@_sqlite_serialized
def daily_article_upsert(
    instance_id: str,
    entry_date: str,
    article_title: str,
    article_url: str,
    article_snippet: str,
    hook: str,
    candidates_json: str,
    themes_json: str,
    search_queries_json: str,
    status: str = "pending",
) -> int:
    """Insert or replace the daily article for a given date. Returns the row id."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_articles
           (instance_id, entry_date, article_title, article_url, article_snippet,
            hook, candidates_json, themes_json, search_queries_json, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            instance_id, entry_date[:10], article_title, article_url,
            article_snippet, hook, candidates_json, themes_json,
            search_queries_json, status,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM daily_articles WHERE instance_id=? AND entry_date=?",
        (instance_id, entry_date[:10]),
    ).fetchone()
    return row[0] if row else 0


@_sqlite_serialized
def daily_article_get(instance_id: str, entry_date: str) -> dict | None:
    """Return the daily article row as a dict, or None."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT id, instance_id, entry_date, article_title, article_url,
                  article_snippet, hook, candidates_json, themes_json,
                  search_queries_json, status, created_at
           FROM daily_articles
           WHERE instance_id=? AND entry_date=?""",
        (instance_id, entry_date[:10]),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "instance_id": row[1],
        "entry_date": row[2],
        "article_title": row[3],
        "article_url": row[4],
        "article_snippet": row[5],
        "hook": row[6],
        "candidates_json": row[7],
        "themes_json": row[8],
        "search_queries_json": row[9],
        "status": row[10],
        "created_at": row[11],
    }


@_sqlite_serialized
def daily_article_update_status(instance_id: str, entry_date: str, status: str) -> bool:
    """Update the status of a daily article. Returns True if a row was updated."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE daily_articles SET status=? WHERE instance_id=? AND entry_date=?",
        (status, instance_id, entry_date[:10]),
    )
    conn.commit()
    return cur.rowcount > 0
