"""
api/core/local_db.py

SQLite backend for desktop mode — replaces Supabase entirely.
All data lives in ~/.tos-summarizer/data.db.

Fixes applied vs initial version:
  1. WAL mode + busy_timeout — prevents "database is locked" errors when
     background ingest writes status while chat router is reading.
  2. Schema migrations via user_version PRAGMA — safe to add columns in
     future releases without breaking existing users' databases.
  3. Explicit JSON column registry — only known JSON columns are
     serialised/deserialised, not any string that starts with [ or {.
"""

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Schema version ────────────────────────────────────────────────────────────
# Bump this integer whenever you add/change columns.
# _MIGRATIONS list must have exactly this many entries.
_SCHEMA_VERSION = 1

# ── Explicit JSON columns per table ──────────────────────────────────────────
# Only these columns are JSON-serialised on write and deserialised on read.
# Avoids false positives from service names like '{"test"}' being parsed.
_JSON_COLUMNS: dict[str, set[str]] = {
    "documents": set(),
    "chats": {"sources"},
    "summaries": {"sources"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueryResult:
    """Mimics Supabase PostgREST response: result.data"""
    def __init__(self, data: Any):
        self.data = data


class QueryBuilder:
    """
    Fluent query builder matching the Supabase client chaining API:
        db.table("documents").select("*").eq("user_id", uid).order(...).execute()

    Every router that calls database.supa_admin.table(...) works identically
    in both server (Supabase) and desktop (LocalDB) mode.
    """

    def __init__(self, conn: sqlite3.Connection, table: str):
        self._conn = conn
        self._table = table
        self._op: str = "select"
        self._columns: str = "*"
        self._filters: list[tuple] = []
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._data: Optional[dict] = None
        self._single: bool = False
        self._limit: Optional[int] = None

    def select(self, columns: str = "*") -> "QueryBuilder":
        self._op = "select"
        self._columns = columns
        return self

    def insert(self, data: dict) -> "QueryBuilder":
        self._op = "insert"
        self._data = data
        return self

    def update(self, data: dict) -> "QueryBuilder":
        self._op = "update"
        self._data = data
        return self

    def delete(self) -> "QueryBuilder":
        self._op = "delete"
        return self

    def eq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append(("eq", column, value))
        return self

    def order(self, column: str, desc: bool = False) -> "QueryBuilder":
        self._order_col = column
        self._order_desc = desc
        return self

    def single(self) -> "QueryBuilder":
        self._single = True
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def execute(self) -> QueryResult:
        try:
            if self._op == "select":
                return self._execute_select()
            elif self._op == "insert":
                return self._execute_insert()
            elif self._op == "update":
                return self._execute_update()
            elif self._op == "delete":
                return self._execute_delete()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                # WAL mode + busy_timeout should prevent this, but log if it
                # still happens so we know to increase the timeout
                logger.error(
                    "SQLite lock timeout on %s.%s — consider increasing busy_timeout",
                    self._table, self._op,
                )
            raise

    def _json_cols(self) -> set[str]:
        return _JSON_COLUMNS.get(self._table, set())

    def _where_clause(self) -> tuple[str, list]:
        if not self._filters:
            return "", []
        parts, params = [], []
        for (op, col, val) in self._filters:
            if op == "eq":
                parts.append(f"{col} = ?")
                params.append(val)
        return "WHERE " + " AND ".join(parts), params

    def _execute_select(self) -> QueryResult:
        where, params = self._where_clause()
        order = ""
        if self._order_col:
            direction = "DESC" if self._order_desc else "ASC"
            order = f"ORDER BY {self._order_col} {direction}"
        limit = f"LIMIT {self._limit}" if self._limit else ""
        sql = f"SELECT {self._columns} FROM {self._table} {where} {order} {limit}".strip()

        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        # Deserialise only known JSON columns
        json_cols = self._json_cols()
        for row in rows:
            for col in json_cols:
                if col in row and isinstance(row[col], str):
                    try:
                        row[col] = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Leave as-is if it fails

        if self._single:
            if not rows:
                raise sqlite3.OperationalError("No rows found (single)")
            return QueryResult(rows[0])
        return QueryResult(rows)

    def _execute_insert(self) -> QueryResult:
        data = dict(self._data)
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        if "created_at" not in data:
            data["created_at"] = _now()

        # Serialise only known JSON columns
        json_cols = self._json_cols()
        serialised = {
            k: json.dumps(v) if k in json_cols and isinstance(v, (dict, list)) else v
            for k, v in data.items()
        }

        cols = ", ".join(serialised.keys())
        placeholders = ", ".join(["?"] * len(serialised))
        sql = f"INSERT INTO {self._table} ({cols}) VALUES ({placeholders})"
        self._conn.execute(sql, list(serialised.values()))
        self._conn.commit()
        return QueryResult(data)

    def _execute_update(self) -> QueryResult:
        where, where_params = self._where_clause()
        json_cols = self._json_cols()
        serialised = {
            k: json.dumps(v) if k in json_cols and isinstance(v, (dict, list)) else v
            for k, v in self._data.items()
        }
        set_clause = ", ".join([f"{k} = ?" for k in serialised.keys()])
        sql = f"UPDATE {self._table} SET {set_clause} {where}"
        self._conn.execute(sql, list(serialised.values()) + where_params)
        self._conn.commit()
        return QueryResult(None)

    def _execute_delete(self) -> QueryResult:
        where, params = self._where_clause()
        sql = f"DELETE FROM {self._table} {where}"
        self._conn.execute(sql, params)
        self._conn.commit()
        return QueryResult(None)


class LocalDB:
    """
    Drop-in replacement for the Supabase client in desktop mode.

    Usage is identical to the Supabase client:
        LocalDB().table("documents").select("*").eq("user_id", uid).execute()
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)

        # check_same_thread=False allows background ingest thread + async
        # request handlers to share the connection safely under WAL mode.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # ── Fix 1: WAL mode ───────────────────────────────────────────────────
        # Default SQLite journal mode locks the entire DB file on any write.
        # This causes "database is locked" errors when the background ingest
        # thread updates document status while the chat router reads chats.
        #
        # WAL (Write-Ahead Logging) allows concurrent readers + one writer,
        # which matches this app's access pattern exactly.
        self._conn.execute("PRAGMA journal_mode=WAL")

        # ── Fix 2: Busy timeout ───────────────────────────────────────────────
        # If a write lock collision does occur (e.g. two background ingests
        # running simultaneously), SQLite waits up to 5 seconds before raising
        # OperationalError instead of failing immediately.
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._conn.commit()
        self._migrate()
        logger.info("LocalDB initialised at %s (WAL mode)", self._path)

    # ── Fix 3: Schema migrations via user_version ─────────────────────────────
    # user_version is a SQLite PRAGMA that persists an integer in the DB file.
    # On every startup we check the current version and run only the migrations
    # that haven't been applied yet.
    #
    # How to add a new column in a future release:
    #   1. Write a new migration string in _MIGRATIONS
    #   2. Bump _SCHEMA_VERSION by 1
    #   That's it — existing users' DBs are upgraded automatically on next launch.
    _MIGRATIONS = [
        # Version 1 — initial schema
        """
        CREATE TABLE IF NOT EXISTS documents (
            id           TEXT PRIMARY KEY,
            user_id      TEXT NOT NULL DEFAULT 'local',
            filename     TEXT,
            service_name TEXT,
            doc_type     TEXT,
            s3_key       TEXT,
            pinecone_ns  TEXT,
            status       TEXT DEFAULT 'processing',
            error_reason TEXT,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            user_id     TEXT NOT NULL DEFAULT 'local',
            role        TEXT NOT NULL,
            content     TEXT,
            sources     TEXT,
            created_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id           TEXT PRIMARY KEY,
            document_id  TEXT NOT NULL,
            user_id      TEXT NOT NULL DEFAULT 'local',
            topic_label  TEXT,
            summary_text TEXT,
            sources      TEXT,
            created_at   TEXT
        );
        """,
        # Version 2 example (add this when you need a new column):
        # "ALTER TABLE documents ADD COLUMN rating INTEGER DEFAULT NULL;",
    ]

    def _migrate(self):
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        pending = self._MIGRATIONS[current:]

        if not pending:
            return

        for i, migration_sql in enumerate(pending):
            version = current + i + 1
            logger.info("applying DB migration to version %d", version)
            self._conn.executescript(migration_sql)
            # user_version must be set with string formatting — PRAGMA
            # doesn't support ? placeholders
            self._conn.execute(f"PRAGMA user_version = {version}")
            self._conn.commit()

        logger.info("DB schema up to date at version %d", current + len(pending))

    def table(self, name: str) -> QueryBuilder:
        return QueryBuilder(self._conn, name)

    # ── Auth stub — desktop has no login ──────────────────────────────────────
    # get_current_user returns {"user_id": "local-user"} in desktop mode
    # before it ever calls supa_admin.auth, so this stub is a safety net only.
    class _AuthStub:
        class _FakeUser:
            id = "local-user"
            email = "local@desktop"

        class _FakeSession:
            access_token = "desktop-token"
            refresh_token = "desktop-token"

        class _FakeRes:
            user = None
            session = None

        def sign_up(self, _) -> "_AuthStub._FakeRes":
            res = LocalDB._AuthStub._FakeRes()
            res.user = LocalDB._AuthStub._FakeUser()
            return res

        def sign_in_with_password(self, _) -> "_AuthStub._FakeRes":
            res = LocalDB._AuthStub._FakeRes()
            res.user = LocalDB._AuthStub._FakeUser()
            res.session = LocalDB._AuthStub._FakeSession()
            return res

        def get_user(self, _token: str) -> "_AuthStub._FakeRes":
            res = LocalDB._AuthStub._FakeRes()
            res.user = LocalDB._AuthStub._FakeUser()
            return res

    auth = _AuthStub()