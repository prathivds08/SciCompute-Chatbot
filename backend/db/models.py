"""
SQLite database schema — sessions, messages, jobs, parameters, artifacts.

Uses SQLAlchemy Core (no ORM session management overhead) for a lightweight
local database that tracks conversation history and simulation provenance.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.config import settings


def _utcnow() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
#  Schema DDL
# ------------------------------------------------------------------ #

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL,
    last_msg_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content     TEXT    NOT NULL,
    agent_used  TEXT,
    tool_calls  TEXT,               -- JSON-serialised list of tool invocations
    timestamp   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    workflow_type   TEXT    NOT NULL CHECK(workflow_type IN ('dft', 'training', 'md', 'analysis')),
    slurm_job_id    TEXT,
    status          TEXT    NOT NULL DEFAULT 'created'
                           CHECK(status IN ('created', 'submitted', 'running', 'completed', 'failed', 'cancelled')),
    submitted_at    TEXT,
    completed_at    TEXT,
    params_json     TEXT,           -- JSON blob of all simulation parameters
    output_path     TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS parameters (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    key     TEXT    NOT NULL,
    value   TEXT    NOT NULL,
    unit    TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    session_id  TEXT    REFERENCES sessions(id) ON DELETE CASCADE,
    type        TEXT    NOT NULL CHECK(type IN ('plot', 'input_file', 'report', 'data', 'other')),
    path        TEXT    NOT NULL,
    description TEXT
);
"""


# ------------------------------------------------------------------ #
#  Database helper class
# ------------------------------------------------------------------ #

class Database:
    """Thin wrapper around a SQLite connection with helper methods."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.SQLITE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    _ALLOWED_JOB_COLUMNS = {
        "slurm_job_id", "status", "submitted_at", "completed_at",
        "params_json", "output_path", "error_message",
    }

    # -- lifecycle ---------------------------------------------------

    def connect(self) -> "Database":
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # -- sessions ----------------------------------------------------

    def create_session(self, session_id: str, title: str = "New Chat") -> dict:
        now = _utcnow()
        self.conn.execute(
            "INSERT INTO sessions (id, title, created_at, last_msg_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        self.conn.commit()
        return {"id": session_id, "title": title, "created_at": now}

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY last_msg_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    # -- messages ----------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_used: Optional[str] = None,
        tool_calls: Optional[list[dict]] = None,
    ) -> int:
        now = _utcnow()
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, agent_used, tool_calls, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, agent_used, json.dumps(tool_calls) if tool_calls else None, now),
        )
        # Touch session last_msg_at
        self.conn.execute("UPDATE sessions SET last_msg_at = ? WHERE id = ?", (now, session_id))
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- jobs --------------------------------------------------------

    def create_job(
        self,
        session_id: str,
        workflow_type: str,
        params: Optional[dict[str, Any]] = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO jobs (session_id, workflow_type, params_json) VALUES (?, ?, ?)",
            (session_id, workflow_type, json.dumps(params) if params else None),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        invalid = set(fields.keys()) - self._ALLOWED_JOB_COLUMNS
        if invalid:
            raise ValueError(f"Disallowed job update columns: {invalid}")
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        self.conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def get_jobs(self, session_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- artifacts ---------------------------------------------------

    def add_artifact(
        self,
        session_id: str,
        artifact_type: str,
        path: str,
        description: str = "",
        job_id: Optional[int] = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO artifacts (job_id, session_id, type, path, description) VALUES (?, ?, ?, ?, ?)",
            (job_id, session_id, artifact_type, path, description),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
