"""
database.py
===========
SQLite-backed persistence for Socratic-OT Streamlit app.

Tables:
  users        — student_id, password_hash, created_at
  sessions     — session_id, student_id, title, started_at, ended_at, data_json
  chat_messages— id, session_id, role, content, created_at

Usage:
  from src.database import Database
  db = Database()                     # opens/creates socratic_ot.db
  db.register("alice", "secret")
  ok, msg = db.login("alice", "secret")
  sid = db.create_session("alice", "Occipital Lobe")
  db.append_message(sid, "user", "What lobe handles vision?")
  db.append_message(sid, "assistant", "Great question — let me give you a clue...")
  db.save_session_memory(sid, weak_topics=[...], mastered_topics=[...], ...)
"""

import os
import json
import sqlite3
import bcrypt
from datetime import datetime
from contextlib import contextmanager

# ── Path ─────────────────────────────────────────────────────────────────────

def _db_path() -> str:
    # Locally: repo_root/data/socratic_ot.db
    # Streamlit Cloud: /tmp/socratic_ot.db  (ephemeral but fine for demo)
    base = os.environ.get("DB_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "socratic_ot.db")


# ── Database class ────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: str = None):
        self.path = path or _db_path()
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    student_id    TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    student_id   TEXT NOT NULL,
                    title        TEXT NOT NULL DEFAULT 'New Chat',
                    started_at   TEXT NOT NULL,
                    ended_at     TEXT,
                    -- session memory fields (updated as session progresses)
                    weak_topics_json     TEXT DEFAULT '[]',
                    mastered_topics_json TEXT DEFAULT '[]',
                    confused_terms_json  TEXT DEFAULT '{}',
                    topic_scores_json    TEXT DEFAULT '{}',
                    FOREIGN KEY (student_id) REFERENCES users(student_id)
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,   -- 'user' or 'assistant'
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)

    # ── Auth ──────────────────────────────────────────────────────────────────

    def register(self, student_id: str, password: str) -> tuple[bool, str]:
        """Register a new student. Returns (success, message)."""
        student_id = student_id.strip().lower()
        if not student_id or not password:
            return False, "Student ID and password cannot be empty."
        if len(password) < 4:
            return False, "Password must be at least 4 characters."
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE student_id = ?", (student_id,)
            ).fetchone()
            if existing:
                return False, f"Student ID '{student_id}' is already taken."
            pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (student_id, password_hash, created_at) VALUES (?, ?, ?)",
                (student_id, pw_hash, datetime.now().isoformat())
            )
        return True, "Account created! You can now log in."

    def login(self, student_id: str, password: str) -> tuple[bool, str]:
        """Verify credentials. Returns (success, message)."""
        student_id = student_id.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE student_id = ?", (student_id,)
            ).fetchone()
        if not row:
            return False, "Student ID not found."
        if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            return True, "Login successful."
        return False, "Incorrect password."

    def student_exists(self, student_id: str) -> bool:
        student_id = student_id.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE student_id = ?", (student_id,)
            ).fetchone()
        return row is not None

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, student_id: str, title: str = "New Chat") -> str:
        """Create a new chat session. Returns session_id."""
        import uuid
        session_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, student_id, title, started_at) VALUES (?, ?, ?, ?)",
                (session_id, student_id.strip().lower(), title, datetime.now().isoformat())
            )
        return session_id

    def update_session_title(self, session_id: str, title: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (title[:60], session_id)
            )

    def end_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id)
            )

    def get_sessions(self, student_id: str) -> list[dict]:
        """Return all sessions for a student, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, title, started_at, ended_at "
                "FROM sessions WHERE student_id = ? ORDER BY started_at DESC",
                (student_id.strip().lower(),)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Messages ──────────────────────────────────────────────────────────────

    def append_message(self, session_id: str, role: str, content: str):
        """Append one message to a session."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.now().isoformat())
            )

    def get_messages(self, session_id: str) -> list[dict]:
        """Return all messages for a session in order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Session memory (weak topics, mastery) ─────────────────────────────────

    def save_session_memory(
        self,
        session_id: str,
        weak_topics: list,
        mastered_topics: list,
        confused_terms: dict,
        topic_scores: dict,
    ):
        with self._conn() as conn:
            conn.execute(
                """UPDATE sessions SET
                    weak_topics_json     = ?,
                    mastered_topics_json = ?,
                    confused_terms_json  = ?,
                    topic_scores_json    = ?
                WHERE session_id = ?""",
                (
                    json.dumps(weak_topics),
                    json.dumps(mastered_topics),
                    json.dumps(confused_terms),
                    json.dumps(topic_scores),
                    session_id,
                )
            )

    def get_dashboard(self, student_id: str) -> dict:
        """
        Aggregate all sessions for a student into a weak-spot dashboard.
        Mirrors SessionMemory.get_dashboard() but reads from SQLite.
        """
        from collections import Counter
        student_id = student_id.strip().lower()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT weak_topics_json, mastered_topics_json, "
                "confused_terms_json, started_at "
                "FROM sessions WHERE student_id = ? ORDER BY started_at ASC",
                (student_id,)
            ).fetchall()

        if not rows:
            return {"student_id": student_id, "sessions": 0}

        all_confused = Counter()
        # Track the most recent outcome per topic (rows ordered ASC by started_at)
        # Later sessions overwrite earlier ones — most recent outcome wins.
        topic_outcome = {}  # key: lower-cased topic, value: ("weak"|"mastered", display_name)

        for row in rows:
            weak_list     = json.loads(row["weak_topics_json"] or "[]")
            mastered_list = json.loads(row["mastered_topics_json"] or "[]")
            for t in weak_list:
                if t.strip():
                    topic_outcome[t.strip().lower()] = ("weak", t.strip())
            for t in mastered_list:
                if t.strip():
                    topic_outcome[t.strip().lower()] = ("mastered", t.strip())
            for term, cnt in json.loads(row["confused_terms_json"] or "{}").items():
                all_confused[term] += cnt

        final_weak     = [v[1] for v in topic_outcome.values() if v[0] == "weak"]
        final_mastered = [v[1] for v in topic_outcome.values() if v[0] == "mastered"]

        weak_set     = {t.lower() for t in final_weak}
        priority = [t for t in final_weak if t.lower() in weak_set]

        return {
            "student_id":      student_id,
            "sessions":        len(rows),
            "last_session":    rows[-1]["started_at"][:10],
            "weak_topics":     final_weak,
            "mastered_topics": final_mastered,
            "priority_review": priority[:5],
            "top_confused":    all_confused.most_common(5),
        }
