"""
memory_store.py
----------------
SQLite-backed storage layer for Memora.

Handles:
- Schema creation
- CRUD for memories (subject/predicate/object triples)
- CRUD for raw messages
- Storage of embeddings as BLOBs (numpy arrays serialized)

Design notes:
- Uses TEXT ids (uuid4 strings) so ids are stable across processes.
- is_deleted is a soft-delete flag so the dashboard can "undo" or show history.
- importance_score exists so retrieval can later blend recency/importance/similarity.
"""

import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memora.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source_message_id TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            importance_score REAL DEFAULT 0.5,
            is_deleted INTEGER DEFAULT 0,
            FOREIGN KEY (source_message_id) REFERENCES messages(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id),
            embedding BLOB NOT NULL,
            text_content TEXT NOT NULL
        )
    """)

    # Simple FTS index over the flattened triple text, used as a lexical
    # fallback / boost alongside vector similarity in retrieval.py
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(memory_id UNINDEXED, text_content)
    """)

    conn.commit()
    conn.close()


def add_message(user_id: str, conversation_id: str, role: str, content: str) -> str:
    conn = get_connection()
    msg_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO messages (id, user_id, conversation_id, role, content) VALUES (?, ?, ?, ?, ?)",
        (msg_id, user_id, conversation_id, role, content),
    )
    conn.commit()
    conn.close()
    return msg_id


def get_conversation_messages(user_id: str, conversation_id: str, limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM messages
           WHERE user_id = ? AND conversation_id = ?
           ORDER BY timestamp ASC LIMIT ?""",
        (user_id, conversation_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_memory(
    user_id: str,
    subject: str,
    predicate: str,
    object_: str,
    confidence: float = 1.0,
    source_message_id: Optional[str] = None,
    importance_score: float = 0.5,
) -> str:
    conn = get_connection()
    mem_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO memories
           (id, user_id, subject, predicate, object, confidence, source_message_id, importance_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (mem_id, user_id, subject, predicate, object_, confidence, source_message_id, importance_score),
    )
    conn.commit()
    conn.close()
    return mem_id

def find_similar_memory(user_id: str, subject: str, predicate: str):
    """Check for an existing memory with the same subject+predicate before inserting
    a new one, so repeated facts update in place instead of piling up as duplicates."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM memories
           WHERE user_id = ? AND subject = ? AND predicate = ? AND is_deleted = 0
           ORDER BY created_at DESC LIMIT 1""",
        (user_id, subject, predicate),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def add_memory_embedding(memory_id: str, embedding_bytes: bytes, text_content: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, text_content) VALUES (?, ?, ?)",
        (memory_id, embedding_bytes, text_content),
    )
    conn.execute(
        "INSERT INTO memories_fts (memory_id, text_content) VALUES (?, ?)",
        (memory_id, text_content),
    )
    conn.commit()
    conn.close()


def get_all_memories(user_id: str, include_deleted: bool = False):
    conn = get_connection()
    query = "SELECT * FROM memories WHERE user_id = ?"
    if not include_deleted:
        query += " AND is_deleted = 0"
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_memories_with_embeddings(user_id: str):
    """Join memories with their embeddings for retrieval.py to consume."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT m.id, m.subject, m.predicate, m.object, m.importance_score,
                  m.created_at, e.embedding, e.text_content
           FROM memories m
           JOIN memory_embeddings e ON m.id = e.memory_id
           WHERE m.user_id = ? AND m.is_deleted = 0""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def soft_delete_memory(memory_id: str) -> None:
    conn = get_connection()
    conn.execute("UPDATE memories SET is_deleted = 1 WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()


def update_memory(memory_id: str, subject: str, predicate: str, object_: str) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE memories SET subject = ?, predicate = ?, object = ? WHERE id = ?",
        (subject, predicate, object_, memory_id),
    )
    conn.commit()
    conn.close()


def keyword_search(user_id: str, query: str, limit: int = 10):
    """Lexical fallback search using FTS5, scoped to this user's memories."""
    # Sanitize: FTS5 MATCH treats punctuation as query syntax (AND/OR/NOT, quotes,
    # NEAR, etc). Strip anything that isn't alphanumeric/space, then wrap each
    # word so it's treated as a literal token, not an operator.
    import re
    words = re.findall(r"[A-Za-z0-9]+", query)
    if not words:
        return []
    safe_query = " OR ".join(f'"{w}"' for w in words)

    conn = get_connection()
    rows = conn.execute(
        """SELECT m.* FROM memories m
           JOIN memories_fts f ON m.id = f.memory_id
           WHERE f.text_content MATCH ? AND m.user_id = ? AND m.is_deleted = 0
           LIMIT ?""",
        (safe_query, user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

MULTI_VALUED_PREDICATES = {
    "career_goal", "works_on", "participates_in", "enjoys", "has_skill", "prefers", "owns"
}
def find_exact_memory(user_id: str, subject: str, predicate: str, object_: str):
    """Check if this exact triple already exists — used for multi-valued predicates,
    where we want new *distinct* values to persist but exact repeats to be skipped."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM memories
           WHERE user_id = ? AND subject = ? AND predicate = ? AND object = ? AND is_deleted = 0
           LIMIT 1""",
        (user_id, subject, predicate, object_),
    ).fetchone()
    conn.close()
    return dict(row) if row else None
if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
