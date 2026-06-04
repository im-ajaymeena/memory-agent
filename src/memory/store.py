import json
import pathlib
import sqlite3
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np

from .models import Category, MemoryRecord, Source

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT    PRIMARY KEY,
    text                TEXT    NOT NULL,
    embedding           TEXT    NOT NULL,
    category            TEXT    NOT NULL,
    intent_label        TEXT    NOT NULL DEFAULT '',
    entities            TEXT    NOT NULL DEFAULT '[]',
    contextual_markers  TEXT    NOT NULL DEFAULT '[]',
    timestamp_created   REAL    NOT NULL,
    timestamp_updated   REAL    NOT NULL,
    source              TEXT    NOT NULL,
    source_trust        INTEGER NOT NULL,
    is_current          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_is_current ON memories(is_current);
CREATE INDEX IF NOT EXISTS idx_category   ON memories(category);
CREATE INDEX IF NOT EXISTS idx_updated    ON memories(timestamp_updated);
"""


class MemoryStore:
    def __init__(self, db_path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Migration: add contextual_markers when opening a pre-existing database
            cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            if "contextual_markers" not in cols:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN contextual_markers TEXT NOT NULL DEFAULT '[]'"
                )

    # ── writes ──────────────────────────────────────────────────────────────

    def add(self, record: MemoryRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, text, embedding, category, intent_label, entities,
                    contextual_markers, timestamp_created, timestamp_updated,
                    source, source_trust, is_current)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,1)""",
                (
                    record.id,
                    record.text,
                    json.dumps(record.embedding),
                    record.category.value,
                    record.intent_label,
                    json.dumps(record.entities),
                    json.dumps(record.contextual_markers),
                    record.timestamp_created,
                    record.timestamp_updated,
                    record.source.value,
                    record.source_trust,
                ),
            )

    def update(self, memory_id: str, new_text: str, new_embedding: list[float]) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE memories
                   SET text=?, embedding=?, timestamp_updated=?
                   WHERE id=? AND is_current=1""",
                (new_text, json.dumps(new_embedding), time.time(), memory_id),
            )

    def soft_delete(self, memory_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET is_current=0 WHERE id=?",
                (memory_id,),
            )

    # ── reads ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        category: Category | None = None,
    ) -> list[MemoryRecord]:
        rows = self._active_rows(category)
        if not rows:
            return []
        matrix = np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
        scores = matrix @ query_embedding  # cosine sim (both L2-normalised)
        top_k = min(k, len(rows))
        indices = np.argsort(scores)[::-1][:top_k]
        return [_row_to_record(rows[i]) for i in indices]

    def get_by_id(self, memory_id: str) -> MemoryRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def all_active(self) -> list[MemoryRecord]:
        return [_row_to_record(r) for r in self._active_rows()]

    def count_active(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_current=1"
            ).fetchone()[0]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _active_rows(self, category: Category | None = None) -> list[sqlite3.Row]:
        with self._conn() as conn:
            if category:
                return conn.execute(
                    "SELECT * FROM memories WHERE is_current=1 AND category=?",
                    (category.value,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM memories WHERE is_current=1"
            ).fetchall()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    r = MemoryRecord(
        id=row["id"],
        text=row["text"],
        category=Category(row["category"]),
        source=Source(row["source"]),
        intent_label=row["intent_label"] or "",
        entities=json.loads(row["entities"]),
        contextual_markers=json.loads(row["contextual_markers"]),
        embedding=json.loads(row["embedding"]),
        timestamp_created=row["timestamp_created"],
        timestamp_updated=row["timestamp_updated"],
        is_current=bool(row["is_current"]),
    )
    return r
