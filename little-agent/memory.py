#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    payload_json TEXT NOT NULL,
    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_occurred_at
    ON events(occurred_at);

CREATE INDEX IF NOT EXISTS idx_events_type_occurred_at
    ON events(type, occurred_at);

CREATE INDEX IF NOT EXISTS idx_events_source_occurred_at
    ON events(source, occurred_at);
"""


class MemoryStore:
    """
    Append-only event store backed by SQLite.

    V1:
      - sensory/action events are stored as immutable facts
      - no summarization
      - no embeddings
      - no consolidation yet
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def append_event(self, event: dict[str, Any]) -> None:
        required = ("id", "type", "source", "occurred_at", "payload")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"missing required event fields: {missing}")

        payload_json = json.dumps(
            event["payload"],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    id,
                    schema_version,
                    type,
                    source,
                    occurred_at,
                    started_at,
                    ended_at,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["id"],
                    int(event.get("schema_version", 1)),
                    event["type"],
                    event["source"],
                    event["occurred_at"],
                    event.get("started_at"),
                    event.get("ended_at"),
                    payload_json,
                ),
            )
            conn.commit()

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id,
                    schema_version,
                    type,
                    source,
                    occurred_at,
                    started_at,
                    ended_at,
                    payload_json
                FROM events
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict[str, Any]] = []

        for row in reversed(rows):
            result.append(
                {
                    "id": row["id"],
                    "schema_version": row["schema_version"],
                    "type": row["type"],
                    "source": row["source"],
                    "occurred_at": row["occurred_at"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "payload": json.loads(row["payload_json"]),
                }
            )

        return result


async def memory_worker(
    queue: asyncio.Queue[dict[str, Any]],
    store: MemoryStore,
) -> None:
    """
    Consume EventBus events and persist them.

    SQLite work runs via asyncio.to_thread so the main event loop
    stays responsive.
    """
    while True:
        event = await queue.get()

        try:
            await asyncio.to_thread(store.append_event, event)
        finally:
            queue.task_done()
