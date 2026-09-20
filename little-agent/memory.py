#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
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

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('active', 'closed')),
    started_at TEXT NOT NULL,
    last_event_at TEXT NOT NULL,
    ended_at TEXT,
    event_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    consolidation_json TEXT,
    ready_for_consolidation INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_episodes_status
    ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_started_at
    ON episodes(started_at);

CREATE TABLE IF NOT EXISTS episode_events (
    episode_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    utterance_seq INTEGER,
    PRIMARY KEY (episode_id, event_id),
    UNIQUE (episode_id, position),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episode_events_event_id
    ON episode_events(event_id);
CREATE INDEX IF NOT EXISTS idx_episode_events_utterance_seq
    ON episode_events(episode_id, utterance_seq);

CREATE TABLE IF NOT EXISTS episode_working_memory (
    episode_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 0,
    upto_utterance_seq INTEGER NOT NULL DEFAULT -1,
    summary TEXT NOT NULL DEFAULT '',
    state_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);
"""


class _ManagedConnection(sqlite3.Connection):
    """sqlite3 connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=5.0,
            factory=_ManagedConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            # Run migrations that must happen before indexes in SCHEMA.
            self._migrate_pre_schema(conn)
            conn.executescript(SCHEMA)
            self._migrate_post_schema(conn)
            conn.commit()

    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        if not self._table_exists(conn, table):
            return set()
        return {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _migrate_pre_schema(self, conn: sqlite3.Connection) -> None:
        episode_columns = self._columns(conn, "episodes")
        if episode_columns:
            if "consolidation_json" not in episode_columns:
                conn.execute(
                    "ALTER TABLE episodes ADD COLUMN consolidation_json TEXT"
                )
            if "ready_for_consolidation" not in episode_columns:
                # Old closed episodes are complete by definition because the
                # old process had no delayed-STT state to recover.
                conn.execute(
                    "ALTER TABLE episodes ADD COLUMN "
                    "ready_for_consolidation INTEGER NOT NULL DEFAULT 1"
                )

        relation_columns = self._columns(conn, "episode_events")
        if relation_columns and "utterance_seq" not in relation_columns:
            conn.execute(
                "ALTER TABLE episode_events ADD COLUMN utterance_seq INTEGER"
            )

    def _migrate_post_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE episodes SET ready_for_consolidation=0 "
            "WHERE status='active'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_ready "
            "ON episodes(status, ready_for_consolidation)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episode_events_utterance_seq "
            "ON episode_events(episode_id, utterance_seq)"
        )

    def open_episode(self, episode_id: str, started_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO episodes (
                    id, status, started_at, last_event_at,
                    event_count, ready_for_consolidation
                ) VALUES (?, 'active', ?, ?, 0, 0)
                """,
                (episode_id, started_at, started_at),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO episode_working_memory (
                    episode_id, version, upto_utterance_seq,
                    summary, state_json
                ) VALUES (?, 0, -1, '', '{}')
                """,
                (episode_id,),
            )
            conn.commit()

    def persist_event(
        self,
        event: dict[str, Any],
        episode_id: str | None,
        position: int | None,
        utterance_seq: int | None,
        touch_activity: bool,
    ) -> None:
        required = ("id", "type", "source", "occurred_at", "payload")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"missing required event fields: {missing}")

        payload_json = json.dumps(
            event["payload"], ensure_ascii=False, separators=(",", ":")
        )

        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    id, schema_version, type, source, occurred_at,
                    started_at, ended_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

            if episode_id is not None:
                if position is None:
                    raise ValueError("position is required when episode_id is set")

                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO episode_events (
                        episode_id, event_id, position, utterance_seq
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (episode_id, event["id"], position, utterance_seq),
                )

                if cursor.rowcount:
                    if touch_activity:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET last_event_at=?, event_count=event_count+1,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=?
                            """,
                            (event["occurred_at"], episode_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET event_count=event_count+1,
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=?
                            """,
                            (episode_id,),
                        )

            conn.commit()

    def persist_attached_event(
        self,
        event: dict[str, Any],
        episode_id: str,
        utterance_seq: int | None = None,
        touch_activity: bool = False,
    ) -> int | None:
        """Persist an event and attach it at the next episode position.

        Used for late internal/action events whose in-memory EpisodeRuntime has
        already been released. Position allocation and insertion happen in one
        SQLite transaction. Returns the assigned position, or None when the
        target episode no longer exists (the raw event is still preserved).
        """
        required = ("id", "type", "source", "occurred_at", "payload")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"missing required event fields: {missing}")

        payload_json = json.dumps(
            event["payload"], ensure_ascii=False, separators=(",", ":")
        )

        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM episodes WHERE id=?",
                (episode_id,),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        id, schema_version, type, source, occurred_at,
                        started_at, ended_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                return None

            position_row = conn.execute(
                """
                SELECT COALESCE(MAX(position), -1) + 1 AS next_position
                FROM episode_events
                WHERE episode_id=?
                """,
                (episode_id,),
            ).fetchone()
            position = int(position_row["next_position"])

            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    id, schema_version, type, source, occurred_at,
                    started_at, ended_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO episode_events (
                    episode_id, event_id, position, utterance_seq
                ) VALUES (?, ?, ?, ?)
                """,
                (episode_id, event["id"], position, utterance_seq),
            )

            if cursor.rowcount:
                if touch_activity:
                    conn.execute(
                        """
                        UPDATE episodes
                        SET last_event_at=?, event_count=event_count+1,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                        """,
                        (event["occurred_at"], episode_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE episodes
                        SET event_count=event_count+1,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id=?
                        """,
                        (episode_id,),
                    )

            conn.commit()
            return position

    def append_event(self, event: dict[str, Any]) -> None:
        self.persist_event(event, None, None, None, False)

    def close_episode(
        self,
        episode_id: str,
        ended_at: str,
        ready_for_consolidation: bool = True,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET status='closed', ended_at=?, ready_for_consolidation=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='active'
                """,
                (ended_at, 1 if ready_for_consolidation else 0, episode_id),
            )
            conn.commit()

    def mark_episode_ready(self, episode_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET ready_for_consolidation=1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='closed'
                """,
                (episode_id,),
            )
            conn.commit()

    def get_active_episode(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, status, started_at, last_event_at, ended_at,
                       event_count, ready_for_consolidation
                FROM episodes
                WHERE status='active'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row is not None else None

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, status, started_at, last_event_at, ended_at,
                       event_count, summary, consolidation_json,
                       ready_for_consolidation
                FROM episodes WHERE id=?
                """,
                (episode_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_next_utterance_seq(self, episode_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(utterance_seq) AS max_seq
                FROM episode_events
                WHERE episode_id=? AND utterance_seq IS NOT NULL
                """,
                (episode_id,),
            ).fetchone()
        if row is None or row["max_seq"] is None:
            return 0
        return int(row["max_seq"]) + 1

    def ensure_working_memory(self, episode_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO episode_working_memory (
                    episode_id, version, upto_utterance_seq, summary, state_json
                ) VALUES (?, 0, -1, '', '{}')
                """,
                (episode_id,),
            )
            conn.commit()

    def get_working_memory(self, episode_id: str) -> dict[str, Any]:
        self.ensure_working_memory(episode_id)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT episode_id, version, upto_utterance_seq,
                       summary, state_json, updated_at
                FROM episode_working_memory
                WHERE episode_id=?
                """,
                (episode_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"working memory missing for episode {episode_id}")

        item = dict(row)
        try:
            item["state"] = json.loads(item["state_json"] or "{}")
        except json.JSONDecodeError:
            item["state"] = {}
        return item

    def save_working_memory_cas(
        self,
        episode_id: str,
        expected_version: int,
        upto_utterance_seq: int,
        summary: str,
        state: dict[str, Any],
    ) -> bool:
        encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE episode_working_memory
                SET version=version+1,
                    upto_utterance_seq=?,
                    summary=?,
                    state_json=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE episode_id=?
                  AND version=?
                  AND upto_utterance_seq < ?
                """,
                (
                    upto_utterance_seq,
                    summary,
                    encoded,
                    episode_id,
                    expected_version,
                    upto_utterance_seq,
                ),
            )
            conn.commit()
            return cursor.rowcount == 1

    def episode_final_utterances(
        self,
        episode_id: str,
        after_seq: int = -1,
        upto_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT ee.utterance_seq, ee.position,
                   e.id, e.type, e.source, e.occurred_at,
                   e.started_at, e.ended_at, e.payload_json
            FROM episode_events ee
            JOIN events e ON e.id=ee.event_id
            WHERE ee.episode_id=?
              AND e.type='speech.final'
              AND ee.utterance_seq IS NOT NULL
              AND ee.utterance_seq > ?
        """
        params: list[Any] = [episode_id, after_seq]
        if upto_seq is not None:
            sql += " AND ee.utterance_seq <= ?"
            params.append(upto_seq)
        sql += " ORDER BY ee.utterance_seq, ee.position"

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "utterance_seq": int(row["utterance_seq"]),
                "position": int(row["position"]),
                "id": row["id"],
                "type": row["type"],
                "source": row["source"],
                "occurred_at": row["occurred_at"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def legacy_semantic_events(self, episode_id: str) -> list[dict[str, Any]]:
        """Fallback for episodes created before utterance_seq existed."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ee.position, e.id, e.type, e.source, e.occurred_at,
                       e.started_at, e.ended_at, e.payload_json
                FROM episode_events ee
                JOIN events e ON e.id=ee.event_id
                WHERE ee.episode_id=?
                  AND e.type IN ('speech', 'speech.final')
                ORDER BY ee.position
                """,
                (episode_id,),
            ).fetchall()
        return [
            {
                "utterance_seq": None,
                "position": int(row["position"]),
                "id": row["id"],
                "type": row["type"],
                "source": row["source"],
                "occurred_at": row["occurred_at"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def episode_events(self, episode_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ee.position, ee.utterance_seq,
                       e.id, e.schema_version, e.type, e.source,
                       e.occurred_at, e.started_at, e.ended_at, e.payload_json
                FROM episode_events ee
                JOIN events e ON e.id=ee.event_id
                WHERE ee.episode_id=?
                ORDER BY ee.position
                """,
                (episode_id,),
            ).fetchall()
        return [
            {
                "position": row["position"],
                "utterance_seq": row["utterance_seq"],
                "id": row["id"],
                "schema_version": row["schema_version"],
                "type": row["type"],
                "source": row["source"],
                "occurred_at": row["occurred_at"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, status, started_at, last_event_at, ended_at,
                       event_count, summary, consolidation_json,
                       ready_for_consolidation
                FROM episodes ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, schema_version, type, source, occurred_at,
                       started_at, ended_at, payload_json
                FROM events ORDER BY occurred_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
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
            for row in reversed(rows)
        ]


async def persistence_worker(
    queue: asyncio.Queue[dict[str, Any]],
    store: MemoryStore,
) -> None:
    while True:
        command = await queue.get()
        try:
            op = command["op"]
            if op == "open_episode":
                await asyncio.to_thread(
                    store.open_episode,
                    command["episode_id"],
                    command["started_at"],
                )
            elif op == "persist_event":
                await asyncio.to_thread(
                    store.persist_event,
                    command["event"],
                    command.get("episode_id"),
                    command.get("position"),
                    command.get("utterance_seq"),
                    bool(command.get("touch_activity", False)),
                )
            elif op == "persist_attached_event":
                await asyncio.to_thread(
                    store.persist_attached_event,
                    command["event"],
                    command["episode_id"],
                    command.get("utterance_seq"),
                    bool(command.get("touch_activity", False)),
                )
            elif op == "close_episode":
                await asyncio.to_thread(
                    store.close_episode,
                    command["episode_id"],
                    command["ended_at"],
                    bool(command.get("ready", False)),
                )
            elif op == "mark_episode_ready":
                await asyncio.to_thread(
                    store.mark_episode_ready,
                    command["episode_id"],
                )
            else:
                raise ValueError(f"unknown persistence op: {op}")
        except Exception as exc:
            print(
                f"[persistence] failed op={command.get('op')}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            queue.task_done()
