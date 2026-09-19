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
    PRIMARY KEY (episode_id, event_id),
    UNIQUE (episode_id, position),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episode_events_event_id
    ON episode_events(event_id);
"""


class MemoryStore:
    def __init__(
        self,
        db_path: str | Path,
    ) -> None:
        self.db_path = (
            Path(db_path)
            .expanduser()
            .resolve()
        )
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def connect(
        self,
    ) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(
            "PRAGMA foreign_keys=ON"
        )
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.commit()

    def _migrate(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(episodes)"
            ).fetchall()
        }

        if (
            "consolidation_json"
            not in columns
        ):
            conn.execute(
                "ALTER TABLE episodes "
                "ADD COLUMN "
                "consolidation_json TEXT"
            )

        if (
            "ready_for_consolidation"
            not in columns
        ):
            # Existing closed episodes were created before this state
            # existed, so consider them ready.
            conn.execute(
                "ALTER TABLE episodes "
                "ADD COLUMN "
                "ready_for_consolidation "
                "INTEGER NOT NULL DEFAULT 1"
            )

        # Active episodes are never ready.
        conn.execute(
            """
            UPDATE episodes
            SET ready_for_consolidation = 0
            WHERE status = 'active'
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_episodes_ready
            ON episodes(
                status,
                ready_for_consolidation
            )
            """
        )

    def append_event(
        self,
        event: dict[str, Any],
    ) -> None:
        self.persist_event(
            event=event,
            episode_id=None,
            position=None,
            touch_activity=False,
        )

    def open_episode(
        self,
        episode_id: str,
        started_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO episodes (
                    id,
                    status,
                    started_at,
                    last_event_at,
                    event_count,
                    ready_for_consolidation
                )
                VALUES (?, 'active', ?, ?, 0, 0)
                """,
                (
                    episode_id,
                    started_at,
                    started_at,
                ),
            )
            conn.commit()

    def persist_event(
        self,
        event: dict[str, Any],
        episode_id: str | None,
        position: int | None,
        touch_activity: bool,
    ) -> None:
        required = (
            "id",
            "type",
            "source",
            "occurred_at",
            "payload",
        )
        missing = [
            key
            for key in required
            if key not in event
        ]

        if missing:
            raise ValueError(
                "missing required event fields: "
                f"{missing}"
            )

        payload_json = json.dumps(
            event["payload"],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self.connect() as conn:
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
                    int(
                        event.get(
                            "schema_version",
                            1,
                        )
                    ),
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
                    raise ValueError(
                        "position is required "
                        "when episode_id is set"
                    )

                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO
                        episode_events (
                            episode_id,
                            event_id,
                            position
                        )
                    VALUES (?, ?, ?)
                    """,
                    (
                        episode_id,
                        event["id"],
                        position,
                    ),
                )

                if cursor.rowcount:
                    if touch_activity:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET
                                last_event_at = ?,
                                event_count =
                                    event_count + 1,
                                updated_at =
                                    CURRENT_TIMESTAMP
                            WHERE id = ?
                            """,
                            (
                                event[
                                    "occurred_at"
                                ],
                                episode_id,
                            ),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET
                                event_count =
                                    event_count + 1,
                                updated_at =
                                    CURRENT_TIMESTAMP
                            WHERE id = ?
                            """,
                            (
                                episode_id,
                            ),
                        )

            conn.commit()

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
                SET
                    status = 'closed',
                    ended_at = ?,
                    ready_for_consolidation = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND status = 'active'
                """,
                (
                    ended_at,
                    1
                    if ready_for_consolidation
                    else 0,
                    episode_id,
                ),
            )
            conn.commit()

    def mark_episode_ready(
        self,
        episode_id: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET
                    ready_for_consolidation = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND status = 'closed'
                """,
                (
                    episode_id,
                ),
            )
            conn.commit()

    def get_active_episode(
        self,
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    status,
                    started_at,
                    last_event_at,
                    ended_at,
                    event_count,
                    ready_for_consolidation
                FROM episodes
                WHERE status = 'active'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def recent_events(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
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

        result: list[
            dict[str, Any]
        ] = []

        for row in reversed(rows):
            result.append(
                {
                    "id": row["id"],
                    "schema_version": (
                        row["schema_version"]
                    ),
                    "type": row["type"],
                    "source": row["source"],
                    "occurred_at": (
                        row["occurred_at"]
                    ),
                    "started_at": (
                        row["started_at"]
                    ),
                    "ended_at": (
                        row["ended_at"]
                    ),
                    "payload": json.loads(
                        row["payload_json"]
                    ),
                }
            )

        return result

    def recent_episodes(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    status,
                    started_at,
                    last_event_at,
                    ended_at,
                    event_count,
                    summary,
                    consolidation_json,
                    ready_for_consolidation
                FROM episodes
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[
            dict[str, Any]
        ] = []

        for row in reversed(rows):
            item = dict(row)

            raw = item.get(
                "consolidation_json"
            )

            if raw:
                try:
                    item[
                        "consolidation"
                    ] = json.loads(raw)
                except json.JSONDecodeError:
                    item[
                        "consolidation"
                    ] = None
            else:
                item[
                    "consolidation"
                ] = None

            result.append(item)

        return result

    def episode_events(
        self,
        episode_id: str,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ee.position,
                    e.id,
                    e.schema_version,
                    e.type,
                    e.source,
                    e.occurred_at,
                    e.started_at,
                    e.ended_at,
                    e.payload_json
                FROM episode_events ee
                JOIN events e
                  ON e.id = ee.event_id
                WHERE ee.episode_id = ?
                ORDER BY ee.position
                """,
                (
                    episode_id,
                ),
            ).fetchall()

        return [
            {
                "position": row["position"],
                "id": row["id"],
                "schema_version": (
                    row["schema_version"]
                ),
                "type": row["type"],
                "source": row["source"],
                "occurred_at": (
                    row["occurred_at"]
                ),
                "started_at": (
                    row["started_at"]
                ),
                "ended_at": (
                    row["ended_at"]
                ),
                "payload": json.loads(
                    row["payload_json"]
                ),
            }
            for row in rows
        ]

    # Compatibility helpers for older local scripts.
    def create_episode(
        self,
        episode_id: str,
        started_at: str,
        first_event_id: str,
    ) -> None:
        self.open_episode(
            episode_id,
            started_at,
        )

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO
                    episode_events (
                        episode_id,
                        event_id,
                        position
                    )
                VALUES (?, ?, 0)
                """,
                (
                    episode_id,
                    first_event_id,
                ),
            )

            if cursor.rowcount:
                conn.execute(
                    """
                    UPDATE episodes
                    SET event_count = 1
                    WHERE id = ?
                    """,
                    (
                        episode_id,
                    ),
                )

            conn.commit()

    def append_event_to_episode(
        self,
        episode_id: str,
        event_id: str,
        occurred_at: str,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT event_count
                FROM episodes
                WHERE id = ?
                """,
                (
                    episode_id,
                ),
            ).fetchone()

            if row is None:
                raise ValueError(
                    "episode not found: "
                    f"{episode_id}"
                )

            position = int(
                row["event_count"]
            )

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO
                    episode_events (
                        episode_id,
                        event_id,
                        position
                    )
                VALUES (?, ?, ?)
                """,
                (
                    episode_id,
                    event_id,
                    position,
                ),
            )

            if cursor.rowcount:
                conn.execute(
                    """
                    UPDATE episodes
                    SET
                        last_event_at = ?,
                        event_count =
                            event_count + 1,
                        updated_at =
                            CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        occurred_at,
                        episode_id,
                    ),
                )

            conn.commit()


async def persistence_worker(
    queue: asyncio.Queue[
        dict[str, Any]
    ],
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
                    command.get(
                        "episode_id"
                    ),
                    command.get(
                        "position"
                    ),
                    bool(
                        command.get(
                            "touch_activity",
                            False,
                        )
                    ),
                )

            elif op == "close_episode":
                await asyncio.to_thread(
                    store.close_episode,
                    command["episode_id"],
                    command["ended_at"],
                    bool(
                        command.get(
                            "ready",
                            False,
                        )
                    ),
                )

            elif op == (
                "mark_episode_ready"
            ):
                await asyncio.to_thread(
                    store.mark_episode_ready,
                    command["episode_id"],
                )

            else:
                raise ValueError(
                    f"unknown persistence op: {op}"
                )

        except Exception as exc:
            print(
                "[persistence] command failed "
                f"op={command.get('op')}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        finally:
            queue.task_done()


async def memory_worker(
    queue: asyncio.Queue[
        dict[str, Any]
    ],
    store: MemoryStore,
) -> None:
    """
    Legacy raw-event worker kept for compatibility.
    New core.py uses persistence_worker instead.
    """
    while True:
        event = await queue.get()

        try:
            await asyncio.to_thread(
                store.append_event,
                event,
            )
        finally:
            queue.task_done()
