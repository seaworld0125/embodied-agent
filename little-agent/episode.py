#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory import MemoryStore


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ActiveEpisode:
    id: str
    started_at: str
    last_event_at: str
    event_count: int


class EpisodeBuilder:
    """
    V1 episode segmentation rule:

    - first event opens an episode
    - events arriving within idle_timeout_sec belong to the same episode
    - if no event arrives for idle_timeout_sec, the episode closes
    - if a new event arrives after a larger gap, close old + open new

    This is intentionally simple temporal segmentation.
    """

    def __init__(
        self,
        store: MemoryStore,
        idle_timeout_sec: float = 15.0,
    ) -> None:
        self.store = store
        self.idle_timeout_sec = idle_timeout_sec
        self.active: ActiveEpisode | None = None

    async def recover(self) -> None:
        row = await asyncio.to_thread(
            self.store.get_active_episode
        )

        if row is None:
            return

        last_dt = parse_iso(row["last_event_at"])
        age = (
            datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)
        ).total_seconds()

        if age >= self.idle_timeout_sec:
            await asyncio.to_thread(
                self.store.close_episode,
                row["id"],
                row["last_event_at"],
            )

            print(
                f"[episode] recovered stale episode "
                f"{row['id']} -> closed",
                file=sys.stderr,
                flush=True,
            )
            return

        self.active = ActiveEpisode(
            id=row["id"],
            started_at=row["started_at"],
            last_event_at=row["last_event_at"],
            event_count=int(row["event_count"]),
        )

        print(
            f"[episode] recovered active episode "
            f"{self.active.id}",
            file=sys.stderr,
            flush=True,
        )

    async def add_event(
        self,
        event: dict[str, Any],
    ) -> None:
        event_id = event["id"]
        occurred_at = event["occurred_at"]

        if self.active is None:
            await self._start_episode(
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return

        previous = parse_iso(
            self.active.last_event_at
        )
        current = parse_iso(occurred_at)

        gap = (current - previous).total_seconds()

        if gap > self.idle_timeout_sec:
            await self.close_active(
                ended_at=self.active.last_event_at
            )

            await self._start_episode(
                event_id=event_id,
                occurred_at=occurred_at,
            )
            return

        await asyncio.to_thread(
            self.store.append_event_to_episode,
            self.active.id,
            event_id,
            occurred_at,
        )

        self.active.last_event_at = occurred_at
        self.active.event_count += 1

        print(
            f"[episode] append "
            f"id={self.active.id} "
            f"count={self.active.event_count}",
            file=sys.stderr,
            flush=True,
        )

    async def _start_episode(
        self,
        event_id: str,
        occurred_at: str,
    ) -> None:
        episode_id = str(uuid4())

        await asyncio.to_thread(
            self.store.create_episode,
            episode_id,
            occurred_at,
            event_id,
        )

        self.active = ActiveEpisode(
            id=episode_id,
            started_at=occurred_at,
            last_event_at=occurred_at,
            event_count=1,
        )

        print(
            f"[episode] start "
            f"id={episode_id}",
            file=sys.stderr,
            flush=True,
        )

    async def close_active(
        self,
        ended_at: str | None = None,
    ) -> None:
        if self.active is None:
            return

        final_ended_at = (
            ended_at
            or self.active.last_event_at
            or now_iso()
        )

        episode_id = self.active.id
        event_count = self.active.event_count

        await asyncio.to_thread(
            self.store.close_episode,
            episode_id,
            final_ended_at,
        )

        self.active = None

        print(
            f"[episode] close "
            f"id={episode_id} "
            f"events={event_count}",
            file=sys.stderr,
            flush=True,
        )


async def episode_worker(
    queue: asyncio.Queue[dict[str, Any]],
    store: MemoryStore,
    idle_timeout_sec: float = 15.0,
) -> None:
    builder = EpisodeBuilder(
        store=store,
        idle_timeout_sec=idle_timeout_sec,
    )

    await builder.recover()

    while True:
        try:
            event = await asyncio.wait_for(
                queue.get(),
                timeout=idle_timeout_sec,
            )
        except asyncio.TimeoutError:
            if builder.active is not None:
                await builder.close_active()
            continue

        try:
            # Important:
            # memory_worker and episode_worker are separate subscribers.
            # Ensure the raw event exists before creating the FK relation.
            await asyncio.to_thread(
                store.append_event,
                event,
            )

            await builder.add_event(event)
        finally:
            queue.task_done()
