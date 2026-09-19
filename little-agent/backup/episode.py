#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from memory import MemoryStore


PublishFn = Callable[
    [dict[str, Any]],
    Awaitable[None],
]


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt


def seconds_between(
    older: str,
    newer: str,
) -> float:
    return (
        parse_iso(newer)
        - parse_iso(older)
    ).total_seconds()


@dataclass
class EpisodeRuntime:
    id: str
    started_at: str
    last_activity_at: str
    last_activity_monotonic: float
    next_position: int = 0
    active_utterances: set[str] = field(
        default_factory=set
    )
    pending_finals: set[str] = field(
        default_factory=set
    )
    closed: bool = False


class ActiveEpisodeManager:
    """
    Realtime episode assignment.

    Important rule:
    episode membership is decided from physical speech lifecycle events
    (speech.started / speech.ended), never from STT completion time.

    speech.final may arrive much later and is routed back to the episode
    already chosen for its utterance_id.
    """

    def __init__(
        self,
        store: MemoryStore,
        persistence_queue: asyncio.Queue[
            dict[str, Any]
        ],
        publish: PublishFn,
        idle_timeout_sec: float = 15.0,
    ) -> None:
        self.store = store
        self.persistence_queue = (
            persistence_queue
        )
        self.publish = publish
        self.idle_timeout_sec = (
            idle_timeout_sec
        )

        self.active: EpisodeRuntime | None = None

        # Keep closed runtime state while a delayed STT result can still
        # arrive for one of its utterances.
        self.states: dict[
            str,
            EpisodeRuntime,
        ] = {}

        self.utterance_episode: dict[
            str,
            str,
        ] = {}

    async def recover(self) -> None:
        row = await asyncio.to_thread(
            self.store.get_active_episode
        )

        if row is None:
            return

        last_activity = row["last_event_at"]

        age = max(
            0.0,
            (
                datetime.now(timezone.utc)
                - parse_iso(last_activity)
            ).total_seconds(),
        )

        if age >= self.idle_timeout_sec:
            # Any in-flight STT belonged to the previous process and can
            # no longer arrive. The episode is safe to finalize.
            await asyncio.to_thread(
                self.store.close_episode,
                row["id"],
                last_activity,
                True,
            )

            print(
                "[episode] recovered stale "
                f"episode {row['id']} -> closed",
                file=sys.stderr,
                flush=True,
            )
            return

        state = EpisodeRuntime(
            id=row["id"],
            started_at=row["started_at"],
            last_activity_at=last_activity,
            last_activity_monotonic=(
                time.monotonic() - age
            ),
            next_position=int(
                row["event_count"]
            ),
        )

        self.active = state
        self.states[state.id] = state

        print(
            "[episode] recovered active "
            f"episode {state.id}",
            file=sys.stderr,
            flush=True,
        )

    async def handle(
        self,
        event: dict[str, Any],
    ) -> None:
        event_type = event.get("type")

        if event_type == "speech.started":
            await self._handle_speech_started(
                event
            )
            return

        if event_type == "speech.ended":
            await self._handle_speech_ended(
                event
            )
            return

        if event_type in (
            "speech.final",
            "speech.failed",
        ):
            await self._handle_perception_result(
                event
            )
            return

        # Compatibility with the pre-refactor ear.
        if event_type == "speech":
            await self._handle_legacy_speech(
                event
            )
            return

        # Unknown/non-episodic raw events are still preserved.
        self._persist_event(
            event=event,
            episode_id=None,
            position=None,
            touch_activity=False,
        )
        await self.publish(event)

    async def idle_loop(self) -> None:
        interval = min(
            0.25,
            max(
                0.05,
                self.idle_timeout_sec / 10.0,
            ),
        )

        while True:
            await asyncio.sleep(interval)

            state = self.active

            if state is None:
                continue

            # A long utterance must not be cut just because it exceeds
            # the normal between-utterance idle timeout.
            if state.active_utterances:
                continue

            idle_for = (
                time.monotonic()
                - state.last_activity_monotonic
            )

            if idle_for >= self.idle_timeout_sec:
                await self._close_state(state)

    async def _handle_speech_started(
        self,
        event: dict[str, Any],
    ) -> None:
        utterance_id = self._utterance_id(
            event
        )
        occurred_at = event["occurred_at"]

        state = self.active

        if (
            state is not None
            and not state.active_utterances
            and seconds_between(
                state.last_activity_at,
                occurred_at,
            ) > self.idle_timeout_sec
        ):
            await self._close_state(state)
            state = None

        if state is None:
            state = self._open_state(
                started_at=occurred_at
            )

        self.utterance_episode[
            utterance_id
        ] = state.id

        state.active_utterances.add(
            utterance_id
        )
        state.pending_finals.add(
            utterance_id
        )
        state.last_activity_at = occurred_at
        state.last_activity_monotonic = (
            time.monotonic()
        )

        await self._assign_and_publish(
            state,
            event,
            touch_activity=True,
        )

    async def _handle_speech_ended(
        self,
        event: dict[str, Any],
    ) -> None:
        utterance_id = self._utterance_id(
            event
        )

        state = self._state_for_utterance(
            utterance_id
        )

        if state is None:
            # Defensive recovery if a speech.started event was lost.
            state = self.active

            if state is None:
                state = self._open_state(
                    started_at=(
                        event.get("started_at")
                        or event["occurred_at"]
                    )
                )

            self.utterance_episode[
                utterance_id
            ] = state.id
            state.pending_finals.add(
                utterance_id
            )

        state.active_utterances.discard(
            utterance_id
        )
        state.last_activity_at = event[
            "occurred_at"
        ]
        state.last_activity_monotonic = (
            time.monotonic()
        )

        await self._assign_and_publish(
            state,
            event,
            touch_activity=True,
        )

    async def _handle_perception_result(
        self,
        event: dict[str, Any],
    ) -> None:
        utterance_id = self._utterance_id(
            event
        )

        state = self._state_for_utterance(
            utterance_id
        )

        if state is None:
            print(
                "[episode] late perception has "
                "no utterance mapping "
                f"utterance={utterance_id}",
                file=sys.stderr,
                flush=True,
            )

            self._persist_event(
                event=event,
                episode_id=None,
                position=None,
                touch_activity=False,
            )
            await self.publish(event)
            return

        await self._assign_and_publish(
            state,
            event,
            touch_activity=False,
        )

        state.pending_finals.discard(
            utterance_id
        )
        self.utterance_episode.pop(
            utterance_id,
            None,
        )

        # Closing an episode and completing perception are independent.
        # Consolidation becomes legal only after both have happened.
        if (
            state.closed
            and not state.pending_finals
        ):
            self._mark_ready(state.id)
            self.states.pop(
                state.id,
                None,
            )

    async def _handle_legacy_speech(
        self,
        event: dict[str, Any],
    ) -> None:
        occurred_at = event["occurred_at"]
        state = self.active

        if (
            state is not None
            and seconds_between(
                state.last_activity_at,
                occurred_at,
            ) > self.idle_timeout_sec
        ):
            await self._close_state(state)
            state = None

        if state is None:
            state = self._open_state(
                started_at=occurred_at
            )

        state.last_activity_at = occurred_at
        state.last_activity_monotonic = (
            time.monotonic()
        )

        await self._assign_and_publish(
            state,
            event,
            touch_activity=True,
        )

    def _open_state(
        self,
        started_at: str,
    ) -> EpisodeRuntime:
        episode_id = str(uuid4())

        state = EpisodeRuntime(
            id=episode_id,
            started_at=started_at,
            last_activity_at=started_at,
            last_activity_monotonic=(
                time.monotonic()
            ),
        )

        self.active = state
        self.states[state.id] = state

        self.persistence_queue.put_nowait(
            {
                "op": "open_episode",
                "episode_id": state.id,
                "started_at": started_at,
            }
        )

        print(
            f"[episode] start id={state.id}",
            file=sys.stderr,
            flush=True,
        )

        return state

    async def _close_state(
        self,
        state: EpisodeRuntime,
    ) -> None:
        if state.closed:
            return

        state.closed = True

        if (
            self.active is not None
            and self.active.id == state.id
        ):
            self.active = None

        ready = not state.pending_finals

        self.persistence_queue.put_nowait(
            {
                "op": "close_episode",
                "episode_id": state.id,
                "ended_at": (
                    state.last_activity_at
                ),
                "ready": ready,
            }
        )

        print(
            f"[episode] close id={state.id} "
            f"pending_stt="
            f"{len(state.pending_finals)} "
            f"ready={ready}",
            file=sys.stderr,
            flush=True,
        )

        if ready:
            self.states.pop(
                state.id,
                None,
            )

    async def _assign_and_publish(
        self,
        state: EpisodeRuntime,
        event: dict[str, Any],
        touch_activity: bool,
    ) -> None:
        enriched = dict(event)
        enriched["episode_id"] = state.id
        enriched[
            "episode_position"
        ] = state.next_position

        state.next_position += 1

        self._persist_event(
            event=enriched,
            episode_id=state.id,
            position=enriched[
                "episode_position"
            ],
            touch_activity=touch_activity,
        )

        await self.publish(enriched)

    def _persist_event(
        self,
        event: dict[str, Any],
        episode_id: str | None,
        position: int | None,
        touch_activity: bool,
    ) -> None:
        self.persistence_queue.put_nowait(
            {
                "op": "persist_event",
                "event": event,
                "episode_id": episode_id,
                "position": position,
                "touch_activity": (
                    touch_activity
                ),
            }
        )

    def _mark_ready(
        self,
        episode_id: str,
    ) -> None:
        self.persistence_queue.put_nowait(
            {
                "op": "mark_episode_ready",
                "episode_id": episode_id,
            }
        )

        print(
            "[episode] perception complete "
            f"id={episode_id} "
            "ready_for_consolidation=true",
            file=sys.stderr,
            flush=True,
        )

    def _state_for_utterance(
        self,
        utterance_id: str,
    ) -> EpisodeRuntime | None:
        episode_id = (
            self.utterance_episode.get(
                utterance_id
            )
        )

        if episode_id is None:
            return None

        return self.states.get(
            episode_id
        )

    @staticmethod
    def _utterance_id(
        event: dict[str, Any],
    ) -> str:
        payload = event.get(
            "payload",
            {},
        )
        utterance_id = payload.get(
            "utterance_id"
        )

        if not utterance_id:
            raise ValueError(
                "speech lifecycle event missing "
                "payload.utterance_id"
            )

        return str(utterance_id)
