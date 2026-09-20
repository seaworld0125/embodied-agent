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


PublishFn = Callable[[dict[str, Any]], Awaitable[None]]


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def seconds_between(older: str, newer: str) -> float:
    return (parse_iso(newer) - parse_iso(older)).total_seconds()


@dataclass
class EpisodeRuntime:
    id: str
    started_at: str
    last_activity_at: str
    last_activity_monotonic: float
    next_position: int = 0
    next_utterance_seq: int = 0
    active_utterances: set[str] = field(default_factory=set)
    active_agent_actions: set[str] = field(default_factory=set)
    pending_finals: set[str] = field(default_factory=set)
    closed: bool = False


class ActiveEpisodeManager:
    """
    Assigns episode membership from physical speech timing.

    A speech.started event gets an immutable utterance_seq immediately.
    speech.ended / speech.final / speech.failed reuse that same sequence even
    when STT finishes much later or after the episode is already closed.
    """

    def __init__(
        self,
        store: MemoryStore,
        persistence_queue: asyncio.Queue[dict[str, Any]],
        publish: PublishFn,
        idle_timeout_sec: float = 15.0,
    ) -> None:
        self.store = store
        self.persistence_queue = persistence_queue
        self.publish = publish
        self.idle_timeout_sec = idle_timeout_sec

        self.active: EpisodeRuntime | None = None
        self.states: dict[str, EpisodeRuntime] = {}
        self.utterance_episode: dict[str, str] = {}
        self.utterance_seq: dict[str, int] = {}

    async def recover(self) -> None:
        row = await asyncio.to_thread(self.store.get_active_episode)
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
            await asyncio.to_thread(
                self.store.close_episode,
                row["id"],
                last_activity,
                True,
            )
            print(
                f"[episode] recovered stale episode {row['id']} -> closed",
                file=sys.stderr,
                flush=True,
            )
            return

        next_seq = await asyncio.to_thread(
            self.store.get_next_utterance_seq,
            row["id"],
        )

        state = EpisodeRuntime(
            id=row["id"],
            started_at=row["started_at"],
            last_activity_at=last_activity,
            last_activity_monotonic=time.monotonic() - age,
            next_position=int(row["event_count"]),
            next_utterance_seq=next_seq,
        )
        self.active = state
        self.states[state.id] = state
        print(
            f"[episode] recovered active episode {state.id} "
            f"next_utterance_seq={next_seq}",
            file=sys.stderr,
            flush=True,
        )

    async def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "speech.started":
            await self._handle_speech_started(event)
        elif event_type == "speech.ended":
            await self._handle_speech_ended(event)
        elif event_type in ("speech.final", "speech.failed"):
            await self._handle_perception_result(event)
        elif event_type == "speech":
            await self._handle_legacy_speech(event)
        else:
            self._persist_event(
                event=event,
                episode_id=None,
                position=None,
                utterance_seq=None,
                touch_activity=False,
            )
            await self.publish(event)

    async def publish_internal_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Persist and publish agent/internal events.

        Cognitive events such as agent.intent and reasoner.failed do not extend
        the interaction timeout. Physical agent speech does: while the agent is
        audibly speaking, the current episode stays alive, and the 15s idle
        clock restarts when speech ends or fails.

        While an episode runtime is still resident, position allocation stays
        in RAM so it cannot collide with sensory events already queued for
        persistence. If runtime state has been released, the persistence worker
        attaches the late event at the next DB position without reopening the
        closed episode.
        """
        episode_id_raw = event.get("episode_id")
        episode_id = str(episode_id_raw) if episode_id_raw else None
        enriched = dict(event)
        event_type = str(enriched.get("type", ""))
        is_physical_agent_speech = event_type in {
            "agent.speech.started",
            "agent.speech.ended",
            "agent.speech.failed",
        }

        if episode_id is None:
            self._persist_event(
                event=enriched,
                episode_id=None,
                position=None,
                utterance_seq=None,
                touch_activity=False,
            )
            await self.publish(enriched)
            return

        state = self.states.get(episode_id)
        if state is not None:
            touch_activity = is_physical_agent_speech and not state.closed
            if touch_activity:
                self._apply_agent_action_lifecycle(state, enriched)

            enriched["episode_position"] = state.next_position
            enriched["utterance_seq"] = None
            state.next_position += 1

            self._persist_event(
                event=enriched,
                episode_id=episode_id,
                position=enriched["episode_position"],
                utterance_seq=None,
                touch_activity=touch_activity,
            )
        else:
            # Late cognitive/action events remain part of the historical
            # episode, but a closed episode is never reopened or extended.
            self.persistence_queue.put_nowait(
                {
                    "op": "persist_attached_event",
                    "event": enriched,
                    "episode_id": episode_id,
                    "utterance_seq": None,
                    "touch_activity": False,
                }
            )

        await self.publish(enriched)

    def _apply_agent_action_lifecycle(
        self,
        state: EpisodeRuntime,
        event: dict[str, Any],
    ) -> None:
        event_type = str(event.get("type", ""))
        action_id = str(event.get("payload", {}).get("action_id", ""))

        if event_type == "agent.speech.started" and action_id:
            state.active_agent_actions.add(action_id)
        elif event_type in ("agent.speech.ended", "agent.speech.failed"):
            if action_id:
                state.active_agent_actions.discard(action_id)

        state.last_activity_at = event["occurred_at"]
        state.last_activity_monotonic = time.monotonic()

    async def idle_loop(self) -> None:
        interval = min(0.25, max(0.05, self.idle_timeout_sec / 10.0))
        while True:
            await asyncio.sleep(interval)
            state = self.active
            if state is None:
                continue
            if state.active_utterances or state.active_agent_actions:
                continue

            idle_for = time.monotonic() - state.last_activity_monotonic
            if idle_for >= self.idle_timeout_sec:
                await self._close_state(state)

    async def _handle_speech_started(self, event: dict[str, Any]) -> None:
        utterance_id = self._utterance_id(event)
        occurred_at = event["occurred_at"]
        state = self.active

        if (
            state is not None
            and not state.active_utterances
            and seconds_between(state.last_activity_at, occurred_at)
            > self.idle_timeout_sec
        ):
            await self._close_state(state)
            state = None

        if state is None:
            state = self._open_state(occurred_at)

        seq = state.next_utterance_seq
        state.next_utterance_seq += 1

        self.utterance_episode[utterance_id] = state.id
        self.utterance_seq[utterance_id] = seq
        state.active_utterances.add(utterance_id)
        state.pending_finals.add(utterance_id)
        state.last_activity_at = occurred_at
        state.last_activity_monotonic = time.monotonic()

        await self._assign_and_publish(
            state,
            event,
            utterance_seq=seq,
            touch_activity=True,
        )

    async def _handle_speech_ended(self, event: dict[str, Any]) -> None:
        utterance_id = self._utterance_id(event)
        state, seq = self._state_and_seq_for_utterance(utterance_id)

        if state is None or seq is None:
            state = self.active
            if state is None:
                state = self._open_state(
                    event.get("started_at") or event["occurred_at"]
                )
            seq = state.next_utterance_seq
            state.next_utterance_seq += 1
            self.utterance_episode[utterance_id] = state.id
            self.utterance_seq[utterance_id] = seq
            state.pending_finals.add(utterance_id)

        state.active_utterances.discard(utterance_id)
        state.last_activity_at = event["occurred_at"]
        state.last_activity_monotonic = time.monotonic()

        await self._assign_and_publish(
            state,
            event,
            utterance_seq=seq,
            touch_activity=True,
        )

    async def _handle_perception_result(self, event: dict[str, Any]) -> None:
        utterance_id = self._utterance_id(event)
        state, seq = self._state_and_seq_for_utterance(utterance_id)

        if state is None or seq is None:
            print(
                "[episode] late perception has no utterance mapping "
                f"utterance={utterance_id}",
                file=sys.stderr,
                flush=True,
            )
            self._persist_event(
                event=event,
                episode_id=None,
                position=None,
                utterance_seq=None,
                touch_activity=False,
            )
            await self.publish(event)
            return

        await self._assign_and_publish(
            state,
            event,
            utterance_seq=seq,
            touch_activity=False,
        )

        state.pending_finals.discard(utterance_id)
        self.utterance_episode.pop(utterance_id, None)
        self.utterance_seq.pop(utterance_id, None)

        if state.closed and not state.pending_finals:
            self._mark_ready(state.id)
            self.states.pop(state.id, None)

    async def _handle_legacy_speech(self, event: dict[str, Any]) -> None:
        occurred_at = event["occurred_at"]
        state = self.active

        if (
            state is not None
            and seconds_between(state.last_activity_at, occurred_at)
            > self.idle_timeout_sec
        ):
            await self._close_state(state)
            state = None

        if state is None:
            state = self._open_state(occurred_at)

        seq = state.next_utterance_seq
        state.next_utterance_seq += 1
        state.last_activity_at = occurred_at
        state.last_activity_monotonic = time.monotonic()

        await self._assign_and_publish(
            state,
            event,
            utterance_seq=seq,
            touch_activity=True,
        )

    def _open_state(self, started_at: str) -> EpisodeRuntime:
        episode_id = str(uuid4())
        state = EpisodeRuntime(
            id=episode_id,
            started_at=started_at,
            last_activity_at=started_at,
            last_activity_monotonic=time.monotonic(),
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

    async def _close_state(self, state: EpisodeRuntime) -> None:
        if state.closed:
            return
        state.closed = True
        if self.active is not None and self.active.id == state.id:
            self.active = None

        ready = not state.pending_finals
        self.persistence_queue.put_nowait(
            {
                "op": "close_episode",
                "episode_id": state.id,
                "ended_at": state.last_activity_at,
                "ready": ready,
            }
        )
        print(
            f"[episode] close id={state.id} "
            f"pending_stt={len(state.pending_finals)} ready={ready}",
            file=sys.stderr,
            flush=True,
        )

        if ready:
            self.states.pop(state.id, None)

    async def _assign_and_publish(
        self,
        state: EpisodeRuntime,
        event: dict[str, Any],
        *,
        utterance_seq: int | None,
        touch_activity: bool,
    ) -> None:
        enriched = dict(event)
        enriched["episode_id"] = state.id
        enriched["episode_position"] = state.next_position
        enriched["utterance_seq"] = utterance_seq
        state.next_position += 1

        self._persist_event(
            event=enriched,
            episode_id=state.id,
            position=enriched["episode_position"],
            utterance_seq=utterance_seq,
            touch_activity=touch_activity,
        )
        await self.publish(enriched)

    def _persist_event(
        self,
        event: dict[str, Any],
        episode_id: str | None,
        position: int | None,
        utterance_seq: int | None,
        touch_activity: bool,
    ) -> None:
        self.persistence_queue.put_nowait(
            {
                "op": "persist_event",
                "event": event,
                "episode_id": episode_id,
                "position": position,
                "utterance_seq": utterance_seq,
                "touch_activity": touch_activity,
            }
        )

    def _mark_ready(self, episode_id: str) -> None:
        self.persistence_queue.put_nowait(
            {
                "op": "mark_episode_ready",
                "episode_id": episode_id,
            }
        )
        print(
            f"[episode] perception complete id={episode_id} "
            "ready_for_consolidation=true",
            file=sys.stderr,
            flush=True,
        )

    def _state_and_seq_for_utterance(
        self,
        utterance_id: str,
    ) -> tuple[EpisodeRuntime | None, int | None]:
        episode_id = self.utterance_episode.get(utterance_id)
        seq = self.utterance_seq.get(utterance_id)
        if episode_id is None:
            return None, seq
        return self.states.get(episode_id), seq

    @staticmethod
    def _utterance_id(event: dict[str, Any]) -> str:
        utterance_id = event.get("payload", {}).get("utterance_id")
        if not utterance_id:
            raise ValueError(
                "speech lifecycle event missing payload.utterance_id"
            )
        return str(utterance_id)
