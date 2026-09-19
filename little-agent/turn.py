#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from reasoner import RealtimeReasoner, StaleReasoningInput


PublishFn = Callable[[dict[str, Any]], Awaitable[None]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class TurnCoordinator:
    """
    Converts speech lifecycle events into realtime reasoning turns.

    Episode boundary and response boundary are intentionally different:
      - episode: long interaction memory boundary (~15s idle)
      - turn: short response grace window (~0.5s)

    New speech invalidates older reasoning through a generation watermark.
    In-flight HTTP generation is not force-killed; stale results are discarded.
    """

    def __init__(
        self,
        reasoner: RealtimeReasoner,
        event_queue: asyncio.Queue[dict[str, Any]],
        publish: PublishFn,
        *,
        grace_ms: int = 500,
    ) -> None:
        self.reasoner = reasoner
        self.event_queue = event_queue
        self.publish = publish
        self.grace_sec = max(0.0, grace_ms / 1000.0)

        self._generation = 0
        self._latest_final_seq = -1
        self._active_utterances: set[str] = set()
        self._pending_grace: asyncio.Task[None] | None = None
        self._reasoning_tasks: set[asyncio.Task[None]] = set()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def latest_final_seq(self) -> int:
        return self._latest_final_seq

    async def event_loop(self) -> None:
        while True:
            event = await self.event_queue.get()
            try:
                event_type = event.get("type")

                if event_type == "speech.started":
                    self._on_speech_started(event)
                elif event_type == "speech.ended":
                    self._on_speech_ended(event)
                elif event_type == "speech.final":
                    self._on_speech_final(event)
                elif event_type == "speech.failed":
                    self._on_speech_failed(event)
            finally:
                self.event_queue.task_done()

    def _on_speech_started(self, event: dict[str, Any]) -> None:
        utterance_id = str(
            event.get("payload", {}).get("utterance_id", "")
        )
        if utterance_id:
            self._active_utterances.add(utterance_id)

        self._generation += 1
        self._cancel_pending_grace()

        log(
            f"[turn] interrupt generation={self._generation} "
            f"seq={event.get('utterance_seq')}"
        )

    def _on_speech_ended(self, event: dict[str, Any]) -> None:
        utterance_id = str(
            event.get("payload", {}).get("utterance_id", "")
        )
        if utterance_id:
            self._active_utterances.discard(utterance_id)

    def _on_speech_failed(self, event: dict[str, Any]) -> None:
        utterance_id = str(
            event.get("payload", {}).get("utterance_id", "")
        )
        if utterance_id:
            self._active_utterances.discard(utterance_id)

    def _on_speech_final(self, event: dict[str, Any]) -> None:
        seq = event.get("utterance_seq")
        if not isinstance(seq, int):
            return

        self._latest_final_seq = max(self._latest_final_seq, seq)
        self._generation += 1
        generation = self._generation

        self._cancel_pending_grace()
        task = asyncio.create_task(
            self._after_grace(event, generation),
            name=f"turn-grace:{event.get('episode_id')}:{seq}",
        )
        self._pending_grace = task

    def _cancel_pending_grace(self) -> None:
        task = self._pending_grace
        if task is not None and not task.done():
            task.cancel()
        self._pending_grace = None

    async def _after_grace(
        self,
        event: dict[str, Any],
        generation: int,
    ) -> None:
        try:
            await asyncio.sleep(self.grace_sec)
        except asyncio.CancelledError:
            return

        if generation != self._generation:
            return
        if self._active_utterances:
            return

        if self._pending_grace is asyncio.current_task():
            self._pending_grace = None

        task = asyncio.create_task(
            self._run_reasoning(event, generation),
            name=(
                f"reasoning:{event.get('episode_id')}:"
                f"{event.get('utterance_seq')}"
            ),
        )
        self._reasoning_tasks.add(task)
        task.add_done_callback(self._reasoning_tasks.discard)

    async def _run_reasoning(
        self,
        event: dict[str, Any],
        generation: int,
    ) -> None:
        episode_id = str(event.get("episode_id", ""))
        seq = event.get("utterance_seq")

        try:
            result = await self.reasoner.reason(event)
        except StaleReasoningInput as exc:
            log(
                f"[turn] stale before inference result episode="
                f"{episode_id[:8]} seq={seq}: {exc}"
            )
            return
        except Exception as exc:
            # A failed S2 turn must never kill the sensory/memory loop.
            if generation == self._generation:
                await self.publish(
                    self._failed_event(
                        episode_id=episode_id,
                        based_on_seq=seq,
                        error=exc,
                    )
                )
            return

        if not self._is_current(
            generation,
            result.based_on_utterance_seq,
        ):
            log(
                f"[turn] stale result discarded episode={episode_id[:8]} "
                f"based_on={result.based_on_utterance_seq} "
                f"latest={self._latest_final_seq} "
                f"generation={generation}->{self._generation}"
            )
            return

        if result.intent == "wait":
            log(
                f"[turn] wait episode={episode_id[:8]} "
                f"seq={result.based_on_utterance_seq}"
            )
            return

        if result.intent == "respond":
            await self.publish(
                self._intent_event(
                    episode_id=result.episode_id,
                    based_on_seq=result.based_on_utterance_seq,
                    response=result.response,
                )
            )

    def _is_current(
        self,
        generation: int,
        based_on_seq: int,
    ) -> bool:
        return (
            generation == self._generation
            and based_on_seq == self._latest_final_seq
            and not self._active_utterances
        )

    @staticmethod
    def _intent_event(
        *,
        episode_id: str,
        based_on_seq: int,
        response: str,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "agent.intent",
            "source": "reasoner",
            "occurred_at": now,
            "started_at": now,
            "ended_at": now,
            "episode_id": episode_id,
            "payload": {
                "intent": "respond",
                "response": response,
                "based_on_utterance_seq": based_on_seq,
            },
        }

    @staticmethod
    def _failed_event(
        *,
        episode_id: str,
        based_on_seq: Any,
        error: Exception,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "reasoner.failed",
            "source": "reasoner",
            "occurred_at": now,
            "started_at": now,
            "ended_at": now,
            "episode_id": episode_id or None,
            "payload": {
                "based_on_utterance_seq": based_on_seq,
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        }
