#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from episode import ActiveEpisodeManager
from memory import MemoryStore, persistence_worker


BASE_TIME = datetime(2026, 9, 20, 6, 0, 0, tzinfo=timezone.utc)


def iso(seconds: float = 0.0) -> str:
    return (BASE_TIME + timedelta(seconds=seconds)).isoformat()


def speech_event(
    event_type: str,
    utterance_id: str,
    at_sec: float,
    *,
    text: str | None = None,
) -> dict:
    payload: dict = {"utterance_id": utterance_id}
    if text is not None:
        payload.update(
            {
                "text": text,
                "language": "ko",
                "duration_ms": 500,
                "queue_wait_ms": 0,
                "stt_latency_ms": 10,
            }
        )

    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": event_type,
        "source": "ear",
        "occurred_at": iso(at_sec),
        "started_at": iso(max(0.0, at_sec - 0.5)),
        "ended_at": None if event_type == "speech.started" else iso(at_sec),
        "payload": payload,
    }


def agent_intent(episode_id: str, based_on: int = 0) -> dict:
    now = iso(2.0)
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
            "response": "테스트 응답",
            "based_on_utterance_seq": based_on,
        },
    }


def agent_speech_event(
    event_type: str,
    episode_id: str,
    action_id: str,
    at_sec: float,
    *,
    status: str | None = None,
) -> dict:
    now = iso(at_sec)
    payload = {
        "action_id": action_id,
        "intent_event_id": "intent-1",
        "based_on_utterance_seq": 0,
        "text": "테스트 응답",
        "engine": "macos.say",
    }
    if status is not None:
        payload["status"] = status
    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": event_type,
        "source": "mouth",
        "occurred_at": now,
        "started_at": now,
        "ended_at": None if event_type == "agent.speech.started" else now,
        "episode_id": episode_id,
        "payload": payload,
    }


class ReasonerPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "agent.db")
        self.store.initialize()

        self.persistence_queue: asyncio.Queue[dict] = asyncio.Queue()
        self.published: list[dict] = []

        async def publish(event: dict) -> None:
            self.published.append(event)

        self.manager = ActiveEpisodeManager(
            store=self.store,
            persistence_queue=self.persistence_queue,
            publish=publish,
            idle_timeout_sec=15.0,
        )
        self.persistence_task = asyncio.create_task(
            persistence_worker(self.persistence_queue, self.store)
        )

    async def asyncTearDown(self) -> None:
        await self.persistence_queue.join()
        self.persistence_task.cancel()
        await asyncio.gather(self.persistence_task, return_exceptions=True)
        self.temp.cleanup()

    async def _make_completed_utterance(self) -> str:
        await self.manager.handle(
            speech_event("speech.started", "u0", 0.0)
        )
        episode_id = self.manager.active.id  # type: ignore[union-attr]
        await self.manager.handle(
            speech_event("speech.ended", "u0", 1.0)
        )
        await self.manager.handle(
            speech_event("speech.final", "u0", 1.0, text="질문")
        )
        await self.persistence_queue.join()
        return episode_id

    async def test_agent_intent_is_persisted_in_active_episode(self) -> None:
        episode_id = await self._make_completed_utterance()
        before = self.store.get_episode(episode_id)
        assert before is not None
        last_activity_before = before["last_event_at"]

        event = agent_intent(episode_id)
        await self.manager.publish_internal_event(event)
        await self.persistence_queue.join()

        events = self.store.episode_events(episode_id)
        intents = [item for item in events if item["type"] == "agent.intent"]

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0]["source"], "reasoner")
        self.assertEqual(intents[0]["payload"]["response"], "테스트 응답")
        self.assertIsNone(intents[0]["utterance_seq"])

        after = self.store.get_episode(episode_id)
        assert after is not None
        self.assertEqual(after["last_event_at"], last_activity_before)
        self.assertGreater(after["event_count"], before["event_count"])

        published = [e for e in self.published if e["type"] == "agent.intent"]
        self.assertEqual(len(published), 1)
        self.assertIn("episode_position", published[0])

    async def test_late_agent_intent_is_attached_after_runtime_is_released(self) -> None:
        episode_id = await self._make_completed_utterance()
        state = self.manager.active
        assert state is not None

        await self.manager._close_state(state)
        await self.persistence_queue.join()

        self.assertNotIn(episode_id, self.manager.states)
        closed = self.store.get_episode(episode_id)
        assert closed is not None
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["ready_for_consolidation"], 1)

        before_positions = [e["position"] for e in self.store.episode_events(episode_id)]
        event = agent_intent(episode_id)
        await self.manager.publish_internal_event(event)
        await self.persistence_queue.join()

        events = self.store.episode_events(episode_id)
        intents = [item for item in events if item["type"] == "agent.intent"]
        self.assertEqual(len(intents), 1)
        self.assertEqual(
            intents[0]["position"],
            max(before_positions) + 1,
        )

        after = self.store.get_episode(episode_id)
        assert after is not None
        self.assertEqual(after["status"], "closed")
        self.assertEqual(after["last_event_at"], closed["last_event_at"])


    async def test_agent_speech_is_persisted_and_touches_episode_activity(self) -> None:
        episode_id = await self._make_completed_utterance()
        before = self.store.get_episode(episode_id)
        assert before is not None

        action_id = "action-1"
        started = agent_speech_event(
            "agent.speech.started", episode_id, action_id, 5.0
        )
        await self.manager.publish_internal_event(started)
        await self.persistence_queue.join()

        state = self.manager.states[episode_id]
        self.assertIn(action_id, state.active_agent_actions)
        after_start = self.store.get_episode(episode_id)
        assert after_start is not None
        self.assertEqual(after_start["last_event_at"], iso(5.0))

        ended = agent_speech_event(
            "agent.speech.ended",
            episode_id,
            action_id,
            7.0,
            status="completed",
        )
        await self.manager.publish_internal_event(ended)
        await self.persistence_queue.join()

        self.assertNotIn(action_id, state.active_agent_actions)
        after_end = self.store.get_episode(episode_id)
        assert after_end is not None
        self.assertEqual(after_end["last_event_at"], iso(7.0))

        events = self.store.episode_events(episode_id)
        types = [e["type"] for e in events]
        self.assertIn("agent.speech.started", types)
        self.assertIn("agent.speech.ended", types)

    async def test_active_agent_speech_prevents_idle_episode_close(self) -> None:
        self.manager.idle_timeout_sec = 0.05
        episode_id = await self._make_completed_utterance()
        action_id = "action-hold"

        idle_task = asyncio.create_task(self.manager.idle_loop())
        try:
            await self.manager.publish_internal_event(
                agent_speech_event(
                    "agent.speech.started", episode_id, action_id, 5.0
                )
            )
            await self.persistence_queue.join()
            await asyncio.sleep(0.09)
            self.assertIsNotNone(self.manager.active)
            self.assertIn(
                action_id,
                self.manager.states[episode_id].active_agent_actions,
            )

            await self.manager.publish_internal_event(
                agent_speech_event(
                    "agent.speech.ended",
                    episode_id,
                    action_id,
                    7.0,
                    status="completed",
                )
            )
            await self.persistence_queue.join()
            await asyncio.sleep(0.09)
            self.assertIsNone(self.manager.active)
        finally:
            idle_task.cancel()
            await asyncio.gather(idle_task, return_exceptions=True)

    async def test_reasoner_failed_without_episode_is_still_preserved_raw(self) -> None:
        now = iso(3.0)
        event = {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "reasoner.failed",
            "source": "reasoner",
            "occurred_at": now,
            "started_at": now,
            "ended_at": now,
            "episode_id": None,
            "payload": {
                "error_type": "ValueError",
                "error": "bad json",
            },
        }

        await self.manager.publish_internal_event(event)
        await self.persistence_queue.join()

        raw = self.store.recent_events(20)
        failures = [item for item in raw if item["type"] == "reasoner.failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["payload"]["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main(verbosity=2)
