#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import llm_client
from consolidation import (
    FINAL_SYSTEM_PROMPT,
    build_final_prompt,
    fetch_unconsolidated_episode_ids,
)
from context import build_realtime_context, speaker_role
from episode import ActiveEpisodeManager
from llm_client import (
    LLMRequestBroker,
    PRIORITY_FINAL,
    PRIORITY_REALTIME,
    PRIORITY_ROLLING,
)
from memory import MemoryStore, persistence_worker
from rolling import (
    ROLLING_SYSTEM_PROMPT,
    RollingMemoryService,
    _EpisodeRollState,
)


BASE_TIME = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def iso(seconds: float = 0.0) -> str:
    return (BASE_TIME + timedelta(seconds=seconds)).isoformat()


def speech_event(
    event_type: str,
    utterance_id: str,
    at_sec: float,
    *,
    started_sec: float | None = None,
    text: str | None = None,
) -> dict:
    payload: dict = {"utterance_id": utterance_id}

    if text is not None:
        payload.update(
            {
                "text": text,
                "language": "ko",
                "duration_ms": 800,
                "queue_wait_ms": 0,
                "stt_latency_ms": 10,
            }
        )

    occurred_at = iso(at_sec)

    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": event_type,
        "source": "ear",
        "occurred_at": occurred_at,
        "started_at": iso(at_sec if started_sec is None else started_sec),
        "ended_at": None if event_type == "speech.started" else occurred_at,
        "payload": payload,
    }


def persist_final(
    store: MemoryStore,
    episode_id: str,
    seq: int,
    text: str,
    *,
    position: int | None = None,
) -> None:
    event = speech_event(
        "speech.final",
        f"u{seq}",
        at_sec=float(seq + 1),
        started_sec=float(seq),
        text=text,
    )

    store.persist_event(
        event=event,
        episode_id=episode_id,
        position=seq if position is None else position,
        utterance_seq=seq,
        touch_activity=False,
    )


class FakeBroker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return """
        {
          "summary": "외부 화자가 에이전트의 기억 구조를 논의하고 있다.",
          "topics": ["기억", "에이전트"],
          "provisional_facts": [
            {
              "text": "외부 화자가 working memory 구조를 논의했다.",
              "confidence": 0.95
            }
          ],
          "open_threads": ["다음 구현 단계"]
        }
        """


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "agent.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_store_connection_context_closes_handle(self) -> None:
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)

        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_working_memory_cas_rejects_stale_result(self) -> None:
        episode_id = "episode-cas"
        self.store.open_episode(episode_id, iso())

        saved = self.store.save_working_memory_cas(
            episode_id=episode_id,
            expected_version=0,
            upto_utterance_seq=2,
            summary="최신 요약",
            state={"summary": "최신 요약"},
        )
        stale_saved = self.store.save_working_memory_cas(
            episode_id=episode_id,
            expected_version=0,
            upto_utterance_seq=3,
            summary="늦게 도착한 오래된 결과",
            state={"summary": "늦게 도착한 오래된 결과"},
        )

        self.assertTrue(saved)
        self.assertFalse(stale_saved)

        working = self.store.get_working_memory(episode_id)
        self.assertEqual(working["version"], 1)
        self.assertEqual(working["upto_utterance_seq"], 2)
        self.assertEqual(working["summary"], "최신 요약")

    def test_realtime_context_keeps_raw_tail_when_working_memory_is_stale(self) -> None:
        episode_id = "episode-context"
        self.store.open_episode(episode_id, iso())

        persist_final(self.store, episode_id, 0, "첫 번째 발화")
        persist_final(self.store, episode_id, 1, "두 번째 최신 발화")
        persist_final(self.store, episode_id, 2, "세 번째 최신 발화")

        saved = self.store.save_working_memory_cas(
            episode_id=episode_id,
            expected_version=0,
            upto_utterance_seq=0,
            summary="첫 번째 발화까지만 압축됨",
            state={"topics": ["테스트"]},
        )
        self.assertTrue(saved)

        context = build_realtime_context(self.store, episode_id)

        self.assertTrue(context["working_memory_is_stale"])
        self.assertEqual(
            context["working_memory"]["authoritative_through_utterance_seq"],
            0,
        )
        self.assertEqual(
            [item["utterance_seq"] for item in context["raw_tail"]],
            [1, 2],
        )
        self.assertEqual(
            context["raw_tail"][0]["speaker_role"],
            "external_speaker",
        )
        self.assertIn("overrides", context["precedence_rule"])

    def test_ear_is_external_speaker_not_agent(self) -> None:
        self.assertEqual(speaker_role("ear"), "external_speaker")
        self.assertEqual(speaker_role("agent"), "agent")
        self.assertIn(
            "이것은 에이전트 자신의 발화가 아니다",
            ROLLING_SYSTEM_PROMPT,
        )
        self.assertIn(
            "이것은 에이전트 자신의 발화가 아니다",
            FINAL_SYSTEM_PROMPT,
        )

    def test_final_consolidation_only_sees_ready_episodes(self) -> None:
        not_ready = "episode-not-ready"
        ready = "episode-ready"

        self.store.open_episode(not_ready, iso())
        self.store.close_episode(not_ready, iso(10), False)

        self.store.open_episode(ready, iso(20))
        self.store.close_episode(ready, iso(30), True)

        ids = fetch_unconsolidated_episode_ids(self.store)
        self.assertNotIn(not_ready, ids)
        self.assertIn(ready, ids)

        self.store.mark_episode_ready(not_ready)
        ids = fetch_unconsolidated_episode_ids(self.store)
        self.assertIn(not_ready, ids)

    def test_final_prompt_marks_ear_tail_as_external_speaker(self) -> None:
        episode_id = "episode-final-prompt"
        self.store.open_episode(episode_id, iso())
        persist_final(self.store, episode_id, 0, "내가 들려준 외부 발화")

        episode = self.store.get_episode(episode_id)
        working = self.store.get_working_memory(episode_id)
        tail = self.store.episode_final_utterances(episode_id, after_seq=-1)

        assert episode is not None
        prompt = build_final_prompt(episode, working, tail)

        self.assertIn('"speaker_role": "external_speaker"', prompt)
        self.assertNotIn('"speaker_role": "agent"', prompt)


class EpisodeLifecycleTests(unittest.IsolatedAsyncioTestCase):
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
        with self.assertRaises(asyncio.CancelledError):
            await self.persistence_task
        self.temp.cleanup()

    async def test_delayed_final_stays_in_original_episode_and_unlocks_consolidation(self) -> None:
        utterance_id = "u1"

        await self.manager.handle(speech_event("speech.started", utterance_id, 0))
        await self.manager.handle(
            speech_event("speech.ended", utterance_id, 1, started_sec=0)
        )

        episode_id = self.manager.active.id
        await self.persistence_queue.join()

        state = self.manager.active
        assert state is not None

        # Close the episode while STT is still pending.
        await self.manager._close_state(state)
        await self.persistence_queue.join()

        row = self.store.get_episode(episode_id)
        assert row is not None
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["ready_for_consolidation"], 0)

        # The transcript arrives after the episode is already closed.
        await self.manager.handle(
            speech_event(
                "speech.final",
                utterance_id,
                1,
                started_sec=0,
                text="늦게 도착한 STT 결과",
            )
        )
        await self.persistence_queue.join()

        row = self.store.get_episode(episode_id)
        assert row is not None
        self.assertEqual(row["ready_for_consolidation"], 1)

        finals = [
            event
            for event in self.store.episode_events(episode_id)
            if event["type"] == "speech.final"
        ]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["utterance_seq"], 0)
        self.assertEqual(finals[0]["payload"]["text"], "늦게 도착한 STT 결과")

    async def test_late_final_does_not_jump_into_new_episode(self) -> None:
        await self.manager.handle(speech_event("speech.started", "u1", 0))
        old_episode_id = self.manager.active.id
        await self.manager.handle(speech_event("speech.ended", "u1", 1, started_sec=0))

        old_state = self.manager.active
        assert old_state is not None
        await self.manager._close_state(old_state)

        await self.manager.handle(speech_event("speech.started", "u2", 20))
        new_episode_id = self.manager.active.id
        self.assertNotEqual(old_episode_id, new_episode_id)

        # U1 finishes recognition after U2 already opened a new episode.
        await self.manager.handle(
            speech_event(
                "speech.final",
                "u1",
                1,
                started_sec=0,
                text="첫 에피소드의 늦은 결과",
            )
        )
        await self.persistence_queue.join()

        old_finals = self.store.episode_final_utterances(old_episode_id)
        new_finals = self.store.episode_final_utterances(new_episode_id)

        self.assertEqual(
            [item["payload"]["text"] for item in old_finals],
            ["첫 에피소드의 늦은 결과"],
        )
        self.assertEqual(new_finals, [])

    async def test_utterance_sequence_is_assigned_at_speech_start(self) -> None:
        for seq, utterance_id in enumerate(("u1", "u2", "u3")):
            await self.manager.handle(
                speech_event("speech.started", utterance_id, seq * 2)
            )
            await self.manager.handle(
                speech_event(
                    "speech.ended",
                    utterance_id,
                    seq * 2 + 1,
                    started_sec=seq * 2,
                )
            )

        self.assertEqual(
            self.manager.utterance_seq,
            {"u1": 0, "u2": 1, "u3": 2},
        )

        started = [
            event for event in self.published if event["type"] == "speech.started"
        ]
        self.assertEqual(
            [event["utterance_seq"] for event in started],
            [0, 1, 2],
        )


class RollingMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "agent.db")
        self.store.initialize()

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_rolling_updates_only_through_target_sequence(self) -> None:
        episode_id = "episode-roll"
        self.store.open_episode(episode_id, iso())
        persist_final(self.store, episode_id, 0, "첫 번째")
        persist_final(self.store, episode_id, 1, "두 번째")
        persist_final(self.store, episode_id, 2, "세 번째")

        broker = FakeBroker()
        service = RollingMemoryService(
            store=self.store,
            broker=broker,  # type: ignore[arg-type]
            event_queue=asyncio.Queue(),
            batch_size=3,
            max_delay_sec=8.0,
        )

        result = await service._roll_episode(episode_id, target_seq=2)
        self.assertTrue(result)

        working = self.store.get_working_memory(episode_id)
        self.assertEqual(working["upto_utterance_seq"], 2)
        self.assertEqual(working["version"], 1)
        self.assertIn("외부 화자", working["summary"])

        self.assertEqual(len(broker.calls), 1)
        call = broker.calls[0]
        self.assertEqual(call["priority"], PRIORITY_ROLLING)
        self.assertIn(
            "이것은 에이전트 자신의 발화가 아니다",
            call["messages"][0]["content"],
        )
        self.assertIn(
            '"speaker_role": "external_speaker"',
            call["messages"][1]["content"],
        )

    async def test_rolling_request_is_single_flight_and_coalesced(self) -> None:
        service = RollingMemoryService(
            store=self.store,
            broker=FakeBroker(),  # type: ignore[arg-type]
            event_queue=asyncio.Queue(),
        )
        state = _EpisodeRollState(latest_seq=5, pending_count=5)

        service._request("episode-1", state)
        service._request("episode-1", state)

        self.assertTrue(state.queued)
        self.assertEqual(service.jobs.qsize(), 1)

        state.queued = False
        state.in_flight = True
        service._request("episode-1", state)
        self.assertEqual(service.jobs.qsize(), 1)


class LLMBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_priority_rolling_before_final(self) -> None:
        broker = LLMRequestBroker(base_url="http://unused", model="test")
        call_order: list[str] = []

        def fake_post(
            base_url,
            model,
            messages,
            timeout_sec,
            temperature,
            max_tokens,
            enable_thinking=None,
        ):
            marker = messages[0]["content"]
            call_order.append(marker)
            return marker

        with patch.object(
            llm_client,
            "post_chat_completion",
            side_effect=fake_post,
        ):
            final_task = asyncio.create_task(
                broker.complete(
                    [{"role": "user", "content": "final"}],
                    priority=PRIORITY_FINAL,
                    label="final",
                )
            )
            rolling_task = asyncio.create_task(
                broker.complete(
                    [{"role": "user", "content": "rolling"}],
                    priority=PRIORITY_ROLLING,
                    label="rolling",
                )
            )

            await asyncio.sleep(0)
            self.assertEqual(broker.queue.qsize(), 2)

            worker = asyncio.create_task(broker.worker())
            results = await asyncio.gather(
                rolling_task,
                final_task,
            )

            self.assertEqual(results, ["rolling", "final"])
            self.assertEqual(call_order, ["rolling", "final"])

            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker


if __name__ == "__main__":
    unittest.main(verbosity=2)
