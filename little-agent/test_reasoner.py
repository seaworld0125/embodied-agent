#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import llm_client
from llm_client import (
    LLMRequestBroker,
    PRIORITY_REALTIME,
    PRIORITY_ROLLING,
)
from memory import MemoryStore
from reasoner import (
    REASONER_SYSTEM_PROMPT,
    ReasoningResult,
    RealtimeReasoner,
)
from turn import TurnCoordinator


BASE_TIME = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def iso(seconds: float = 0.0) -> str:
    return (BASE_TIME + timedelta(seconds=seconds)).isoformat()


def speech_event(
    event_type: str,
    seq: int,
    *,
    text: str = "",
    utterance_id: str | None = None,
    episode_id: str = "episode-1",
) -> dict:
    utterance_id = utterance_id or f"u{seq}"
    payload = {"utterance_id": utterance_id}
    if event_type == "speech.final":
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
        "occurred_at": iso(seq + 1),
        "started_at": iso(seq),
        "ended_at": None if event_type == "speech.started" else iso(seq + 1),
        "episode_id": episode_id,
        "episode_position": seq * 3,
        "utterance_seq": seq,
        "payload": payload,
    }


def persist_final(
    store: MemoryStore,
    episode_id: str,
    seq: int,
    text: str,
) -> dict:
    event = speech_event(
        "speech.final",
        seq,
        text=text,
        episode_id=episode_id,
    )
    store.persist_event(
        event=event,
        episode_id=episode_id,
        position=seq,
        utterance_seq=seq,
        touch_activity=False,
    )
    return event


class FakeBroker:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.response


class ReasonerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "agent.db")
        self.store.initialize()
        self.episode_id = "episode-reasoner"
        self.store.open_episode(self.episode_id, iso())

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_reasoner_uses_working_memory_plus_raw_tail_and_external_speaker(self) -> None:
        persist_final(self.store, self.episode_id, 0, "이전 발화")
        current = persist_final(self.store, self.episode_id, 1, "지금 질문이야")
        self.store.save_working_memory_cas(
            episode_id=self.episode_id,
            expected_version=0,
            upto_utterance_seq=0,
            summary="이전 발화까지만 압축됨",
            state={"topics": ["테스트"]},
        )

        broker = FakeBroker(
            '{"intent":"respond","response":"응답할게."}'
        )
        reasoner = RealtimeReasoner(
            self.store,
            broker,  # type: ignore[arg-type]
        )

        result = await reasoner.reason(current)

        self.assertEqual(result.intent, "respond")
        self.assertEqual(result.response, "응답할게.")
        self.assertEqual(result.based_on_utterance_seq, 1)
        self.assertEqual(len(broker.calls), 1)
        call = broker.calls[0]
        self.assertEqual(call["priority"], PRIORITY_REALTIME)
        self.assertIn(
            "이것은 에이전트 자신의 발화가 아니다",
            call["messages"][0]["content"],
        )
        self.assertIn(
            '"speaker_role": "external_speaker"',
            call["messages"][1]["content"],
        )
        self.assertIn("이전 발화까지만 압축됨", call["messages"][1]["content"])
        self.assertIn("지금 질문이야", call["messages"][1]["content"])

    def test_reasoner_prompt_contract_marks_ear_as_external(self) -> None:
        self.assertIn(
            "source=ear 또는 speaker_role=external_speaker",
            REASONER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "이것은 에이전트 자신의 발화가 아니다",
            REASONER_SYSTEM_PROMPT,
        )


class FakeReasoner:
    def __init__(
        self,
        *,
        intent: str = "respond",
        response: str = "테스트 응답",
        error: Exception | None = None,
        block: bool = False,
    ) -> None:
        self.intent = intent
        self.response = response
        self.error = error
        self.block = block
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def reason(self, event: dict) -> ReasoningResult:
        self.calls.append(event)
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return ReasoningResult(
            episode_id=str(event["episode_id"]),
            based_on_utterance_seq=int(event["utterance_seq"]),
            intent=self.intent,
            response=self.response if self.intent == "respond" else "",
        )


class TurnCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.published: list[dict] = []

        async def publish(event: dict) -> None:
            self.published.append(event)

        self.publish = publish
        self.tasks: list[asyncio.Task] = []

    async def asyncTearDown(self) -> None:
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def start(self, reasoner: FakeReasoner, grace_ms: int = 20) -> TurnCoordinator:
        coordinator = TurnCoordinator(
            reasoner=reasoner,  # type: ignore[arg-type]
            event_queue=self.queue,
            publish=self.publish,
            grace_ms=grace_ms,
        )
        task = asyncio.create_task(coordinator.event_loop())
        self.tasks.append(task)
        return coordinator

    async def test_new_speech_during_grace_cancels_reasoning(self) -> None:
        reasoner = FakeReasoner()
        self.start(reasoner, grace_ms=50)

        await self.queue.put(
            speech_event("speech.final", 0, text="첫 발화")
        )
        await asyncio.sleep(0.01)
        await self.queue.put(speech_event("speech.started", 1))
        await asyncio.sleep(0.08)

        self.assertEqual(len(reasoner.calls), 0)
        self.assertFalse(any(e["type"] == "agent.intent" for e in self.published))

    async def test_inflight_old_reasoning_result_is_discarded_after_new_speech(self) -> None:
        reasoner = FakeReasoner(block=True)
        self.start(reasoner, grace_ms=5)

        await self.queue.put(
            speech_event("speech.final", 0, text="첫 질문")
        )
        await asyncio.wait_for(reasoner.started.wait(), timeout=0.2)

        await self.queue.put(speech_event("speech.started", 1))
        await asyncio.sleep(0.01)
        reasoner.release.set()
        await asyncio.sleep(0.03)

        self.assertFalse(any(e["type"] == "agent.intent" for e in self.published))

    async def test_respond_publishes_agent_intent(self) -> None:
        reasoner = FakeReasoner(intent="respond", response="안녕.")
        self.start(reasoner, grace_ms=5)

        await self.queue.put(
            speech_event("speech.final", 0, text="안녕")
        )
        await asyncio.sleep(0.04)

        intents = [e for e in self.published if e["type"] == "agent.intent"]
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0]["payload"]["response"], "안녕.")
        self.assertEqual(
            intents[0]["payload"]["based_on_utterance_seq"],
            0,
        )

    async def test_wait_does_not_publish_agent_intent(self) -> None:
        reasoner = FakeReasoner(intent="wait")
        self.start(reasoner, grace_ms=5)

        await self.queue.put(
            speech_event("speech.final", 0, text="그냥 혼잣말")
        )
        await asyncio.sleep(0.04)

        self.assertFalse(any(e["type"] == "agent.intent" for e in self.published))

    async def test_reasoner_failure_becomes_event_without_killing_loop(self) -> None:
        reasoner = FakeReasoner(error=ValueError("bad json"))
        self.start(reasoner, grace_ms=5)

        await self.queue.put(
            speech_event("speech.final", 0, text="질문")
        )
        await asyncio.sleep(0.04)

        failures = [e for e in self.published if e["type"] == "reasoner.failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["payload"]["error_type"], "ValueError")


class LLMLaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_can_finish_while_background_request_is_running(self) -> None:
        broker = LLMRequestBroker(
            base_url="http://unused",
            model="test",
            realtime_concurrency=2,
        )
        background_started = threading.Event()
        release_background = threading.Event()

        def fake_post(
            base_url,
            model,
            messages,
            timeout_sec,
            temperature,
            max_tokens,
        ):
            marker = messages[0]["content"]
            if marker == "background":
                background_started.set()
                if not release_background.wait(timeout=2):
                    raise TimeoutError("test background was not released")
                return "background"
            if marker == "realtime":
                return "realtime"
            raise AssertionError(marker)

        with patch.object(
            llm_client,
            "post_chat_completion",
            side_effect=fake_post,
        ):
            worker = asyncio.create_task(broker.worker())
            try:
                background = asyncio.create_task(
                    broker.complete(
                        [{"role": "user", "content": "background"}],
                        priority=PRIORITY_ROLLING,
                        label="background",
                    )
                )

                started = await asyncio.to_thread(
                    background_started.wait,
                    1.0,
                )
                self.assertTrue(started)

                realtime = asyncio.create_task(
                    broker.complete(
                        [{"role": "user", "content": "realtime"}],
                        priority=PRIORITY_REALTIME,
                        label="realtime",
                    )
                )

                self.assertEqual(
                    await asyncio.wait_for(realtime, timeout=0.5),
                    "realtime",
                )
                self.assertFalse(background.done())

                release_background.set()
                self.assertEqual(
                    await asyncio.wait_for(background, timeout=0.5),
                    "background",
                )
            finally:
                release_background.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
