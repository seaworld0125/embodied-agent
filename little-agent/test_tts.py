#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from tts import MacOSSayTTS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def intent_event(text: str = "안녕하세요") -> dict[str, Any]:
    now = now_iso()
    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": "agent.intent",
        "source": "reasoner",
        "occurred_at": now,
        "started_at": now,
        "ended_at": now,
        "episode_id": "episode-tts",
        "payload": {
            "intent": "respond",
            "response": text,
            "based_on_utterance_seq": 3,
        },
    }


def external_started() -> dict[str, Any]:
    now = now_iso()
    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": "speech.started",
        "source": "ear",
        "occurred_at": now,
        "started_at": now,
        "ended_at": None,
        "episode_id": "episode-tts",
        "utterance_seq": 4,
        "payload": {"utterance_id": "u4"},
    }


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self._done = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def complete(self, code: int = 0) -> None:
        if self.returncode is None:
            self.returncode = code
            self._done.set()

    def terminate(self) -> None:
        self.terminated = True
        self.complete(-15)

    def kill(self) -> None:
        self.killed = True
        self.complete(-9)


class FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []
        self.created = asyncio.Event()

    async def __call__(self, *args: Any, **kwargs: Any) -> FakeProcess:
        process = FakeProcess()
        self.calls.append((args, kwargs))
        self.processes.append(process)
        self.created.set()
        return process


async def wait_until(predicate, timeout: float = 0.5) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=timeout)


class TTSTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.events: list[dict[str, Any]] = []
        self.factory = FakeFactory()

        async def publish(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.service = MacOSSayTTS(
            event_queue=self.queue,
            publish=publish,
            command="/usr/bin/say",
            voice="Yuna",
            rate=180,
            process_factory=self.factory,
            force_kill_after_sec=0.05,
        )
        self.loop_task = asyncio.create_task(self.service.event_loop())

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.loop_task.cancel()
        await asyncio.gather(self.loop_task, return_exceptions=True)

    async def test_intent_produces_started_and_completed_events(self) -> None:
        intent = intent_event("테스트 음성")
        await self.queue.put(intent)
        await asyncio.wait_for(self.factory.created.wait(), timeout=0.2)
        await wait_until(
            lambda: any(e["type"] == "agent.speech.started" for e in self.events)
        )

        args, _ = self.factory.calls[0]
        self.assertEqual(args[0], "/usr/bin/say")
        self.assertIn("Yuna", args)
        self.assertIn("180", args)
        self.assertEqual(args[-1], "테스트 음성")

        self.factory.processes[0].complete(0)
        await wait_until(
            lambda: any(e["type"] == "agent.speech.ended" for e in self.events)
        )

        started = next(e for e in self.events if e["type"] == "agent.speech.started")
        ended = next(e for e in self.events if e["type"] == "agent.speech.ended")
        self.assertEqual(started["source"], "mouth")
        self.assertEqual(ended["payload"]["status"], "completed")
        self.assertEqual(
            started["payload"]["action_id"],
            ended["payload"]["action_id"],
        )
        self.assertEqual(ended["payload"]["intent_event_id"], intent["id"])
        self.assertEqual(ended["payload"]["based_on_utterance_seq"], 3)

    async def test_external_speech_started_interrupts_active_tts(self) -> None:
        await self.queue.put(intent_event("긴 응답"))
        await asyncio.wait_for(self.factory.created.wait(), timeout=0.2)
        await wait_until(lambda: self.service.is_speaking)

        process = self.factory.processes[0]
        await self.queue.put(external_started())

        await wait_until(
            lambda: any(e["type"] == "agent.speech.ended" for e in self.events)
        )
        ended = next(e for e in self.events if e["type"] == "agent.speech.ended")
        self.assertTrue(process.terminated)
        self.assertEqual(ended["payload"]["status"], "interrupted")
        self.assertEqual(
            ended["payload"]["interrupt_reason"],
            "external_speech_started",
        )

    async def test_disabled_tts_ignores_agent_intent(self) -> None:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        factory = FakeFactory()
        published: list[dict[str, Any]] = []

        async def publish(event: dict[str, Any]) -> None:
            published.append(event)

        service = MacOSSayTTS(
            event_queue=queue,
            publish=publish,
            enabled=False,
            process_factory=factory,
        )
        task = asyncio.create_task(service.event_loop())
        try:
            await queue.put(intent_event())
            await asyncio.sleep(0.03)
            self.assertEqual(factory.calls, [])
            self.assertEqual(published, [])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
