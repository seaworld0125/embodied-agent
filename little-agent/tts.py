#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4


PublishFn = Callable[[dict[str, Any]], Awaitable[None]]


class ProcessLike(Protocol):
    returncode: int | None

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[..., Awaitable[ProcessLike]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class ActiveSpeech:
    action_id: str
    episode_id: str | None
    intent_event_id: str
    based_on_utterance_seq: int | None
    text: str
    started_at: str
    started_monotonic: float
    process: ProcessLike
    interrupt_reason: str | None = None


class MacOSSayTTS:
    """Realtime TTS adapter around macOS `say`.

    The service consumes the shared event stream. `agent.intent` starts speech.
    An external `speech.started` event immediately interrupts active speech so
    the user can barge in. Speech lifecycle events are published back through
    the supplied durable publish function.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue[dict[str, Any]],
        publish: PublishFn,
        *,
        command: str = "/usr/bin/say",
        voice: str | None = None,
        rate: int | None = None,
        enabled: bool = True,
        process_factory: ProcessFactory | None = None,
        force_kill_after_sec: float = 0.5,
    ) -> None:
        self.event_queue = event_queue
        self.publish = publish
        self.command = command
        self.voice = voice
        self.rate = rate
        self.enabled = enabled
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self.force_kill_after_sec = max(0.05, force_kill_after_sec)

        self._generation = 0
        self._active: ActiveSpeech | None = None
        self._speech_tasks: set[asyncio.Task[None]] = set()
        self._kill_tasks: set[asyncio.Task[None]] = set()

    @property
    def is_speaking(self) -> bool:
        return self._active is not None and self._active.process.returncode is None

    async def event_loop(self) -> None:
        while True:
            event = await self.event_queue.get()
            try:
                event_type = event.get("type")

                if event_type == "agent.intent":
                    await self._on_intent(event)
                elif event_type == "speech.started" and event.get("source") == "ear":
                    await self._on_external_speech_started(event)
            finally:
                self.event_queue.task_done()

    async def _on_intent(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        payload = event.get("payload", {})
        if payload.get("intent") != "respond":
            return

        text = str(payload.get("response", "")).strip()
        if not text:
            return

        # A newer intent supersedes any speech that is still playing.
        self._generation += 1
        generation = self._generation
        await self.interrupt("superseded_by_new_intent")

        task = asyncio.create_task(
            self._speak(event, text, generation),
            name=f"tts:{str(event.get('id', ''))[:8]}",
        )
        self._speech_tasks.add(task)
        task.add_done_callback(self._speech_tasks.discard)

    async def _on_external_speech_started(self, event: dict[str, Any]) -> None:
        if not self.enabled or not self.is_speaking:
            return

        self._generation += 1
        utterance_id = event.get("payload", {}).get("utterance_id")
        log(f"[tts] barge-in utterance={utterance_id}")
        await self.interrupt("external_speech_started")

    async def _speak(
        self,
        intent_event: dict[str, Any],
        text: str,
        generation: int,
    ) -> None:
        if generation != self._generation:
            return

        action_id = str(uuid4())
        episode_id_raw = intent_event.get("episode_id")
        episode_id = str(episode_id_raw) if episode_id_raw else None
        payload = intent_event.get("payload", {})
        based_on = payload.get("based_on_utterance_seq")
        based_on_seq = int(based_on) if isinstance(based_on, int) else None

        args = [self.command]
        if self.voice:
            args.extend(["-v", self.voice])
        if self.rate is not None:
            args.extend(["-r", str(self.rate)])
        args.append(text)

        try:
            process = await self.process_factory(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            await self.publish(
                self._failed_event(
                    action_id=action_id,
                    episode_id=episode_id,
                    intent_event_id=str(intent_event.get("id", "")),
                    based_on_seq=based_on_seq,
                    text=text,
                    started_at=None,
                    error=exc,
                )
            )
            return

        if generation != self._generation:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            return

        started_at = utc_now_iso()
        active = ActiveSpeech(
            action_id=action_id,
            episode_id=episode_id,
            intent_event_id=str(intent_event.get("id", "")),
            based_on_utterance_seq=based_on_seq,
            text=text,
            started_at=started_at,
            started_monotonic=time.monotonic(),
            process=process,
        )
        self._active = active

        await self.publish(self._started_event(active))
        log(
            f"[tts] start action={action_id[:8]} "
            f"episode={(episode_id or '-')[:8]} chars={len(text)}"
        )

        return_code = await process.wait()
        ended_at = utc_now_iso()
        duration_ms = int(
            max(0.0, time.monotonic() - active.started_monotonic) * 1000
        )

        if self._active is active:
            self._active = None

        if active.interrupt_reason is not None:
            await self.publish(
                self._ended_event(
                    active,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                    return_code=return_code,
                    status="interrupted",
                    interrupt_reason=active.interrupt_reason,
                )
            )
            log(
                f"[tts] interrupted action={action_id[:8]} "
                f"reason={active.interrupt_reason}"
            )
            return

        if return_code == 0:
            await self.publish(
                self._ended_event(
                    active,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                    return_code=return_code,
                    status="completed",
                    interrupt_reason=None,
                )
            )
            log(f"[tts] end action={action_id[:8]} duration={duration_ms}ms")
            return

        await self.publish(
            self._failed_event(
                action_id=active.action_id,
                episode_id=active.episode_id,
                intent_event_id=active.intent_event_id,
                based_on_seq=active.based_on_utterance_seq,
                text=active.text,
                started_at=active.started_at,
                error=RuntimeError(f"say exited with code {return_code}"),
                ended_at=ended_at,
                duration_ms=duration_ms,
                return_code=return_code,
            )
        )

    async def interrupt(self, reason: str) -> None:
        active = self._active
        if active is None or active.process.returncode is not None:
            return

        if active.interrupt_reason is None:
            active.interrupt_reason = reason

        try:
            active.process.terminate()
        except ProcessLookupError:
            return

        task = asyncio.create_task(
            self._force_kill(active.process),
            name=f"tts-force-kill:{active.action_id[:8]}",
        )
        self._kill_tasks.add(task)
        task.add_done_callback(self._kill_tasks.discard)

    async def _force_kill(self, process: ProcessLike) -> None:
        await asyncio.sleep(self.force_kill_after_sec)
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    async def close(self) -> None:
        self._generation += 1
        await self.interrupt("shutdown")

        tasks = list(self._speech_tasks)
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.force_kill_after_sec + 0.5,
                )
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        for task in list(self._kill_tasks):
            if not task.done():
                task.cancel()
        if self._kill_tasks:
            await asyncio.gather(*self._kill_tasks, return_exceptions=True)

    def _common_payload(self, active: ActiveSpeech) -> dict[str, Any]:
        return {
            "action_id": active.action_id,
            "intent_event_id": active.intent_event_id,
            "based_on_utterance_seq": active.based_on_utterance_seq,
            "text": active.text,
            "voice": self.voice,
            "rate": self.rate,
            "engine": "macos.say",
        }

    def _started_event(self, active: ActiveSpeech) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "agent.speech.started",
            "source": "mouth",
            "occurred_at": active.started_at,
            "started_at": active.started_at,
            "ended_at": None,
            "episode_id": active.episode_id,
            "payload": self._common_payload(active),
        }

    def _ended_event(
        self,
        active: ActiveSpeech,
        *,
        ended_at: str,
        duration_ms: int,
        return_code: int,
        status: str,
        interrupt_reason: str | None,
    ) -> dict[str, Any]:
        payload = self._common_payload(active)
        payload.update(
            {
                "status": status,
                "duration_ms": duration_ms,
                "return_code": return_code,
                "interrupt_reason": interrupt_reason,
            }
        )
        return {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "agent.speech.ended",
            "source": "mouth",
            "occurred_at": ended_at,
            "started_at": active.started_at,
            "ended_at": ended_at,
            "episode_id": active.episode_id,
            "payload": payload,
        }

    def _failed_event(
        self,
        *,
        action_id: str,
        episode_id: str | None,
        intent_event_id: str,
        based_on_seq: int | None,
        text: str,
        started_at: str | None,
        error: Exception,
        ended_at: str | None = None,
        duration_ms: int | None = None,
        return_code: int | None = None,
    ) -> dict[str, Any]:
        now = ended_at or utc_now_iso()
        return {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "agent.speech.failed",
            "source": "mouth",
            "occurred_at": now,
            "started_at": started_at,
            "ended_at": now,
            "episode_id": episode_id,
            "payload": {
                "action_id": action_id,
                "intent_event_id": intent_event_id,
                "based_on_utterance_seq": based_on_seq,
                "text": text,
                "voice": self.voice,
                "rate": self.rate,
                "engine": "macos.say",
                "duration_ms": duration_ms,
                "return_code": return_code,
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        }
