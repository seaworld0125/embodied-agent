#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from consolidation import final_consolidation_worker
from episode import ActiveEpisodeManager
from llm_client import LLMRequestBroker
from memory import MemoryStore, persistence_worker
from reasoner import RealtimeReasoner
from rolling import RollingMemoryService
from turn import TurnCoordinator


@dataclass(frozen=True)
class Subscription:
    name: str
    queue: asyncio.Queue[dict[str, Any]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscription] = []

    def subscribe(
        self,
        name: str,
        maxsize: int = 256,
    ) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(Subscription(name=name, queue=queue))
        return queue

    async def publish(self, event: dict[str, Any]) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                print(
                    f"[core] dropping event for slow subscriber={subscriber.name}",
                    file=sys.stderr,
                    flush=True,
                )


async def pipe_ear_stderr(stream: asyncio.StreamReader) -> None:
    while True:
        line = await stream.readline()
        if not line:
            return
        print(
            line.decode("utf-8", errors="replace").rstrip(),
            file=sys.stderr,
            flush=True,
        )


async def read_ear_events(
    stream: asyncio.StreamReader,
    episode_manager: ActiveEpisodeManager,
) -> None:
    while True:
        line = await stream.readline()
        if not line:
            return

        raw = line.decode("utf-8", errors="replace").strip()
        if not raw:
            continue

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            print(
                f"[core] invalid JSON from ear: {raw!r}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if not isinstance(event, dict) or "type" not in event:
            continue

        # Realtime path: RAM mutation + put_nowait queues only.
        await episode_manager.handle(event)


async def diagnostics_worker(
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    while True:
        event = await queue.get()
        try:
            event_type = event.get("type")
            payload = event.get("payload", {})
            episode_id = event.get("episode_id")
            episode = episode_id[:8] if episode_id else "-"
            position = event.get("episode_position", "-")
            seq = event.get("utterance_seq", "-")

            if event_type == "speech.started":
                print(
                    f"[core][speech:start] episode={episode} "
                    f"seq={seq} pos={position} "
                    f"utterance={payload.get('utterance_id')}",
                    flush=True,
                )
            elif event_type == "speech.ended":
                print(
                    f"[core][speech:end] episode={episode} "
                    f"seq={seq} pos={position} "
                    f"duration={payload.get('duration_ms')}ms",
                    flush=True,
                )
            elif event_type in ("speech.final", "speech"):
                print(
                    f"[core][speech:final] episode={episode} "
                    f"seq={seq} pos={position} "
                    f'"{payload.get("text", "")}" '
                    f"stt={payload.get('stt_latency_ms')}ms "
                    f"queue={payload.get('queue_wait_ms')}ms",
                    flush=True,
                )
            elif event_type == "speech.failed":
                print(
                    f"[core][speech:failed] episode={episode} "
                    f"seq={seq} pos={position} "
                    f"reason={payload.get('reason')}",
                    flush=True,
                )
            elif event_type == "agent.intent":
                print(
                    f"[core][agent:intent] episode={episode} "
                    f"based_on={payload.get('based_on_utterance_seq')} "
                    f'"{payload.get("response", "")}"',
                    flush=True,
                )
            elif event_type == "reasoner.failed":
                print(
                    f"[core][reasoner:failed] episode={episode} "
                    f"based_on={payload.get('based_on_utterance_seq')} "
                    f"{payload.get('error_type')}: {payload.get('error')}",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            queue.task_done()


def build_ear_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(args.ear).expanduser().resolve()),
        "--whisper",
        str(Path(args.whisper).expanduser().resolve()),
        "--model",
        str(Path(args.model).expanduser().resolve()),
        "--language",
        args.language,
        "--threads",
        str(args.threads),
        "--vad-threshold",
        str(args.vad_threshold),
        "--min-silence-ms",
        str(args.min_silence_ms),
        "--speech-pad-ms",
        str(args.speech_pad_ms),
        "--max-utterance-sec",
        str(args.max_utterance_sec),
        "--stt-queue-max",
        str(args.stt_queue_max),
    ]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    return cmd


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ear", default="./ear.py")
    parser.add_argument(
        "--whisper",
        default=str(home / "whisper.cpp/build/bin/whisper-cli"),
    )
    parser.add_argument(
        "--model",
        default=str(home / "whisper.cpp/models/ggml-small.bin"),
    )
    parser.add_argument("--memory-db", default="./data/agent.db")
    parser.add_argument("--episode-idle-sec", type=float, default=15.0)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=int, default=600)
    parser.add_argument("--speech-pad-ms", type=int, default=200)
    parser.add_argument("--max-utterance-sec", type=float, default=20.0)
    parser.add_argument("--stt-queue-max", type=int, default=8)

    parser.add_argument("--llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--llm-model", default="local")
    parser.add_argument("--llm-realtime-concurrency", type=int, default=2)

    parser.add_argument("--turn-grace-ms", type=int, default=500)
    parser.add_argument("--reasoner-temperature", type=float, default=0.4)
    parser.add_argument("--reasoner-max-tokens", type=int, default=512)
    parser.add_argument("--reasoner-timeout-sec", type=float, default=60.0)

    parser.add_argument("--rolling-batch", type=int, default=3)
    parser.add_argument("--rolling-delay-sec", type=float, default=8.0)
    parser.add_argument("--final-poll-sec", type=float, default=3.0)
    return parser


async def run(args: argparse.Namespace) -> None:
    store = MemoryStore(args.memory_db)
    await asyncio.to_thread(store.initialize)

    print(f"[core] memory db: {store.db_path}", file=sys.stderr, flush=True)
    print(
        f"[core] episode idle timeout: {args.episode_idle_sec}s",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[core] turn grace: {args.turn_grace_ms}ms",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[core] rolling: batch={args.rolling_batch} "
        f"delay={args.rolling_delay_sec}s",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[core] LLM lanes: realtime + background. "
        "For real concurrency, run llama-server with -np 2 or more.",
        file=sys.stderr,
        flush=True,
    )

    bus = EventBus()
    diagnostics_queue = bus.subscribe("diagnostics")
    rolling_event_queue = bus.subscribe("rolling", maxsize=512)
    turn_event_queue = bus.subscribe("turns", maxsize=512)

    persistence_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    episode_manager = ActiveEpisodeManager(
        store=store,
        persistence_queue=persistence_queue,
        publish=bus.publish,
        idle_timeout_sec=args.episode_idle_sec,
    )
    await episode_manager.recover()

    broker = LLMRequestBroker(
        base_url=args.llm_url,
        model=args.llm_model,
        realtime_concurrency=args.llm_realtime_concurrency,
    )

    reasoner = RealtimeReasoner(
        store=store,
        broker=broker,
        temperature=args.reasoner_temperature,
        max_tokens=args.reasoner_max_tokens,
        timeout_sec=args.reasoner_timeout_sec,
    )
    turns = TurnCoordinator(
        reasoner=reasoner,
        event_queue=turn_event_queue,
        publish=bus.publish,
        grace_ms=args.turn_grace_ms,
    )

    rolling = RollingMemoryService(
        store=store,
        broker=broker,
        event_queue=rolling_event_queue,
        batch_size=args.rolling_batch,
        max_delay_sec=args.rolling_delay_sec,
    )

    process = await asyncio.create_subprocess_exec(
        *build_ear_command(args),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    tasks = [
        asyncio.create_task(
            persistence_worker(persistence_queue, store),
            name="persistence",
        ),
        asyncio.create_task(
            episode_manager.idle_loop(),
            name="episode-idle",
        ),
        asyncio.create_task(
            read_ear_events(process.stdout, episode_manager),
            name="ear-events",
        ),
        asyncio.create_task(
            pipe_ear_stderr(process.stderr),
            name="ear-stderr",
        ),
        asyncio.create_task(
            diagnostics_worker(diagnostics_queue),
            name="diagnostics",
        ),
        asyncio.create_task(
            broker.worker(),
            name="llm-background-lane",
        ),
        asyncio.create_task(
            turns.event_loop(),
            name="turn-coordinator",
        ),
        asyncio.create_task(
            rolling.event_loop(),
            name="rolling-events",
        ),
        asyncio.create_task(
            rolling.timer_loop(),
            name="rolling-timer",
        ),
        asyncio.create_task(
            rolling.worker_loop(),
            name="rolling-worker",
        ),
        asyncio.create_task(
            final_consolidation_worker(
                store,
                broker,
                poll_interval_sec=args.final_poll_sec,
            ),
            name="final-consolidation",
        ),
    ]

    try:
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(f"ear process exited with code {return_code}")
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[core] stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
