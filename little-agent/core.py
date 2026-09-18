#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory import MemoryStore, memory_worker


@dataclass(frozen=True)
class Subscription:
    name: str
    queue: asyncio.Queue[dict[str, Any]]


class EventBus:
    """
    Minimal in-process fan-out event bus.
    Each subscriber gets its own queue.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscription] = []

    def subscribe(
        self,
        name: str,
        maxsize: int = 256,
    ) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(
            Subscription(name=name, queue=queue)
        )
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

        message = line.decode("utf-8", errors="replace").rstrip()
        print(message, file=sys.stderr, flush=True)


async def read_ear_events(
    stream: asyncio.StreamReader,
    bus: EventBus,
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
            print(
                f"[core] invalid event shape: {event!r}",
                file=sys.stderr,
                flush=True,
            )
            continue

        await bus.publish(event)


async def diagnostics_worker(
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    while True:
        event = await queue.get()

        try:
            if event.get("type") == "speech":
                payload = event.get("payload", {})
                text = payload.get("text", "")
                duration = payload.get("duration_ms")

                print(
                    f'[core][speech] "{text}" ({duration} ms)',
                    flush=True,
                )
            else:
                print(
                    "[core][event] "
                    + json.dumps(event, ensure_ascii=False),
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
    ]

    if args.device is not None:
        cmd.extend(["--device", args.device])

    return cmd


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()

    parser = argparse.ArgumentParser(
        description=(
            "little-agent core: launches sensory processes, "
            "routes events, and persists memory"
        )
    )

    parser.add_argument("--ear", default="./ear.py")
    parser.add_argument(
        "--whisper",
        default=str(home / "whisper.cpp/build/bin/whisper-cli"),
    )
    parser.add_argument(
        "--model",
        default=str(home / "whisper.cpp/models/ggml-small.bin"),
    )
    parser.add_argument(
        "--memory-db",
        default="./data/agent.db",
    )
    parser.add_argument("--language", default="ko")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=int, default=600)
    parser.add_argument("--speech-pad-ms", type=int, default=200)

    return parser


async def run(args: argparse.Namespace) -> None:
    ear_path = Path(args.ear).expanduser().resolve()
    if not ear_path.exists():
        raise SystemExit(f"ear.py not found: {ear_path}")

    store = MemoryStore(args.memory_db)
    await asyncio.to_thread(store.initialize)

    print(
        f"[core] memory db: {store.db_path}",
        file=sys.stderr,
        flush=True,
    )

    bus = EventBus()

    diagnostics_queue = bus.subscribe("diagnostics")
    memory_queue = bus.subscribe("memory", maxsize=1024)

    command = build_ear_command(args)

    print("[core] starting ear process", file=sys.stderr, flush=True)

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert process.stdout is not None
    assert process.stderr is not None

    tasks = [
        asyncio.create_task(
            read_ear_events(process.stdout, bus),
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
            memory_worker(memory_queue, store),
            name="memory",
        ),
    ]

    try:
        return_code = await process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"ear process exited with code {return_code}"
            )

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
