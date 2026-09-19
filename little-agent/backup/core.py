#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from episode import ActiveEpisodeManager
from memory import (
    MemoryStore,
    persistence_worker,
)


@dataclass(frozen=True)
class Subscription:
    name: str
    queue: asyncio.Queue[
        dict[str, Any]
    ]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[
            Subscription
        ] = []

    def subscribe(
        self,
        name: str,
        maxsize: int = 256,
    ) -> asyncio.Queue[
        dict[str, Any]
    ]:
        queue: asyncio.Queue[
            dict[str, Any]
        ] = asyncio.Queue(
            maxsize=maxsize
        )

        self._subscribers.append(
            Subscription(
                name=name,
                queue=queue,
            )
        )

        return queue

    async def publish(
        self,
        event: dict[str, Any],
    ) -> None:
        # Fanout itself never waits on a slow consumer.
        for subscriber in self._subscribers:
            try:
                subscriber.queue.put_nowait(
                    event
                )
            except asyncio.QueueFull:
                print(
                    "[core] dropping event "
                    "for slow subscriber="
                    f"{subscriber.name}",
                    file=sys.stderr,
                    flush=True,
                )


async def pipe_ear_stderr(
    stream: asyncio.StreamReader,
) -> None:
    while True:
        line = await stream.readline()

        if not line:
            return

        message = line.decode(
            "utf-8",
            errors="replace",
        ).rstrip()

        print(
            message,
            file=sys.stderr,
            flush=True,
        )


async def read_ear_events(
    stream: asyncio.StreamReader,
    episode_manager: (
        ActiveEpisodeManager
    ),
) -> None:
    while True:
        line = await stream.readline()

        if not line:
            return

        raw = line.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not raw:
            continue

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            print(
                "[core] invalid JSON "
                f"from ear: {raw!r}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if (
            not isinstance(event, dict)
            or "type" not in event
        ):
            continue

        # This path only mutates in-memory episode state and enqueues
        # persistence work. It does not wait for SQLite or STT.
        await episode_manager.handle(
            event
        )


async def diagnostics_worker(
    queue: asyncio.Queue[
        dict[str, Any]
    ],
) -> None:
    while True:
        event = await queue.get()

        try:
            event_type = event.get(
                "type"
            )
            payload = event.get(
                "payload",
                {},
            )
            episode_id = event.get(
                "episode_id"
            )
            short_episode = (
                episode_id[:8]
                if episode_id
                else "-"
            )
            position = event.get(
                "episode_position",
                "-",
            )

            if event_type == (
                "speech.started"
            ):
                print(
                    "[core][speech:start] "
                    f"episode={short_episode} "
                    f"pos={position} "
                    "utterance="
                    f"{payload.get('utterance_id')}",
                    flush=True,
                )

            elif event_type == (
                "speech.ended"
            ):
                print(
                    "[core][speech:end] "
                    f"episode={short_episode} "
                    f"pos={position} "
                    f"duration="
                    f"{payload.get('duration_ms')}ms",
                    flush=True,
                )

            elif event_type in (
                "speech.final",
                "speech",
            ):
                print(
                    "[core][speech:final] "
                    f"episode={short_episode} "
                    f"pos={position} "
                    f'"{payload.get("text", "")}" '
                    "stt="
                    f"{payload.get('stt_latency_ms')}ms "
                    "queue="
                    f"{payload.get('queue_wait_ms')}ms",
                    flush=True,
                )

            elif event_type == (
                "speech.failed"
            ):
                print(
                    "[core][speech:failed] "
                    f"episode={short_episode} "
                    f"pos={position} "
                    "reason="
                    f"{payload.get('reason')}",
                    flush=True,
                )

        finally:
            queue.task_done()


def build_ear_command(
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(
            Path(args.ear)
            .expanduser()
            .resolve()
        ),
        "--whisper",
        str(
            Path(args.whisper)
            .expanduser()
            .resolve()
        ),
        "--model",
        str(
            Path(args.model)
            .expanduser()
            .resolve()
        ),
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
        cmd.extend(
            [
                "--device",
                args.device,
            ]
        )

    return cmd


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ear",
        default="./ear.py",
    )
    parser.add_argument(
        "--whisper",
        default=str(
            home
            / "whisper.cpp/build/bin/whisper-cli"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(
            home
            / "whisper.cpp/models/ggml-small.bin"
        ),
    )
    parser.add_argument(
        "--memory-db",
        default="./data/agent.db",
    )
    parser.add_argument(
        "--episode-idle-sec",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--language",
        default="ko",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-utterance-sec",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--stt-queue-max",
        type=int,
        default=8,
    )

    return parser


async def run(
    args: argparse.Namespace,
) -> None:
    store = MemoryStore(
        args.memory_db
    )

    await asyncio.to_thread(
        store.initialize
    )

    print(
        "[core] memory db: "
        f"{store.db_path}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[core] episode idle timeout: "
        f"{args.episode_idle_sec}s",
        file=sys.stderr,
        flush=True,
    )

    bus = EventBus()
    diagnostics_queue = bus.subscribe(
        "diagnostics"
    )

    # Intentionally unbounded for now:
    # realtime episode assignment must never wait on SQLite.
    # We observe backlog and can add batching/backpressure later.
    persistence_queue: asyncio.Queue[
        dict[str, Any]
    ] = asyncio.Queue()

    episode_manager = ActiveEpisodeManager(
        store=store,
        persistence_queue=(
            persistence_queue
        ),
        publish=bus.publish,
        idle_timeout_sec=(
            args.episode_idle_sec
        ),
    )

    await episode_manager.recover()

    process = await (
        asyncio.create_subprocess_exec(
            *build_ear_command(args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    )

    assert process.stdout is not None
    assert process.stderr is not None

    tasks = [
        asyncio.create_task(
            persistence_worker(
                persistence_queue,
                store,
            ),
            name="persistence",
        ),
        asyncio.create_task(
            episode_manager.idle_loop(),
            name="episode-idle",
        ),
        asyncio.create_task(
            read_ear_events(
                process.stdout,
                episode_manager,
            ),
            name="ear-events",
        ),
        asyncio.create_task(
            pipe_ear_stderr(
                process.stderr
            ),
            name="ear-stderr",
        ),
        asyncio.create_task(
            diagnostics_worker(
                diagnostics_queue
            ),
            name="diagnostics",
        ),
    ]

    try:
        return_code = await process.wait()

        if return_code != 0:
            raise RuntimeError(
                "ear process exited "
                f"with code {return_code}"
            )

    finally:
        if process.returncode is None:
            process.terminate()

            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )


def main() -> None:
    args = build_parser().parse_args()

    try:
        asyncio.run(
            run(args)
        )
    except KeyboardInterrupt:
        print(
            "\n[core] stopped",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
