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
from config import load_config, value
from episode import ActiveEpisodeManager
from llm_client import LLMRequestBroker
from memory import MemoryStore, persistence_worker
from reasoner import RealtimeReasoner
from rolling import RollingMemoryService
from turn import TurnCoordinator
from tts import MacOSSayTTS


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
                    f"mode={payload.get('reasoning_mode', 'fast')} "
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
            elif event_type == "agent.speech.started":
                print(
                    f"[core][agent:speech:start] episode={episode} "
                    f"action={str(payload.get('action_id', ''))[:8]} "
                    f'"{payload.get("text", "")}"',
                    flush=True,
                )
            elif event_type == "agent.speech.ended":
                print(
                    f"[core][agent:speech:end] episode={episode} "
                    f"action={str(payload.get('action_id', ''))[:8]} "
                    f"status={payload.get('status')} "
                    f"duration={payload.get('duration_ms')}ms",
                    flush=True,
                )
            elif event_type == "agent.speech.failed":
                print(
                    f"[core][agent:speech:failed] episode={episode} "
                    f"action={str(payload.get('action_id', ''))[:8]} "
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


def build_parser(
    config: dict[str, Any],
    config_path: str = "./config.toml",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path)

    parser.add_argument("--ear", default=value(config, "paths", "ear"))
    parser.add_argument("--whisper", default=value(config, "paths", "whisper"))
    parser.add_argument("--model", default=value(config, "paths", "whisper_model"))
    parser.add_argument("--memory-db", default=value(config, "paths", "memory_db"))

    parser.add_argument(
        "--episode-idle-sec",
        type=float,
        default=value(config, "episode", "idle_sec"),
    )
    parser.add_argument("--language", default=value(config, "audio", "language"))
    parser.add_argument("--threads", type=int, default=value(config, "audio", "threads"))
    parser.add_argument("--device", default=value(config, "audio", "device"))

    parser.add_argument(
        "--vad-threshold", type=float, default=value(config, "vad", "threshold")
    )
    parser.add_argument(
        "--min-silence-ms", type=int, default=value(config, "vad", "min_silence_ms")
    )
    parser.add_argument(
        "--speech-pad-ms", type=int, default=value(config, "vad", "speech_pad_ms")
    )
    parser.add_argument(
        "--max-utterance-sec",
        type=float,
        default=value(config, "vad", "max_utterance_sec"),
    )
    parser.add_argument(
        "--stt-queue-max", type=int, default=value(config, "vad", "stt_queue_max")
    )

    parser.add_argument("--llm-url", default=value(config, "llm", "url"))
    parser.add_argument("--llm-model", default=value(config, "llm", "model"))
    parser.add_argument(
        "--llm-realtime-concurrency",
        type=int,
        default=value(config, "llm", "realtime_concurrency"),
    )

    parser.add_argument(
        "--turn-grace-ms", type=int, default=value(config, "turn", "grace_ms")
    )
    parser.add_argument(
        "--reasoner-temperature",
        type=float,
        default=value(config, "reasoner", "fast", "temperature"),
    )
    parser.add_argument(
        "--reasoner-max-tokens",
        type=int,
        default=value(config, "reasoner", "fast", "max_tokens"),
    )
    parser.add_argument(
        "--reasoner-timeout-sec",
        type=float,
        default=value(config, "reasoner", "fast", "timeout_sec"),
    )
    parser.add_argument(
        "--reasoner-thinking",
        action=argparse.BooleanOptionalAction,
        default=value(config, "reasoner", "fast", "thinking"),
    )
    parser.add_argument(
        "--deliberate-enabled",
        action=argparse.BooleanOptionalAction,
        default=value(config, "reasoner", "deliberate", "enabled"),
    )
    parser.add_argument(
        "--deliberate-temperature",
        type=float,
        default=value(config, "reasoner", "deliberate", "temperature"),
    )
    parser.add_argument(
        "--deliberate-max-tokens",
        type=int,
        default=value(config, "reasoner", "deliberate", "max_tokens"),
    )
    parser.add_argument(
        "--deliberate-timeout-sec",
        type=float,
        default=value(config, "reasoner", "deliberate", "timeout_sec"),
    )
    parser.add_argument(
        "--deliberate-thinking",
        action=argparse.BooleanOptionalAction,
        default=value(config, "reasoner", "deliberate", "thinking"),
    )

    parser.add_argument("--tts-command", default=value(config, "tts", "command"))
    parser.add_argument("--tts-voice", default=value(config, "tts", "voice"))
    parser.add_argument("--tts-rate", type=int, default=value(config, "tts", "rate"))
    parser.add_argument(
        "--tts-enabled",
        action=argparse.BooleanOptionalAction,
        default=value(config, "tts", "enabled"),
    )
    # Backward-compatible alias from v5.
    parser.add_argument("--no-tts", action="store_false", dest="tts_enabled")

    parser.add_argument(
        "--rolling-batch", type=int, default=value(config, "rolling", "batch")
    )
    parser.add_argument(
        "--rolling-delay-sec",
        type=float,
        default=value(config, "rolling", "delay_sec"),
    )
    parser.add_argument(
        "--rolling-temperature",
        type=float,
        default=value(config, "rolling", "temperature"),
    )
    parser.add_argument(
        "--rolling-max-tokens",
        type=int,
        default=value(config, "rolling", "max_tokens"),
    )
    parser.add_argument(
        "--rolling-timeout-sec",
        type=float,
        default=value(config, "rolling", "timeout_sec"),
    )
    parser.add_argument(
        "--rolling-thinking",
        action=argparse.BooleanOptionalAction,
        default=value(config, "rolling", "thinking"),
    )

    parser.add_argument(
        "--final-poll-sec", type=float, default=value(config, "final", "poll_sec")
    )
    parser.add_argument(
        "--final-temperature",
        type=float,
        default=value(config, "final", "temperature"),
    )
    parser.add_argument(
        "--final-max-tokens",
        type=int,
        default=value(config, "final", "max_tokens"),
    )
    parser.add_argument(
        "--final-timeout-sec",
        type=float,
        default=value(config, "final", "timeout_sec"),
    )
    parser.add_argument(
        "--final-thinking",
        action=argparse.BooleanOptionalAction,
        default=value(config, "final", "thinking"),
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    store = MemoryStore(args.memory_db)
    await asyncio.to_thread(store.initialize)

    print(f"[core] memory db: {store.db_path}", file=sys.stderr, flush=True)
    print(f"[core] config: {args.config}", file=sys.stderr, flush=True)
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
        f"[core] reasoner fast: thinking={args.reasoner_thinking} "
        f"tokens={args.reasoner_max_tokens}; deliberate: "
        f"enabled={args.deliberate_enabled} thinking={args.deliberate_thinking} "
        f"tokens={args.deliberate_max_tokens}",
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
    tts_event_queue = bus.subscribe("tts", maxsize=512)

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
        thinking=args.reasoner_thinking,
        deliberate_enabled=args.deliberate_enabled,
        deliberate_temperature=args.deliberate_temperature,
        deliberate_max_tokens=args.deliberate_max_tokens,
        deliberate_timeout_sec=args.deliberate_timeout_sec,
        deliberate_thinking=args.deliberate_thinking,
    )
    turns = TurnCoordinator(
        reasoner=reasoner,
        event_queue=turn_event_queue,
        # Reasoner outputs become part of the agent's durable episode history
        # before being fanned out to the rest of the runtime.
        publish=episode_manager.publish_internal_event,
        grace_ms=args.turn_grace_ms,
    )

    rolling = RollingMemoryService(
        store=store,
        broker=broker,
        event_queue=rolling_event_queue,
        batch_size=args.rolling_batch,
        max_delay_sec=args.rolling_delay_sec,
        llm_temperature=args.rolling_temperature,
        llm_max_tokens=args.rolling_max_tokens,
        llm_timeout_sec=args.rolling_timeout_sec,
        llm_thinking=args.rolling_thinking,
    )

    tts = MacOSSayTTS(
        event_queue=tts_event_queue,
        publish=episode_manager.publish_internal_event,
        command=args.tts_command,
        voice=args.tts_voice,
        rate=args.tts_rate,
        enabled=args.tts_enabled,
    )
    print(
        f"[core] TTS: {'enabled' if args.tts_enabled else 'disabled'} "
        f"command={args.tts_command} voice={args.tts_voice or 'system'} "
        f"rate={args.tts_rate or 'system'}",
        file=sys.stderr,
        flush=True,
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
            tts.event_loop(),
            name="tts",
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
                temperature=args.final_temperature,
                max_tokens=args.final_max_tokens,
                timeout_sec=args.final_timeout_sec,
                enable_thinking=args.final_thinking,
            ),
            name="final-consolidation",
        ),
    ]

    try:
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(f"ear process exited with code {return_code}")
    finally:
        await tts.close()

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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="./config.toml")
    pre_args, _ = pre_parser.parse_known_args()

    config_path = Path(pre_args.config).expanduser()
    config = load_config(config_path)
    if not config_path.exists():
        print(
            f"[core] config not found: {config_path}; using built-in defaults",
            file=sys.stderr,
            flush=True,
        )

    args = build_parser(config, str(config_path)).parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[core] stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
