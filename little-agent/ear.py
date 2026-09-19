#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

from stt import (
    Utterance,
    WhisperConfig,
    WhisperSTTWorker,
    publish_jsonl,
)


SAMPLE_RATE = 16000
CHUNK_SIZE = 512
PRE_ROLL_MS = 250
AUDIO_QUEUE_MAX = 512
DEFAULT_STT_QUEUE_MAX = 8


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize_device(value: str | None):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description=(
            "Async Ear: continuous microphone/VAD capture "
            "with background whisper.cpp transcription"
        )
    )
    parser.add_argument(
        "--whisper",
        default=str(home / "whisper.cpp/build/bin/whisper-cli"),
    )
    parser.add_argument(
        "--model",
        default=str(home / "whisper.cpp/models/ggml-small.bin"),
    )
    parser.add_argument("--language", default="ko")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=int, default=600)
    parser.add_argument("--speech-pad-ms", type=int, default=200)
    parser.add_argument("--max-utterance-sec", type=float, default=20.0)
    parser.add_argument(
        "--stt-queue-max",
        type=int,
        default=DEFAULT_STT_QUEUE_MAX,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    whisper_cli = Path(args.whisper).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not whisper_cli.exists():
        raise SystemExit(f"whisper-cli not found: {whisper_cli}")

    if not model_path.exists():
        raise SystemExit(f"whisper model not found: {model_path}")

    device = normalize_device(args.device)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(
        maxsize=AUDIO_QUEUE_MAX
    )
    stt_queue: queue.Queue[Utterance] = queue.Queue(
        maxsize=args.stt_queue_max
    )

    stt_worker = WhisperSTTWorker(
        config=WhisperConfig(
            whisper_cli=whisper_cli,
            model=model_path,
            language=args.language,
            threads=args.threads,
            sample_rate=SAMPLE_RATE,
        ),
        utterance_queue=stt_queue,
        publish=publish_jsonl,
    )

    dropped_audio_chunks = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal dropped_audio_chunks

        if status:
            log(f"[ear][audio] {status}")

        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped_audio_chunks += 1
            if dropped_audio_chunks % 50 == 1:
                log(
                    "[ear] audio queue overflow "
                    f"dropped_chunks={dropped_audio_chunks}"
                )

    log("[ear] loading Silero VAD...")
    vad_model = load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=args.vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    pre_roll_chunks = max(
        1,
        int((PRE_ROLL_MS / 1000) * SAMPLE_RATE / CHUNK_SIZE),
    )
    pre_roll = deque(maxlen=pre_roll_chunks)

    recording = False
    utterance_chunks: list[np.ndarray] = []
    utterance_id: str | None = None
    started_at: str | None = None
    started_monotonic: float | None = None

    stt_worker.start()

    log(
        "[ear] listening asynchronously "
        f"(device={device!r}, "
        f"threshold={args.vad_threshold}, "
        f"silence={args.min_silence_ms}ms, "
        f"stt_queue_max={args.stt_queue_max})"
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            device=device,
        ):
            while True:
                chunk = audio_queue.get()
                vad_event = vad(
                    torch.from_numpy(chunk),
                    return_seconds=False,
                )

                if not recording:
                    pre_roll.append(chunk)

                if vad_event and "start" in vad_event and not recording:
                    recording = True
                    utterance_id = str(uuid4())
                    started_at = utc_now_iso()
                    started_monotonic = time.monotonic()
                    utterance_chunks = list(pre_roll)
                    pre_roll.clear()

                    log(f"[ear] speech:start id={utterance_id}")
                    continue

                if recording:
                    utterance_chunks.append(chunk)

                reached_max = (
                    recording
                    and started_monotonic is not None
                    and (
                        time.monotonic() - started_monotonic
                    ) >= args.max_utterance_sec
                )

                ended_by_vad = bool(
                    vad_event
                    and "end" in vad_event
                    and recording
                )

                if not (ended_by_vad or reached_max):
                    continue

                ended_at = utc_now_iso()
                duration_ms = int(
                    (
                        time.monotonic()
                        - (started_monotonic or time.monotonic())
                    )
                    * 1000
                )

                audio = (
                    np.concatenate(utterance_chunks)
                    if utterance_chunks
                    else np.array([], dtype=np.float32)
                )

                finished_id = utterance_id or str(uuid4())
                started_at_value = started_at or ended_at

                log(
                    f"[ear] speech:end id={finished_id} "
                    f"reason={'max-duration' if reached_max else 'vad'}"
                )

                # Reset immediately. No STT work happens on this path.
                recording = False
                utterance_chunks = []
                utterance_id = None
                started_at = None
                started_monotonic = None
                pre_roll.clear()

                if audio.size == 0:
                    vad.reset_states()
                    continue

                item = Utterance(
                    id=finished_id,
                    audio=audio,
                    started_at=started_at_value,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                    enqueued_monotonic_ms=monotonic_ms(),
                )

                try:
                    stt_queue.put_nowait(item)
                    log(
                        f"[ear] queued stt id={finished_id} "
                        f"pending={stt_queue.qsize()}"
                    )
                except queue.Full:
                    # Preserve sensory-loop liveness over blocking.
                    log(
                        "[ear] STT QUEUE FULL: "
                        f"dropped utterance id={finished_id}"
                    )

                vad.reset_states()

    except KeyboardInterrupt:
        log("[ear] stopping")

    finally:
        stt_worker.stop()


if __name__ == "__main__":
    main()
