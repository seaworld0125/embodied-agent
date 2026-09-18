#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import tempfile
import time
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad


SAMPLE_RATE = 16_000
CHUNK_SIZE = 512  # 32 ms at 16 kHz
PRE_ROLL_MS = 250
MAX_QUEUE_CHUNKS = 256


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    # stdout is reserved for machine-readable JSONL events.
    print(message, file=sys.stderr, flush=True)


def publish(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def write_wav(audio: np.ndarray, path: Path) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def transcribe(
    audio: np.ndarray,
    whisper_cli: Path,
    model: Path,
    language: str,
    threads: int,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        write_wav(audio, wav_path)

        cmd = [
            str(whisper_cli),
            "-m", str(model),
            "-f", str(wav_path),
            "-l", language,
            "-t", str(threads),
            "-nt",
            "-np",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            log(f"[ear] whisper failed: {result.stderr.strip()}")
            return ""

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return " ".join(lines).strip()

    finally:
        wav_path.unlink(missing_ok=True)


def normalize_device(value: str | None):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def flush_queue(audio_queue: queue.Queue) -> None:
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            return


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()

    parser = argparse.ArgumentParser(
        description="Local microphone -> Silero VAD -> whisper.cpp -> JSONL SpeechEvent"
    )
    parser.add_argument(
        "--whisper",
        default=str(home / "whisper.cpp/build/bin/whisper-cli"),
        help="Path to whisper-cli",
    )
    parser.add_argument(
        "--model",
        default=str(home / "whisper.cpp/models/ggml-small.bin"),
        help="Path to whisper.cpp model",
    )
    parser.add_argument("--language", default="ko")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice input device index or name. Default: system input",
    )

    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=int, default=600)
    parser.add_argument("--speech-pad-ms", type=int, default=200)
    parser.add_argument("--max-utterance-sec", type=float, default=20.0)

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

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE_CHUNKS)

    def audio_callback(indata, frames, time_info, status):
        if status:
            log(f"[ear][audio] {status}")

        chunk = indata[:, 0].copy()
        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            # V0 is intentionally lossy while downstream is busy.
            pass

    log("[ear] loading Silero VAD...")
    vad_model = load_silero_vad()

    vad = VADIterator(
        vad_model,
        threshold=args.vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    pre_roll_chunks = max(1, int((PRE_ROLL_MS / 1000) * SAMPLE_RATE / CHUNK_SIZE))
    pre_roll = deque(maxlen=pre_roll_chunks)

    recording = False
    utterance: list[np.ndarray] = []
    started_at: str | None = None
    started_monotonic: float | None = None

    log(
        "[ear] listening "
        f"(device={device!r}, threshold={args.vad_threshold}, "
        f"silence={args.min_silence_ms}ms)"
    )

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
            tensor = torch.from_numpy(chunk)

            vad_event = vad(tensor, return_seconds=False)

            if not recording:
                pre_roll.append(chunk)

            if vad_event and "start" in vad_event and not recording:
                recording = True
                started_at = now_iso()
                started_monotonic = time.monotonic()
                utterance = list(pre_roll)
                pre_roll.clear()
                log("[ear] speech:start")

            elif recording:
                utterance.append(chunk)

            reached_max = (
                recording
                and started_monotonic is not None
                and (time.monotonic() - started_monotonic) >= args.max_utterance_sec
            )

            ended_by_vad = bool(vad_event and "end" in vad_event and recording)

            if not (ended_by_vad or reached_max):
                continue

            ended_at = now_iso()
            duration_sec = (
                time.monotonic() - started_monotonic
                if started_monotonic is not None
                else len(utterance) * CHUNK_SIZE / SAMPLE_RATE
            )

            if reached_max:
                log("[ear] speech:end (max duration)")
            else:
                log("[ear] speech:end")

            audio = np.concatenate(utterance) if utterance else np.array([], dtype=np.float32)

            recording = False
            utterance = []
            pre_roll.clear()

            text = ""
            if audio.size > 0:
                text = transcribe(
                    audio=audio,
                    whisper_cli=whisper_cli,
                    model=model_path,
                    language=args.language,
                    threads=args.threads,
                )

            if text:
                event = {
                    "schema_version": 1,
                    "id": str(uuid4()),
                    "type": "speech",
                    "source": "ear",
                    "occurred_at": ended_at,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "payload": {
                        "text": text,
                        "language": args.language,
                        "duration_ms": int(duration_sec * 1000),
                    },
                }
                publish(event)
                log(f'[ear] heard: "{text}"')
            else:
                log("[ear] heard: <empty>")

            # V0 is half-duplex: audio arriving during Whisper inference is discarded.
            flush_queue(audio_queue)
            vad.reset_states()
            started_at = None
            started_monotonic = None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[ear] stopped")
