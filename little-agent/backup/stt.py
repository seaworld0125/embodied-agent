#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Callable
from uuid import uuid4

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(slots=True)
class Utterance:
    id: str
    audio: np.ndarray
    started_at: str
    ended_at: str
    duration_ms: int
    enqueued_monotonic_ms: int


@dataclass(slots=True)
class WhisperConfig:
    whisper_cli: Path
    model: Path
    language: str = "ko"
    threads: int = 6
    sample_rate: int = 16000


class WhisperSTTWorker:
    def __init__(
        self,
        config: WhisperConfig,
        utterance_queue: queue.Queue[Utterance],
        publish: Callable[[dict], None],
    ) -> None:
        self.config = config
        self.utterance_queue = utterance_queue
        self.publish = publish
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name="whisper-stt",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        log("[stt] worker started")

        while not self._stop.is_set():
            try:
                utterance = self.utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process(utterance)
            except Exception as exc:
                log(
                    f"[stt] failed utterance={utterance.id}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self.utterance_queue.task_done()

        log("[stt] worker stopped")

    def _process(self, utterance: Utterance) -> None:
        stt_started_at = utc_now_iso()
        stt_started_ms = monotonic_ms()

        queue_wait_ms = max(
            0,
            stt_started_ms - utterance.enqueued_monotonic_ms,
        )

        log(
            f"[stt] start id={utterance.id} "
            f"queue_wait={queue_wait_ms}ms "
            f"pending={self.utterance_queue.qsize()}"
        )

        text = transcribe(
            audio=utterance.audio,
            config=self.config,
        )

        stt_finished_at = utc_now_iso()
        stt_latency_ms = max(
            0,
            monotonic_ms() - stt_started_ms,
        )

        if not text:
            log(
                f"[stt] empty id={utterance.id} "
                f"latency={stt_latency_ms}ms"
            )
            return

        event = {
            "schema_version": 1,
            "id": str(uuid4()),
            "type": "speech",
            "source": "ear",
            "occurred_at": utterance.ended_at,
            "started_at": utterance.started_at,
            "ended_at": utterance.ended_at,
            "payload": {
                "utterance_id": utterance.id,
                "text": text,
                "language": self.config.language,
                "duration_ms": utterance.duration_ms,
                "queue_wait_ms": queue_wait_ms,
                "stt_latency_ms": stt_latency_ms,
                "stt_started_at": stt_started_at,
                "stt_finished_at": stt_finished_at,
            },
        }

        self.publish(event)

        log(
            f'[stt] done id={utterance.id} '
            f'latency={stt_latency_ms}ms '
            f'text="{text}"'
        )


def transcribe(audio: np.ndarray, config: WhisperConfig) -> str:
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        wav_path = Path(tmp.name)

    try:
        write_wav(audio, wav_path, config.sample_rate)

        result = subprocess.run(
            [
                str(config.whisper_cli),
                "-m", str(config.model),
                "-f", str(wav_path),
                "-l", config.language,
                "-t", str(config.threads),
                "-nt",
                "-np",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or f"whisper-cli exited {result.returncode}"
            )

        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        return " ".join(lines).strip()

    finally:
        wav_path.unlink(missing_ok=True)


def write_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def publish_jsonl(event: dict) -> None:
    print(
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
