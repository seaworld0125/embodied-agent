#!/usr/bin/env python3

import argparse
import queue
import subprocess
import tempfile
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad


SAMPLE_RATE = 16_000
CHUNK_SIZE = 512  # 32 ms @ 16 kHz; Silero streaming-friendly chunk size


def write_wav(audio: np.ndarray, path: Path) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def transcribe(audio: np.ndarray, whisper_cli: Path, model_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        write_wav(audio, wav_path)

        result = subprocess.run(
            [
                str(whisper_cli),
                "-m", str(model_path),
                "-f", str(wav_path),
                "-l", "ko",
                "-nt",
                "-np",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print("[whisper:error]")
            print(result.stderr.strip())
            return ""

        return " ".join(
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ).strip()

    finally:
        wav_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime microphone -> Silero VAD -> whisper.cpp transcription"
    )
    parser.add_argument(
        "--whisper",
        type=Path,
        default=Path.home() / "whisper.cpp/build/bin/whisper-cli",
        help="Path to whisper-cli",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "whisper.cpp/models/ggml-small.bin",
        help="Path to whisper.cpp model",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice input device index or name; default = macOS default input",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-ms", type=int, default=600)
    parser.add_argument("--speech-pad-ms", type=int, default=200)
    parser.add_argument("--pre-roll-ms", type=int, default=250)
    parser.add_argument("--max-utterance-sec", type=float, default=20.0)
    args = parser.parse_args()

    if not args.whisper.exists():
        raise SystemExit(f"whisper-cli not found: {args.whisper}")
    if not args.model.exists():
        raise SystemExit(f"whisper model not found: {args.model}")

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    torch.set_num_threads(1)
    vad_model = load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=args.threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

    pre_roll_chunks = max(1, int((args.pre_roll_ms / 1000) * SAMPLE_RATE / CHUNK_SIZE))
    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
    max_chunks = max(1, int(args.max_utterance_sec * SAMPLE_RATE / CHUNK_SIZE))

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    print("[ear] loading Silero VAD: done")
    print("[ear] listening")
    print(f"[ear] threshold={args.threshold} min_silence={args.min_silence_ms}ms")
    print("[ear] Ctrl+C to stop")

    recording = False
    utterance: list[np.ndarray] = []

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            device=device,
        ):
            while True:
                chunk = audio_queue.get()
                tensor = torch.from_numpy(chunk)
                event = vad(tensor, return_seconds=False)

                if not recording:
                    pre_roll.append(chunk)

                if event and "start" in event and not recording:
                    recording = True
                    utterance = list(pre_roll)
                    pre_roll.clear()
                    print("[speech:start]")
                    continue

                if recording:
                    utterance.append(chunk)

                if recording and len(utterance) >= max_chunks:
                    print("[speech:max-duration]")
                    event = {"end": 0}

                if event and "end" in event and recording:
                    recording = False
                    audio = np.concatenate(utterance) if utterance else np.array([], dtype=np.float32)
                    utterance = []
                    pre_roll.clear()

                    print(f"[speech:end] {len(audio) / SAMPLE_RATE:.2f}s")

                    if len(audio) == 0:
                        vad.reset_states()
                        continue

                    text = transcribe(audio, args.whisper, args.model)
                    if text:
                        print(f'[heard] "{text}"')
                    else:
                        print("[heard] <empty>")

                    # V0 is intentionally half-duplex while Whisper is running.
                    # Drop stale microphone chunks captured during transcription.
                    while True:
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    vad.reset_states()

    except KeyboardInterrupt:
        print("\n[ear] stopped")


if __name__ == "__main__":
    main()
