#!/usr/bin/env python3
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "ear": "./ear.py",
        "whisper": "~/whisper.cpp/build/bin/whisper-cli",
        "whisper_model": "~/whisper.cpp/models/ggml-small.bin",
        "memory_db": "./data/agent.db",
    },
    "audio": {
        "language": "ko",
        "threads": 6,
        "device": None,
    },
    "vad": {
        "threshold": 0.70,
        "min_silence_ms": 600,
        "speech_pad_ms": 200,
        "max_utterance_sec": 20.0,
        "stt_queue_max": 8,
    },
    "episode": {
        "idle_sec": 15.0,
    },
    "llm": {
        "url": "http://127.0.0.1:8080",
        "model": "local",
        "realtime_concurrency": 2,
    },
    "turn": {
        "grace_ms": 300,
    },
    "reasoner": {
        "fast": {
            "temperature": 0.30,
            "max_tokens": 192,
            "timeout_sec": 30.0,
            "thinking": False,
        },
        "deliberate": {
            "enabled": True,
            "temperature": 0.35,
            "max_tokens": 768,
            "timeout_sec": 60.0,
            "thinking": True,
        },
    },
    "rolling": {
        "batch": 3,
        "delay_sec": 8.0,
        "temperature": 0.15,
        "max_tokens": 800,
        "timeout_sec": 120.0,
        "thinking": True,
    },
    "final": {
        "poll_sec": 3.0,
        "temperature": 0.15,
        "max_tokens": 1024,
        "timeout_sec": 120.0,
        "thinking": True,
    },
    "tts": {
        "enabled": True,
        "command": "/usr/bin/say",
        "voice": None,
        "rate": None,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load TOML config and merge it over built-in defaults.

    Missing default config files are allowed so the program still starts with
    built-in defaults. An explicitly supplied path should be checked by the
    caller if strict behavior is desired.
    """
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(path).expanduser()
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    with config_path.open("rb") as fh:
        loaded = tomllib.load(fh)

    if not isinstance(loaded, dict):
        raise ValueError("config root must be a TOML table")

    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(merged)
    return merged


def value(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def validate_config(config: dict[str, Any]) -> None:
    threshold = float(value(config, "vad", "threshold"))
    if not 0.0 < threshold < 1.0:
        raise ValueError("vad.threshold must be between 0 and 1")

    if int(value(config, "vad", "min_silence_ms")) < 0:
        raise ValueError("vad.min_silence_ms must be >= 0")
    if int(value(config, "turn", "grace_ms")) < 0:
        raise ValueError("turn.grace_ms must be >= 0")
    if float(value(config, "episode", "idle_sec")) <= 0:
        raise ValueError("episode.idle_sec must be > 0")

    for section in (
        ("reasoner", "fast"),
        ("reasoner", "deliberate"),
    ):
        node = value(config, *section)
        if int(node["max_tokens"]) <= 0:
            raise ValueError(f"{'.'.join(section)}.max_tokens must be > 0")
        if float(node["timeout_sec"]) <= 0:
            raise ValueError(f"{'.'.join(section)}.timeout_sec must be > 0")

    for section in ("rolling", "final"):
        node = value(config, section)
        if int(node["max_tokens"]) <= 0:
            raise ValueError(f"{section}.max_tokens must be > 0")
        if float(node["timeout_sec"]) <= 0:
            raise ValueError(f"{section}.timeout_sec must be > 0")
