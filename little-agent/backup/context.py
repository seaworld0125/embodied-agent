#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from memory import MemoryStore


def speaker_role(source: str) -> str:
    if source == "ear":
        return "external_speaker"
    if source in {"agent", "mouth", "tts"}:
        return "agent"
    return "unknown"


def utterance_view(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload", {})
    return {
        "utterance_seq": event.get("utterance_seq"),
        "occurred_at": event.get("occurred_at"),
        "source": event.get("source"),
        "speaker_role": speaker_role(str(event.get("source", ""))),
        "text": str(payload.get("text", "")).strip(),
    }


def build_realtime_context(
    store: MemoryStore,
    episode_id: str,
) -> dict[str, Any]:
    """
    Safe context snapshot for future realtime reasoning.

    The working summary is authoritative only through upto_utterance_seq.
    Every newer recognized utterance is returned verbatim in raw_tail.
    Therefore a stale working summary increases context size, but does not
    remove newer evidence from the reasoner.
    """
    working = store.get_working_memory(episode_id)
    upto = int(working["upto_utterance_seq"])
    tail_events = store.episode_final_utterances(
        episode_id,
        after_seq=upto,
    )
    tail = [utterance_view(event) for event in tail_events]

    fresh_through = upto
    if tail:
        fresh_through = max(
            int(item["utterance_seq"])
            for item in tail
            if item["utterance_seq"] is not None
        )

    return {
        "episode_id": episode_id,
        "working_memory": {
            "version": int(working["version"]),
            "authoritative_through_utterance_seq": upto,
            "summary": working["summary"],
            "state": working["state"],
        },
        "raw_tail": tail,
        "fresh_through_utterance_seq": fresh_through,
        "working_memory_is_stale": bool(tail),
        "precedence_rule": (
            "raw_tail is newer evidence and overrides conflicting or "
            "superseded claims in working_memory"
        ),
    }
