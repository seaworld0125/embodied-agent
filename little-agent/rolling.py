#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from context import utterance_view
from llm_client import LLMRequestBroker, PRIORITY_ROLLING
from memory import MemoryStore


ROLLING_SYSTEM_PROMPT = """너는 embodied agent의 현재 에피소드 working memory를 갱신하는 내부 기억 과정이다.

중요한 세계 모델 규칙:
- source=ear 또는 speaker_role=external_speaker 인 음성은 에이전트가 마이크로 들은 외부 화자의 발화다.
- 이것은 에이전트 자신의 발화가 아니다.
- 별도의 화자 식별 정보가 없으면 그 사람을 특정 인물이라고 단정하지 말고 '외부 화자' 또는 '대화 상대'로 취급하라.
- 에이전트 자신의 발화는 명시적으로 source=agent/mouth/tts 등으로 표시된 경우에만 그렇게 취급하라.

working memory 규칙:
- PREVIOUS WORKING MEMORY는 과거 raw utterance를 압축한 가변 캐시다. 틀리거나 오래되었을 수 있다.
- NEW RAW UTTERANCES는 더 최신의 직접 증거다.
- 둘이 충돌하면 반드시 NEW RAW UTTERANCES를 우선한다.
- 새 정보가 이전 추정/사실을 정정하면 오래된 내용을 제거하거나 수정한다.
- 주어진 정보에 없는 사실, 감정, 관계, 목적을 만들어내지 마라.
- 아직 확정되지 않은 사실은 provisional_facts에만 두고 confidence를 낮춰라.
- 결과는 JSON object 하나만 출력하고 마크다운 코드펜스를 쓰지 마라.
- 모든 자연어 값은 고유명사/기술용어를 제외하고 한국어로 작성하라.

출력 형식:
{
  "summary": "현재 대화/상황을 이해하기 위한 짧은 한국어 요약",
  "topics": ["현재 주제"],
  "provisional_facts": [
    {"text": "현재까지 유용한 잠정 사실", "confidence": 0.0}
  ],
  "open_threads": ["아직 이어지고 있거나 해결되지 않은 맥락"]
}
"""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        strip_code_fence(text),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ValueError("could not extract JSON object from rolling output")


def clamp(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def normalize_state(value: dict[str, Any]) -> dict[str, Any]:
    summary = str(value.get("summary", "")).strip()
    topics_raw = value.get("topics", [])
    topics = (
        [str(item).strip() for item in topics_raw if str(item).strip()]
        if isinstance(topics_raw, list)
        else []
    )

    facts: list[dict[str, Any]] = []
    facts_raw = value.get("provisional_facts", [])
    if isinstance(facts_raw, list):
        for item in facts_raw:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    facts.append({"text": text, "confidence": 0.5})
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    facts.append(
                        {
                            "text": text,
                            "confidence": clamp(item.get("confidence"), 0.5),
                        }
                    )

    threads_raw = value.get("open_threads", [])
    threads = (
        [str(item).strip() for item in threads_raw if str(item).strip()]
        if isinstance(threads_raw, list)
        else []
    )

    return {
        "summary": summary,
        "topics": topics,
        "provisional_facts": facts,
        "open_threads": threads,
    }


def build_prompt(
    working: dict[str, Any],
    utterances: list[dict[str, Any]],
    target_seq: int,
) -> str:
    previous = {
        "version": int(working["version"]),
        "upto_utterance_seq": int(working["upto_utterance_seq"]),
        "summary": working["summary"],
        "state": working["state"],
    }
    raw = [utterance_view(event) for event in utterances]

    return (
        "아래 working memory를 새 raw utterance를 반영해 갱신하라.\n"
        "PREVIOUS WORKING MEMORY는 그것의 upto_utterance_seq까지만 압축한다.\n"
        "그 이후의 NEW RAW UTTERANCES가 항상 더 최신이며 충돌 시 우선한다.\n\n"
        f"TARGET_UPTO_UTTERANCE_SEQ: {target_seq}\n\n"
        "PREVIOUS WORKING MEMORY:\n"
        f"{json.dumps(previous, ensure_ascii=False, indent=2)}\n\n"
        "NEW RAW UTTERANCES:\n"
        f"{json.dumps(raw, ensure_ascii=False, indent=2)}\n"
    )


@dataclass
class _EpisodeRollState:
    latest_seq: int = -1
    pending_count: int = 0
    first_pending_at: float | None = None
    queued: bool = False
    in_flight: bool = False


class RollingMemoryService:
    def __init__(
        self,
        store: MemoryStore,
        broker: LLMRequestBroker,
        event_queue: asyncio.Queue[dict[str, Any]],
        batch_size: int = 3,
        max_delay_sec: float = 8.0,
        llm_temperature: float = 0.15,
        llm_max_tokens: int = 800,
        llm_timeout_sec: float = 120.0,
        llm_thinking: bool = True,
    ) -> None:
        self.store = store
        self.broker = broker
        self.event_queue = event_queue
        self.batch_size = max(1, batch_size)
        self.max_delay_sec = max(0.5, max_delay_sec)
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout_sec = llm_timeout_sec
        self.llm_thinking = llm_thinking
        self.jobs: asyncio.Queue[str] = asyncio.Queue()
        self.states: dict[str, _EpisodeRollState] = {}

    async def event_loop(self) -> None:
        while True:
            event = await self.event_queue.get()
            try:
                if event.get("type") != "speech.final":
                    continue

                episode_id = event.get("episode_id")
                seq = event.get("utterance_seq")
                if not episode_id or not isinstance(seq, int):
                    continue

                state = self.states.setdefault(
                    str(episode_id), _EpisodeRollState()
                )
                state.latest_seq = max(state.latest_seq, seq)
                state.pending_count += 1
                if state.first_pending_at is None:
                    state.first_pending_at = time.monotonic()

                if state.pending_count >= self.batch_size:
                    self._request(str(episode_id), state)
            finally:
                self.event_queue.task_done()

    async def timer_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            for episode_id, state in list(self.states.items()):
                if state.pending_count <= 0 or state.first_pending_at is None:
                    continue
                if now - state.first_pending_at >= self.max_delay_sec:
                    self._request(episode_id, state)

    async def worker_loop(self) -> None:
        while True:
            episode_id = await self.jobs.get()
            state = self.states.setdefault(episode_id, _EpisodeRollState())
            state.queued = False
            state.in_flight = True
            target_seq = state.latest_seq

            # New arrivals while this inference runs are counted separately.
            state.pending_count = 0
            state.first_pending_at = None

            try:
                success = await self._roll_episode(episode_id, target_seq)
                if not success:
                    # Preserve eventual progress. A DB lag or transient LLM
                    # failure should cause a later retry, not data loss.
                    state.pending_count = max(1, state.pending_count)
                    if state.first_pending_at is None:
                        state.first_pending_at = time.monotonic()
            except Exception as exc:
                log(
                    f"[rolling] failed episode={episode_id} "
                    f"target={target_seq}: {type(exc).__name__}: {exc}"
                )
                state.pending_count = max(1, state.pending_count)
                if state.first_pending_at is None:
                    state.first_pending_at = time.monotonic()
            finally:
                state.in_flight = False
                self.jobs.task_done()

            if state.pending_count >= self.batch_size:
                self._request(episode_id, state)

    def _request(self, episode_id: str, state: _EpisodeRollState) -> None:
        if state.queued or state.in_flight:
            return
        state.queued = True
        self.jobs.put_nowait(episode_id)

    async def _roll_episode(self, episode_id: str, target_seq: int) -> bool:
        working = await asyncio.to_thread(
            self.store.get_working_memory,
            episode_id,
        )
        upto = int(working["upto_utterance_seq"])
        if target_seq <= upto:
            return True

        utterances: list[dict[str, Any]] = []
        for _ in range(20):
            utterances = await asyncio.to_thread(
                self.store.episode_final_utterances,
                episode_id,
                upto,
                target_seq,
            )
            if utterances and max(
                int(item["utterance_seq"]) for item in utterances
            ) >= target_seq:
                break
            await asyncio.sleep(0.05)

        if not utterances:
            log(
                f"[rolling] DB tail not visible yet episode={episode_id} "
                f"after={upto} target={target_seq}"
            )
            return False

        # If some earlier utterance failed STT, sequence gaps are legitimate.
        visible_target = max(int(item["utterance_seq"]) for item in utterances)
        if visible_target < target_seq:
            return False

        prompt = build_prompt(working, utterances, target_seq)
        log(
            f"[rolling] start episode={episode_id} "
            f"v={working['version']} {upto}->{target_seq} "
            f"new={len(utterances)}"
        )

        raw = await self.broker.complete(
            [
                {"role": "system", "content": ROLLING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            priority=PRIORITY_ROLLING,
            label=f"rolling:{episode_id}:{target_seq}",
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout_sec=self.llm_timeout_sec,
            enable_thinking=self.llm_thinking,
        )
        normalized = normalize_state(extract_json_object(raw))
        if not normalized["summary"]:
            raise ValueError("rolling model returned empty summary")

        saved = await asyncio.to_thread(
            self.store.save_working_memory_cas,
            episode_id,
            int(working["version"]),
            target_seq,
            normalized["summary"],
            normalized,
        )

        if not saved:
            log(
                f"[rolling] stale result discarded episode={episode_id} "
                f"expected_version={working['version']} target={target_seq}"
            )
            return True

        log(
            f"[rolling] done episode={episode_id} upto={target_seq} "
            f'summary="{normalized["summary"]}"'
        )
        return True
