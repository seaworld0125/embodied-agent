#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from typing import Any

from context import build_realtime_context, utterance_view
from llm_client import LLMRequestBroker, PRIORITY_REALTIME
from memory import MemoryStore


REASONER_SYSTEM_PROMPT = """너는 embodied agent의 실시간 S2 reasoner다.

역할:
- 현재 외부 화자의 발화와 최신 대화 맥락을 이해한다.
- 지금 에이전트가 응답해야 하는지 판단한다.
- 응답한다면 자연스럽고 짧은 한국어 응답 내용을 만든다.
- DB, episode lifecycle, queue, TTS 프로세스 같은 orchestration을 직접 지시하지 않는다.

세계 모델 규칙:
- source=ear 또는 speaker_role=external_speaker 인 음성은 에이전트가 마이크로 들은 외부 화자의 발화다.
- 이것은 에이전트 자신의 발화가 아니다.
- 별도의 화자 식별 정보가 없으면 특정 인물이라고 단정하지 말고 '외부 화자' 또는 '대화 상대'로 취급한다.
- 에이전트 자신의 발화는 source=agent/mouth/tts 등으로 명시된 경우에만 그렇게 취급한다.

context 규칙:
- WORKING MEMORY는 upto_utterance_seq까지의 압축 캐시이며 틀리거나 오래되었을 수 있다.
- RAW TAIL과 CURRENT UTTERANCE는 더 최신의 직접 증거다.
- WORKING MEMORY와 RAW TAIL/CURRENT UTTERANCE가 충돌하면 반드시 최신 raw evidence를 우선한다.
- 입력에 없는 사실, 관계, 감정, 사용자 정체성을 만들어내지 않는다.
- CURRENT UTTERANCE에 질문이나 응답 요구가 없고 굳이 반응할 필요가 없다면 intent=wait를 선택할 수 있다.

출력은 JSON object 하나만 사용한다. 마크다운 코드펜스를 쓰지 마라.

출력 형식:
{
  "intent": "respond" | "wait",
  "response": "respond일 때 말할 한국어 문장. wait이면 빈 문자열"
}
"""


class StaleReasoningInput(RuntimeError):
    pass


@dataclass(frozen=True)
class ReasoningResult:
    episode_id: str
    based_on_utterance_seq: int
    intent: str
    response: str


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

    raise ValueError("could not extract JSON object from realtime reasoning output")


def normalize_result(value: dict[str, Any]) -> tuple[str, str]:
    intent = str(value.get("intent", "")).strip().lower()
    response = str(value.get("response", "")).strip()

    if intent not in {"respond", "wait"}:
        raise ValueError(f"unsupported reasoner intent: {intent!r}")

    if intent == "respond" and not response:
        raise ValueError("reasoner returned respond with empty response")

    if intent == "wait":
        response = ""

    return intent, response


def build_reasoner_prompt(
    context: dict[str, Any],
    current_event: dict[str, Any],
) -> str:
    current = utterance_view(current_event)
    current_seq = int(current_event["utterance_seq"])

    # Keep the current utterance explicit even if it is already present in the
    # raw tail. Older tail items provide immediate turn-local context.
    tail_before_current = [
        item
        for item in context.get("raw_tail", [])
        if isinstance(item.get("utterance_seq"), int)
        and int(item["utterance_seq"]) < current_seq
    ]

    view = {
        "episode_id": context["episode_id"],
        "working_memory": context["working_memory"],
        "raw_tail_before_current": tail_before_current,
        "current_utterance": current,
        "precedence_rule": context["precedence_rule"],
    }

    return (
        "다음은 현재 interaction의 context snapshot이다.\n"
        "CURRENT UTTERANCE에 대해 지금 응답할지 판단하라.\n"
        "최신 raw evidence가 working memory와 충돌하면 최신 raw evidence를 우선하라.\n\n"
        + json.dumps(view, ensure_ascii=False, indent=2)
    )


class RealtimeReasoner:
    def __init__(
        self,
        store: MemoryStore,
        broker: LLMRequestBroker,
        *,
        temperature: float = 0.4,
        max_tokens: int = 512,
        timeout_sec: float = 60.0,
    ) -> None:
        self.store = store
        self.broker = broker
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    async def reason(
        self,
        current_event: dict[str, Any],
    ) -> ReasoningResult:
        if current_event.get("type") != "speech.final":
            raise ValueError("realtime reasoner requires speech.final")

        episode_id = current_event.get("episode_id")
        seq = current_event.get("utterance_seq")

        if not episode_id or not isinstance(seq, int):
            raise ValueError("speech.final missing episode_id/utterance_seq")

        context = await asyncio.to_thread(
            build_realtime_context,
            self.store,
            str(episode_id),
        )

        fresh_through = int(
            context.get("fresh_through_utterance_seq", -1)
        )
        if fresh_through > seq:
            raise StaleReasoningInput(
                f"context already contains newer utterance "
                f"fresh_through={fresh_through} based_on={seq}"
            )

        prompt = build_reasoner_prompt(context, current_event)

        log(
            f"[reasoner] start episode={str(episode_id)[:8]} "
            f"seq={seq} working_upto="
            f"{context['working_memory']['authoritative_through_utterance_seq']} "
            f"tail={len(context['raw_tail'])}"
        )

        raw = await self.broker.complete(
            [
                {"role": "system", "content": REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            priority=PRIORITY_REALTIME,
            label=f"realtime:{episode_id}:{seq}",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_sec=self.timeout_sec,
        )

        intent, response = normalize_result(extract_json_object(raw))

        log(
            f"[reasoner] done episode={str(episode_id)[:8]} "
            f"seq={seq} intent={intent}"
        )

        return ReasoningResult(
            episode_id=str(episode_id),
            based_on_utterance_seq=seq,
            intent=intent,
            response=response,
        )
