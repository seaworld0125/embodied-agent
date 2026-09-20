#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from context import build_realtime_context, utterance_view
from llm_client import LLMRequestBroker, PRIORITY_REALTIME
from memory import MemoryStore


REASONER_SYSTEM_PROMPT = """너는 embodied agent의 빠른 S2 realtime reasoner다.

역할:
- 현재 외부 화자의 발화와 최신 대화 맥락을 이해한다.
- 일반적인 대화 turn은 즉시 respond/wait로 처리한다.
- 복잡한 다단계 추론, 중요한 비교/설계 판단, 또는 짧은 fast pass로 신뢰하기 어려운 문제만 intent=deliberate로 승격한다.
- 단순 인사, 확인, 짧은 질문, 명확한 사실 응답을 deliberate로 보내지 않는다.
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
  "intent": "respond" | "wait" | "deliberate",
  "response": "respond일 때 말할 짧은 한국어 문장. wait/deliberate이면 빈 문자열",
  "reason": "deliberate일 때만 짧은 승격 이유. 그 외에는 빈 문자열"
}
"""

DELIBERATE_SYSTEM_PROMPT = """너는 embodied agent의 S2 deliberate reasoner다.
빠른 S2가 이 turn은 더 깊은 추론이 필요하다고 판단해 승격했다.

역할:
- 같은 최신 context를 바탕으로 충분히 검토한 뒤 최종 응답을 만든다.
- working memory보다 최신 raw evidence를 우선한다.
- 입력에 없는 사실을 만들지 않는다.
- source=ear 또는 speaker_role=external_speaker는 에이전트가 들은 외부 화자의 발화이며 에이전트 자신의 발화가 아니다.
- orchestration 명령을 만들지 않는다.

출력은 JSON object 하나만 사용하고 마크다운 코드펜스를 쓰지 마라.
출력 형식:
{
  "intent": "respond" | "wait",
  "response": "respond일 때 말할 자연스러운 한국어 응답. wait이면 빈 문자열"
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
    mode: str = "fast"


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


def normalize_result(
    value: dict[str, Any],
    *,
    allow_deliberate: bool = False,
) -> tuple[str, str, str]:
    intent = str(value.get("intent", "")).strip().lower()
    response = str(value.get("response", "")).strip()
    reason = str(value.get("reason", "")).strip()

    allowed = {"respond", "wait"}
    if allow_deliberate:
        allowed.add("deliberate")
    if intent not in allowed:
        raise ValueError(f"unsupported reasoner intent: {intent!r}")

    if intent == "respond" and not response:
        raise ValueError("reasoner returned respond with empty response")
    if intent != "respond":
        response = ""
    if intent != "deliberate":
        reason = ""

    return intent, response, reason


def build_reasoner_prompt(
    context: dict[str, Any],
    current_event: dict[str, Any],
) -> str:
    current = utterance_view(current_event)
    current_seq = int(current_event["utterance_seq"])
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


def _thinking_control(prompt: str, enabled: bool) -> str:
    # Qwen3 soft switch. We also send chat_template_kwargs at request level;
    # keeping both makes behavior robust across llama.cpp versions/templates.
    return prompt.rstrip() + ("\n\n/think" if enabled else "\n\n/no_think")


class RealtimeReasoner:
    def __init__(
        self,
        store: MemoryStore,
        broker: LLMRequestBroker,
        *,
        temperature: float = 0.30,
        max_tokens: int = 192,
        timeout_sec: float = 30.0,
        thinking: bool = False,
        deliberate_enabled: bool = True,
        deliberate_temperature: float = 0.35,
        deliberate_max_tokens: int = 768,
        deliberate_timeout_sec: float = 60.0,
        deliberate_thinking: bool = True,
    ) -> None:
        self.store = store
        self.broker = broker
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.thinking = thinking
        self.deliberate_enabled = deliberate_enabled
        self.deliberate_temperature = deliberate_temperature
        self.deliberate_max_tokens = deliberate_max_tokens
        self.deliberate_timeout_sec = deliberate_timeout_sec
        self.deliberate_thinking = deliberate_thinking

    async def reason(self, current_event: dict[str, Any]) -> ReasoningResult:
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
        fresh_through = int(context.get("fresh_through_utterance_seq", -1))
        if fresh_through > seq:
            raise StaleReasoningInput(
                f"context already contains newer utterance "
                f"fresh_through={fresh_through} based_on={seq}"
            )

        prompt = build_reasoner_prompt(context, current_event)
        started = time.monotonic()
        log(
            f"[reasoner] fast start episode={str(episode_id)[:8]} seq={seq} "
            f"thinking={self.thinking} working_upto="
            f"{context['working_memory']['authoritative_through_utterance_seq']} "
            f"tail={len(context['raw_tail'])}"
        )

        raw = await self.broker.complete(
            [
                {
                    "role": "system",
                    "content": _thinking_control(REASONER_SYSTEM_PROMPT, self.thinking),
                },
                {"role": "user", "content": prompt},
            ],
            priority=PRIORITY_REALTIME,
            label=f"realtime-fast:{episode_id}:{seq}",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_sec=self.timeout_sec,
            enable_thinking=self.thinking,
        )
        intent, response, reason = normalize_result(
            extract_json_object(raw),
            allow_deliberate=True,
        )
        fast_ms = int((time.monotonic() - started) * 1000)

        if intent != "deliberate":
            log(
                f"[reasoner] fast done episode={str(episode_id)[:8]} "
                f"seq={seq} intent={intent} latency={fast_ms}ms"
            )
            return ReasoningResult(
                episode_id=str(episode_id),
                based_on_utterance_seq=seq,
                intent=intent,
                response=response,
                mode="fast",
            )

        if not self.deliberate_enabled:
            log(
                f"[reasoner] deliberate requested but disabled "
                f"episode={str(episode_id)[:8]} seq={seq}"
            )
            # If escalation is disabled, fail closed to silence rather than
            # improvising an answer the fast model explicitly distrusted.
            return ReasoningResult(
                episode_id=str(episode_id),
                based_on_utterance_seq=seq,
                intent="wait",
                response="",
                mode="fast",
            )

        log(
            f"[reasoner] escalate episode={str(episode_id)[:8]} seq={seq} "
            f"fast_latency={fast_ms}ms reason={reason[:120]!r}"
        )
        deliberate_prompt = (
            prompt
            + "\n\nFAST PASS ESCALATION REASON:\n"
            + (reason or "빠른 판단에서 더 깊은 추론이 필요하다고 판단됨")
            + "\n\n이제 최종 respond 또는 wait만 결정하라. deliberate를 다시 출력하지 마라."
        )
        deep_started = time.monotonic()
        deep_raw = await self.broker.complete(
            [
                {
                    "role": "system",
                    "content": _thinking_control(
                        DELIBERATE_SYSTEM_PROMPT,
                        self.deliberate_thinking,
                    ),
                },
                {"role": "user", "content": deliberate_prompt},
            ],
            priority=PRIORITY_REALTIME,
            label=f"realtime-deliberate:{episode_id}:{seq}",
            temperature=self.deliberate_temperature,
            max_tokens=self.deliberate_max_tokens,
            timeout_sec=self.deliberate_timeout_sec,
            enable_thinking=self.deliberate_thinking,
        )
        deep_intent, deep_response, _ = normalize_result(
            extract_json_object(deep_raw),
            allow_deliberate=False,
        )
        deep_ms = int((time.monotonic() - deep_started) * 1000)
        log(
            f"[reasoner] deliberate done episode={str(episode_id)[:8]} "
            f"seq={seq} intent={deep_intent} latency={deep_ms}ms"
        )
        return ReasoningResult(
            episode_id=str(episode_id),
            based_on_utterance_seq=seq,
            intent=deep_intent,
            response=deep_response,
            mode="deliberate",
        )
