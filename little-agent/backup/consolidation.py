#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any

from context import utterance_view
from llm_client import LLMRequestBroker, PRIORITY_FINAL
from memory import MemoryStore


FINAL_SYSTEM_PROMPT = """너는 embodied agent가 종료된 경험을 장기 기억 후보로 정리하는 내부 기억 과정이다.

세계 모델 규칙:
- source=ear 또는 speaker_role=external_speaker 인 음성은 에이전트가 마이크로 들은 외부 화자의 발화다.
- 이것은 에이전트 자신의 발화가 아니다.
- 별도의 화자 식별 정보가 없으면 특정 인물이라고 단정하지 말고 '외부 화자' 또는 '대화 상대'로 표현하라.
- 에이전트 자신의 발화는 명시적으로 source=agent/mouth/tts 등으로 기록된 경우에만 그렇게 취급하라.

기억 규칙:
- WORKING MEMORY는 이전 raw 경험을 압축한 잠정 상태이며 완전한 원본이 아니다.
- RAW TAIL은 working memory 이후의 더 최신 직접 증거다.
- 둘이 충돌하면 RAW TAIL을 우선한다.
- 실제 입력에서 뒷받침되지 않는 사실, 감정, 관계, 선호, 목표를 새로 만들어내지 마라.
- 외부 화자가 말했다는 것과 객관적 사실을 구분하라.
- 장기간 다시 쓸 가치가 낮은 사소한 표현은 facts에 넣지 않아도 된다.
- 결과는 JSON object 하나만 출력하고 마크다운 코드펜스를 쓰지 마라.
- 모든 자연어 값은 고유명사/기술용어를 제외하고 한국어로 작성하라.

출력 형식:
{
  "summary": "종료된 경험의 핵심을 한국어 1~3문장으로 요약",
  "topics": ["주제1", "주제2"],
  "facts": [
    {
      "text": "미래에 유용할 수 있는 사실",
      "stability": 0.0,
      "confidence": 0.0
    }
  ],
  "importance": 0.0,
  "unresolved": ["아직 이어질 수 있는 질문이나 맥락"]
}
"""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    raise ValueError("could not extract JSON object from final consolidation")


def clamp(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = str(result.get("summary", "")).strip()

    topics_raw = result.get("topics", [])
    topics = (
        [str(item).strip() for item in topics_raw if str(item).strip()]
        if isinstance(topics_raw, list)
        else []
    )

    facts: list[dict[str, Any]] = []
    facts_raw = result.get("facts", [])
    if isinstance(facts_raw, list):
        for item in facts_raw:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    facts.append(
                        {"text": text, "stability": 0.5, "confidence": 0.5}
                    )
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    facts.append(
                        {
                            "text": text,
                            "stability": clamp(item.get("stability"), 0.5),
                            "confidence": clamp(item.get("confidence"), 0.5),
                        }
                    )

    unresolved_raw = result.get("unresolved", [])
    unresolved = (
        [str(item).strip() for item in unresolved_raw if str(item).strip()]
        if isinstance(unresolved_raw, list)
        else []
    )

    return {
        "schema_version": 2,
        "summary": summary,
        "topics": topics,
        "facts": facts,
        "importance": clamp(result.get("importance"), 0.0),
        "unresolved": unresolved,
        "consolidated_at": utc_now_iso(),
    }


def fetch_unconsolidated_episode_ids(
    store: MemoryStore,
    limit: int = 10,
) -> list[str]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM episodes
            WHERE status='closed'
              AND ready_for_consolidation=1
              AND (consolidation_json IS NULL OR TRIM(consolidation_json)='')
            ORDER BY started_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [str(row["id"]) for row in rows]


def save_final(
    store: MemoryStore,
    episode_id: str,
    result: dict[str, Any],
) -> None:
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE episodes
            SET summary=?, consolidation_json=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND status='closed' AND ready_for_consolidation=1
            """,
            (result["summary"], encoded, episode_id),
        )
        conn.commit()


def build_final_prompt(
    episode: dict[str, Any],
    working: dict[str, Any],
    tail: list[dict[str, Any]],
    legacy: list[dict[str, Any]] | None = None,
) -> str:
    working_view = {
        "version": int(working["version"]),
        "upto_utterance_seq": int(working["upto_utterance_seq"]),
        "summary": working["summary"],
        "state": working["state"],
    }

    if legacy is not None:
        raw_view = [utterance_view(item) for item in legacy]
        source_note = "LEGACY RAW UTTERANCES는 이 에피소드의 전체 인식 발화다."
    else:
        raw_view = [utterance_view(item) for item in tail]
        source_note = (
            "RAW TAIL은 WORKING MEMORY의 upto_utterance_seq 이후에 발생한 "
            "최신 외부 발화다."
        )

    return f"""다음 종료된 에피소드를 장기 기억 후보로 consolidation 하라.

EPISODE:
{json.dumps(episode, ensure_ascii=False, indent=2)}

WORKING MEMORY:
{json.dumps(working_view, ensure_ascii=False, indent=2)}

{source_note}
RAW UTTERANCES / TAIL:
{json.dumps(raw_view, ensure_ascii=False, indent=2)}
"""


async def consolidate_episode(
    store: MemoryStore,
    broker: LLMRequestBroker,
    episode_id: str,
) -> bool:
    episode = await asyncio.to_thread(store.get_episode, episode_id)
    if episode is None:
        return False
    if episode["status"] != "closed" or not episode["ready_for_consolidation"]:
        return False

    working = await asyncio.to_thread(store.get_working_memory, episode_id)
    upto = int(working["upto_utterance_seq"])
    tail = await asyncio.to_thread(
        store.episode_final_utterances,
        episode_id,
        upto,
        None,
    )

    legacy: list[dict[str, Any]] | None = None
    if upto < 0 and not tail:
        legacy = await asyncio.to_thread(store.legacy_semantic_events, episode_id)
        if not legacy:
            log(f"[final] no semantic speech episode={episode_id}")
            return False

    prompt = build_final_prompt(episode, working, tail, legacy)
    log(
        f"[final] start episode={episode_id} "
        f"working_upto={upto} tail={len(tail)}"
    )

    try:
        raw = await broker.complete(
            [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            priority=PRIORITY_FINAL,
            label=f"final:{episode_id}",
            temperature=0.15,
            max_tokens=1024,
        )
        normalized = normalize_result(extract_json_object(raw))
        if not normalized["summary"]:
            raise ValueError("final model returned empty summary")

        await asyncio.to_thread(save_final, store, episode_id, normalized)
        log(
            f"[final] done episode={episode_id} "
            f"importance={normalized['importance']:.2f} "
            f'summary="{normalized["summary"]}"'
        )
        return True
    except Exception as exc:
        log(
            f"[final] failed episode={episode_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


async def final_consolidation_worker(
    store: MemoryStore,
    broker: LLMRequestBroker,
    poll_interval_sec: float = 3.0,
) -> None:
    log("[final] worker started")
    while True:
        episode_ids = await asyncio.to_thread(
            fetch_unconsolidated_episode_ids,
            store,
            10,
        )
        if not episode_ids:
            await asyncio.sleep(poll_interval_sec)
            continue

        for episode_id in episode_ids:
            ok = await consolidate_episode(store, broker, episode_id)
            if not ok:
                await asyncio.sleep(poll_interval_sec)
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./data/agent.db")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--poll-sec", type=float, default=3.0)
    return parser


async def run_standalone(args: argparse.Namespace) -> None:
    store = MemoryStore(args.db)
    await asyncio.to_thread(store.initialize)
    broker = LLMRequestBroker(args.llm_url, args.model)
    await asyncio.gather(
        broker.worker(),
        final_consolidation_worker(store, broker, args.poll_sec),
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run_standalone(args))
    except KeyboardInterrupt:
        log("[final] stopped")


if __name__ == "__main__":
    main()
