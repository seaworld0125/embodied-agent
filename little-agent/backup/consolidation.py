#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from memory import MemoryStore


SYSTEM_PROMPT = """너는 이 에이전트 자신의 기억을 정리하는 내부 인지 과정이다.

주어진 에피소드는 에이전트가 실제로 경험한 raw event들이다.
너의 역할은 이 경험을 미래에 다시 사용할 수 있는 기억으로 압축하는 것이다.

핵심 규칙:
- 주어진 이벤트에 없는 사실을 추측하지 마라.
- 일시적인 사실과 지속적으로 유용한 사실을 구분하라.
- 사용자의 말은 "사용자가 당시 그렇게 말했다"는 사실로 취급하라.
- 확실하지 않은 내용은 confidence를 낮춰라.
- 감정, 선호, 관계, 목표를 근거 없이 만들어내지 마라.
- 결과는 반드시 JSON object 하나만 출력하라.
- 마크다운 코드펜스는 사용하지 마라.

언어 규칙:
- JSON key는 아래에 지정된 영문 key를 그대로 사용한다.
- summary 값은 반드시 자연스러운 한국어로 작성한다.
- topics 배열의 모든 값은 반드시 한국어로 작성한다.
- facts[].text의 모든 값은 반드시 한국어로 작성한다.
- unresolved 배열의 모든 값은 반드시 한국어로 작성한다.
- 제품명, 프로젝트명, API명, 라이브러리명 등 고유명사/기술 용어를 제외하고 영어 문장을 사용하지 마라.

출력 형식:
{
  "summary": "이 에피소드의 핵심을 한국어 1~3문장으로 요약",
  "topics": ["주제1", "주제2"],
  "facts": [
    {
      "text": "미래에 유용할 수 있는 사실을 한국어로 작성",
      "stability": 0.0,
      "confidence": 0.0
    }
  ],
  "importance": 0.0,
  "unresolved": ["아직 해결되지 않은 질문이나 후속 맥락"]
}

점수 의미:
- stability: 이 사실이 장기간 유지될 가능성
- confidence: 이벤트가 이 사실을 얼마나 직접적으로 뒷받침하는지
- importance: 이 에피소드 전체가 미래 행동/대화에 얼마나 유용한지
"""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_unconsolidated_episode_ids(
    store: MemoryStore,
    limit: int = 10,
) -> list[str]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM episodes
            WHERE status = 'closed'
              AND (
                    consolidation_json IS NULL
                    OR TRIM(consolidation_json) = ''
                  )
            ORDER BY started_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [str(row["id"]) for row in rows]


def get_episode_metadata(
    store: MemoryStore,
    episode_id: str,
) -> dict[str, Any] | None:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT
                id,
                status,
                started_at,
                ended_at,
                event_count,
                summary,
                consolidation_json
            FROM episodes
            WHERE id = ?
            """,
            (episode_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def save_consolidation(
    store: MemoryStore,
    episode_id: str,
    result: dict[str, Any],
) -> None:
    summary_text = str(
        result.get("summary", "")
    ).strip()

    encoded = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with store.connect() as conn:
        conn.execute(
            """
            UPDATE episodes
            SET
                summary = ?,
                consolidation_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND status = 'closed'
            """,
            (
                summary_text,
                encoded,
                episode_id,
            ),
        )
        conn.commit()


def event_to_line(event: dict[str, Any]) -> str:
    event_type = event.get("type", "unknown")
    source = event.get("source", "unknown")
    occurred_at = event.get("occurred_at", "")
    payload = event.get("payload", {})

    if event_type == "speech":
        text = payload.get("text", "")
        duration = payload.get("duration_ms")

        return (
            f"- [{occurred_at}] "
            f"source={source} type=speech "
            f"duration_ms={duration}: {text}"
        )

    return (
        f"- [{occurred_at}] "
        f"source={source} type={event_type}: "
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def build_user_prompt(
    episode: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    event_lines = "\n".join(
        event_to_line(event)
        for event in events
    )

    return f"""다음 에피소드를 consolidation 하라.

episode_id: {episode["id"]}
started_at: {episode["started_at"]}
ended_at: {episode["ended_at"]}
event_count: {episode["event_count"]}

RAW EVENTS:
{event_lines}
"""


def post_chat_completion(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_sec: float,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }

    data = json.dumps(
        body,
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout_sec,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"llama-server returned no choices: {payload}"
        )

    message = choices[0].get("message") or {}
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"llama-server returned empty content: {payload}"
        )

    return content.strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()

    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    decoder = json.JSONDecoder()

    for index, char in enumerate(cleaned):
        if char != "{":
            continue

        try:
            value, _ = decoder.raw_decode(
                cleaned[index:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            return value

    raise ValueError(
        "could not extract JSON object from model output: "
        + text[:500]
    )


def clamp_score(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default

    return max(0.0, min(1.0, score))


def normalize_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    summary = str(
        result.get("summary", "")
    ).strip()

    topics_raw = result.get("topics", [])
    topics = (
        [
            str(topic).strip()
            for topic in topics_raw
            if str(topic).strip()
        ]
        if isinstance(topics_raw, list)
        else []
    )

    facts_raw = result.get("facts", [])
    facts: list[dict[str, Any]] = []

    if isinstance(facts_raw, list):
        for item in facts_raw:
            if isinstance(item, str):
                text = item.strip()

                if text:
                    facts.append(
                        {
                            "text": text,
                            "stability": 0.5,
                            "confidence": 0.5,
                        }
                    )

                continue

            if not isinstance(item, dict):
                continue

            text = str(
                item.get("text", "")
            ).strip()

            if not text:
                continue

            facts.append(
                {
                    "text": text,
                    "stability": clamp_score(
                        item.get("stability"),
                        0.5,
                    ),
                    "confidence": clamp_score(
                        item.get("confidence"),
                        0.5,
                    ),
                }
            )

    unresolved_raw = result.get(
        "unresolved",
        [],
    )

    unresolved = (
        [
            str(item).strip()
            for item in unresolved_raw
            if str(item).strip()
        ]
        if isinstance(unresolved_raw, list)
        else []
    )

    return {
        "schema_version": 1,
        "summary": summary,
        "topics": topics,
        "facts": facts,
        "importance": clamp_score(
            result.get("importance"),
            0.0,
        ),
        "unresolved": unresolved,
        "consolidated_at": utc_now_iso(),
    }


async def consolidate_episode(
    store: MemoryStore,
    episode_id: str,
    base_url: str,
    model: str,
    timeout_sec: float,
    temperature: float,
) -> bool:
    episode = await asyncio.to_thread(
        get_episode_metadata,
        store,
        episode_id,
    )

    if episode is None:
        log(
            f"[consolidation] episode not found: "
            f"{episode_id}"
        )
        return False

    if episode["status"] != "closed":
        return False

    events = await asyncio.to_thread(
        store.episode_events,
        episode_id,
    )

    if not events:
        log(
            f"[consolidation] no events: "
            f"{episode_id}"
        )
        return False

    prompt = build_user_prompt(
        episode,
        events,
    )

    log(
        f"[consolidation] start "
        f"id={episode_id} "
        f"events={len(events)}"
    )

    try:
        raw_output = await asyncio.to_thread(
            post_chat_completion,
            base_url,
            model,
            prompt,
            timeout_sec,
            temperature,
        )

        parsed = extract_json_object(
            raw_output
        )

        normalized = normalize_result(
            parsed
        )

        if not normalized["summary"]:
            raise ValueError(
                "model returned an empty summary"
            )

        await asyncio.to_thread(
            save_consolidation,
            store,
            episode_id,
            normalized,
        )

        log(
            f"[consolidation] done "
            f"id={episode_id} "
            f"importance="
            f"{normalized['importance']:.2f} "
            f'summary="{normalized["summary"]}"'
        )

        return True

    except urllib.error.URLError as exc:
        log(
            f"[consolidation] llama-server unavailable: "
            f"{exc}"
        )
        return False

    except Exception as exc:
        log(
            f"[consolidation] failed "
            f"id={episode_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


async def consolidation_worker(
    store: MemoryStore,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "local",
    poll_interval_sec: float = 3.0,
    timeout_sec: float = 120.0,
    temperature: float = 0.2,
) -> None:
    """
    Consolidate closed episodes whose consolidation_json is still empty.

    DB polling is intentional:
    - episode lifecycle remains owned by EpisodeBuilder
    - unfinished consolidation survives process restarts
    """

    log(
        "[consolidation] worker started "
        f"url={base_url}"
    )

    while True:
        episode_ids = await asyncio.to_thread(
            fetch_unconsolidated_episode_ids,
            store,
            10,
        )

        if not episode_ids:
            await asyncio.sleep(
                poll_interval_sec
            )
            continue

        for episode_id in episode_ids:
            success = await consolidate_episode(
                store=store,
                episode_id=episode_id,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
                temperature=temperature,
            )

            if not success:
                await asyncio.sleep(
                    poll_interval_sec
                )
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate closed episodes using "
            "the local llama-server."
        )
    )

    parser.add_argument(
        "--db",
        default="./data/agent.db",
    )

    parser.add_argument(
        "--llm-url",
        default="http://127.0.0.1:8080",
    )

    parser.add_argument(
        "--model",
        default="local",
    )

    parser.add_argument(
        "--poll-sec",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    return parser


async def run_standalone(
    args: argparse.Namespace,
) -> None:
    store = MemoryStore(args.db)

    await asyncio.to_thread(
        store.initialize
    )

    await consolidation_worker(
        store=store,
        base_url=args.llm_url,
        model=args.model,
        poll_interval_sec=args.poll_sec,
        timeout_sec=args.timeout_sec,
        temperature=args.temperature,
    )


def main() -> None:
    args = build_parser().parse_args()

    try:
        asyncio.run(
            run_standalone(args)
        )
    except KeyboardInterrupt:
        log("[consolidation] stopped")


if __name__ == "__main__":
    main()
