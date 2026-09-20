#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from config import load_config
from core import build_parser
from llm_client import post_chat_completion
from memory import MemoryStore
from reasoner import RealtimeReasoner


class SequenceBroker:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        return self.responses.pop(0)


def speech_final(episode_id: str, seq: int, text: str) -> dict:
    return {
        "schema_version": 1,
        "id": str(uuid4()),
        "type": "speech.final",
        "source": "ear",
        "occurred_at": "2026-09-20T00:00:01+00:00",
        "started_at": "2026-09-20T00:00:00+00:00",
        "ended_at": "2026-09-20T00:00:01+00:00",
        "episode_id": episode_id,
        "episode_position": seq,
        "utterance_seq": seq,
        "payload": {
            "utterance_id": f"u{seq}",
            "text": text,
            "language": "ko",
        },
    }


class ConfigTests(unittest.TestCase):
    def test_toml_overrides_defaults_and_cli_can_override_toml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "agent.toml"
            path.write_text(
                """
[vad]
threshold = 0.73

[turn]
grace_ms = 275

[reasoner.fast]
max_tokens = 144
thinking = false

[reasoner.deliberate]
enabled = true
thinking = true
max_tokens = 700
""".strip(),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["vad"]["threshold"], 0.73)
            self.assertEqual(config["turn"]["grace_ms"], 275)
            self.assertEqual(config["reasoner"]["fast"]["max_tokens"], 144)
            self.assertFalse(config["reasoner"]["fast"]["thinking"])
            self.assertTrue(config["reasoner"]["deliberate"]["thinking"])

            args = build_parser(config, str(path)).parse_args(
                ["--vad-threshold", "0.81", "--turn-grace-ms", "190"]
            )
            self.assertEqual(args.vad_threshold, 0.81)
            self.assertEqual(args.turn_grace_ms, 190)
            self.assertEqual(args.reasoner_max_tokens, 144)


class ReasonerEscalationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "agent.db")
        self.store.initialize()
        self.episode_id = "episode-config-test"
        self.store.open_episode(self.episode_id, "2026-09-20T00:00:00+00:00")
        self.current = speech_final(self.episode_id, 0, "이 설계의 race condition을 분석해줘")
        self.store.persist_event(
            self.current,
            self.episode_id,
            0,
            0,
            False,
        )

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_fast_path_is_non_thinking(self) -> None:
        broker = SequenceBroker([
            '{"intent":"respond","response":"바로 답할게.","reason":""}'
        ])
        reasoner = RealtimeReasoner(
            self.store,
            broker,  # type: ignore[arg-type]
            thinking=False,
        )
        result = await reasoner.reason(self.current)
        self.assertEqual(result.mode, "fast")
        self.assertEqual(result.intent, "respond")
        self.assertEqual(len(broker.calls), 1)
        self.assertFalse(broker.calls[0]["enable_thinking"])
        self.assertIn("/no_think", broker.calls[0]["messages"][0]["content"])

    async def test_deliberate_escalation_switches_thinking_on(self) -> None:
        broker = SequenceBroker([
            '{"intent":"deliberate","response":"","reason":"다단계 분석 필요"}',
            '{"intent":"respond","response":"세 가지 race가 보여.","reason":""}',
        ])
        reasoner = RealtimeReasoner(
            self.store,
            broker,  # type: ignore[arg-type]
            thinking=False,
            deliberate_enabled=True,
            deliberate_thinking=True,
        )
        result = await reasoner.reason(self.current)
        self.assertEqual(result.mode, "deliberate")
        self.assertEqual(result.response, "세 가지 race가 보여.")
        self.assertEqual(len(broker.calls), 2)
        self.assertFalse(broker.calls[0]["enable_thinking"])
        self.assertTrue(broker.calls[1]["enable_thinking"])
        self.assertIn("/think", broker.calls[1]["messages"][0]["content"])

    async def test_deliberate_request_waits_when_escalation_disabled(self) -> None:
        broker = SequenceBroker([
            '{"intent":"deliberate","response":"","reason":"복잡함"}'
        ])
        reasoner = RealtimeReasoner(
            self.store,
            broker,  # type: ignore[arg-type]
            deliberate_enabled=False,
        )
        result = await reasoner.reason(self.current)
        self.assertEqual(result.intent, "wait")
        self.assertEqual(len(broker.calls), 1)


class LLMRequestBodyTests(unittest.TestCase):
    def test_enable_thinking_is_request_scoped(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = post_chat_completion(
                "http://localhost:8080",
                "local",
                [{"role": "user", "content": "hello"}],
                5.0,
                0.2,
                64,
                False,
            )
        self.assertEqual(result, "ok")
        self.assertEqual(
            captured["body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
