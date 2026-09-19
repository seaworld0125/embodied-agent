#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import itertools
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any


PRIORITY_REALTIME = 0
PRIORITY_ROLLING = 10
PRIORITY_FINAL = 20


def post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: float,
    temperature: float,
    max_tokens: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"llama-server returned no choices: {payload}")

    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"llama-server returned empty content: {payload}")

    return content.strip()


@dataclass(order=True)
class _QueuedRequest:
    priority: int
    order: int
    future: asyncio.Future[str] = field(compare=False)
    messages: list[dict[str, str]] = field(compare=False)
    temperature: float = field(compare=False)
    max_tokens: int = field(compare=False)
    timeout_sec: float = field(compare=False)
    label: str = field(compare=False)


class LLMRequestBroker:
    """
    One local model, multiple cognitive workloads.

    Lower priority number wins:
      0  realtime reasoning (reserved for later)
      10 rolling working-memory consolidation
      20 final long-term consolidation

    V1 intentionally executes one request at a time. This prevents background
    memory work from saturating a 16 GB edge machine. Later, this can be
    replaced with multiple llama-server slots without changing callers.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local",
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.queue: asyncio.PriorityQueue[_QueuedRequest] = (
            asyncio.PriorityQueue()
        )
        self._counter = itertools.count()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        priority: int,
        label: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_sec: float = 120.0,
    ) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        await self.queue.put(
            _QueuedRequest(
                priority=priority,
                order=next(self._counter),
                future=future,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                label=label,
            )
        )

        return await future

    async def worker(self) -> None:
        while True:
            request = await self.queue.get()

            try:
                result = await asyncio.to_thread(
                    post_chat_completion,
                    self.base_url,
                    self.model,
                    request.messages,
                    request.timeout_sec,
                    request.temperature,
                    request.max_tokens,
                )

                if not request.future.cancelled():
                    request.future.set_result(result)

            except Exception as exc:
                if not request.future.cancelled():
                    request.future.set_exception(exc)

            finally:
                self.queue.task_done()
