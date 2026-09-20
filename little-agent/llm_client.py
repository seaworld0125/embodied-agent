#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import itertools
import json
import urllib.request
from dataclasses import dataclass, field


PRIORITY_REALTIME = 0
PRIORITY_ROLLING = 10
PRIORITY_FINAL = 20


class EmptyLLMContentError(RuntimeError):
    def __init__(self, *, finish_reason: str | None, reasoning_content: str | None) -> None:
        snippet = (reasoning_content or "").strip().replace("\n", " ")[:240]
        detail = f" finish_reason={finish_reason!r}"
        if snippet:
            detail += f" reasoning={snippet!r}"
        super().__init__("llama-server returned empty content;" + detail)
        self.finish_reason = finish_reason
        self.reasoning_content = reasoning_content


def post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: float,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool | None = None,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        # llama.cpp forwards this to Qwen3's chat template. Keeping it
        # request-scoped lets fast S2 run non-thinking while memory workers
        # can still use thinking mode on the same server.
        body["chat_template_kwargs"] = {
            "enable_thinking": bool(enable_thinking),
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

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        reasoning_content = message.get("reasoning_content")
        raise EmptyLLMContentError(
            finish_reason=choice.get("finish_reason"),
            reasoning_content=(
                reasoning_content if isinstance(reasoning_content, str) else None
            ),
        )

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
    enable_thinking: bool | None = field(compare=False, default=None)


class LLMRequestBroker:
    """Two logical lanes over one llama-server.

    Realtime requests bypass the background priority queue. Rolling/final
    remain serialized in priority order. Thinking mode is request-scoped, so
    all lanes can share one Qwen3 llama-server without one global switch.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local",
        realtime_concurrency: int = 2,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.queue: asyncio.PriorityQueue[_QueuedRequest] = asyncio.PriorityQueue()
        self._counter = itertools.count()
        self._realtime_slots = asyncio.Semaphore(max(1, realtime_concurrency))
        self._realtime_active = 0
        self._realtime_idle = asyncio.Event()
        self._realtime_idle.set()

    @property
    def realtime_active(self) -> int:
        return self._realtime_active

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        priority: int,
        label: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_sec: float = 120.0,
        enable_thinking: bool | None = None,
    ) -> str:
        if priority <= PRIORITY_REALTIME:
            return await self._complete_realtime(
                messages=messages,
                label=label,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                enable_thinking=enable_thinking,
            )

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
                enable_thinking=enable_thinking,
            )
        )
        return await future

    async def _complete_realtime(
        self,
        *,
        messages: list[dict[str, str]],
        label: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
        enable_thinking: bool | None,
    ) -> str:
        del label
        async with self._realtime_slots:
            self._realtime_active += 1
            self._realtime_idle.clear()
            try:
                return await asyncio.to_thread(
                    post_chat_completion,
                    self.base_url,
                    self.model,
                    messages,
                    timeout_sec,
                    temperature,
                    max_tokens,
                    enable_thinking,
                )
            finally:
                self._realtime_active -= 1
                if self._realtime_active <= 0:
                    self._realtime_active = 0
                    self._realtime_idle.set()

    async def worker(self) -> None:
        while True:
            request = await self.queue.get()
            try:
                await self._realtime_idle.wait()
                result = await asyncio.to_thread(
                    post_chat_completion,
                    self.base_url,
                    self.model,
                    request.messages,
                    request.timeout_sec,
                    request.temperature,
                    request.max_tokens,
                    request.enable_thinking,
                )
                if not request.future.cancelled():
                    request.future.set_result(result)
            except Exception as exc:
                if not request.future.cancelled():
                    request.future.set_exception(exc)
            finally:
                self.queue.task_done()
