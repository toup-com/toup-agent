"""Matched deterministic latency probe for the actual realtime voice handler.

Run this file in a fresh Python process with ``TOUP_BENCH_SOURCE`` pointing at
the backend checkout to measure.  It contacts no provider, database, or device;
fake sockets drive the real ``realtime_voice_ws`` event readers.

Example:
    TOUP_BENCH_SOURCE=/path/to/toup-platform \
      python tests/benchmark_voice_lifecycle.py --iterations 15 --delay-ms 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock


SOURCE = Path(os.environ.get("TOUP_BENCH_SOURCE", Path(__file__).parents[2])).resolve()
sys.path.insert(0, str(SOURCE / "backend"))

from app.api import ws_realtime as voice  # noqa: E402
from app.db import database  # noqa: E402


class FakeSocket:
    def __init__(self):
        self.inbound: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[dict] = []
        self.sent_at: list[float] = []
        self.changed = asyncio.Event()
        self.closed = False

    def feed(self, event: dict | None) -> None:
        self.inbound.put_nowait(json.dumps(event) if event is not None else None)

    async def _receive(self) -> str | None:
        return await self.inbound.get()

    async def _send(self, event: dict) -> None:
        await asyncio.sleep(0)
        self.sent.append(event)
        self.sent_at.append(time.perf_counter())
        self.changed.set()

    async def wait_for(self, predicate, *, after: int = 0) -> tuple[dict, float]:
        async with asyncio.timeout(5.0):
            while True:
                self.changed.clear()
                for index, event in enumerate(self.sent[after:], start=after):
                    if predicate(event):
                        return event, self.sent_at[index]
                await self.changed.wait()

    async def close(self, **_) -> None:
        self.closed = True


class ClientSocket(FakeSocket):
    scope = {"subprotocols": ["toup.auth.v1", "bearer.benchmark-token"]}

    async def accept(self, **_) -> None:
        pass

    async def receive_text(self) -> str | None:
        return await self._receive()

    async def send_json(self, event: dict) -> None:
        await self._send(event)


class ProviderSocket(FakeSocket):
    async def send(self, raw: str) -> None:
        await self._send(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        raw = await self._receive()
        if raw is None:
            raise StopAsyncIteration
        return raw


class DelayedThink:
    def __init__(self, delay: float):
        self.delay = delay
        self.entered = asyncio.Event()
        self.calls = 0

    async def __call__(self, *_, **__):
        self.calls += 1
        self.entered.set()
        await asyncio.sleep(self.delay)
        return "matched research result", "benchmark-model"


class DelayedSave:
    def __init__(self, delay: float):
        self.delay = delay
        self.entered = asyncio.Event()
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, _user_id, _session_id, user_text, assistant_text, **_):
        self.calls.append((user_text, assistant_text))
        self.entered.set()
        await asyncio.sleep(self.delay)


def configure(provider: ProviderSocket) -> None:
    voice._authenticate_ws = AsyncMock(return_value="benchmark-user")
    voice._get_user_openai_key_ex = AsyncMock(return_value=("fake-key", True))
    voice._get_or_create_voice_session = AsyncMock(return_value="benchmark-session")
    voice._ensure_vps_user = AsyncMock()
    voice._get_vps_info = AsyncMock(return_value=None)
    voice.build_realtime_instructions = AsyncMock(return_value="Benchmark instructions")
    voice.resolve_voice_language = AsyncMock(return_value=None)
    voice._maybe_meter_response = lambda *_: None
    voice._resolve_v2_for_user = lambda _: False
    voice._lang_hint_enabled = lambda: False
    voice.websockets.connect = AsyncMock(return_value=provider)
    voice.settings.voice_context_from_agent = False
    voice.settings.voice_context_shadow = False
    voice.settings.auto_extract_memories = False
    voice.settings.security_leak_filter = False
    voice._instr_cache = {
        "benchmark-user": (
            "Benchmark instructions", voice.REALTIME_TOOLS, time.monotonic(),
        ),
    }

    def no_database():
        raise RuntimeError("Database intentionally unavailable in latency probe")

    database.async_session_maker = no_database


class Session:
    def __init__(self):
        self.client = ClientSocket()
        self.provider = ProviderSocket()
        configure(self.provider)
        self.task = asyncio.create_task(voice.realtime_voice_ws(
            self.client,
            token=None,
            session_id="benchmark-session",
            onboarding=False,
        ))

    async def ready(self) -> None:
        await self.client.wait_for(lambda event: event.get("type") == "ready")
        await self.provider.wait_for(
            lambda event: event.get("type") == "session.update"
            and bool(event.get("session", {}).get("tools")),
        )

    async def stop(self) -> None:
        self.client.feed({"type": "stop"})
        await asyncio.wait_for(asyncio.shield(self.task), timeout=5.0)


async def function_reader_sample(delay: float) -> dict[str, float]:
    think = DelayedThink(delay)
    voice._think = think
    voice._save_voice_messages = AsyncMock()
    session = Session()
    await session.ready()
    item = {
        "type": "function_call",
        "name": "think",
        "call_id": "call-a",
        "arguments": json.dumps({"task": "Run matched research"}),
    }
    submitted_at = time.perf_counter()
    session.provider.feed({
        "type": "response.created", "response": {"id": "response-a"},
    })
    session.provider.feed({
        "type": "response.output_item.done",
        "response_id": "response-a",
        "item": item,
    })
    _, ack_at = await session.client.wait_for(
        lambda event: event.get("type") == "tool_call.started",
    )
    await asyncio.wait_for(think.entered.wait(), timeout=1.0)
    session.provider.feed({
        "type": "response.done",
        "response": {"id": "response-a", "status": "completed", "output": [item]},
    })
    control_at = time.perf_counter()
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    _, vad_at = await session.client.wait_for(
        lambda event: event.get("type") == "speech_started",
    )
    completed, completed_at = await session.client.wait_for(
        lambda event: event.get("type") == "tool_call.completed",
    )
    output, output_at = await session.provider.wait_for(
        lambda event: event.get("type") == "conversation.item.create"
        and event.get("item", {}).get("call_id") == "call-a",
    )
    assert think.calls == 1
    assert completed.get("result_preview") == "matched research result"
    assert output["item"]["output"] == "matched research result"
    assert sum(e.get("type") == "tool_call.completed" for e in session.client.sent) == 1
    await session.stop()
    return {
        "ack_ms": (ack_at - submitted_at) * 1000,
        "vad_ms": (vad_at - control_at) * 1000,
        "useful_result_ms": (completed_at - submitted_at) * 1000,
        "function_output_ms": (output_at - submitted_at) * 1000,
    }


async def persistence_reader_sample(delay: float) -> dict[str, float]:
    save = DelayedSave(delay)
    voice._save_voice_messages = save
    voice._think = AsyncMock(return_value=("unused", "benchmark-model"))
    session = Session()
    await session.ready()
    session.provider.feed({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "user-item",
        "transcript": "Only Toronto, please",
    })
    await asyncio.wait_for(save.entered.wait(), timeout=1.0)
    session.provider.feed({
        "type": "response.created", "response": {"id": "response-persist"},
    })
    session.provider.feed({
        "type": "response.output_audio_transcript.delta",
        "response_id": "response-persist",
        "delta": "Understood",
    })
    control_at = time.perf_counter()
    session.provider.feed({
        "type": "response.done",
        "response": {"id": "response-persist", "status": "completed", "output": []},
    })
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    _, done_at = await session.client.wait_for(
        lambda event: event.get("type") == "response_done",
    )
    _, vad_at = await session.client.wait_for(
        lambda event: event.get("type") == "speech_started",
    )
    await session.stop()
    assert save.calls == [
        ("Only Toronto, please", ""),
        ("", "Understood"),
    ]
    assert sum(e.get("type") == "transcript" for e in session.client.sent) == 1
    assert sum(e.get("type") == "response_done" for e in session.client.sent) == 1
    return {
        "response_done_ms": (done_at - control_at) * 1000,
        "vad_ms": (vad_at - control_at) * 1000,
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for key in samples[0]:
        values = sorted(sample[key] for sample in samples)
        result[key] = {
            "median": round(statistics.median(values), 3),
            "p95_nearest_rank": round(values[math.ceil(0.95 * len(values)) - 1], 3),
            "min": round(values[0], 3),
            "max": round(values[-1], 3),
        }
    return result


async def main(iterations: int, delay_ms: float) -> None:
    delay = delay_ms / 1000
    # Match the chat harness: exercise module/session setup once, then report
    # only the warm-process repetitions below.
    await function_reader_sample(delay)
    await persistence_reader_sample(delay)
    function_samples = [
        await function_reader_sample(delay) for _ in range(iterations)
    ]
    persistence_samples = [
        await persistence_reader_sample(delay) for _ in range(iterations)
    ]
    print(json.dumps({
        "source": str(SOURCE),
        "iterations": iterations,
        "warmup_samples_per_scenario": 1,
        "delay_ms": delay_ms,
        "percentile_method": "nearest-rank",
        "conditions": "warm process; fresh fake socket session per sample; no external I/O",
        "quality": {
            "function_result_exactly_once": True,
            "function_output_exactly_once": True,
            "transcript_exactly_once": True,
            "ordered_user_then_assistant_persistence": True,
        },
        "function_reader": summarize(function_samples),
        "persistence_reader": summarize(persistence_samples),
        "raw_samples": {
            "function_reader": [
                {key: round(value, 3) for key, value in sample.items()}
                for sample in function_samples
            ],
            "persistence_reader": [
                {key: round(value, 3) for key, value in sample.items()}
                for sample in persistence_samples
            ],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--delay-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    asyncio.run(main(args.iterations, args.delay_ms))
