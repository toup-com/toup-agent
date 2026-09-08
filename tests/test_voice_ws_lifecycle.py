"""Run the actual voice WebSocket handler with asynchronous fake transports.

No provider, database, hosted agent, or device is contacted. These tests exercise
the reader loops and their task/persistence integration, not copied AST logic.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import ws_realtime as voice


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


class FakeSocket:
    def __init__(self):
        self.inbound = asyncio.Queue()
        self.sent = []
        self.changed = asyncio.Event()
        self.closed = False
        self.receive_cancelled = False

    def feed(self, event):
        self.inbound.put_nowait(json.dumps(event) if event is not None else None)

    async def _receive(self):
        try:
            return await self.inbound.get()
        except asyncio.CancelledError:
            self.receive_cancelled = True
            raise

    async def _send(self, event):
        await asyncio.sleep(0)  # real suspension exposes loop/task races
        self.sent.append(event)
        self.changed.set()

    async def wait_for(self, predicate, *, after=0):
        async with asyncio.timeout(1.0):
            while True:
                self.changed.clear()
                for event in self.sent[after:]:
                    if predicate(event):
                        return event
                await self.changed.wait()

    async def close(self, **_):
        self.closed = True


class ClientSocket(FakeSocket):
    scope = {"subprotocols": ["toup.auth.v1", "bearer.test-token"]}

    async def accept(self, **_):
        pass

    async def receive_text(self):
        return await self._receive()

    async def send_json(self, event):
        await self._send(event)


class ProviderSocket(FakeSocket):
    async def send(self, raw):
        await self._send(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self._receive()
        if raw is None:
            raise StopAsyncIteration
        return raw


class BlockedOperation:
    def __init__(self, result):
        self.result = result
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.cancelled = False

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.entered.set()
        try:
            await self.release.wait()
            return self.result
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Session:
    def __init__(self, client, provider):
        self.client = client
        self.provider = provider
        self.releases = []
        self.task = asyncio.create_task(voice.realtime_voice_ws(
            client, token=None, session_id="session-test", onboarding=False,
        ))

    def block(self, monkeypatch, name, result):
        op = BlockedOperation(result)
        monkeypatch.setattr(voice, name, op)
        self.releases.append(op.release)
        return op

    async def ready(self):
        await self.client.wait_for(lambda e: e["type"] == "ready")
        await self.provider.wait_for(
            lambda e: e["type"] == "session.update" and bool(e["session"].get("tools")),
        )

    async def stop(self):
        self.client.feed({"type": "stop"})
        await asyncio.wait_for(asyncio.shield(self.task), timeout=1.0)

    def begin_call(self, name="think", call_id="call-a", response_id="r", *, created=True):
        item = {
            "type": "function_call", "name": name, "call_id": call_id,
            "arguments": json.dumps({"task": "Research this", "query": "music"}),
        }
        if created:
            self.provider.feed({"type": "response.created", "response": {"id": response_id}})
        self.provider.feed({"type": "response.output_item.done", "response_id": response_id, "item": item})
        return item

    def finish_response(self, items=(), response_id="r", status="completed"):
        self.provider.feed({"type": "response.done", "response": {
            "id": response_id, "status": status, "output": list(items),
        }})


@pytest_asyncio.fixture
async def session(monkeypatch):
    client, provider = ClientSocket(), ProviderSocket()
    monkeypatch.setattr(voice, "_authenticate_ws", AsyncMock(return_value="test-user"))
    monkeypatch.setattr(voice, "_get_user_openai_key_ex", AsyncMock(return_value=("fake-key", True)))
    monkeypatch.setattr(voice, "_get_or_create_voice_session", AsyncMock(return_value="session-test"))
    monkeypatch.setattr(voice, "_ensure_vps_user", AsyncMock())
    monkeypatch.setattr(voice, "_get_vps_info", AsyncMock(return_value=None))
    monkeypatch.setattr(voice, "build_realtime_instructions", AsyncMock(return_value="Test instructions"))
    monkeypatch.setattr(voice, "resolve_voice_language", AsyncMock(return_value=None))
    monkeypatch.setattr(voice, "_save_voice_messages", AsyncMock())
    monkeypatch.setattr(voice, "_maybe_meter_response", lambda *_: None)
    monkeypatch.setattr(voice, "_resolve_v2_for_user", lambda _: False)
    monkeypatch.setattr(voice, "_lang_hint_enabled", lambda: False)
    monkeypatch.setattr(voice.websockets, "connect", AsyncMock(return_value=provider))
    monkeypatch.setattr(voice.settings, "voice_context_from_agent", False)
    monkeypatch.setattr(voice.settings, "voice_context_shadow", False)
    monkeypatch.setattr(voice.settings, "auto_extract_memories", False)
    monkeypatch.setattr(voice.settings, "security_leak_filter", False)
    monkeypatch.setattr(voice, "_instr_cache", {
        "test-user": ("Test instructions", voice.REALTIME_TOOLS, time.monotonic()),
    })
    # Tool configuration DB unavailability has an existing safe fallback. This
    # fake enforces the no-database contract while exercising that real path.
    from app.db import database

    def no_database():
        raise RuntimeError("Database intentionally unavailable in socket test")

    monkeypatch.setattr(database, "async_session_maker", no_database)
    running = Session(client, provider)
    await running.ready()
    yield running
    for release in running.releases:
        release.set()
    if not running.task.done():
        await running.stop()
    else:
        await running.task


@pytest.mark.parametrize("tool,helper,result", [
    ("think", "_think", ("Completed research with sources", "test-model")),
    ("play_media", "_play_media_direct", ("Playback started", None)),
])
async def test_blocked_function_does_not_block_speech_or_microphone(
    session, monkeypatch, tool, helper, result,
):
    op = session.block(monkeypatch, helper, result)
    item = session.begin_call(name=tool)
    await asyncio.wait_for(op.entered.wait(), timeout=1.0)
    session.finish_response([item])
    session.provider.feed({"type": "response.output_item.done", "response_id": "r", "item": item})
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    session.client.feed({"type": "audio", "data": "microphone-frame"})
    await session.client.wait_for(lambda e: e["type"] == "speech_started")
    await session.provider.wait_for(lambda e: e["type"] == "input_audio_buffer.append")
    assert len(op.calls) == 1
    assert not op.cancelled
    assert not any(e["type"] == "tool_call.completed" for e in session.client.sent)
    op.release.set()
    await session.client.wait_for(lambda e: e["type"] == "tool_call.completed")
    await session.provider.wait_for(lambda e: e["type"] == "conversation.item.create")
    assert not any(e["type"] == "response.create" for e in session.provider.sent)
    await session.stop()
    assert session.provider.closed and session.client.closed
    assert session.provider.receive_cancelled


async def test_interrupt_stops_speech_without_cancelling_work(session, monkeypatch):
    op = session.block(monkeypatch, "_think", ("Result remains available", "test-model"))
    item = session.begin_call()
    await asyncio.wait_for(op.entered.wait(), timeout=1.0)
    session.client.feed({"type": "interrupt"})
    await session.provider.wait_for(lambda e: e["type"] == "response.cancel")
    assert not op.cancelled
    session.provider.feed({
        "type": "response.output_audio.delta", "response_id": "r", "delta": "stale-audio",
    })
    session.finish_response([item], status="cancelled")
    op.release.set()
    await session.client.wait_for(lambda e: e["type"] == "tool_call.completed")
    await session.provider.wait_for(lambda e: e["type"] == "conversation.item.create")
    assert not any(e["type"] in {"response.create"} for e in session.provider.sent)
    assert not any(e.get("data") == "stale-audio" for e in session.client.sent)
    # A subsequent utterance resumes speech and leaves the prior task intact.
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    session.provider.feed({"type": "input_audio_buffer.speech_stopped"})
    session.provider.feed({"type": "response.created", "response": {"id": "new"}})
    session.provider.feed({
        "type": "response.output_audio.delta", "response_id": "new", "delta": "new-audio",
    })
    await session.client.wait_for(lambda e: e.get("data") == "new-audio")
    assert len(op.calls) == 1


async def test_slow_persistence_does_not_block_response_speech_or_microphone(session, monkeypatch):
    op = session.block(monkeypatch, "_save_voice_messages", None)
    transcript = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "user-item", "transcript": "Only Toronto, please",
    }
    session.provider.feed(transcript)
    await asyncio.wait_for(op.entered.wait(), timeout=1.0)
    session.provider.feed(transcript)  # repeated input may not persist twice
    session.provider.feed({"type": "response.created", "response": {"id": "r"}})
    session.provider.feed({
        "type": "response.output_audio_transcript.delta", "response_id": "r", "delta": "Understood",
    })
    session.finish_response()
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    session.client.feed({"type": "audio", "data": "next-utterance"})
    await session.client.wait_for(lambda e: e["type"] == "response_done")
    await session.client.wait_for(lambda e: e["type"] == "speech_started")
    await session.provider.wait_for(lambda e: e["type"] == "input_audio_buffer.append")
    assert len(op.calls) == 1  # assistant save queued behind user save
    op.release.set()
    await session.stop()  # drains the serial persistence worker
    assert [args[2:4] for args, _ in op.calls] == [
        ("Only Toronto, please", ""), ("", "Understood"),
    ]
    assert len([e for e in session.client.sent if e["type"] == "transcript"]) == 1


async def test_out_of_order_function_batch_has_one_followup(session, monkeypatch):
    think = AsyncMock(return_value=("Done", "test-model"))
    monkeypatch.setattr(voice, "_think", think)
    items = [{
        "type": "function_call", "name": "think", "call_id": call_id,
        "arguments": '{"task":"Research"}',
    } for call_id in ("a", "b")]
    session.provider.feed({"type": "response.created", "response": {"id": "r"}})
    session.finish_response(items)  # full response precedes item notifications
    session.provider.feed({"type": "response.output_item.done", "response_id": "r", "item": items[0]})
    await session.client.wait_for(lambda e: e["type"] == "tool_call.completed")
    assert not any(e["type"] == "conversation.item.create" for e in session.provider.sent)
    session.provider.feed({"type": "response.output_item.done", "response_id": "r", "item": items[1]})
    await session.provider.wait_for(lambda e: e["type"] == "response.create")
    outputs = [e["item"] for e in session.provider.sent if e["type"] == "conversation.item.create"]
    assert [o["call_id"] for o in outputs] == ["a", "b"]
    assert think.await_count == 2
    session.finish_response(items)  # duplicate completion cannot speak again
    await session.stop()
    assert len([e for e in session.provider.sent if e["type"] == "response.create"]) == 1


async def test_stop_reaps_both_readers_and_never_starts_queued_native_mutation(session, monkeypatch):
    blocked = session.block(monkeypatch, "_think", ("unused", "test-model"))
    play = AsyncMock(return_value=("must not run", None))
    monkeypatch.setattr(voice, "_play_media_direct", play)
    session.begin_call()
    await asyncio.wait_for(blocked.entered.wait(), timeout=1.0)
    session.begin_call(name="play_media", call_id="queued", created=False)
    session.provider.feed({"type": "input_audio_buffer.speech_started"})
    await session.client.wait_for(lambda e: e["type"] == "speech_started")
    await session.stop()
    assert blocked.cancelled
    play.assert_not_awaited()
    assert session.provider.receive_cancelled
    assert session.client.closed and session.provider.closed


async def test_provider_disconnect_stops_waiting_client_reader(session):
    session.provider.feed(None)
    await asyncio.wait_for(asyncio.shield(session.task), timeout=1.0)
    assert session.client.receive_cancelled
    assert session.client.closed and session.provider.closed


async def test_late_audio_after_done_and_old_done_do_not_corrupt_new_response(session):
    session.provider.feed({"type": "response.created", "response": {"id": "old"}})
    session.finish_response(response_id="old")
    session.provider.feed({"type": "response.created", "response": {"id": "new"}})
    session.provider.feed({
        "type": "response.output_audio_transcript.delta", "response_id": "new", "delta": "New caption",
    })
    session.provider.feed({
        "type": "response.output_audio.delta", "response_id": "old", "delta": "late-old-audio",
    })
    session.finish_response(response_id="old")
    session.finish_response(response_id="new")
    result = await session.client.wait_for(lambda e: e["type"] == "response_done")
    assert result["text"] == "New caption"
    assert not any(e.get("data") == "late-old-audio" for e in session.client.sent)
