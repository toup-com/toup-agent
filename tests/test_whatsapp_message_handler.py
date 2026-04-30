"""Unit tests for app.agent.channels.shared.message_handler.

The shared dispatch handler is the bridge between BaseChannel adapters
(Discord/Slack/WhatsApp) and AgentRunner. It is the production-critical
correctness layer — without it, every inbound webhook lands on the
floor with a "no callback set" warning.

These tests use a fake BaseChannel subclass and a mocked AgentRunner
so the handler exercises real code paths (session cache, dispatch,
error handling, redaction) without needing a database or LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
)


# ── Test fakes ───────────────────────────────────────────────────


class FakeChannel(BaseChannel):
    """In-memory channel that records sent texts + typing calls."""

    def __init__(self, channel_type: ChannelType = ChannelType.WHATSAPP):
        super().__init__(channel_type)
        self.sent: List[Tuple[str, str]] = []
        self.typing_calls: List[str] = []
        self._typing_should_raise = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(
        self, chat_id: str, text: str, parse_mode: Optional[str] = None
    ) -> None:
        self.sent.append((chat_id, text))

    async def send_typing(self, chat_id: str) -> None:
        if self._typing_should_raise:
            raise RuntimeError("typing failed")
        self.typing_calls.append(chat_id)


def _fake_runner(
    *,
    response_text: str = "ok",
    response_session_id: str = "sess-1234",
    raise_exc: Optional[BaseException] = None,
) -> MagicMock:
    """Build a MagicMock AgentRunner whose run() returns / raises."""
    resp = MagicMock()
    resp.text = response_text
    resp.session_id = response_session_id
    resp.tokens_total = 0
    resp.tool_calls = []
    resp.processing_time_ms = 1
    runner = MagicMock()
    if raise_exc is not None:
        runner.run = AsyncMock(side_effect=raise_exc)
    else:
        runner.run = AsyncMock(return_value=resp)
    return runner


def _inbound(
    text: str = "hi",
    chat_id: str = "+14155552671",
    media: Optional[List[str]] = None,
    channel_type: ChannelType = ChannelType.WHATSAPP,
) -> InboundMessage:
    return InboundMessage(
        channel=channel_type,
        channel_user_id=chat_id,
        channel_chat_id=chat_id,
        text=text,
        media_paths=media or [],
    )


# ── Happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_runs_agent_and_sends_reply():
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner(response_text="hello back", response_session_id="sess-A")
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    await handler(_inbound("hi there"))

    assert ch.sent == [("+14155552671", "hello back")]
    runner.run.assert_awaited_once()
    kwargs = runner.run.await_args.kwargs
    assert kwargs["user_message"] == "hi there"
    assert kwargs["user_id"] == "u1"
    assert kwargs["channel"] == "whatsapp"
    # client_tz must NOT be passed — the handler relies on the
    # User.timezone fallback in resolve_day_chat_id_for_now so
    # WhatsApp turns share Day-as-Chat with web/mobile.
    assert "client_tz" not in kwargs


@pytest.mark.asyncio
async def test_typing_indicator_failure_does_not_block_run():
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    ch._typing_should_raise = True
    runner = _fake_runner()
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    await handler(_inbound())

    # Typing failed but the agent still ran and reply still landed.
    runner.run.assert_awaited_once()
    assert len(ch.sent) == 1


@pytest.mark.asyncio
async def test_session_cached_across_turns():
    """First turn produces session_id 'A', second turn must reuse it."""
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner(response_session_id="sess-A")
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")

    await handler(_inbound("first"))
    # Replace runner.run for second call: assert it's called with cached session_id.
    runner.run = AsyncMock(
        return_value=MagicMock(
            text="reply",
            session_id="sess-A",
            tokens_total=0,
            tool_calls=[],
            processing_time_ms=1,
        )
    )
    await handler(_inbound("second"))

    second_kwargs = runner.run.await_args.kwargs
    assert second_kwargs["session_id"] == "sess-A"


# ── Skip paths ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_and_no_media_skips():
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner()
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    await handler(_inbound(text="", media=[]))

    runner.run.assert_not_called()
    assert ch.sent == []


@pytest.mark.asyncio
async def test_media_only_message_runs_agent():
    """A photo with no caption is still a real user turn."""
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner()
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    await handler(_inbound(text="", media=["/tmp/cat.jpg"]))

    runner.run.assert_awaited_once()
    kwargs = runner.run.await_args.kwargs
    assert kwargs["media_paths"] == ["/tmp/cat.jpg"]


@pytest.mark.asyncio
async def test_missing_user_id_skips():
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner()
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="")
    await handler(_inbound())

    runner.run.assert_not_called()
    assert ch.sent == []


# ── Error / cancellation paths ───────────────────────────────────


@pytest.mark.asyncio
async def test_agent_exception_sends_user_safe_apology_and_swallows():
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner(raise_exc=RuntimeError("internal kaboom"))
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")

    # Must not raise — exception is caught, generic apology delivered.
    await handler(_inbound())

    assert len(ch.sent) == 1
    sent_chat, sent_text = ch.sent[0]
    # No internals leaked.
    assert "kaboom" not in sent_text
    assert "RuntimeError" not in sent_text
    # User-friendly tone.
    assert "wrong" in sent_text.lower() or "try again" in sent_text.lower()


@pytest.mark.asyncio
async def test_cancellation_propagates():
    """asyncio.CancelledError must NOT be swallowed — it has to bubble
    so any awaiter (typically the FastAPI request lifecycle) can act
    on a real cancellation.
    """
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner(raise_exc=asyncio.CancelledError())
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")

    with pytest.raises(asyncio.CancelledError):
        await handler(_inbound())


@pytest.mark.asyncio
async def test_empty_reply_does_not_send():
    """If AgentResponse.text is empty/whitespace we must not post a
    blank message back to the user.
    """
    from app.agent.channels.shared.message_handler import make_channel_handler

    ch = FakeChannel()
    runner = _fake_runner(response_text="   ")
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    await handler(_inbound())

    assert ch.sent == []


@pytest.mark.asyncio
async def test_send_failure_after_run_does_not_raise():
    """If channel.send_text raises (network, Meta down, etc.), the
    handler must log and swallow — we already wrote the persisted
    Message inside agent_runner so a retry is not safe."""
    from app.agent.channels.shared.message_handler import make_channel_handler

    class BrokenSendChannel(FakeChannel):
        async def send_text(self, *a, **kw):
            raise RuntimeError("Meta down")

    ch = BrokenSendChannel()
    runner = _fake_runner()
    handler = make_channel_handler(channel=ch, agent_runner=runner, user_id="u1")
    # Must not raise.
    await handler(_inbound())
