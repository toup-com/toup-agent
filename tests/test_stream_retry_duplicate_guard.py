"""F-12 — a mid-stream retry must never replay text the consumer already has.

Both streaming paths in openai_agent_service put their retry loop AROUND the
`async for`. A connection error at chunk 500 therefore restarted the request
and yielded a SECOND, independently-generated answer into the same consumer,
appended to what the user was already reading. That doubled text is what got
persisted, what memory extraction later read, and what was billed twice.

These tests drive the real `create_message_stream` with a fake OpenAI client,
so they exercise the actual retry/except structure rather than a model of it.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.openai_agent_service import (
    OpenAIAgentService,
    _abort_rather_than_replay,
)

import httpx
from openai import APIConnectionError, RateLimitError


def _rate_limit_error() -> RateLimitError:
    """The SDK derives .request from the response, so it needs a real one."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


@pytest.fixture(autouse=True)
def _chat_wire_and_no_credit_gate(monkeypatch):
    """These tests exercise the CHAT wire; keep the credit pre-flight and the
    responses-wire branch out of the way."""
    monkeypatch.setattr(settings, "openai_wire_api", "chat", raising=False)
    import app.services.credit_reporter as _cr
    monkeypatch.setattr(_cr, "raise_if_exhausted", lambda *a, **k: None)


# ── unit: the decision helper ────────────────────────────────────────


class _Exc(Exception):
    pass


def test_no_output_yet_still_retries():
    """The common case — connect/rate-limit at request time — is untouched."""
    assert _abort_rather_than_replay(False, _Exc(), 0) is False


def test_after_output_aborts():
    assert _abort_rather_than_replay(True, _Exc(), 1) is True


def test_kill_switch_restores_replay(monkeypatch):
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", False, raising=False)
    assert _abort_rather_than_replay(True, _Exc(), 1) is False


# ── integration: the real stream loop ────────────────────────────────


def _chunk(text: str = "", *, finish: str | None = None, cid: str = "chatcmpl-1"):
    """Minimal duck-typed chat-completions chunk."""
    class _D:
        content = text
        tool_calls = None
    class _C:
        delta = _D()
        finish_reason = finish
    class _Chunk:
        id = cid
        choices = [_C()]
        usage = None
    return _Chunk()


class _Stream:
    """Async-iterable that yields some chunks then optionally explodes."""

    def __init__(self, chunks, raise_after=None):
        self._chunks = chunks
        self._raise_after = raise_after

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, c in enumerate(self._chunks):
            yield c
            if self._raise_after is not None and i == self._raise_after:
                raise APIConnectionError(request=None)


class _FakeCompletions:
    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return self._streams.pop(0)


class _FakeClient:
    def __init__(self, streams):
        self.chat = type("chat", (), {"completions": _FakeCompletions(streams)})()


def _service(streams) -> OpenAIAgentService:
    """Real service object, fake transport — the retry/except structure under
    test is the shipped one."""
    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc.client = _FakeClient(streams)
    svc.default_model = "gpt-5.5"
    svc.default_max_tokens = 256
    svc._keys = type("k", (), {"refresh": lambda self: None})()
    svc._ensure_client = lambda: None
    return svc


async def _drain(svc, **kw):
    out = []
    async for ev in svc.create_message_stream(
        messages=[{"role": "user", "content": "hi"}], system="s", **kw
    ):
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_failure_after_partial_output_raises_instead_of_duplicating(monkeypatch):
    """The regression itself: first stream emits "Hello world" then drops.
    Pre-fix the retry emitted a whole second answer after it."""
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", True, raising=False)
    first = _Stream([_chunk("Hello "), _chunk("world")], raise_after=1)
    second = _Stream([_chunk("Hello "), _chunk("world"), _chunk("", finish="stop")])
    svc = _service([first, second])

    collected = []
    with pytest.raises(APIConnectionError):
        async for ev in svc.create_message_stream(
            messages=[{"role": "user", "content": "hi"}], system="s"
        ):
            collected.append(ev)

    text = "".join(e.text for e in collected if e.type == "text")
    assert text == "Hello world", "consumer keeps exactly what it already had"
    assert svc.client.chat.completions.calls == 1, "no second request was made"


@pytest.mark.asyncio
async def test_failure_before_any_output_still_retries(monkeypatch):
    """A drop before the first delta must retry exactly as it always did."""
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", True, raising=False)

    class _DeadStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            raise APIConnectionError(request=None)
            yield  # pragma: no cover

    good = _Stream([_chunk("recovered"), _chunk("", finish="stop")])
    svc = _service([_DeadStream(), good])

    events = await _drain(svc)
    text = "".join(e.text for e in events if e.type == "text")
    assert text == "recovered"
    assert svc.client.chat.completions.calls == 2, "retry must still happen"


@pytest.mark.asyncio
async def test_kill_switch_restores_the_duplicating_behaviour(monkeypatch):
    """Proves the flag is a real kill switch — and documents what it buys
    back: the doubled answer."""
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", False, raising=False)
    first = _Stream([_chunk("Hello "), _chunk("world")], raise_after=1)
    second = _Stream([_chunk("Hello "), _chunk("world"), _chunk("", finish="stop")])
    svc = _service([first, second])

    events = await _drain(svc)
    text = "".join(e.text for e in events if e.type == "text")
    assert text == "Hello worldHello world", "flag off = historical duplication"
    assert svc.client.chat.completions.calls == 2


@pytest.mark.asyncio
async def test_rate_limit_after_output_also_aborts(monkeypatch):
    """Rate limits normally hit before output, but if one lands mid-stream
    the same rule applies."""
    monkeypatch.setattr(settings, "llm_stream_duplicate_guard", True, raising=False)

    class _RLStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _chunk("partial")
            raise _rate_limit_error()

    svc = _service([_RLStream(), _Stream([_chunk("again", finish="stop")])])
    collected = []
    with pytest.raises(RateLimitError):
        async for ev in svc.create_message_stream(
            messages=[{"role": "user", "content": "hi"}], system="s"
        ):
            collected.append(ev)
    assert "".join(e.text for e in collected if e.type == "text") == "partial"
    assert svc.client.chat.completions.calls == 1
