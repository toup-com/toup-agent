"""Unit tests for POST /api/v1/internal/agent-turn.

This is the internal, X-Agent-Key-gated endpoint the realtime-voice `think`
path hops to so voice runs the user's FULL agent (every tool/skill/connector)
instead of the tool-less /api/chat completion. Pin:
  * platform process (run_mode != "agent") ⇒ 404 (invisible to probers)
  * missing / wrong X-Agent-Key ⇒ 401
  * valid key ⇒ runs _agent_runner.run with channel="voice" and honours `save`
    (voice passes save=False; the voice handler owns persistence)
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import settings
import app.api.api_v1 as av1
from app.api.api_v1 import ChatRequest, internal_agent_turn


class _FakeRequest:
    def __init__(self, agent_key: str | None):
        self.headers = {} if agent_key is None else {"X-Agent-Key": agent_key}


class _FakeResponse:
    text = "The newest model is X, and I checked your inbox."
    session_id = "sess-9"
    tokens_input = 10
    tokens_output = 20
    tokens_total = 30
    model = "gpt-5.5"
    tool_calls = [{"name": "gmail.search"}, {"name": "web.search"}]
    processing_time_ms = 1234


class _FakeRunner:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


@pytest.fixture
def agent_mode(monkeypatch):
    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "secret-agent-key")
    monkeypatch.setattr(settings, "user_id", "owner-1")
    # Keep the model id verbatim so the assertion is deterministic.
    monkeypatch.setattr(settings, "security_leak_filter", False)


async def test_platform_process_returns_404(monkeypatch):
    monkeypatch.setattr(settings, "run_mode", "platform")
    with pytest.raises(HTTPException) as ei:
        await internal_agent_turn(ChatRequest(message="hi"), _FakeRequest("secret-agent-key"))
    assert ei.value.status_code == 404


async def test_missing_key_rejected(agent_mode):
    with pytest.raises(HTTPException) as ei:
        await internal_agent_turn(ChatRequest(message="hi"), _FakeRequest(None))
    assert ei.value.status_code == 401


async def test_wrong_key_rejected(agent_mode):
    with pytest.raises(HTTPException) as ei:
        await internal_agent_turn(ChatRequest(message="hi"), _FakeRequest("nope"))
    assert ei.value.status_code == 401


async def test_runs_full_agent_with_voice_channel_and_save_flag(agent_mode, monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setattr(av1, "_agent_runner", runner)

    resp = await internal_agent_turn(
        ChatRequest(message="what's the newest model?", session_id="sess-9", save=False),
        _FakeRequest("secret-agent-key"),
    )

    assert resp.text == _FakeResponse.text
    assert resp.model == "gpt-5.5"
    assert resp.tool_calls == 2               # len() of the runner's tool_calls list
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["user_id"] == "owner-1"       # resolved from settings.user_id, not a header
    assert call["session_id"] == "sess-9"
    assert call["channel"] == "voice"
    assert call["save_user_message"] is False
    assert call["save_assistant_message"] is False


async def test_no_runner_returns_503(agent_mode, monkeypatch):
    monkeypatch.setattr(av1, "_agent_runner", None)
    with pytest.raises(HTTPException) as ei:
        await internal_agent_turn(ChatRequest(message="hi"), _FakeRequest("secret-agent-key"))
    assert ei.value.status_code == 503
