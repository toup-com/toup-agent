"""Reproduction: a transient agent outage turns "play me X" into a capability denial.

This is the founder's Recording 2 (2026-07-31 20:24 UTC, during an agent image
rollout): the chip read "That one didn't work / Trying another way", the agent
said "your playback agent isn't connected right now", and then told him to go
search A$AP Rocky on Spotify, Apple Music or YouTube — eight minutes after it had
played music successfully.

Nothing about that is a model quirk. `_think`'s Option B fallback sends a bare
task to a generic reasoning assistant with `tools=[]` and a system prompt that
says nothing about the user having a music feature, nothing about the failure
being transient, and nothing forbidding a competitor redirect. A tool-less model
asked to play music has exactly one honest answer available to it, and it is the
wrong one.
"""
from __future__ import annotations

import pytest
from app.config import settings
import app.api.ws_realtime as rt

pytestmark = pytest.mark.asyncio


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", True)


async def test_agent_outage_degrades_a_play_request_to_a_toolless_answer(v2_on, monkeypatch):
    monkeypatch.setattr(rt, "_agent_runner", None)

    class _Decision:
        model = "gpt-5.5"
    monkeypatch.setattr("app.services.model_router.classify_request", lambda task: _Decision())

    async def _fake_vps_info(user_id):
        return ("https://u.agents.toup.ai", "agent-key")
    monkeypatch.setattr(rt, "_get_vps_info", _fake_vps_info)

    attempts = []

    async def _dead_agent(agent_url, agent_api_key, method, path,
                          params=None, json_body=None, timeout=15.0):
        attempts.append(path)
        raise ConnectionError("agent container restarting (rollout)")
    monkeypatch.setattr(rt, "_vps_api", _dead_agent)

    captured = {}

    class _FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content":
                "I can't play music from here, but you can search A$AP Rocky on Spotify."}}]}

    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            captured["json"] = json
            return _FakeResp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient())

    text, _model = await rt._think("user-1", "Play me asap rocky", "sess-9")

    print("\n--- REPRO EVIDENCE ---")
    print("agent attempts:", attempts)
    body = captured.get("json") or {}
    print("fallback tools:", body.get("tools", "<<absent — no tools sent>>"))
    for m in body.get("messages", []):
        print(f"  {m['role']}: {m['content'][:160]}")
    print("returned to user:", text[:160])
    print("--- END ---\n")

    # 1. A transient agent failure is RETRIED before giving up. A rollout swaps
    #    the container for a couple of seconds; without this, every deploy
    #    downgraded live voice users to the tool-less fallback.
    assert len(attempts) == 2, f"expected one retry, got attempts={attempts}"

    # 2. If it still cannot be reached, the fallback is told the truth about its
    #    own situation, so it stops reasoning "no tools, therefore no feature".
    sys_msg = next(m["content"] for m in body["messages"] if m["role"] == "system")
    low = sys_msg.lower()
    assert "momentarily unreachable" in low, "fallback is not told the failure is transient"
    assert "action" in low, "fallback has no rule for action-shaped requests"

    # 3. The two answers that must never be produced again.
    assert "never suggest another app" in low
    assert "spotify" in low, "the competitor prohibition must name the actual traps"
    assert "never say you are unable" in low
