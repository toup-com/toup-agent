"""Round 24 turn-safety + honest-health for rollouts.

WHY THIS EXISTS — the 2026-08-23 P0

The legacy bridge `/upgrade` is a `docker rm -f`: SIGKILL, no drain. An image
push that landed while the founder was mid-reply truncated the turn — chat went
silent while the container was pulled and recreated. The structural fix that
needs neither host access nor an auth change: the tenant reports how many turns
are in flight (`/agent/health.active_turns`, count only), and the rollout waits
for that to reach 0 before the SIGKILL upgrade, up to a bounded grace.

Run:
  cd backend && pytest tests/test_r24_turn_safety.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.rollout_service as RS  # noqa: E402


class _Resp:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body


class _FakeClient:
    """Serves a scripted sequence of health bodies, then repeats the last."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        i = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[i]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(RS.asyncio, "sleep", AsyncMock())


def _install_client(monkeypatch, client):
    monkeypatch.setattr(RS.httpx, "AsyncClient", lambda *a, **k: client)


@pytest.mark.asyncio
async def test_idle_immediately_returns_true(monkeypatch):
    _install_client(monkeypatch, _FakeClient([_Resp(200, {"active_turns": 0})]))
    idle, last = await RS._wait_for_tenant_idle("https://a.example", grace_s=90)
    assert idle is True and last == 0


@pytest.mark.asyncio
async def test_a_live_turn_is_waited_out(monkeypatch):
    # Busy, busy, then idle — the wait must poll past the busy reads.
    c = _FakeClient([
        _Resp(200, {"active_turns": 1}),
        _Resp(200, {"active_turns": 1}),
        _Resp(200, {"active_turns": 0}),
    ])
    _install_client(monkeypatch, c)
    idle, last = await RS._wait_for_tenant_idle("https://a.example", grace_s=90)
    assert idle is True and last == 0
    assert c.calls >= 3  # it did not give up on the first busy read


@pytest.mark.asyncio
async def test_a_wedged_turn_times_out_and_proceeds(monkeypatch):
    # Always busy: the wait must return (False) so the caller upgrades anyway —
    # a rollout cannot wait forever on a stuck turn.
    monkeypatch.setattr(RS.time, "time", _fake_clock(step=40))
    _install_client(monkeypatch, _FakeClient([_Resp(200, {"active_turns": 2})]))
    idle, last = await RS._wait_for_tenant_idle("https://a.example", grace_s=90)
    assert idle is False and last == 2


@pytest.mark.asyncio
async def test_zero_grace_disables_the_wait(monkeypatch):
    # No client should even be constructed.
    def _boom(*a, **k):
        raise AssertionError("no health poll when grace is 0")
    monkeypatch.setattr(RS.httpx, "AsyncClient", _boom)
    idle, last = await RS._wait_for_tenant_idle("https://a.example", grace_s=0)
    assert idle is True and last == 0


@pytest.mark.asyncio
async def test_an_unreachable_agent_is_treated_as_idle(monkeypatch):
    class _Err(_FakeClient):
        async def get(self, url):
            raise RS.httpx.ConnectError("no route")
    _install_client(monkeypatch, _Err([]))
    idle, _ = await RS._wait_for_tenant_idle("https://a.example", grace_s=90)
    assert idle is True  # never hold a rollout for a health endpoint that is down


@pytest.mark.asyncio
async def test_a_non_200_is_not_waited_on(monkeypatch):
    _install_client(monkeypatch, _FakeClient([_Resp(503, {})]))
    idle, _ = await RS._wait_for_tenant_idle("https://a.example", grace_s=90)
    assert idle is True  # unhealthy is the health gate's problem, not the wait's


def test_health_exposes_active_turns_count_only():
    # The count must be a plain int helper with no identities in its contract.
    from app.api import ws_chat
    assert hasattr(ws_chat, "active_turn_count")
    assert isinstance(ws_chat.active_turn_count(), int)


def _fake_clock(step):
    """A monotonic clock that advances `step` seconds per call, so a bounded
    grace elapses within a few polls without real waiting."""
    t = {"n": 0.0}

    def _now():
        t["n"] += step
        return t["n"]
    return _now
