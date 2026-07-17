"""Interim Live Activity progress (2026-07-16 granular-progress fix).

The bar used to jump 0→45→100 because progress only moved once per
~300s tick. TurnProgressEmitter emits throttled progress rows at tool
boundaries; these tests pin the interpolation, throttle, gate, and
payload contract the LA lane consumes.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def notify_calls(monkeypatch):
    calls: list[dict] = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "outbox-row"

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", fake_notify)
    return calls


def _mk(**kw):
    from app.agent.turn_progress import TurnProgressEmitter

    defaults = dict(
        mission_id="m-1", mission_title="Autopilot: Test",
        base_progress=40, ceiling=60, route="mission-control",
        min_interval_s=8.0,
    )
    defaults.update(kw)
    return TurnProgressEmitter(**defaults)


@pytest.mark.asyncio
async def test_first_tool_start_emits_progress_row(notify_calls):
    em = _mk()
    await em.on_tool_start("web_search")
    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "progress"
    assert call["priority"] == "low"
    assert call["dedup_key"] == "m-1:progress"
    assert call["body"] == "Searching the web…"
    data = call["data"]
    assert data["mission_id"] == "m-1"
    assert data["route"] == "mission-control"
    assert 40 < data["progress"] < 60


@pytest.mark.asyncio
async def test_interpolation_is_monotonic_and_below_ceiling(notify_calls):
    em = _mk(min_interval_s=0.0)
    last = 40
    for i in range(30):
        await em.on_tool_start("exec")
    values = [c["data"]["progress"] for c in notify_calls]
    assert values == sorted(values), "progress must never move backwards"
    assert all(v < 60 for v in values), "interim progress never reaches ceiling"
    assert em.last_emitted_progress == values[-1]


@pytest.mark.asyncio
async def test_throttle_suppresses_rapid_boundaries(notify_calls):
    em = _mk(min_interval_s=60.0)
    await em.on_tool_start("web_search")
    await em.on_tool_start("web_fetch")
    await em.on_tool_start("exec")
    assert len(notify_calls) == 1, "only the first emission passes the window"
    em.force_next()
    await em.on_tool_start("web_fetch")
    assert len(notify_calls) == 2, "force_next resets the throttle"


@pytest.mark.asyncio
async def test_gate_false_suppresses_everything(notify_calls):
    flags = {"gone": False}
    em = _mk(gate=lambda: flags["gone"], min_interval_s=0.0)
    await em.on_tool_start("web_search")
    assert notify_calls == [], "client still connected → no lock-screen card"
    flags["gone"] = True
    await em.on_tool_start("web_fetch")
    assert len(notify_calls) == 1


@pytest.mark.asyncio
async def test_notify_failure_never_raises(monkeypatch):
    async def exploding(**kwargs):
        raise RuntimeError("outbox down")

    import app.services.agent_notify_client as anc
    monkeypatch.setattr(anc, "notify", exploding)
    em = _mk()
    await em.on_tool_start("web_search")  # must not raise


def test_wiring_pins():
    """The emitter is only useful if the two call sites actually wire
    it: autopilot ticks (always-on) and chat turns (gated)."""
    import inspect
    from app.agent.routines import autopilot_handler
    from app.api import ws_chat

    hsrc = inspect.getsource(autopilot_handler)
    assert "TurnProgressEmitter" in hsrc
    assert "on_tool_start=_emitter.on_tool_start" in hsrc
    assert "_emitter.last_emitted_progress" in hsrc

    wsrc = inspect.getsource(ws_chat)
    assert "TurnProgressEmitter" in wsrc
    assert 'gate=lambda: _turn_flags["client_gone"]' in wsrc
    assert "_turn_emitter.force_next()" in wsrc


def test_progress_fastlane_flag_and_ingest_pin():
    import inspect
    from app.api import agent_notify
    from app.config import settings

    assert settings.notification_progress_fastlane_enabled is True
    src = inspect.getsource(agent_notify)
    assert "notification_progress_fastlane_enabled" in src
    assert "_dispatch_row" in src
