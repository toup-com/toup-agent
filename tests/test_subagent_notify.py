"""Spawned-job phone-surface events (subagent_orchestrator._notify_job_event).

The Live Activity lane consumes exactly this contract — mission_id +
mission_title + urgent + timer/dismissal hints in data — so pin it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def notify_calls(monkeypatch):
    calls: list[dict] = []

    async def fake_notify(**kwargs):
        calls.append(kwargs)
        return "outbox-row-id"

    import app.services.agent_notify_client as client
    monkeypatch.setattr(client, "notify", fake_notify)
    return calls


@pytest.mark.asyncio
async def test_started_event_shape(notify_calls):
    from app.agent.subagent_orchestrator import _notify_job_event

    await _notify_job_event(
        job_id="job-123", label="PM tools comparison",
        kind="mission_started", title="🛠 Working on: PM tools comparison",
        body="Research the 3 best…", timer_end_ms=1_752_000_000_000,
        dedup_suffix="started",
    )
    assert len(notify_calls) == 1
    call = notify_calls[0]
    assert call["event_kind"] == "mission_started"
    assert call["dedup_key"] == "job-123:started"
    data = call["data"]
    assert data["mission_id"] == "job-123"
    assert data["mission_title"] == "PM tools comparison"
    assert data["kind"] == "job"
    # Interactive origin: quiet hours must never defer the card.
    assert data["urgent"] is True
    assert data["timer_end_ms"] == 1_752_000_000_000
    assert "progress" not in data


@pytest.mark.asyncio
async def test_completed_event_carries_dismissal(notify_calls):
    from app.agent.subagent_orchestrator import _notify_job_event

    await _notify_job_event(
        job_id="job-123", label="PM tools comparison",
        kind="mission_completed", title="✅ Done: PM tools comparison",
        body="Asana wins.", progress=100, dismiss_after_s=900,
        dedup_suffix="completed",
    )
    data = notify_calls[0]["data"]
    assert data["progress"] == 100
    assert data["dismiss_after_s"] == 900


@pytest.mark.asyncio
async def test_notify_failure_never_raises(monkeypatch):
    async def exploding_notify(**kwargs):
        raise RuntimeError("outbox down")

    import app.services.agent_notify_client as client
    monkeypatch.setattr(client, "notify", exploding_notify)

    from app.agent.subagent_orchestrator import _notify_job_event

    # Contract: a job must never fail on notification plumbing.
    await _notify_job_event(
        job_id="j", label=None, kind="mission_failed",
        title="⚠️ Didn't finish: background task", dedup_suffix="failed",
    )


@pytest.mark.asyncio
async def test_autopilot_parent_spawns_are_not_urgent(notify_calls):
    """Spawns fired by a mission tick must NOT bypass quiet hours —
    a 3am autopilot spawn is background work, not an awake user."""
    from app.agent.subagent_orchestrator import _notify_job_event

    await _notify_job_event(
        job_id="job-9", label="Night research",
        kind="mission_started", title="🛠 Working on: Night research",
        dedup_suffix="started", urgent=False,
    )
    assert notify_calls[0]["data"]["urgent"] is False
