"""Stalled-job reaper (app.agent.job_reaper).

Pins the honesty contract: a running job whose last sign of life is
older than STALE_AFTER gets failed + a mission_failed phone event;
anything fresh, actively heartbeating (job_events), or token-paused
survives untouched.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

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


async def _make_job(user_id: str, *, age: timedelta, status: str = "running",
                    paused: bool = False) -> str:
    from app.db import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=job_id,
            user_id=user_id,
            title="Top 10 CRM comparison",
            prompt="research",
            job_type="agent_task",
            status=status,
            created_at=now - age,
            paused_at=(now - age) if paused else None,
        ))
        await db.commit()
    return job_id


@pytest.mark.asyncio
async def test_stalled_running_job_is_stopped_and_notified(test_user_id, notify_calls):
    """A stall is `cancelled`, not `failed`.

    Changed deliberately 2026-07-31. The dominant cause of a stall is a turn
    that went away mid-flight — the voice relay cancels its SSE generator the
    instant the caller hangs up — by which point the agent has usually already
    delivered the answer out loud. Calling that "Failed" is the exact lie this
    pipeline exists to remove.

    The notification KIND stays `mission_failed`: only a terminal notification
    closes a Live Activity card and `KNOWN_NOTIFY_KINDS` is a closed enum
    validated at ingest, so it is the only terminal lane besides
    `mission_completed`. The user-facing COPY is what changed.
    """
    from app.agent.job_reaper import sweep_stalled_jobs
    from app.db import async_session_maker
    from app.db.models import BuildJob

    job_id = await _make_job(test_user_id, age=timedelta(minutes=45))

    reaped = await sweep_stalled_jobs()
    assert reaped == 1

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "cancelled"
        assert job.completed_at is not None
        # Taxonomy fields must be populated, or the API's read-time
        # classification falls through to `unknown` — "Something went wrong.
        # We've been notified" — which is worse than the text it replaced.
        assert job.error_class == "turn_interrupted"
        assert "conversation ended" in (job.user_message or "").lower()
        # Operator detail is kept, but never on a user-facing field.
        assert "Stalled" in (job.technical_detail or "")
        assert "Stalled" not in (job.user_message or "")

    (call,) = [c for c in notify_calls if c["data"]["mission_id"] == job_id]
    assert call["event_kind"] == "mission_failed", "the only terminal card-closing lane"
    assert call["dedup_key"] == f"{job_id}:stalled"
    assert call["data"]["dismiss_after_s"] == 900
    assert "Didn't finish" not in call["data"].get("title", "")


@pytest.mark.asyncio
async def test_recent_heartbeat_survives(test_user_id, notify_calls):
    from app.agent.job_reaper import sweep_stalled_jobs
    from app.db import async_session_maker
    from app.db.models import BuildJob, JobEvent

    job_id = await _make_job(test_user_id, age=timedelta(minutes=45))
    async with async_session_maker() as db:
        db.add(JobEvent(
            job_id=job_id, user_id=test_user_id, kind="info",
            label="Progress: 2/4 steps",
            ts=datetime.utcnow() - timedelta(minutes=5),
        ))
        await db.commit()

    assert await sweep_stalled_jobs() == 0
    async with async_session_maker() as db:
        assert (await db.get(BuildJob, job_id)).status == "running"
    assert notify_calls == []


@pytest.mark.asyncio
async def test_fresh_and_paused_jobs_survive(test_user_id, notify_calls):
    from app.agent.job_reaper import sweep_stalled_jobs
    from app.db import async_session_maker
    from app.db.models import BuildJob

    fresh = await _make_job(test_user_id, age=timedelta(minutes=5))
    paused = await _make_job(
        test_user_id, age=timedelta(hours=3), paused=True,
    )

    assert await sweep_stalled_jobs() == 0
    async with async_session_maker() as db:
        assert (await db.get(BuildJob, fresh)).status == "running"
        assert (await db.get(BuildJob, paused)).status == "running"
    assert notify_calls == []
