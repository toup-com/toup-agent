"""Restart recovery is boot-critical and decides what the user sees.

Before the Mission Control overhaul this logic lived untested inline in
`agent_main.py`'s lifespan and flipped EVERY `running` and `queued` row to
`failed` with "Agent restarted during execution" — 7 of the 11 user-visible
failures on the founder's tenant (64%), none of them real.

Every test here maps to a rule that, if broken, produces a specific
production symptom. The symptom is named in each docstring.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest

from app.agent.job_recovery import (
    MAX_RESTART_REQUEUES,
    REQUEUE_COUNT_KEY,
    recover_orphaned_jobs,
)
from app.agent.job_status import (
    ERR_INFRA_INTERRUPTED,
    ERR_INFRA_UNRECOVERABLE,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_WAITING_ON_USER,
)

OLD = datetime.utcnow() - timedelta(hours=2)      # outside the grace window
FRESH = datetime.utcnow() - timedelta(minutes=1)  # inside it


async def _user() -> str:
    from app.db import User, async_session_maker

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"{uid[:8]}@t.test", name="T",
            hashed_password="x", role="beta_user",
        ))
        await db.commit()
    return uid


async def _job(user_id: str, **kw):
    from app.db import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    payload = dict(
        id=jid, user_id=user_id, title=kw.pop("title", "a task"),
        prompt="p", status=STATUS_RUNNING, created_at=OLD,
    )
    payload.update(kw)
    async with async_session_maker() as db:
        db.add(BuildJob(**payload))
        await db.commit()
    return jid


async def _get(job_id: str):
    from app.db import async_session_maker
    from app.db.models import BuildJob

    async with async_session_maker() as db:
        return await db.get(BuildJob, job_id)


@pytest.fixture()
def sm():
    from app.db import async_session_maker

    return async_session_maker


# ── the 64% fix ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queued_jobs_are_never_touched(sm):
    """SYMPTOM: a job that never started shows as "Failed".

    The old sweep included `queued` in its filter, so a job merely waiting
    was killed by an unrelated container roll. Pure loss.
    """
    uid = await _user()
    jid = await _job(uid, status=STATUS_QUEUED, source_kind="routine")

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_QUEUED
    assert row.error_class is None
    assert row.user_message is None
    assert out.touched == 0


@pytest.mark.asyncio
async def test_trigger_job_is_requeued_silently(sm):
    """Triggers are the ONE kind with a queued-row drain, so they can
    genuinely auto-resume. That must be invisible: no error fields set."""
    uid = await _user()
    jid = await _job(uid, source_kind="trigger")

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_QUEUED
    assert row.error_class is None, "an auto-resumed job must stay invisible"
    assert row.user_message is None
    assert row.error_message is None
    assert row.completed_at is None, "a resuming job is not finished"
    assert (row.config_json or {})[REQUEUE_COUNT_KEY] == 1
    assert out.requeued == 1
    assert out.interrupted == [] and out.gave_up == []


@pytest.mark.parametrize("kind", ["routine", "subagent", "chat_intent", "manual", None])
@pytest.mark.asyncio
async def test_non_drainable_kinds_are_interrupted_not_stranded(sm, kind):
    """SYMPTOM: a job parked in "Queued" forever with a Live Activity card
    that never closes.

    Only `trigger` has a drain loop. Re-queueing anything else would strand
    it, so these must terminalise — visibly and honestly — instead.
    """
    uid = await _user()
    jid = await _job(uid, source_kind=kind)

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_FAILED
    assert row.error_class == ERR_INFRA_INTERRUPTED
    assert row.user_message and "restart" in row.user_message.lower()
    assert row.completed_at is not None
    # Must be reported so the caller can END the lock screen card.
    assert [j[0] for j in out.interrupted] == [jid]
    assert out.requeued == 0


@pytest.mark.asyncio
async def test_raw_marker_never_written_to_a_user_field(sm):
    """SYMPTOM: "Agent restarted during execution" on the user's screen."""
    uid = await _user()
    for kind in ("trigger", "routine"):
        jid = await _job(uid, source_kind=kind)
        await recover_orphaned_jobs(sm)
        row = await _get(jid)
        for field in (row.user_message, row.error_message):
            assert "restarted during execution" not in (field or "")


# ── blue-green safety ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_running_jobs_are_left_to_the_draining_sibling(sm):
    """SYMPTOM: a job executes twice during a blue-green roll.

    The incoming container boots against the SAME tenant DB while the
    outgoing one still drains. Touching a fresh `running` row would
    re-dispatch work that is currently in flight.
    """
    uid = await _user()
    jid = await _job(uid, source_kind="trigger", created_at=FRESH)

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_RUNNING, "fresh in-flight job must be untouched"
    assert out.skipped_in_grace == 1
    assert out.touched == 0


# ── runaway protection ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_requeue_is_capped_then_surfaced_honestly(sm):
    """SYMPTOM: a poison job re-queues forever, crash-looping the container.

    At the cap it becomes a real (reported) failure rather than looping.
    """
    uid = await _user()
    jid = await _job(
        uid, source_kind="trigger",
        config_json={REQUEUE_COUNT_KEY: MAX_RESTART_REQUEUES},
    )

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_FAILED
    assert row.error_class == ERR_INFRA_UNRECOVERABLE
    assert row.user_message and "several attempts" in row.user_message
    assert [j[0] for j in out.gave_up] == [jid]


@pytest.mark.asyncio
async def test_requeue_counter_increments_across_restarts(sm):
    """The counter must survive in config_json, not reset each boot."""
    uid = await _user()
    jid = await _job(uid, source_kind="trigger")

    for expected in (1, 2, 3):
        out = await recover_orphaned_jobs(sm)
        row = await _get(jid)
        assert (row.config_json or {})[REQUEUE_COUNT_KEY] == expected
        assert out.requeued == 1
        # Put it back in flight to simulate the next crash.
        from app.db import async_session_maker
        async with async_session_maker() as db:
            j = await db.get(type(row), jid)
            j.status = STATUS_RUNNING
            j.created_at = OLD
            await db.commit()

    # Fourth restart hits the cap.
    out = await recover_orphaned_jobs(sm)
    row = await _get(jid)
    assert row.status == STATUS_FAILED
    assert row.error_class == ERR_INFRA_UNRECOVERABLE


@pytest.mark.asyncio
async def test_attempt_column_is_not_used_as_the_requeue_counter(sm):
    """`attempt` is the routine-retry counter; conflating them corrupts
    both. The re-queue count lives in config_json."""
    uid = await _user()
    jid = await _job(uid, source_kind="trigger", attempt=2)

    await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.attempt == 2, "routine retry count must be preserved"
    assert (row.config_json or {})[REQUEUE_COUNT_KEY] == 1


# ── progress / step hygiene ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_running_steps_rewind_to_pending_on_requeue(sm):
    """SYMPTOM: a job stuck reading "2/5" forever (founder's, for 79 days).

    A half-finished step left as `running` is a phantom that never resolves.
    """
    uid = await _user()
    steps = [
        {"id": "1", "type": "search", "label": "a", "status": "done"},
        {"id": "2", "type": "code", "label": "b", "status": "running",
         "started_at": OLD.isoformat()},
        {"id": "3", "type": "output", "label": "c", "status": "pending"},
    ]
    jid = await _job(uid, source_kind="trigger", steps_json=json.dumps(steps))

    await recover_orphaned_jobs(sm)

    row = await _get(jid)
    got = json.loads(row.steps_json)
    assert [s["status"] for s in got] == ["done", "pending", "pending"]
    assert "started_at" not in got[1], "stale start time would skew duration"


@pytest.mark.asyncio
async def test_malformed_steps_json_does_not_break_recovery(sm):
    """A container must boot even with a corrupt row."""
    uid = await _user()
    jid = await _job(uid, source_kind="trigger", steps_json="{not json")

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == STATUS_QUEUED
    assert out.requeued == 1


# ── isolation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status", [STATUS_COMPLETED, STATUS_FAILED, STATUS_WAITING_ON_USER, "paused"],
)
@pytest.mark.asyncio
async def test_terminal_and_parked_rows_are_untouched(sm, status):
    """SYMPTOM: a job waiting on the user gets clobbered by a restart.

    `waiting_on_user` is durable by design — the requirement outlives the
    container — so a boot must not disturb it.
    """
    uid = await _user()
    jid = await _job(uid, status=status, source_kind="routine")

    out = await recover_orphaned_jobs(sm)

    row = await _get(jid)
    assert row.status == status
    assert out.touched == 0


@pytest.mark.asyncio
async def test_mixed_fleet_is_partitioned_correctly(sm):
    """One boot, every case at once — the realistic shape."""
    uid = await _user()
    trigger = await _job(uid, source_kind="trigger", title="gmail fire")
    routine = await _job(uid, source_kind="routine", title="daily digest")
    fresh = await _job(uid, source_kind="trigger", created_at=FRESH, title="in flight")
    queued = await _job(uid, status=STATUS_QUEUED, source_kind="routine", title="waiting")
    poison = await _job(
        uid, source_kind="trigger", title="poison",
        config_json={REQUEUE_COUNT_KEY: MAX_RESTART_REQUEUES},
    )

    out = await recover_orphaned_jobs(sm)

    assert (await _get(trigger)).status == STATUS_QUEUED
    assert (await _get(routine)).status == STATUS_FAILED
    assert (await _get(fresh)).status == STATUS_RUNNING
    assert (await _get(queued)).status == STATUS_QUEUED
    assert (await _get(poison)).status == STATUS_FAILED

    assert out.requeued == 1
    assert [j[0] for j in out.interrupted] == [routine]
    assert [j[0] for j in out.gave_up] == [poison]
    assert out.skipped_in_grace == 1


@pytest.mark.asyncio
async def test_notification_targets_cover_every_terminalised_row(sm):
    """The load-bearing invariant tying recovery to the Dynamic Island.

    A Live Activity card is closed ONLY by a terminal notification — a DB
    write never touches it. So every row this sweep terminalises must be
    reported back, or its card lingers for hours on stale progress.
    """
    uid = await _user()
    ids = [
        await _job(uid, source_kind="routine"),
        await _job(uid, source_kind="subagent"),
        await _job(
            uid, source_kind="trigger",
            config_json={REQUEUE_COUNT_KEY: MAX_RESTART_REQUEUES},
        ),
    ]

    out = await recover_orphaned_jobs(sm)

    reported = {j[0] for j in out.interrupted} | {j[0] for j in out.gave_up}
    assert reported == set(ids)
    # Every reported tuple must carry what the notifier needs.
    for jid, title, user_id in out.interrupted + out.gave_up:
        assert jid and user_id
        assert isinstance(title, str)


@pytest.mark.asyncio
async def test_app_status_unstuck_when_a_build_dies(sm):
    """SYMPTOM: an app pinned on "building" forever after a restart."""
    from app.db import async_session_maker
    from app.db.models import App as AppModel

    uid = await _user()
    app_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(AppModel(
            id=app_id, user_id=uid, name="app", slug=f"s{app_id[:6]}",
            status="building", app_dir=f"/tmp/{app_id}",
        ))
        await db.commit()
    await _job(uid, source_kind="manual", app_id=app_id)

    await recover_orphaned_jobs(sm)

    async with async_session_maker() as db:
        assert (await db.get(AppModel, app_id)).status == "error"
