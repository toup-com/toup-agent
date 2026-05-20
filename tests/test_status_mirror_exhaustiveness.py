"""Status mapping exhaustiveness contract.

The unified-jobs arc replaced four bespoke status enums with the
closed BuildJob (status, outcome) tuple. Each runner translates
its terminal status → (job_status, job_outcome) combination.

Originally these tests pinned the mirror helper functions
(``_mirror_event_terminal_to_job`` /
``_mirror_run_terminal_to_job``) that dual-wrote terminal state
from the legacy table onto BuildJob. PR #49 of the cutover arc
removed those helpers along with the legacy dual-write — the
runners now inline the same mapping in ``_dispatch_one`` (triggers)
and ``_finalize_run`` (routines).

The pin we still need: the closed ``(status, outcome)`` mapping
table itself. A new status added to a source's enum without
extending the runner's mapping would silently produce a row with
``status='completed', outcome=<the new value>`` (the conservative
fallback both runners take). These tests exercise the runners
end-to-end and assert the resulting BuildJob row's (status,
outcome) tuple matches the documented table.

Why this matters: the closed BuildJob enum is the read-side
contract for the activity feed and Mission Control. If a new
status leaks through as ``outcome='success'`` when it shouldn't,
Mission Control reports a "delivered" run that actually skipped.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-status-mirror")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000sm001")


USER_ID = "00000000-0000-0000-0000-0000000sm001"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Per-test fresh DB. Only the tables the runners touch
    end-to-end (BuildJob + Trigger + Routine + User). PR #49: no
    TriggerEvent / RoutineRun in the cutover paths."""
    from app.db.database import rebind_database
    from app.config import settings

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Routine, Trigger, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Trigger, Routine):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (Routine, Trigger, JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers — seed the minimum rows the runner exercises.
# ──────────────────────────────────────────────────────────────────────


async def _seed_user_and_trigger_job() -> tuple[str, str]:
    """Seed User + Trigger + BuildJob (status=queued) with linkage.
    Returns (trigger_id, job_id)."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger, User

    trigger_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"sm-{USER_ID[:8]}@example.com",
                hashed_password="x", timezone="UTC",
            ))
        db.add(Trigger(
            id=trigger_id, user_id=USER_ID, kind="email_received",
            action="agent_handle", enabled=True, config_json={},
            provider_state_json={}, filter_json={}, last_status="never_fired",
            fire_count=0,
        ))
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Mirror test",
            prompt="", job_type="trigger_run", status="queued",
            source_kind="trigger", source_id=trigger_id,
            idempotency_key=f"dedupe-{job_id[:8]}",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return trigger_id, job_id


async def _seed_user_and_routine_job() -> tuple[str, str]:
    """Seed User + Routine + BuildJob (status=running) with linkage.
    Returns (routine_id, job_id)."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine, User

    routine_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"sm-{USER_ID[:8]}@example.com",
                hashed_password="x", timezone="UTC",
            ))
        db.add(Routine(
            id=routine_id, user_id=USER_ID, kind="email_briefing",
            enabled=True, config_json={},
            schedule_cron_local="0 8 * * *",
        ))
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Mirror routine test",
            prompt="", job_type="routine_run", status="running",
            source_kind="routine", source_id=routine_id,
            idempotency_key="2026-05-20",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return routine_id, job_id


async def _read_job(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return await db.get(BuildJob, job_id)


# ──────────────────────────────────────────────────────────────────────
# Trigger runner: status → (job_status, outcome) mapping.
# Exercises ``_dispatch_one`` via a stub handler that returns each
# per-event terminal status in turn.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeTriggerResult:
    status: str = "success"
    per_event_status: dict = field(default_factory=dict)
    summary_message_id: Optional[str] = None
    new_provider_state: Optional[dict] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    metrics: dict = field(default_factory=dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("per_event_status,expected_job_status,expected_outcome", [
    ("success",            "completed", "success"),
    ("failed",             "failed",    None),
    ("skipped_rate_limit", "completed", "skipped_rate_limit"),
    ("skipped_filter",     "completed", "skipped_filter"),
    ("coalesced",          "completed", "coalesced"),
])
async def test_trigger_runner_maps_terminal_status_via_dispatch(
    per_event_status, expected_job_status, expected_outcome,
):
    """For every documented terminal value a handler can return,
    ``_dispatch_one`` writes the matching ``(status, outcome)``
    tuple onto the BuildJob. A regression that, say, maps
    ``failed`` to ``completed`` would crash this test on the
    ``failed`` row — exactly the kind of mis-routing that would
    surface to Mission Control as "delivered" but actually failed.

    PR #49: the mapping is inlined in the runner's step-5 batch
    loop (the mirror helper was deleted)."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    trigger_id, job_id = await _seed_user_and_trigger_job()

    class _StatusHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: per_event_status for e in events},
                error_class=("e" if per_event_status == "failed" else None),
                error_detail=("forced" if per_event_status == "failed" else None),
            )

    KIND_HANDLERS["email_received"] = _StatusHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        ok = await runner._dispatch_one(job_id, attempt_idx=0)
        assert ok is True
    finally:
        KIND_HANDLERS.pop("email_received", None)

    job = await _read_job(job_id)
    assert job.status == expected_job_status, (
        f"per_event_status={per_event_status!r} mapped to "
        f"job.status={job.status!r}, expected {expected_job_status!r}"
    )
    assert job.outcome == expected_outcome, (
        f"per_event_status={per_event_status!r} mapped to "
        f"job.outcome={job.outcome!r}, expected {expected_outcome!r}"
    )


@pytest.mark.asyncio
async def test_trigger_runner_covers_full_enum():
    """``TRIGGER_EVENT_STATUSES`` enumerates every legal value the
    legacy column took. Pin the enum membership so adding a value
    to the source enum without extending the runner's mapping is a
    loud test failure here, not a silent ``outcome=<new-value>``
    leak."""
    from app.db.models.trigger import TRIGGER_EVENT_STATUSES

    # ``queued`` / ``running`` are non-terminal — the mapping is
    # only exercised at terminal-status sites.
    expected_terminals = {
        "success", "failed", "skipped_rate_limit",
        "skipped_filter", "coalesced",
    }
    actual_terminals = TRIGGER_EVENT_STATUSES - {"queued", "running"}
    assert actual_terminals == expected_terminals, (
        f"TRIGGER_EVENT_STATUSES terminal set drifted from the runner's "
        f"mapping expectations: {actual_terminals} vs "
        f"{expected_terminals}. Update both."
    )


@pytest.mark.asyncio
async def test_trigger_runner_error_message_only_on_failure():
    """Invariant: ``error_message`` on BuildJob is populated ONLY
    when the per-event terminal is ``failed``. A success row
    carrying a stale error_message would show as "succeeded with
    error" on the dashboard."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    # Success case: a TriggerResult.error_class set is ignored
    # because per_event_status==success → the runner's mapping
    # writes outcome='success' and skips error_message.
    trigger_id_1, job_id_1 = await _seed_user_and_trigger_job()

    class _SuccessLeakHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: "success" for e in events},
                error_class="rate_limited",
                error_detail="should not surface",
            )

    KIND_HANDLERS["email_received"] = _SuccessLeakHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._dispatch_one(job_id_1, attempt_idx=0)
    finally:
        KIND_HANDLERS.pop("email_received", None)
    job_1 = await _read_job(job_id_1)
    assert job_1.error_message is None, (
        "success row leaked error_message — must be NULL when outcome=success"
    )

    # Failed case: error_message lands on the row.
    trigger_id_2, job_id_2 = await _seed_user_and_trigger_job()

    class _FailHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: "failed" for e in events},
                error_class="rate_limited",
                error_detail="rate limit hit",
            )

    KIND_HANDLERS["email_received"] = _FailHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._dispatch_one(job_id_2, attempt_idx=0)
    finally:
        KIND_HANDLERS.pop("email_received", None)
    job_2 = await _read_job(job_id_2)
    assert "rate limit hit" in (job_2.error_message or ""), (
        "failed row missing error_message — Mission Control needs it"
    )


@pytest.mark.asyncio
async def test_trigger_runner_truncates_long_error_detail():
    """Defensive: error_detail can be unbounded (provider stack
    traces). The BuildJob.error_message column is sized to a
    bounded length; the runner truncates to 1000 chars."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    trigger_id, job_id = await _seed_user_and_trigger_job()
    huge_error = "x" * 5000

    class _HugeFailHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: "failed" for e in events},
                error_class="provider_error",
                error_detail=huge_error,
            )

    KIND_HANDLERS["email_received"] = _HugeFailHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._dispatch_one(job_id, attempt_idx=0)
    finally:
        KIND_HANDLERS.pop("email_received", None)
    job = await _read_job(job_id)
    assert len(job.error_message or "") <= 1000


# ──────────────────────────────────────────────────────────────────────
# Routine runner: status → (job_status, outcome) mapping.
# Exercises ``_finalize_run`` directly with each status combination.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_status,derived_outcome,expected_job_status,expected_outcome", [
    # success: outcome falls back to status when no derived value
    ("success",        None,             "completed", "success"),
    # success with derived outcome (Ticket 2.1 — success_empty / etc.)
    ("success",        "success_empty",  "completed", "success_empty"),
    # partial: derived takes precedence
    ("partial",        None,             "completed", "partial"),
    ("partial",        "partial",        "completed", "partial"),
    # skipped_reauth
    ("skipped_reauth", None,             "completed", "skipped_reauth"),
    # failed: outcome MUST be None (derived value ignored)
    ("failed",         None,             "failed",    None),
    ("failed",         "tool_error",     "failed",    None),
])
async def test_routine_runner_maps_status_via_finalize(
    legacy_status, derived_outcome,
    expected_job_status, expected_outcome,
):
    """Routine mapping (inlined in ``_finalize_run`` after PR #49):

      - status=failed → BuildJob.status=failed, outcome=NULL
                        (regardless of derived outcome)
      - status=success / partial / skipped_reauth →
          BuildJob.status=completed,
          outcome = derived OR raw status (fallback)
    """
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker

    routine_id, job_id = await _seed_user_and_routine_job()
    runner = RoutineRunner(session_maker=async_session_maker)
    await runner._finalize_run(
        job_id,
        status=legacy_status,
        outcome=derived_outcome,
    )

    job = await _read_job(job_id)
    assert job.status == expected_job_status, (
        f"legacy_status={legacy_status!r} outcome={derived_outcome!r} "
        f"mapped to job.status={job.status!r}, expected {expected_job_status!r}"
    )
    assert job.outcome == expected_outcome, (
        f"legacy_status={legacy_status!r} outcome={derived_outcome!r} "
        f"mapped to job.outcome={job.outcome!r}, expected {expected_outcome!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# completed_at always stamped on terminal write.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("per_event_status", ["success", "failed", "skipped_filter"])
async def test_trigger_runner_stamps_completed_at(per_event_status):
    """Every terminal write MUST set completed_at. A NULL
    completed_at after a terminal call would break the dashboard's
    "took N seconds" rendering."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    trigger_id, job_id = await _seed_user_and_trigger_job()

    class _Handler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: per_event_status for e in events},
                error_class=("e" if per_event_status == "failed" else None),
                error_detail=("forced" if per_event_status == "failed" else None),
            )

    KIND_HANDLERS["email_received"] = _Handler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._dispatch_one(job_id, attempt_idx=0)
    finally:
        KIND_HANDLERS.pop("email_received", None)
    job = await _read_job(job_id)
    assert job.completed_at is not None, (
        f"terminal write for {per_event_status!r} forgot completed_at"
    )
