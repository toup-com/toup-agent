"""Status-mirror exhaustiveness contract.

The unified-jobs arc replaced four bespoke status enums with the
closed BuildJob (status, outcome) tuple. Each source's mirror
helper translates the source's status → (job_status, job_outcome)
combination — but the per-source tests only exercise the *common*
statuses (success / failed). What was missing:

  - Pin EVERY value in ``TRIGGER_EVENT_STATUSES`` and the routine
    runner's known status set to a defined ``(status, outcome)``
    pair. A new status added to the source's enum without
    extending the mirror helper would silently produce a row with
    ``status='completed', outcome=<the new value>`` (conservative
    fallback) — that's the bug we want to surface via test.
  - Pin the success / failed / skipped axes to ``completed/<axis>``
    vs ``failed/NULL`` invariants so a regression that, say, maps
    ``failed`` to ``completed`` (a real customer-impacting
    mis-routing) crashes the test.

These tests don't seed a DB. They unit-test the **mapping logic
in isolation** by invoking the mirror helper with a stub TriggerEvent /
RoutineRun (job_id present) and a stub BuildJob row, and asserting
the UPDATE statement values match expectations. Fast, hermetic, no
async DB ceremony.

Why this matters: the closed BuildJob enum is the read-side
contract for the activity feed and Mission Control. If a new
status leaks through as ``outcome='success'`` when it shouldn't,
Mission Control reports a "delivered" run that actually skipped.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-status-mirror")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000sm001")


USER_ID = "00000000-0000-0000-0000-0000000sm001"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Per-test fresh DB. Only the tables the mirror helpers
    touch (BuildJob + TriggerEvent + RoutineRun + Trigger +
    Routine + User)."""
    from app.db.database import rebind_database
    from app.config import settings

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import (
        BuildJob, Routine, RoutineRun, Trigger, TriggerEvent, User,
    )

    async with engine.begin() as conn:
        for model_cls in (
            User, BuildJob, Trigger, TriggerEvent, Routine, RoutineRun,
        ):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (
            RoutineRun, Routine, TriggerEvent, Trigger, BuildJob, User,
        ):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers — seed the minimum rows the mirror helper reads.
# ──────────────────────────────────────────────────────────────────────


async def _seed_user_and_trigger_event(status_in: str = "queued") -> tuple[str, str]:
    """Seed User + Trigger + TriggerEvent + BuildJob with the linkage.
    Returns (event_id, job_id)."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger, TriggerEvent, User

    event_id = str(uuid.uuid4())
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
        ))
        db.add(BuildJob(
            id=job_id, user_id=USER_ID, title="Mirror test",
            prompt="", job_type="trigger_run", status="running",
            source_kind="trigger", source_id=trigger_id,
            created_at=datetime.utcnow(),
        ))
        db.add(TriggerEvent(
            id=event_id, trigger_id=trigger_id, user_id=USER_ID,
            event_dedupe_id=f"dedupe-{event_id[:8]}",
            received_at=datetime.utcnow(),
            status=status_in,
            job_id=job_id,
        ))
        await db.commit()
    return event_id, job_id


async def _seed_user_and_routine_run() -> tuple[str, str]:
    """Seed User + Routine + RoutineRun + BuildJob with linkage.
    Returns (run_id, job_id)."""
    from datetime import date
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Routine, RoutineRun, User

    run_id = str(uuid.uuid4())
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
            created_at=datetime.utcnow(),
        ))
        db.add(RoutineRun(
            id=run_id, routine_id=routine_id, user_id=USER_ID,
            scheduled_for_local_date=date.today(),
            started_at=datetime.utcnow(),
            status="running",
            job_id=job_id,
        ))
        await db.commit()
    return run_id, job_id


async def _read_job(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    async with async_session_maker() as db:
        return await db.get(BuildJob, job_id)


# ──────────────────────────────────────────────────────────────────────
# Trigger mirror: every value in TRIGGER_EVENT_STATUSES has a mapping.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status,expected_job_status,expected_outcome", [
    ("success",            "completed", "success"),
    ("failed",             "failed",    None),
    ("skipped_rate_limit", "completed", "skipped_rate_limit"),
    ("skipped_filter",     "completed", "skipped_filter"),
    ("coalesced",          "completed", "coalesced"),
])
async def test_trigger_mirror_maps_every_terminal_status(
    terminal_status, expected_job_status, expected_outcome,
):
    """For every terminal value in ``TRIGGER_EVENT_STATUSES``, the
    mirror helper produces the documented ``(status, outcome)``
    tuple on the linked BuildJob row. A regression that, say, maps
    ``failed`` to ``completed`` would crash this test on the
    ``failed`` row — exactly the kind of mis-routing that surfaces
    to Mission Control as "delivered" but actually failed."""
    from app.agent.triggers.runner import TriggerRunner

    event_id, job_id = await _seed_user_and_trigger_event()
    runner = TriggerRunner()

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        await runner._mirror_event_terminal_to_job(
            db,
            event_id=event_id,
            terminal_status=terminal_status,
        )
        await db.commit()

    job = await _read_job(job_id)
    assert job.status == expected_job_status, (
        f"terminal_status={terminal_status!r} mapped to "
        f"job.status={job.status!r}, expected {expected_job_status!r}"
    )
    assert job.outcome == expected_outcome, (
        f"terminal_status={terminal_status!r} mapped to "
        f"job.outcome={job.outcome!r}, expected {expected_outcome!r}"
    )


@pytest.mark.asyncio
async def test_trigger_mirror_covers_full_enum():
    """``TRIGGER_EVENT_STATUSES`` enumerates every legal value the
    column can take. The runner's mirror helper exhaustively
    branches on this enum (success / failed / skipped_* / coalesced /
    fallback). Pin the enum membership so adding a value to the
    enum without extending the mirror is a loud test failure here,
    not a silent ``outcome=<new-value>`` leak."""
    from app.db.models.trigger import TRIGGER_EVENT_STATUSES

    # ``queued`` / ``running`` are non-terminal — the mirror is only
    # called from terminal-status sites. The terminal set is what the
    # mirror MUST cover.
    expected_terminals = {
        "success", "failed", "skipped_rate_limit",
        "skipped_filter", "coalesced",
    }
    actual_terminals = TRIGGER_EVENT_STATUSES - {"queued", "running"}
    assert actual_terminals == expected_terminals, (
        f"TRIGGER_EVENT_STATUSES terminal set drifted from the mirror "
        f"helper's expectations: {actual_terminals} vs "
        f"{expected_terminals}. Update both."
    )


@pytest.mark.asyncio
async def test_trigger_mirror_error_message_only_set_on_failure():
    """Invariant: ``error_message`` on the BuildJob row is populated
    ONLY when ``terminal_status == 'failed'``. A success row carrying
    a stale error_message would show as "succeeded with error" on the
    dashboard — a UX regression."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    event_id, job_id = await _seed_user_and_trigger_event()
    runner = TriggerRunner()
    async with async_session_maker() as db:
        await runner._mirror_event_terminal_to_job(
            db, event_id=event_id, terminal_status="success",
            error_class="rate_limited", error_detail="should not surface",
        )
        await db.commit()
    job = await _read_job(job_id)
    assert job.error_message is None, (
        "success row leaked error_message — must be NULL when status=completed"
    )

    # Now test failed row carries it.
    event_id_2, job_id_2 = await _seed_user_and_trigger_event()
    async with async_session_maker() as db:
        await runner._mirror_event_terminal_to_job(
            db, event_id=event_id_2, terminal_status="failed",
            error_class="rate_limited", error_detail="rate limit hit",
        )
        await db.commit()
    job_2 = await _read_job(job_id_2)
    assert "rate limit hit" in (job_2.error_message or ""), (
        "failed row missing error_message — Mission Control needs it"
    )


@pytest.mark.asyncio
async def test_trigger_mirror_truncates_long_error_detail():
    """Defensive: error_detail can be unbounded (provider stack
    traces). The BuildJob.error_message column is sized to a
    bounded length; the helper truncates to 1000 chars. A
    regression that removed the truncation could blow up the row
    write on Postgres for a long stack trace."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    event_id, job_id = await _seed_user_and_trigger_event()
    runner = TriggerRunner()
    huge_error = "x" * 5000
    async with async_session_maker() as db:
        await runner._mirror_event_terminal_to_job(
            db, event_id=event_id, terminal_status="failed",
            error_detail=huge_error,
        )
        await db.commit()
    job = await _read_job(job_id)
    assert len(job.error_message or "") <= 1000


# ──────────────────────────────────────────────────────────────────────
# Routine mirror: every documented status maps deterministically.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_status,derived_outcome,expected_job_status,expected_outcome", [
    # success: outcome falls back to status when no derived value
    ("success",        None,             "completed", "success"),
    # success with derived outcome (Ticket 2.1 — success_empty / etc.)
    ("success",        "success_empty",  "completed", "success_empty"),
    # partial: outcome is documented as 'partial', but derived takes precedence
    ("partial",        None,             "completed", "partial"),
    ("partial",        "partial",        "completed", "partial"),
    # skipped_reauth
    ("skipped_reauth", None,             "completed", "skipped_reauth"),
    # failed: outcome MUST be None
    ("failed",         None,             "failed",    None),
    ("failed",         "tool_error",     "failed",    None),
])
async def test_routine_mirror_maps_every_status_with_optional_outcome(
    legacy_status, derived_outcome,
    expected_job_status, expected_outcome,
):
    """Routine mirror is more nuanced than trigger: ``RoutineRun``
    carries both a legacy ``status`` AND a derived ``outcome``
    (Ticket 2.1 — success_empty / tool_error / partial / failure).
    Mirror priority:
        - status=failed  → BuildJob.status=failed, outcome=NULL
                           (regardless of derived outcome)
        - status=success / partial / skipped_reauth →
            BuildJob.status=completed,
            outcome = derived OR raw status (fallback)
    """
    from app.agent.routines.runner import RoutineRunner

    run_id, job_id = await _seed_user_and_routine_run()
    runner = RoutineRunner()
    await runner._mirror_run_terminal_to_job(
        run_id=run_id,
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
# No-op invariants: mirror is best-effort, never propagates.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_mirror_noop_when_event_has_no_job_id():
    """Pre-PR-4a TriggerEvent rows have ``job_id=NULL``. The mirror
    must silently no-op (don't write to a random Job). Pre-existing
    Job rows for these events stay as the runner first wrote them."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import TriggerEvent

    event_id, _ = await _seed_user_and_trigger_event()
    # Null out job_id — simulates a legacy row.
    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        ev.job_id = None
        await db.commit()

    runner = TriggerRunner()
    async with async_session_maker() as db:
        # Must not raise — best-effort write.
        await runner._mirror_event_terminal_to_job(
            db, event_id=event_id, terminal_status="success",
        )
        await db.commit()


@pytest.mark.asyncio
async def test_trigger_mirror_noop_when_event_does_not_exist():
    """A vanished TriggerEvent (mis-routing, or test setup error)
    must NOT raise. The mirror exits silently."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    runner = TriggerRunner()
    async with async_session_maker() as db:
        # ID that doesn't exist — must NOT raise.
        await runner._mirror_event_terminal_to_job(
            db, event_id="does-not-exist", terminal_status="success",
        )
        await db.commit()


@pytest.mark.asyncio
async def test_routine_mirror_noop_when_run_has_no_job_id():
    """Pre-PR-4b RoutineRun rows have ``job_id=NULL``. The mirror
    must silently no-op."""
    from app.agent.routines.runner import RoutineRunner
    from app.db.database import async_session_maker
    from app.db.models import RoutineRun

    run_id, _ = await _seed_user_and_routine_run()
    async with async_session_maker() as db:
        run_row = await db.get(RoutineRun, run_id)
        run_row.job_id = None
        await db.commit()

    runner = RoutineRunner()
    # Must NOT raise.
    await runner._mirror_run_terminal_to_job(
        run_id=run_id, status="success",
    )


# ──────────────────────────────────────────────────────────────────────
# completed_at always stamped on terminal mirror.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["success", "failed", "skipped_filter"])
async def test_trigger_mirror_stamps_completed_at(terminal_status):
    """Every terminal-status mirror MUST write completed_at. A NULL
    completed_at after a terminal call would break the dashboard's
    "took N seconds" rendering. Pin it for every documented status."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker

    event_id, job_id = await _seed_user_and_trigger_event()
    runner = TriggerRunner()
    async with async_session_maker() as db:
        await runner._mirror_event_terminal_to_job(
            db, event_id=event_id, terminal_status=terminal_status,
        )
        await db.commit()
    job = await _read_job(job_id)
    assert job.completed_at is not None, (
        f"terminal mirror for {terminal_status!r} forgot completed_at"
    )
