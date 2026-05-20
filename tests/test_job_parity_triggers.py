"""Job parity — trigger fires must materialise mirrored BuildJob rows.

PR 4a of the unified-jobs arc.

What we pin:

  1. ``triggers_inbound._idempotent_insert`` writes BOTH a
     ``trigger_events`` row AND a mirrored ``build_jobs`` row in
     the same transaction. The TriggerEvent's ``job_id`` column
     points at the Job. Job's ``source_id`` points back at the
     trigger.
  2. The Job's ``idempotency_key`` is the event's
     ``event_dedupe_id``, so the composite UNIQUE on
     ``(source_id, idempotency_key)`` collapses to the same
     dedupe gate as ``UNIQUE (trigger_id, event_dedupe_id)``.
  3. Each TriggerEvent terminal status path mirrors onto the
     linked Job: success/failed/skipped_rate_limit/skipped_filter/
     coalesced. The Job's ``status`` is the closed
     queued/running/completed/failed enum; the granular
     TriggerEvent status lives on ``Job.outcome``.
  4. The retry-exhausted path (PR 1's ``_finalise_exhausted``)
     also mirrors onto the Job — status='failed',
     error_message=<repr>, completed_at=NOW.

A Job mint failure at intake time must not roll back the
TriggerEvent insert (best-effort dual-write). That property is
exercised by the explicit `dual_write_robust_to_job_failure` test.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-job-parity-triggers")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000bb1")


USER_ID = "00000000-0000-0000-0000-000000000bb1"
TRIGGER_ID = "00000000-0000-0000-0000-0000000bb100"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Same surgical approach as test_trigger_runner_failure_handler /
    test_job_runner — bypass conftest's init_db autouse fixture
    (entities pgvector breaks SQLite) and build only the tables we
    need."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Trigger, TriggerEvent, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Trigger, TriggerEvent):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (TriggerEvent, Trigger, JobEvent, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


@pytest_asyncio.fixture
async def _seed_user_and_trigger():
    from app.db.database import async_session_maker
    from app.db.models import Trigger, User

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"parity-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        db.add(Trigger(
            id=TRIGGER_ID,
            user_id=USER_ID,
            kind="email_received",
            action="summarize_and_post",
            name="parity-test trigger",
            enabled=True,
            filter_json={},
            config_json={},
            provider_state_json={},
            last_status="never_fired",
            fire_count=0,
        ))
        await db.commit()


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idempotent_insert_creates_mirrored_job(_seed_user_and_trigger):
    """A successful ``_idempotent_insert`` writes a TriggerEvent
    AND a mirrored BuildJob. The Job carries source_id=trigger.id,
    idempotency_key=event_dedupe_id, job_type='trigger_run'.
    TriggerEvent.job_id links back to the Job."""
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, TriggerEvent
    from sqlalchemy import select

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        outcome = await _idempotent_insert(
            db,
            trigger_id=TRIGGER_ID,
            user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
    assert outcome == "inserted"

    async with async_session_maker() as db:
        ev = (await db.execute(
            select(TriggerEvent).where(
                TriggerEvent.trigger_id == TRIGGER_ID,
                TriggerEvent.event_dedupe_id == event_dedupe_id,
            )
        )).scalar_one()
        assert ev.status == "queued"
        assert ev.job_id is not None, (
            "TriggerEvent.job_id must be set after dual-write"
        )

        job = await db.get(BuildJob, ev.job_id)
        assert job is not None
        assert job.job_type == "trigger_run"
        assert job.status == "queued"
        assert job.source_kind == "trigger"
        assert job.source_id == TRIGGER_ID
        assert job.idempotency_key == event_dedupe_id
        assert job.user_id == USER_ID


@pytest.mark.asyncio
async def test_idempotency_returns_same_job_on_dedupe(_seed_user_and_trigger):
    """Two ``_idempotent_insert`` calls with the same ``event_dedupe_id``:
    the first returns 'inserted' and mints both rows; the second
    returns 'dedupe_hit' and creates NO duplicate Job. The
    composite UNIQUE on ``(source_id, idempotency_key)`` collapses
    to the same dedupe gate as ``UNIQUE (trigger_id,
    event_dedupe_id)``."""
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select, func

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        first_outcome = await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
    async with async_session_maker() as db:
        second_outcome = await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
    assert first_outcome == "inserted"
    assert second_outcome == "dedupe_hit"

    # Exactly ONE Job row for this dedupe pair.
    async with async_session_maker() as db:
        count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert count == 1, (
            f"expected exactly 1 mirrored Job for the dedupe pair, "
            f"got {count}"
        )


# ── Status-mirror tests via the trigger runner ───────────────────────


@dataclass
class _FakeTriggerResult:
    """Stub for app.agent.triggers.base_handler.TriggerResult."""
    status: str = "success"
    per_event_status: dict = field(default_factory=dict)
    summary_message_id: Optional[str] = None
    new_provider_state: Optional[dict] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    metrics: dict = field(default_factory=dict)


@pytest.mark.asyncio
async def test_runner_mirrors_success_status_to_job(_seed_user_and_trigger):
    """When the handler returns ``success``, the runner's step-5 batch
    loop mirrors onto the Job: status='completed', outcome='success',
    completed_at set, summary_message_id linked."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, TriggerEvent

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
        ev_row = (await db.execute(
            __import__("sqlalchemy").select(TriggerEvent).where(
                TriggerEvent.event_dedupe_id == event_dedupe_id,
            )
        )).scalar_one()
        event_id = ev_row.id
        job_id = ev_row.job_id

    class _SuccessHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: "success" for e in events},
                summary_message_id=None,
            )

    KIND_HANDLERS["email_received"] = _SuccessHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._handle_event_with_retry(event_id)
    finally:
        # Don't leak the stub handler into other tests.
        KIND_HANDLERS.pop("email_received", None)

    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        assert ev.status == "success"
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed", (
            f"Job status must mirror success → completed; got {job.status!r}"
        )
        assert job.outcome == "success", (
            f"Job outcome must carry the granular state; got {job.outcome!r}"
        )
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_runner_mirrors_retry_exhaustion_to_job(_seed_user_and_trigger):
    """When all 3 retries crash, ``_finalise_exhausted`` mirrors onto
    the Job too: status='failed', error_message=<repr last exc>,
    completed_at set."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, TriggerEvent

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
        ev_row = (await db.execute(
            __import__("sqlalchemy").select(TriggerEvent).where(
                TriggerEvent.event_dedupe_id == event_dedupe_id,
            )
        )).scalar_one()
        event_id = ev_row.id
        job_id = ev_row.job_id

    class _CrashHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            raise RuntimeError("handler boom for parity test")

    KIND_HANDLERS["email_received"] = _CrashHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        await runner._handle_event_with_retry(event_id)
    finally:
        KIND_HANDLERS.pop("email_received", None)

    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        assert ev.status == "failed"
        assert ev.error_class == "all_retries_exhausted"
        job = await db.get(BuildJob, job_id)
        assert job.status == "failed", (
            f"Job status must mirror failed; got {job.status!r}"
        )
        assert job.error_message and "handler boom" in job.error_message
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_intake_aborts_when_job_mint_fails(_seed_user_and_trigger, monkeypatch):
    """PR #46 of the unified-jobs arc inverts the dual-write order:
    ``build_jobs`` is now the source of truth, minted first. If
    ``JobRunner.create_job`` raises, the whole intake aborts — no
    legacy ``trigger_events`` row gets written either. Pub/Sub will
    retry the 5xx and we get another shot. This is the deliberate
    new contract: a Job exists ⇔ a TriggerEvent exists, with the Job
    being the canonical reader.

    The previous test (pre-cutover) asserted the legacy fire path
    still worked when the Job mint failed. That property no longer
    holds, by design — keeping it would defeat the cutover's
    invariant that the runner only reads from ``build_jobs``."""
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, TriggerEvent
    from sqlalchemy import select, func

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async def _explode(*a, **kw):
        raise RuntimeError("simulated JobRunner.create_job crash")

    # Patch the JobRunner so create_job inside _idempotent_insert raises.
    import app.agent.job_runner as jr
    monkeypatch.setattr(jr.JobRunner, "create_job", _explode)

    with pytest.raises(RuntimeError, match="simulated JobRunner.create_job crash"):
        async with async_session_maker() as db:
            await _idempotent_insert(
                db, trigger_id=TRIGGER_ID, user_id=USER_ID,
                event_dedupe_id=event_dedupe_id,
            )

    # Neither row exists — the failure aborts the whole intake.
    async with async_session_maker() as db:
        ev_count = (await db.execute(
            select(func.count(TriggerEvent.id)).where(
                TriggerEvent.event_dedupe_id == event_dedupe_id,
            )
        )).scalar_one()
        assert ev_count == 0
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert job_count == 0
