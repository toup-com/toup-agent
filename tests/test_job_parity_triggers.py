"""Job parity — trigger fires materialise BuildJob rows.

Originally PR 4a of the unified-jobs arc, this suite pinned a
dual-write between ``trigger_events`` (legacy) and ``build_jobs``
(new). PR #49 of the cutover arc removed the legacy dual-write —
``build_jobs`` is now the sole source of truth. The tests below
were updated to reflect that: they assert the Job row shape and
the dispatch terminal mapping, with no remaining assertions about
``trigger_events``.

What we pin:

  1. ``triggers_inbound._idempotent_insert`` writes ONE BuildJob
     row per (trigger, event_dedupe_id) pair. ``source_id`` points
     at the trigger; ``idempotency_key`` is the event_dedupe_id;
     ``job_type='trigger_run'``.
  2. The composite UNIQUE on ``(source_id, idempotency_key)`` is
     the dedupe gate.
  3. Success-path dispatch through the runner writes
     ``status='completed'`` + ``outcome='success'`` on the
     BuildJob.
  4. Retry exhaustion path writes ``status='failed'`` +
     ``error_message=<repr>`` + ``completed_at`` on the BuildJob.
  5. A Job mint failure at intake aborts the whole intake — no
     downstream row exists.
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
    """Bypass conftest's init_db autouse fixture (entities pgvector
    breaks SQLite). PR #49: tests no longer touch TriggerEvent, so
    the fixture only sets up User, Trigger, BuildJob, JobEvent."""
    from app.db.database import engine
    from app.db.models import BuildJob, JobEvent, Trigger, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, JobEvent, Trigger):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (Trigger, JobEvent, BuildJob, User):
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
async def test_idempotent_insert_creates_job(_seed_user_and_trigger):
    """A successful ``_idempotent_insert`` writes ONE BuildJob row.
    The Job carries source_id=trigger.id,
    idempotency_key=event_dedupe_id, job_type='trigger_run',
    status='queued'."""
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
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
        job = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert job is not None
        assert job.job_type == "trigger_run"
        assert job.status == "queued"
        assert job.source_kind == "trigger"
        assert job.source_id == TRIGGER_ID
        assert job.idempotency_key == event_dedupe_id
        assert job.user_id == USER_ID


@pytest.mark.asyncio
async def test_idempotency_returns_same_job_on_dedupe(_seed_user_and_trigger):
    """Two ``_idempotent_insert`` calls with the same
    ``event_dedupe_id``: the first returns 'inserted' and mints
    a Job; the second returns 'dedupe_hit' and creates no
    duplicate. The composite UNIQUE on
    ``(source_id, idempotency_key)`` is the sole dedupe gate."""
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

    async with async_session_maker() as db:
        count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert count == 1


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
async def test_runner_writes_success_status_to_job(_seed_user_and_trigger):
    """When the handler returns ``success``, the runner's step-5 batch
    loop writes status='completed', outcome='success', completed_at
    set on the BuildJob."""
    from sqlalchemy import select

    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
        job_row = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        job_id = job_row.id

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
        await runner._handle_event_with_retry(job_id)
    finally:
        # Don't leak the stub handler into other tests.
        KIND_HANDLERS.pop("email_received", None)

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed", (
            f"Job status must be completed on success; got {job.status!r}"
        )
        assert job.outcome == "success", (
            f"Job outcome must carry the granular state; got {job.outcome!r}"
        )
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_runner_writes_retry_exhaustion_to_job(_seed_user_and_trigger):
    """When all 3 retries crash, ``_finalise_exhausted`` writes
    status='failed', error_message=<repr last exc>, completed_at on
    the BuildJob."""
    from sqlalchemy import select

    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
        job_row = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        job_id = job_row.id

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
        await runner._handle_event_with_retry(job_id)
    finally:
        KIND_HANDLERS.pop("email_received", None)

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "failed", (
            f"Job status must be failed; got {job.status!r}"
        )
        assert job.error_message and "handler boom" in job.error_message
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_intake_aborts_when_job_mint_fails(_seed_user_and_trigger, monkeypatch):
    """PR #46 of the unified-jobs arc inverted the dual-write order:
    ``build_jobs`` is the source of truth, minted first. If
    ``JobRunner.create_job`` raises, the whole intake aborts — no
    Job row gets written. Pub/Sub will retry the 5xx and we get
    another shot.

    PR #49 of the cutover arc: with the legacy dual-write removed,
    this property is even simpler — there's no other row to
    consider. The Job is the only row."""
    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
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

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert job_count == 0
