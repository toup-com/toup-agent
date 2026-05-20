"""PR #46 — trigger runner cutover from ``trigger_events`` to
``build_jobs``.

The 9-PR unified-jobs arc dual-wrote every trigger fire into both
tables; PR #46 moved the *reads* on the runner side over to
``build_jobs``, and PR #49 removed the legacy dual-write so the
runner now reads AND writes ``build_jobs`` exclusively.

What this file pins:

  1. ``_restart_sweep`` flips orphan ``build_jobs`` rows (status=
     'running', source_kind='trigger', stale ``created_at``) to
     status='failed' with a clear ``error_message``. PR #49: the
     legacy ``trigger_events`` UPDATE is gone — only the BuildJob
     write remains.
  2. ``_warm_rate_buckets`` seeds the per-trigger limiter from
     ``build_jobs.created_at`` timestamps, grouped by ``source_id``.
  3. ``_fetch_queued_ids`` returns ``BuildJob.id`` values for
     ``status='queued'`` + ``source_kind='trigger'`` rows ordered by
     ``created_at`` ASC.
  4. The intake path (``_idempotent_insert``) is Job-only: a
     second call with the same ``event_dedupe_id`` produces no
     duplicate Job. PR #49 removed the legacy TriggerEvent
     dual-write.
  5. A success-path ``_dispatch_one`` writes ``status='completed'``
     + ``outcome='success'`` onto the BuildJob.

Test fixture pattern matches ``test_job_parity_triggers.py`` — bypass
conftest's full ``init_db()`` (entities pgvector breaks SQLite) and
build only the narrow set of tables these tests need.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-runner-cutover")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-000000000cc1")


USER_ID = "00000000-0000-0000-0000-000000000cc1"
TRIGGER_ID = "00000000-0000-0000-0000-0000000cc100"


# ── Schema setup ──────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Narrow schema bootstrap — same approach as the rest of the
    unified-jobs test suite. We need User, Trigger, BuildJob,
    JobEvent. PR #49 removed the TriggerEvent dependency from the
    cutover paths."""
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
                email=f"cutover-{USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        db.add(Trigger(
            id=TRIGGER_ID,
            user_id=USER_ID,
            kind="email_received",
            # ``action`` is NOT NULL on Trigger; the runner doesn't
            # read it, but the schema enforces presence.
            action="summarize_and_post",
            name="cutover-test trigger",
            enabled=True,
            filter_json={},
            config_json={},
            provider_state_json={},
            last_status="never_fired",
            fire_count=0,
        ))
        await db.commit()


# ── _restart_sweep ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_sweep_terminalises_orphaned_build_jobs(
    _seed_user_and_trigger,
):
    """A BuildJob stuck in ``status='running'`` with a
    ``created_at`` older than ORPHAN_THRESHOLD must be flipped to
    ``status='failed'`` with ``error_message`` containing
    'agent_restarted' and ``completed_at`` populated."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    orphan_id = str(uuid.uuid4())
    fresh_id = str(uuid.uuid4())
    # Older than the 10-minute orphan threshold.
    orphan_age = TriggerRunner.ORPHAN_THRESHOLD + timedelta(minutes=5)
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=orphan_id,
            user_id=USER_ID,
            title="orphan",
            prompt="",
            job_type="trigger_run",
            status="running",
            source_kind="trigger",
            source_id=TRIGGER_ID,
            idempotency_key=f"orphan-{orphan_id[:8]}",
            created_at=datetime.utcnow() - orphan_age,
        ))
        # Control: a fresh 'running' row must NOT be swept.
        db.add(BuildJob(
            id=fresh_id,
            user_id=USER_ID,
            title="fresh",
            prompt="",
            job_type="trigger_run",
            status="running",
            source_kind="trigger",
            source_id=TRIGGER_ID,
            idempotency_key=f"fresh-{fresh_id[:8]}",
            created_at=datetime.utcnow(),
        ))
        await db.commit()

    runner = TriggerRunner(session_maker=async_session_maker)
    swept = await runner._restart_sweep()
    assert swept == 1, f"expected 1 orphan swept, got {swept}"

    async with async_session_maker() as db:
        orphan = await db.get(BuildJob, orphan_id)
        assert orphan.status == "failed"
        assert orphan.error_message and "agent_restarted" in orphan.error_message
        assert orphan.completed_at is not None
        fresh = await db.get(BuildJob, fresh_id)
        assert fresh.status == "running", (
            "fresh in-flight rows must not be swept"
        )


# ── _warm_rate_buckets ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_warm_rate_buckets_seeds_limiter_from_build_jobs(
    _seed_user_and_trigger,
):
    """The runner warms the per-trigger fire history by reading
    recent ``build_jobs`` rows. Three rows for one trigger within
    the last hour, with mixed statuses (completed/failed/running),
    must all be passed to ``limiter.warmup`` as a single batch keyed
    by ``source_id``."""
    from unittest.mock import MagicMock

    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    now = datetime.utcnow()
    rows = [
        ("completed", now - timedelta(minutes=10)),
        ("failed",    now - timedelta(minutes=20)),
        ("running",   now - timedelta(minutes=2)),
        # Outside the 1-hour window — must be excluded.
        ("completed", now - timedelta(hours=2)),
        # status='queued' — excluded (no fire has happened yet).
        ("queued",    now - timedelta(minutes=5)),
    ]
    async with async_session_maker() as db:
        for i, (status, created_at) in enumerate(rows):
            db.add(BuildJob(
                id=str(uuid.uuid4()),
                user_id=USER_ID,
                title=f"row{i}",
                prompt="",
                job_type="trigger_run",
                status=status,
                source_kind="trigger",
                source_id=TRIGGER_ID,
                idempotency_key=f"warm-{i}",
                created_at=created_at,
            ))
        await db.commit()

    runner = TriggerRunner(session_maker=async_session_maker)
    # Replace the limiter so we can spy on its warmup call without
    # touching internal state.
    runner._limiter = MagicMock()
    await runner._warm_rate_buckets()
    runner._limiter.warmup.assert_called_once()
    tid, fires = runner._limiter.warmup.call_args.args
    assert tid == TRIGGER_ID
    assert len(fires) == 3, (
        f"expected 3 in-window completed/failed/running fires, got {len(fires)}"
    )
    # All fires are timestamps — sanity check the type.
    assert all(isinstance(f, float) for f in fires)


# ── _fetch_queued_ids ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_queued_ids_returns_build_job_ids_ordered(
    _seed_user_and_trigger,
):
    """Queued trigger-sourced ``build_jobs`` rows come back in
    ``created_at`` ASC order, excluding any non-queued or
    non-trigger rows."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    base = datetime.utcnow() - timedelta(minutes=30)
    queued_ids: list[tuple[str, datetime]] = []
    for i in range(5):
        jid = str(uuid.uuid4())
        # Stagger created_at so ordering is deterministic.
        ts = base + timedelta(seconds=i)
        queued_ids.append((jid, ts))

    async with async_session_maker() as db:
        # 5 queued + trigger rows.
        for jid, ts in queued_ids:
            db.add(BuildJob(
                id=jid,
                user_id=USER_ID,
                title=f"queued-{jid[:6]}",
                prompt="",
                job_type="trigger_run",
                status="queued",
                source_kind="trigger",
                source_id=TRIGGER_ID,
                idempotency_key=f"queue-{jid[:8]}",
                created_at=ts,
            ))
        # 2 running rows — must be excluded.
        for i in range(2):
            db.add(BuildJob(
                id=str(uuid.uuid4()),
                user_id=USER_ID,
                title=f"running-{i}",
                prompt="",
                job_type="trigger_run",
                status="running",
                source_kind="trigger",
                source_id=TRIGGER_ID,
                idempotency_key=f"running-{i}",
                created_at=datetime.utcnow(),
            ))
        # 1 queued + non-trigger row — must be excluded.
        db.add(BuildJob(
            id=str(uuid.uuid4()),
            user_id=USER_ID,
            title="non-trigger",
            prompt="",
            job_type="agent_task",
            status="queued",
            source_kind="manual",
            source_id=None,
            idempotency_key=None,
            created_at=datetime.utcnow(),
        ))
        await db.commit()

    runner = TriggerRunner(session_maker=async_session_maker)
    out = await runner._fetch_queued_ids(limit=10)
    assert len(out) == 5, f"expected 5 queued trigger rows, got {len(out)}"
    # Ordering check: created_at ASC matches our insertion order.
    expected_order = [jid for jid, _ in queued_ids]
    assert out == expected_order


# ── Intake dedupe ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_intake_dedupe_creates_one_job(_seed_user_and_trigger):
    """Two ``_idempotent_insert`` calls with the same
    ``event_dedupe_id`` produce exactly one BuildJob and the second
    call returns ``"dedupe_hit"``.

    PR #49 removed the legacy ``trigger_events`` dual-write — the
    composite UNIQUE on ``(build_jobs.source_id,
    build_jobs.idempotency_key)`` is the sole dedupe gate."""
    from sqlalchemy import select, func

    from app.api.triggers_inbound import _idempotent_insert
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    event_dedupe_id = f"gmail-msg-{uuid.uuid4().hex[:12]}"

    async with async_session_maker() as db:
        first = await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
    async with async_session_maker() as db:
        second = await _idempotent_insert(
            db, trigger_id=TRIGGER_ID, user_id=USER_ID,
            event_dedupe_id=event_dedupe_id,
        )
    assert first == "inserted"
    assert second == "dedupe_hit"

    async with async_session_maker() as db:
        job_count = (await db.execute(
            select(func.count(BuildJob.id)).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        assert job_count == 1


# ── Dispatch success mirror ───────────────────────────────────────────


@dataclass
class _FakeTriggerResult:
    """Stub for ``app.agent.triggers.base_handler.TriggerResult``."""
    status: str = "success"
    per_event_status: dict = field(default_factory=dict)
    summary_message_id: Optional[str] = None
    new_provider_state: Optional[dict] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    metrics: dict = field(default_factory=dict)


@pytest.mark.asyncio
async def test_dispatch_one_writes_success_to_build_job(_seed_user_and_trigger):
    """``_dispatch_one`` on a queued Job whose handler returns
    success: BuildJob.status='completed' with outcome='success'.

    PR #49: the legacy TriggerEvent mirror is gone — the BuildJob
    UPDATE in the runner's step-5 loop IS the terminal write."""
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
        job = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_id == TRIGGER_ID,
                BuildJob.idempotency_key == event_dedupe_id,
            )
        )).scalar_one()
        job_id = job.id

    class _SuccessHandler:
        kind = "email_received"
        async def execute(self, trigger, events, db):
            return _FakeTriggerResult(
                status="success",
                per_event_status={e.id: "success" for e in events},
            )

    KIND_HANDLERS["email_received"] = _SuccessHandler()
    try:
        runner = TriggerRunner(
            session_maker=async_session_maker,
            retry_delays=(0.001, 0.001, 0.001),
        )
        ok = await runner._dispatch_one(job_id, attempt_idx=0)
    finally:
        KIND_HANDLERS.pop("email_received", None)

    assert ok is True
    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed", (
            f"BuildJob.status must be completed on success; got {job.status!r}"
        )
        assert job.outcome == "success"
        assert job.completed_at is not None
