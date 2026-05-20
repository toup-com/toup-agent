# PR #50 cutover: assertions updated from TriggerEvent → BuildJob. Same invariants, new source-of-truth row.
"""Trigger runner — post-loop failure-handler fix (D7).

Pinned bug:

  `_handle_event_with_retry` in `backend/app/agent/triggers/runner.py`
  loops over `DEFAULT_RETRY_DELAYS = (5.0, 30.0, 120.0)` calling
  `_dispatch_one` per attempt. Pre-fix, when all three attempts crashed
  the loop fell through with NO terminal write — the BuildJob row
  stayed in `status='running'` until the 10-minute orphan sweep flipped
  it to `failed` with an `agent_restarted` marker and no useful
  error context. The user's live "Summarize every new Gmail" trigger
  accumulated 11 such rows.

  Fix: after the retry loop falls through, call `_finalise_exhausted`
  which UPDATEs the row to:
    status='failed'
    error_message='all_retries_exhausted: ' + repr(last_exception)
    completed_at=NOW()
  …and stamps the parent Trigger with last_status='failed', last_error,
  fire_count += 1, last_fired_at.

These two parametrised tests pin both halves of the spec:

  1. All N attempts crash → terminal state matches the spec exactly.
  2. Attempts 1..N-1 crash, attempt N succeeds → row ends in
     status='completed' / outcome='success', NO error fields touched,
     retry count matches.

The tests build a TriggerRunner with `retry_delays=(0.01, 0.01, 0.01)`
so they run in milliseconds, and inject a fake handler into
KIND_HANDLERS that's configurable per test.

PR #50 cutover: the runner no longer writes ``trigger_events``. All
assertions previously made against ``TriggerEvent`` are now made
against ``BuildJob`` with the column mapping:

  TriggerEvent.status='failed'                 → BuildJob.status='failed'
  TriggerEvent.error_class + .error_detail     → BuildJob.error_message
                                                  (single Text column,
                                                  format
                                                  'all_retries_exhausted: '
                                                  + repr(exc))
  TriggerEvent.finished_at                     → BuildJob.completed_at
  TriggerEvent.status='success'                → BuildJob.status='completed'
                                                  + BuildJob.outcome='success'

Trigger-row assertions (last_status, last_error, fire_count,
last_fired_at) are unchanged — the runner still stamps the parent
Trigger the same way.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-runner-failure-handler")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000e0")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")


CONTAINER_USER_ID = "00000000-0000-0000-0000-0000000000e0"
TRIGGER_ID = "00000000-0000-0000-0000-0000000ff000"


# Override the conftest autouse `_reset_database` fixture. The default
# fixture calls ``init_db()`` which tries to create the entire agent
# schema, including the ``entities`` table whose pgvector ``embedding``
# column generates a SQLAlchemy NullType DDL error under SQLite. CI
# defaults the job to RUN_MODE=platform precisely to skip that table —
# but RUN_MODE=platform also excludes ``triggers`` / ``build_jobs``,
# which these tests need.
#
# Solution: build only the 3 tables the trigger-runner tests touch
# (``users``, ``triggers``, ``build_jobs``). PR #50 removed every
# ``trigger_events`` write from the runner, so we no longer create
# that table here. Matches the seed pattern in
# ``test_trigger_runner_cutover.py``.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    from app.db.database import engine
    from app.db.models import BuildJob, Trigger, User

    async with engine.begin() as conn:
        for model_cls in (User, BuildJob, Trigger):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (Trigger, BuildJob, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)
    await engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Source-grep guard — the fix's existence in the codebase.
# ──────────────────────────────────────────────────────────────────────

from pathlib import Path

_RUNNER_SRC = (
    Path(__file__).resolve().parent.parent
    / "app/agent/triggers/runner.py"
).read_text()


def test_runner_source_has_finalise_exhausted_helper():
    """The fix introduces a `_finalise_exhausted` method that the
    retry loop calls on fallthrough. Source-grep so a future refactor
    that removes the helper without an equivalent inline write trips
    the test."""
    assert "_finalise_exhausted" in _RUNNER_SRC, (
        "_finalise_exhausted helper missing from runner.py — the retry "
        "loop fallthrough must call something that marks the row failed."
    )
    assert "all_retries_exhausted" in _RUNNER_SRC, (
        "error marker 'all_retries_exhausted' literal missing from "
        "runner.py — the post-loop write would not match the spec."
    )


# ──────────────────────────────────────────────────────────────────────
# Fixtures.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeHandler:
    """Test handler whose execute() can be configured to crash a
    specified number of times before succeeding (or always crash).

    `fail_first_n=None` → fail on every call (covers test #1).
    `fail_first_n=2` → fail on attempts 0 and 1, succeed on attempt 2
    (covers test #2 — the partial-then-success path).
    """

    fail_first_n: Optional[int] = None
    failure_exc: BaseException = field(default_factory=lambda: RuntimeError("simulated handler crash"))
    call_count: int = 0
    kind: str = "email_received"

    async def execute(self, trigger, events, db):  # noqa: D401
        self.call_count += 1
        should_fail = (
            self.fail_first_n is None
            or self.call_count <= self.fail_first_n
        )
        if should_fail:
            raise self.failure_exc
        # Success path — import locally so import order doesn't matter.
        from app.agent.triggers.base_handler import TriggerResult
        return TriggerResult(
            status="success",
            per_event_status={e.id: "success" for e in events},
            summary_message_id=None,
            new_provider_state=None,
            error_class=None,
            error_detail=None,
            metrics={"fake": True},
        )


@pytest_asyncio.fixture
async def _seed_user_and_trigger():
    """Insert User + Trigger rows for the runner to find. The autouse
    `_reset_database` fixture above gives us a clean schema."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger, User

    async with async_session_maker() as db:
        if await db.get(User, CONTAINER_USER_ID) is None:
            db.add(User(
                id=CONTAINER_USER_ID,
                email=f"runner-failure-{CONTAINER_USER_ID[:8]}@example.com",
                hashed_password="x",
            ))
        db.add(Trigger(
            id=TRIGGER_ID,
            user_id=CONTAINER_USER_ID,
            kind="email_received",
            action="summarize_and_post",
            name="Test trigger",
            enabled=True,
            filter_json={},
            config_json={},
            provider_state_json={},
            last_status="never_fired",
            fire_count=0,
        ))
        await db.commit()
    return {"user_id": CONTAINER_USER_ID, "trigger_id": TRIGGER_ID}


async def _insert_buildjob(trigger_id: str, user_id: str) -> str:
    """Insert a fresh BuildJob in status='queued' as the runner's
    source-of-truth row for one trigger fire. PR #50: this replaces
    the legacy ``_insert_event`` helper that wrote TriggerEvent."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    jid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=jid,
            user_id=user_id,
            title="Test fire",
            prompt="",
            job_type="trigger_run",
            status="queued",
            source_kind="trigger",
            source_id=trigger_id,
            idempotency_key=f"test:{jid[:8]}",
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return jid


# ──────────────────────────────────────────────────────────────────────
# Behavioural tests.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("boom"),
        ValueError("bad data"),
        # Use a custom exception to confirm repr() captures the class name.
        type("HandlerDownstreamError", (Exception,), {})("downstream service is down"),
    ],
)
@pytest.mark.asyncio
async def test_all_retries_exhausted_terminalises_row_and_trigger(
    _seed_user_and_trigger, monkeypatch, exc
):
    """When _dispatch_one raises on every retry attempt, the runner must:

      - flip BuildJob.status from 'running' to 'failed'
      - write error_message containing 'all_retries_exhausted' AND a
        recognisable reference to the last exception (the actual
        message, not NULL — that's the regression we're pinning)
      - stamp completed_at
      - bump Trigger.fire_count by 1
      - set Trigger.last_status='failed' + last_error + last_fired_at

    Pre-fix, ALL of these were broken: the row stayed in 'running' with
    NULL error fields and the parent trigger was never stamped, so
    `fire_count` showed the count of dispatch attempts but `last_status`
    was whatever the previous fire left it (often 'active' or
    'never_fired').

    PR #50: the column for the error string is the single
    ``BuildJob.error_message`` Text column — the old TriggerEvent
    pair (``error_class`` + ``error_detail``) is folded into one
    string of the form ``'all_retries_exhausted: ' + repr(exc)``."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    job_id = await _insert_buildjob(trigger_id, user_id)

    fake = _FakeHandler(fail_first_n=None, failure_exc=exc)
    monkeypatch.setitem(KIND_HANDLERS, "email_received", fake)

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    await runner._handle_event_with_retry(job_id)

    # Handler called once per retry attempt.
    assert fake.call_count == 3, (
        f"expected 3 handler attempts, got {fake.call_count}"
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job is not None, "build_job row vanished"
        assert job.status == "failed", (
            f"expected status='failed' after retries exhausted, "
            f"got {job.status!r} — this is the regression."
        )
        assert job.error_message is not None, (
            "error_message is NULL — pre-fix symptom. The "
            "'all_retries_exhausted' marker plus the repr of the "
            "last exception must be persisted so operators can diagnose."
        )
        assert "all_retries_exhausted" in job.error_message, (
            f"error_message {job.error_message!r} should carry the "
            f"'all_retries_exhausted' marker."
        )
        # The error_message should contain something recognisable from
        # the exception. Each parametrised case has a unique substring.
        assert (
            repr(exc).split("(", 1)[0] in job.error_message
            or str(exc) in job.error_message
        ), (
            f"error_message {job.error_message!r} should reference "
            f"exception {exc!r}"
        )
        assert job.completed_at is not None, "completed_at must be stamped"

        trig = await db.get(Trigger, trigger_id)
        assert trig is not None
        assert trig.last_status == "failed", (
            f"parent trigger.last_status should be 'failed', got {trig.last_status!r}"
        )
        assert trig.last_error and (
            repr(exc).split("(", 1)[0] in trig.last_error or str(exc) in trig.last_error
        ), f"trigger.last_error {trig.last_error!r} should reference exception"
        assert trig.last_fired_at is not None
        assert trig.fire_count == 1, (
            f"trigger.fire_count not bumped: got {trig.fire_count}"
        )


@pytest.mark.parametrize("fail_first_n", [1, 2])
@pytest.mark.asyncio
async def test_partial_failure_then_success_does_not_touch_error_fields(
    _seed_user_and_trigger, monkeypatch, fail_first_n
):
    """When the first `fail_first_n` attempts crash but a later attempt
    succeeds, the BuildJob row must reach status='completed' /
    outcome='success' with a NULL ``error_message`` — i.e. the
    'all_retries_exhausted' marker is NOT written on a path that
    ultimately succeeded. The parent trigger should be promoted to
    last_status='active' the same way a normal first-try success would
    be.

    This pins the inverse regression: don't write the 'all retries
    exhausted' marker on a path that ultimately succeeded."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    job_id = await _insert_buildjob(trigger_id, user_id)

    fake = _FakeHandler(
        fail_first_n=fail_first_n,
        failure_exc=RuntimeError("transient"),
    )
    monkeypatch.setitem(KIND_HANDLERS, "email_received", fake)

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    await runner._handle_event_with_retry(job_id)

    # Handler called fail_first_n times (crashes) + 1 time (success).
    assert fake.call_count == fail_first_n + 1, (
        f"expected {fail_first_n + 1} handler attempts, got {fake.call_count}"
    )

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job is not None
        assert job.status == "completed" and job.outcome == "success", (
            f"after partial-failure-then-success, expected "
            f"status='completed'/outcome='success', got "
            f"status={job.status!r} outcome={job.outcome!r}"
        )
        assert job.error_message is None, (
            f"error_message should be NULL on success — got "
            f"{job.error_message!r}. The all-retries-exhausted marker "
            f"must NOT be written on a path that ultimately succeeded."
        )
        assert job.completed_at is not None

        trig = await db.get(Trigger, trigger_id)
        assert trig is not None
        # The handler's success result.status == "success" should promote
        # the trigger to 'active' (the "real event delivered" signal).
        assert trig.last_status == "active", (
            f"trigger.last_status should be 'active' after a real "
            f"success, got {trig.last_status!r}"
        )
        assert trig.last_error is None, (
            f"trigger.last_error should be cleared on success — got "
            f"{trig.last_error!r}"
        )
        assert trig.fire_count == 1, (
            f"trigger.fire_count should be 1 (one batch dispatched, "
            f"regardless of retry count), got {trig.fire_count}"
        )


@pytest.mark.asyncio
async def test_finalise_exhausted_is_idempotent_against_race(
    _seed_user_and_trigger, monkeypatch
):
    """If a sibling path (drain loop, inline retry) has already
    terminalised the BuildJob by the time `_finalise_exhausted` runs,
    the helper must NOT overwrite the existing terminal state — that
    would clobber the legitimate outcome with an
    'all_retries_exhausted' marker that doesn't reflect what actually
    happened.

    Simulates the race by pre-flipping the BuildJob to
    status='completed'/outcome='success' before calling
    _finalise_exhausted directly."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    job_id = await _insert_buildjob(trigger_id, user_id)

    # Pre-terminalise: pretend a sibling completed this row first.
    from sqlalchemy import update
    async with async_session_maker() as db:
        await db.execute(
            update(BuildJob)
            .where(BuildJob.id == job_id)
            .values(
                status="completed",
                outcome="success",
                completed_at=datetime.utcnow(),
            )
        )
        await db.commit()

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    # Call directly with a pretend last error — must be a no-op.
    await runner._finalise_exhausted(job_id, "should_not_persist")

    async with async_session_maker() as db:
        job = await db.get(BuildJob, job_id)
        assert job.status == "completed" and job.outcome == "success", (
            f"_finalise_exhausted clobbered a terminal row! "
            f"status={job.status!r} outcome={job.outcome!r}"
        )
        assert job.error_message is None

        trig = await db.get(Trigger, trigger_id)
        assert trig.fire_count == 0, (
            f"fire_count should NOT be bumped when the row was already "
            f"terminal — got {trig.fire_count}"
        )
        assert trig.last_status == "never_fired", (
            f"trigger.last_status should be untouched on the race path, "
            f"got {trig.last_status!r}"
        )
