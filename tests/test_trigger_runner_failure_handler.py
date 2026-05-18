"""Trigger runner — post-loop failure-handler fix (D7).

Pinned bug:

  `_handle_event_with_retry` in `backend/app/agent/triggers/runner.py`
  loops over `DEFAULT_RETRY_DELAYS = (5.0, 30.0, 120.0)` calling
  `_dispatch_one` per attempt. Pre-fix, when all three attempts crashed
  the loop fell through with NO terminal write — the TriggerEvent row
  stayed in `status='running'` until the 10-minute orphan sweep flipped
  it to `failed` with `error_class='agent_restarted'` and NULL
  `error_detail`. The user's live "Summarize every new Gmail" trigger
  accumulated 11 such rows.

  Fix: after the retry loop falls through, call `_finalise_exhausted`
  which UPDATEs the row to:
    status='failed'
    error_class='all_retries_exhausted'
    error_detail=repr(last_exception)
    finished_at=NOW()
  …and stamps the parent Trigger with last_status='failed', last_error,
  fire_count += 1, last_fired_at.

These two parametrised tests pin both halves of the spec:

  1. All N attempts crash → terminal state matches the spec exactly.
  2. Attempts 1..N-1 crash, attempt N succeeds → row ends in
     status='success', NO error fields touched, retry count matches.

The tests build a TriggerRunner with `retry_delays=(0.01, 0.01, 0.01)`
so they run in milliseconds, and inject a fake handler into
KIND_HANDLERS that's configurable per test.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-runner-failure-handler")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000e0")
os.environ.setdefault("TRIGGERS_EMAIL_ENABLED", "true")


CONTAINER_USER_ID = "00000000-0000-0000-0000-0000000000e0"
TRIGGER_ID = "00000000-0000-0000-0000-0000000ff000"


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
        "error_class='all_retries_exhausted' literal missing from "
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
    `_reset_database` fixture from conftest gives us a clean schema."""
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


async def _insert_event(trigger_id: str, user_id: str) -> str:
    """Insert a fresh TriggerEvent in status='queued'."""
    from app.db.database import async_session_maker
    from app.db.models import TriggerEvent

    eid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(TriggerEvent(
            id=eid,
            trigger_id=trigger_id,
            user_id=user_id,
            event_dedupe_id=f"test:{eid[:8]}",
            status="queued",
        ))
        await db.commit()
    return eid


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

      - flip TriggerEvent.status from 'running' to 'failed'
      - set error_class='all_retries_exhausted'
      - set error_detail to repr(last exception) (the actual message,
        not NULL — that's the regression we're pinning)
      - stamp finished_at
      - bump Trigger.fire_count by 1
      - set Trigger.last_status='failed' + last_error + last_fired_at

    Pre-fix, ALL of these were broken: the row stayed in 'running' with
    NULL error fields and the parent trigger was never stamped, so
    `fire_count` showed the count of dispatch attempts but `last_status`
    was whatever the previous fire left it (often 'active' or
    'never_fired')."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    event_id = await _insert_event(trigger_id, user_id)

    fake = _FakeHandler(fail_first_n=None, failure_exc=exc)
    monkeypatch.setitem(KIND_HANDLERS, "email_received", fake)

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    await runner._handle_event_with_retry(event_id)

    # Handler called once per retry attempt.
    assert fake.call_count == 3, (
        f"expected 3 handler attempts, got {fake.call_count}"
    )

    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        assert ev is not None, "event row vanished"
        assert ev.status == "failed", (
            f"expected status='failed' after retries exhausted, "
            f"got {ev.status!r} — this is the regression."
        )
        assert ev.error_class == "all_retries_exhausted", (
            f"expected error_class='all_retries_exhausted', "
            f"got {ev.error_class!r}"
        )
        assert ev.error_detail is not None, (
            "error_detail is NULL — pre-fix symptom. The repr of the "
            "last exception must be persisted so operators can diagnose."
        )
        # The error_detail should contain something recognisable from
        # the exception. Each parametrised case has a unique substring.
        assert repr(exc).split("(", 1)[0] in ev.error_detail or str(exc) in ev.error_detail, (
            f"error_detail {ev.error_detail!r} should reference exception {exc!r}"
        )
        assert ev.finished_at is not None, "finished_at must be stamped"

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
    succeeds, the event row must reach status='success' with NO
    error_class / error_detail / 'all_retries_exhausted' marker. The
    parent trigger should be promoted to last_status='active' the same
    way a normal first-try success would be.

    This pins the inverse regression: don't write the 'all retries
    exhausted' marker on a path that ultimately succeeded."""
    from app.agent.triggers.registry import KIND_HANDLERS
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    event_id = await _insert_event(trigger_id, user_id)

    fake = _FakeHandler(
        fail_first_n=fail_first_n,
        failure_exc=RuntimeError("transient"),
    )
    monkeypatch.setitem(KIND_HANDLERS, "email_received", fake)

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    await runner._handle_event_with_retry(event_id)

    # Handler called fail_first_n times (crashes) + 1 time (success).
    assert fake.call_count == fail_first_n + 1, (
        f"expected {fail_first_n + 1} handler attempts, got {fake.call_count}"
    )

    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        assert ev is not None
        assert ev.status == "success", (
            f"after partial-failure-then-success, expected status='success', "
            f"got {ev.status!r}"
        )
        assert ev.error_class is None, (
            f"error_class should be NULL on success — got "
            f"{ev.error_class!r}. The all-retries-exhausted marker "
            f"must NOT be written on a path that ultimately succeeded."
        )
        assert ev.error_detail is None, (
            f"error_detail should be NULL on success — got {ev.error_detail!r}"
        )
        assert ev.finished_at is not None

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
    terminalised the row by the time `_finalise_exhausted` runs, the
    helper must NOT overwrite the existing terminal state — that would
    clobber the legitimate outcome with a 'all_retries_exhausted' that
    doesn't reflect what actually happened.

    Simulates the race by pre-flipping the row to status='success'
    before calling _finalise_exhausted directly."""
    from app.agent.triggers.runner import TriggerRunner
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    trigger_id = _seed_user_and_trigger["trigger_id"]
    user_id = _seed_user_and_trigger["user_id"]
    event_id = await _insert_event(trigger_id, user_id)

    # Pre-terminalise: pretend a sibling completed this row first.
    from sqlalchemy import update
    from datetime import datetime
    async with async_session_maker() as db:
        await db.execute(
            update(TriggerEvent)
            .where(TriggerEvent.id == event_id)
            .values(
                status="success",
                finished_at=datetime.utcnow(),
            )
        )
        await db.commit()

    runner = TriggerRunner(
        session_maker=async_session_maker,
        retry_delays=(0.001, 0.001, 0.001),
    )
    # Call directly with a pretend last error — must be a no-op.
    await runner._finalise_exhausted(event_id, "should_not_persist")

    async with async_session_maker() as db:
        ev = await db.get(TriggerEvent, event_id)
        assert ev.status == "success", (
            f"_finalise_exhausted clobbered a terminal row! "
            f"status={ev.status!r}"
        )
        assert ev.error_class is None
        assert ev.error_detail is None

        trig = await db.get(Trigger, trigger_id)
        assert trig.fire_count == 0, (
            f"fire_count should NOT be bumped when the row was already "
            f"terminal — got {trig.fire_count}"
        )
        assert trig.last_status == "never_fired", (
            f"trigger.last_status should be untouched on the race path, "
            f"got {trig.last_status!r}"
        )
