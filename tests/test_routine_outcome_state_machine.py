"""Ticket 2.1 — outcome state machine acceptance tests.

Four lock points, one per outcome state the runner can produce:

  1. tool_error  — handler returns skipped_reauth (Gmail 401 / reauth).
  2. success_empty — handler returns success + emails_fetched=0.
  3. partial — handler returns success + channel_results show a skip.
  4. success — handler returns success + every channel confirmed delivered.

Each test exercises `_post_terminal` end-to-end and asserts the new
`routine_runs.outcome` column reflects the expected state. The legacy
`status` column is also pinned so we don't accidentally regress
existing dashboard queries.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

import pytest
from sqlalchemy import select


# ── Helpers ──────────────────────────────────────────────────────────


async def _make_user(timezone: str = "America/Toronto") -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@outcome.test",
                hashed_password="x",
                name="Outcome Test",
                timezone=timezone,
            )
        )
        await db.commit()
    return user_id


async def _make_routine(user_id: str, *, kind: str = "email_briefing") -> str:
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind=kind,
                enabled=True,
                schedule_cron_local="58 10 * * *",
                config_json={"delivery_channels": ["website", "telegram"]},
                last_status="never_run",
            )
        )
        await db.commit()
    return rid


def _make_run_id() -> str:
    return str(uuid.uuid4())


async def _seed_run_row(routine_id: str, user_id: str) -> str:
    from app.db import async_session_maker
    from app.db.models import RoutineRun

    run_id = _make_run_id()
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run_id,
                routine_id=routine_id,
                user_id=user_id,
                scheduled_for_local_date=date.today(),
                status="running",
                fire_instant=datetime(2026, 5, 13, 14, 58, 0),
            )
        )
        await db.commit()
    return run_id


async def _fetch_run(run_id: str):
    from app.db import async_session_maker
    from app.db.models import RoutineRun

    async with async_session_maker() as db:
        result = await db.execute(select(RoutineRun).where(RoutineRun.id == run_id))
        return result.scalar_one()


# ── Outcome 1 — tool_error (Gmail reauth required) ──────────────────


@pytest.mark.asyncio
async def test_outcome_tool_error_when_handler_returns_skipped_reauth():
    """Gmail handler returns status='skipped_reauth' (the connector
    returned reauth_required) → outcome must be 'tool_error'.

    The legacy `status` column keeps the prior value for backward compat,
    but `outcome` is the new authoritative state."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)

    result = RoutineResult(
        status="skipped_reauth",
        error_class="reauth_required",
        error_detail="Gmail token expired (401)",
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    run = await _fetch_run(run_id)
    assert run.outcome == "tool_error"
    assert run.status == "skipped_reauth"  # legacy column preserved
    assert run.error_json is not None
    assert run.error_json["class"] == "reauth_required"


# ── Outcome 2 — success_empty (no new emails) ───────────────────────


@pytest.mark.asyncio
async def test_outcome_success_empty_when_no_emails_and_no_watermark():
    """Handler returns success with emails_fetched=0 AND new_watermark=None
    (the `_post_empty` path) → outcome must be 'success_empty'.

    This is distinct from 'success' so dashboards can show "ran cleanly,
    nothing to deliver" vs "delivered N emails"."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)

    result = RoutineResult(
        status="success",
        emails_fetched=0,
        new_watermark=None,
        summary_message_id="msg-empty",
        channel_results={
            "website": {"status": "delivered", "message_id": "msg-empty", "ws_count": 1},
            "telegram": {"status": "delivered", "telegram_message_id": 100},
        },
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    run = await _fetch_run(run_id)
    assert run.outcome == "success_empty"
    assert run.status == "success"
    assert run.error_json is None


# ── Outcome 3 — partial (channel skipped silently) ──────────────────


@pytest.mark.asyncio
async def test_outcome_partial_when_any_channel_skipped():
    """Handler returns success, summary written, but channel_results show
    at least one channel != 'delivered' → outcome must be 'partial'.

    THIS is the Defect A symptom: handler claims success but the user
    didn't actually receive on every configured channel."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)

    result = RoutineResult(
        status="success",
        emails_fetched=5,
        new_watermark={"last_processed_internal_date": "1747159080000"},
        summary_message_id="msg-success",
        channel_results={
            "website": {"status": "delivered", "message_id": "msg-success", "ws_count": 2},
            "telegram": {"status": "skipped", "reason": "no_recipient"},
        },
        tools_invoked=["gmail__list_messages", "gmail__get_message"],
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    run = await _fetch_run(run_id)
    assert run.outcome == "partial", (
        f"Defect A regression: outcome={run.outcome!r}, expected 'partial'. "
        "A channel that skipped silently MUST downgrade outcome to partial."
    )
    assert run.status == "success"  # legacy column unchanged
    assert run.channel_results_json is not None
    assert run.channel_results_json["telegram"]["reason"] == "no_recipient"
    assert "gmail__list_messages" in (run.tools_invoked_json or [])


# ── Outcome 4 — success (every channel confirmed) ───────────────────


@pytest.mark.asyncio
async def test_outcome_success_when_every_channel_delivered():
    """Handler returns success + every channel_results entry has
    status='delivered' → outcome must be 'success'."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)

    result = RoutineResult(
        status="success",
        emails_fetched=5,
        new_watermark={"last_processed_internal_date": "1747159080000"},
        summary_message_id="msg-success",
        channel_results={
            "website": {"status": "delivered", "message_id": "msg-success", "ws_count": 2},
            "telegram": {"status": "delivered", "telegram_message_id": 12345},
        },
        tools_invoked=["gmail__list_messages", "gmail__get_message"],
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    run = await _fetch_run(run_id)
    assert run.outcome == "success"
    assert run.status == "success"
    assert run.channel_results_json["telegram"]["status"] == "delivered"


# ── Outcome 5 — failure (retries exhausted) ─────────────────────────


@pytest.mark.asyncio
async def test_outcome_failure_when_handler_returns_failed():
    """Handler returns status='failed' (rate_limited, provider_down,
    handler exception that exhausted retries) → outcome='failure'."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)

    result = RoutineResult(
        status="failed",
        error_class="provider_down",
        error_detail="Gmail returned 503 after 3 retries",
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    run = await _fetch_run(run_id)
    assert run.outcome == "failure"
    assert run.status == "failed"
    assert run.error_json["class"] == "provider_down"


# ── fire_instant lock — last_run_at uses it, not utcnow ─────────────


@pytest.mark.asyncio
async def test_last_run_at_uses_fire_instant_not_utcnow():
    """Ticket 2.3 contract: `routines.last_run_at` is stamped from
    `routine_runs.fire_instant` (the scheduled APScheduler fire time),
    NOT from `datetime.utcnow()` at end of _post_terminal.

    A slow handler that finishes 10 minutes after the trigger MUST still
    report last_run_at = trigger fire instant."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user()
    routine_id = await _make_routine(user_id)
    run_id = await _seed_run_row(routine_id, user_id)
    # The seeded run row carries fire_instant = 2026-05-13 14:58:00 UTC.
    # Test: even though _post_terminal runs at datetime.utcnow() (which
    # is "now" — many seconds/minutes later), last_run_at must reflect
    # the trigger instant.

    result = RoutineResult(
        status="success",
        emails_fetched=3,
        new_watermark={"last_processed_internal_date": "1747159080000"},
        summary_message_id="msg-1",
        channel_results={"website": {"status": "delivered", "message_id": "msg-1"}},
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)
    await rr._post_terminal(routine, run_id, result)

    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)

    assert routine.last_run_at == datetime(2026, 5, 13, 14, 58, 0), (
        f"Ticket 2.3 regression: last_run_at={routine.last_run_at} "
        "must equal fire_instant=2026-05-13 14:58:00. The post-terminal "
        "stamp must come from APScheduler's scheduled fire time, NOT "
        "from datetime.utcnow() at the end of _post_terminal."
    )
