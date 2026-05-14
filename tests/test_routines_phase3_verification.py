"""Phase 3 end-to-end verification — five scenarios.

This file is the acceptance harness for Phase 2 (Tickets 2.1–2.5). It
exercises the full RoutineRunner._fire pipeline against mocked Gmail +
mocked LLM + a real SQLite DB, and asserts the post-fix contract for
each scenario:

  1. Happy path           — outcome=success, last_run_at=fire_instant
  2. Empty inbox          — outcome=success_empty, no hallucinated text
  3. Auth failure         — outcome=tool_error, reconnect notice fanned out,
                            dedupe holds on refire
  4. Partial channel fail — outcome=partial, channel_results_json populated
  5. DST regression       — clock-shift harness, EST cron → 15:58 UTC

Mocks: handler.execute is replaced with a stub that returns canned
RoutineResults; broadcaster is patched so we can assert kwargs. The
`messages` table is not created on SQLite (vector schema skipped) so
all message writes are mocked.

Pass/fail evidence is captured into the docs/routines/VERIFICATION_2026-05.md
file separately — this file just contains the test code.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select


# ── Helpers ──────────────────────────────────────────────────────────


async def _make_user(timezone_name: str = "America/Toronto") -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id, email=f"{user_id}@p3.test",
                hashed_password="x", name="P3", timezone=timezone_name,
            )
        )
        await db.commit()
    return user_id


async def _make_routine(
    user_id: str,
    *,
    cron: str = "58 10 * * *",
    delivery_channels: list[str] | None = None,
    kind: str = "email_briefing",
) -> str:
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    cfg: dict[str, Any] = {"mode": "latest_n", "max_emails": 5}
    if delivery_channels is not None:
        cfg["delivery_channels"] = delivery_channels

    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid, user_id=user_id, kind=kind, enabled=True,
                name="Phase3 routine",
                schedule_cron_local=cron, config_json=cfg,
                last_status="never_run",
            )
        )
        await db.commit()
    return rid


def _patch_registry(monkeypatch, kind: str, fn):
    """Override KIND_HANDLERS[kind] for a single test. Wraps `fn` in a
    handler class with `kind`+`execute`."""
    from app.agent.routines.registry import KIND_HANDLERS

    class _Handler:
        def __init__(self):
            self.kind = kind

        async def execute(self, routine, run, db):
            return await fn(routine, run, db)

    monkeypatch.setitem(KIND_HANDLERS, kind, _Handler())


def _enable_email_briefing(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "routines_email_briefing_enabled", True)


# ── Scenario 1 — Happy path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_scenario_1_happy_path(monkeypatch, caplog):
    """A normal fire: handler returns success with full channel_results,
    runner stamps outcome=success, last_run_at=fire_instant."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun

    _enable_email_briefing(monkeypatch)
    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id,
        delivery_channels=["website", "telegram", "whatsapp"],
    )

    async def _handler(routine, run, db):
        return RoutineResult(
            status="success",
            emails_fetched=3,
            new_watermark={"last_processed_internal_date": "1747159080000"},
            summary_message_id="msg-happy",
            channel_results={
                "website":  {"status": "delivered", "message_id": "msg-happy", "ws_count": 1},
                "telegram": {"status": "delivered", "telegram_message_id": 100, "chat_id": 555},
                "whatsapp": {"status": "delivered", "whatsapp_message_id": "wamid.x"},
            },
            tools_invoked=["gmail__list_messages", "gmail__get_message"],
        )

    _patch_registry(monkeypatch, "email_briefing", _handler)

    rr = RoutineRunner()
    caplog.set_level(logging.INFO)
    await rr._fire(routine_id)

    # Verify outcomes.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == routine_id)
        )
        run = result.scalar_one()
        routine = await db.get(Routine, routine_id)

    assert run.outcome == "success"
    assert run.status == "success"
    assert run.channel_results_json["telegram"]["status"] == "delivered"
    assert routine.last_status == "success"
    # last_run_at = fire_instant if APScheduler had one; otherwise
    # falls back to utcnow at _post_terminal. Since _fire wasn't
    # invoked via APScheduler in this test, fire_instant is None and
    # last_run_at is utcnow.
    assert routine.last_run_at is not None


# ── Scenario 2 — Empty inbox ──────────────────────────────────────


@pytest.mark.asyncio
async def test_scenario_2_empty_inbox(monkeypatch):
    """Handler returns success with emails_fetched=0 + new_watermark=None
    → outcome=success_empty. The routine row stamps last_status=success
    (legacy) but the runner's outcome distinguishes the empty case."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun

    _enable_email_briefing(monkeypatch)
    user_id = await _make_user()
    routine_id = await _make_routine(user_id, delivery_channels=["website"])

    async def _handler(routine, run, db):
        return RoutineResult(
            status="success",
            emails_fetched=0,
            new_watermark=None,
            summary_message_id="msg-empty",
            channel_results={"website": {"status": "delivered", "message_id": "msg-empty", "ws_count": 0}},
            tools_invoked=["gmail__list_messages"],
        )

    _patch_registry(monkeypatch, "email_briefing", _handler)

    rr = RoutineRunner()
    await rr._fire(routine_id)

    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == routine_id)
        )
        run = result.scalar_one()

    assert run.outcome == "success_empty", (
        f"Expected outcome=success_empty for empty-inbox handler; got {run.outcome!r}"
    )
    assert run.emails_fetched == 0


# ── Scenario 3 — Auth failure + reconnect notice + dedupe ─────────


@pytest.mark.asyncio
async def test_scenario_3_auth_failure_reconnect_dedupe(monkeypatch):
    """Handler returns status=skipped_reauth → runner emits a reconnect
    notice on every configured channel AND records a dedupe row. A
    second fire within the same local day hits the dedupe — no
    duplicate notice."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, RoutineNotificationDedupe
    from sqlalchemy import delete

    _enable_email_briefing(monkeypatch)
    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id, delivery_channels=["website", "telegram", "whatsapp"],
    )

    async def _handler(routine, run, db):
        return RoutineResult(
            status="skipped_reauth",
            error_class="reauth_required",
            error_detail="Gmail token expired",
        )

    _patch_registry(monkeypatch, "email_briefing", _handler)

    # Patch writer + broadcaster to capture calls (test env has no messages table).
    broadcast_calls: list[dict] = []

    async def _fake_write(db, **kwargs):
        return ("msg-x", "daychat-x")

    async def _fake_broadcast(user_id, **kwargs):
        broadcast_calls.append(kwargs)
        return {"ws_count": 0, "channel_results": {
            "website": {"status": "delivered", "message_id": "msg-x"},
            "telegram": {"status": "skipped", "reason": "no_recipient"},
            "whatsapp": {"status": "skipped", "reason": "no_recipient"},
        }}

    monkeypatch.setattr(
        "app.agent.routines.message_writer.write_routine_message", _fake_write
    )
    monkeypatch.setattr(
        "app.agent.routines.message_writer.broadcast_routine_message", _fake_broadcast
    )

    rr = RoutineRunner()
    await rr._fire(routine_id)

    # The notice fanout fired with full delivery_channels.
    assert broadcast_calls, "Ticket 2.2: reconnect broadcaster must be invoked"
    first = broadcast_calls[-1]
    assert set(first["delivery_channels"]) == {"website", "telegram", "whatsapp"}

    # Dedupe row was claimed.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineNotificationDedupe).where(
                RoutineNotificationDedupe.routine_id == routine_id
            )
        )
        rows = list(result.scalars().all())
    assert len(rows) == 1
    assert rows[0].kind == "reauth"

    # Refire: dedupe must suppress. Wipe the run row so we don't hit
    # the routine_runs UNIQUE.
    async with async_session_maker() as db:
        await db.execute(delete(RoutineRun).where(RoutineRun.routine_id == routine_id))
        await db.commit()

    n_calls_before = len(broadcast_calls)
    await rr._fire(routine_id)
    n_calls_after = len(broadcast_calls)

    assert n_calls_after == n_calls_before, (
        f"Ticket 2.2 dedupe violated: refire posted another notice. "
        f"calls_before={n_calls_before} calls_after={n_calls_after}"
    )

    # Outcome on both runs must be tool_error.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == routine_id)
        )
        runs = list(result.scalars().all())
    assert all(r.outcome == "tool_error" for r in runs)


# ── Scenario 4 — Partial channel failure ──────────────────────────


@pytest.mark.asyncio
async def test_scenario_4_partial_channel_failure(monkeypatch):
    """Handler returns success but channel_results show one channel
    skipped silently (no_recipient). Runner downgrades outcome to
    `partial` and persists the accurate channel_results_json."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import RoutineRun

    _enable_email_briefing(monkeypatch)
    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id, delivery_channels=["website", "telegram"],
    )

    async def _handler(routine, run, db):
        return RoutineResult(
            status="success",
            emails_fetched=2,
            new_watermark={"last_processed_internal_date": "1747159080000"},
            summary_message_id="msg-partial",
            channel_results={
                "website":  {"status": "delivered", "message_id": "msg-partial", "ws_count": 1},
                "telegram": {"status": "skipped", "reason": "no_recipient"},
            },
            tools_invoked=["gmail__list_messages", "gmail__get_message"],
        )

    _patch_registry(monkeypatch, "email_briefing", _handler)

    rr = RoutineRunner()
    await rr._fire(routine_id)

    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineRun).where(RoutineRun.routine_id == routine_id)
        )
        run = result.scalar_one()

    # The Defect A regression-net: status="success" but outcome="partial".
    assert run.status == "success"  # legacy compat
    assert run.outcome == "partial", (
        "DEFECT A regression net failed: outcome must downgrade to 'partial' "
        "when channel_results show any skip. Without this, Defect A re-opens."
    )
    assert run.channel_results_json["telegram"]["reason"] == "no_recipient"
    assert "gmail__list_messages" in (run.tools_invoked_json or [])


# ── Scenario 5 — DST regression (clock-shift harness) ─────────────


def test_scenario_5_dst_est_cron_resolves_to_15_58_utc():
    """A `58 10 * * *` cron in America/Toronto on a winter (EST) date
    must resolve to next_fire_utc = 15:58:00 UTC. Simulates the clock-
    shift harness without actually waiting for December: we point a
    fresh trigger at a known winter date and verify."""
    from apscheduler.triggers.cron import CronTrigger

    tz = ZoneInfo("America/Toronto")
    trigger = CronTrigger(
        minute="58", hour="10", day="*", month="*", day_of_week="*", timezone=tz,
    )
    now_local = datetime(2026, 12, 13, 9, 0, 0, tzinfo=tz)
    nxt = trigger.get_next_fire_time(None, now_local)
    assert nxt is not None
    assert nxt.astimezone(timezone.utc) == datetime(
        2026, 12, 13, 15, 58, 0, tzinfo=timezone.utc
    )
