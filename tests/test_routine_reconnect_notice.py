"""Ticket 2.2 — reconnect notice across all configured channels + dedupe.

Pre-fix bug: `_write_nudge` called `broadcast_routine_message` without
`delivery_channels`, so reauth/failure nudges fanned to website only.
Telegram and WhatsApp were silent on tool failures by construction
(runner.py:738-745 — confirmed in DEFECT_A_REPRO.md §1 / Branch C).

Post-fix contract:
  • `_write_nudge` reads `delivery_channels` from `routine.config_json`
    and passes them through to `broadcast_routine_message`. The
    dispatcher fans the notice to Telegram + WhatsApp + website.
  • A `routine_notification_dedupe` row keyed on
    `(routine_id, kind, scope_date)` rate-limits notices to one per
    kind per local day. A second nudge in the same window is silently
    suppressed.
  • A reconnected Gmail (next fire produces success) does NOT inherit
    the dedupe — the dedupe scope is per-(routine, kind, date), not
    per-routine.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any, Optional

import pytest
from sqlalchemy import select

# Every user in this file lives here, and the dedupe rows the runner writes
# are stamped with THIS zone's date — never the test runner's. Keep the two
# uses tied to one constant so they cannot drift apart again.
_TEST_TZ = "America/Toronto"


def _user_local_today() -> date:
    """The date `_write_nudge` will stamp on the dedupe row.

    NOT `date.today()`. The runner resolves the USER's IANA timezone and
    uses `datetime.now(tz).date()` — "one notice per local day" means the
    user's day, which is the whole point of storing a timezone. `date.today()`
    is the *server's* day, and for a UTC-4 user the two disagree for four
    hours out of every twenty-four.

    That gap is exactly how this file failed CI at 02:21 UTC on 2026-08-04:
    the runner wrote scope_date=2026-08-03 (22:21 in Toronto) and the test
    queried for 2026-08-04, found nothing, and reported it as a violated
    dedupe contract. Nothing was wrong with the runner. A test that is green
    twenty hours a day is not a passing test — it is an unfired one.
    """
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(_TEST_TZ)).date()


async def _make_user(timezone: str = _TEST_TZ) -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@reconnect.test",
                hashed_password="x",
                name="Reconnect Test",
                timezone=timezone,
            )
        )
        await db.commit()
    return user_id


async def _make_routine(
    user_id: str,
    *,
    delivery_channels: Optional[list[str]] = None,
) -> str:
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    cfg: dict[str, Any] = {}
    if delivery_channels is not None:
        cfg["delivery_channels"] = list(delivery_channels)
    async with async_session_maker() as db:
        db.add(
            Routine(
                id=rid,
                user_id=user_id,
                kind="email_briefing",
                enabled=True,
                schedule_cron_local="58 10 * * *",
                config_json=cfg,
                last_status="never_run",
            )
        )
        await db.commit()
    return rid


# ── 1. Nudge fan-out — Telegram + WhatsApp + website ──────────────


@pytest.mark.asyncio
async def test_reauth_nudge_fans_out_to_every_configured_channel(caplog, monkeypatch):
    """`_write_nudge` MUST honour delivery_channels. For a routine
    configured with website+telegram+whatsapp, the nudge writer must be
    called AND `broadcast_routine_message` must receive the full
    delivery_channels list (website + telegram + whatsapp).

    The test env doesn't have a `messages` table (vector schema skipped
    on SQLite), so we monkeypatch `write_routine_message` /
    `broadcast_routine_message` and assert the kwargs the runner passes.
    """
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun

    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id,
        delivery_channels=["website", "telegram", "whatsapp"],
    )
    run_id = str(uuid.uuid4())
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

    captured: dict[str, Any] = {}

    async def _fake_write(db, **kwargs):
        captured["write_called"] = True
        return ("msg-fake", "daychat-fake")

    async def _fake_broadcast(user_id, **kwargs):
        captured["broadcast_called"] = True
        captured["delivery_channels"] = list(kwargs.get("delivery_channels") or [])
        captured["routine_name"] = kwargs.get("routine_name")
        captured["content"] = kwargs.get("content")
        return 0  # legacy int return shape

    # Patch the module-level functions the runner imports.
    monkeypatch.setattr(
        "app.agent.routines.message_writer.write_routine_message", _fake_write
    )
    monkeypatch.setattr(
        "app.agent.routines.message_writer.broadcast_routine_message",
        _fake_broadcast,
    )

    result = RoutineResult(
        status="skipped_reauth",
        error_class="reauth_required",
        error_detail="Gmail token expired",
    )
    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)

    caplog.set_level(logging.INFO)
    await rr._post_terminal(routine, run_id, result)

    # Lock #1: the writer fired (nudge attempted).
    assert captured.get("write_called") is True
    # Lock #2: the broadcaster received the full delivery_channels list.
    assert captured.get("broadcast_called") is True
    assert set(captured["delivery_channels"]) == {"website", "telegram", "whatsapp"}, (
        f"Ticket 2.2 regression: nudge broadcast got "
        f"delivery_channels={captured['delivery_channels']!r}. Expected all "
        "three configured channels. The pre-fix code passed no channels "
        "and silenced Telegram + WhatsApp on tool failures."
    )
    # Lock #3: the content carries the reconnect deep-link.
    assert "toup.ai/agent/integrations" in (captured.get("content") or "")


# ── 2. Dedupe — second nudge in the same local day is suppressed ──


@pytest.mark.asyncio
async def test_reauth_nudge_dedupes_within_same_local_day():
    """Two consecutive skipped_reauth fires on the same local day must
    produce exactly ONE nudge — the second hits the
    `routine_notification_dedupe` UNIQUE and is silently suppressed."""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, RoutineNotificationDedupe

    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id, delivery_channels=["website", "telegram"],
    )

    day_at_start = _user_local_today()

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)

    # First nudge: should claim the dedupe slot.
    run1 = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run1, routine_id=routine_id, user_id=user_id,
                scheduled_for_local_date=date.today(), status="running",
                fire_instant=datetime.utcnow(),
            )
        )
        await db.commit()
    await rr._post_terminal(
        routine, run1,
        RoutineResult(status="skipped_reauth", error_class="reauth_required"),
    )

    # Second nudge: same routine, same local date — should be deduped.
    # We delete the existing run row to bypass the routine_runs UNIQUE
    # (we want to exercise the dedupe table, not the run-row constraint).
    async with async_session_maker() as db:
        from sqlalchemy import delete
        await db.execute(delete(RoutineRun).where(RoutineRun.id == run1))
        await db.commit()

    run2 = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run2, routine_id=routine_id, user_id=user_id,
                scheduled_for_local_date=date.today(), status="running",
                fire_instant=datetime.utcnow(),
            )
        )
        await db.commit()
    await rr._post_terminal(
        routine, run2,
        RoutineResult(status="skipped_reauth", error_class="reauth_required"),
    )

    # If the user's midnight passed while those two nudges were being posted,
    # the premise is gone: the second nudge belongs to a different local day
    # and SHOULD get its own row. Skip rather than assert — a test that fails
    # for one second a day teaches nothing, and silently tolerating two rows
    # would blunt the check for the other 86,399.
    if _user_local_today() != day_at_start:
        pytest.skip(
            f"run straddled midnight in {_TEST_TZ} "
            f"({day_at_start} -> {_user_local_today()}); same-day premise void"
        )

    # Exactly one dedupe row exists for (routine_id, "reauth").
    #
    # Deliberately NOT filtered by date: filtering would let a second row
    # written under a different scope_date pass unnoticed, which is the one
    # way this dedupe can actually break in production. Two nudges landing
    # either side of the user's midnight is a real duplicate, and a
    # date-filtered query is blind to it. Count them all, then pin the date.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineNotificationDedupe).where(
                RoutineNotificationDedupe.routine_id == routine_id,
                RoutineNotificationDedupe.kind == "reauth",
            )
        )
        rows = list(result.scalars().all())
    assert len(rows) == 1, (
        f"Ticket 2.2 dedupe contract violated: found {len(rows)} rows for "
        "(routine, kind=reauth). Expected exactly 1 — the second nudge must "
        "be suppressed by the UNIQUE on (routine_id, kind, scope_date). "
        f"scope_dates seen: {[r.scope_date for r in rows]}"
    )
    assert rows[0].scope_date == day_at_start, (
        f"dedupe row is stamped {rows[0].scope_date}, but the user's local "
        f"date is {day_at_start}. The 24h notice window is scoped to the "
        "USER's day — a server-local scope_date silently shifts the window "
        "for every user outside UTC."
    )


# ── 2b. The dedupe day is the USER's day, at every hour ────────────


def _zone_whose_date_differs_from_the_server() -> tuple[str, date]:
    """An IANA zone whose current date is NOT the server's, plus that date.

    There is always one. Kiritimati is UTC+14 and Niue is UTC-11, so for any
    UTC hour 0-9 Niue has already rolled back a day, and for 10-23 Kiritimati
    has rolled forward. One of the two disagrees with the server at every
    instant of every day.

    That is what makes the test below a real guard rather than a lottery
    ticket. The `America/Toronto` test above only exercises the difference
    between the user's day and the server's for the four hours a day that
    Toronto and UTC disagree — which is precisely why a server-local
    scope_date survived in CI until a run happened to land at 02:21 UTC.
    """
    server_today = date.today()
    for name in ("Pacific/Kiritimati", "Pacific/Niue"):
        from zoneinfo import ZoneInfo

        local = datetime.now(ZoneInfo(name)).date()
        if local != server_today:
            return name, local
    raise AssertionError(  # unreachable — see the docstring
        f"neither +14 nor -11 differs from the server date {server_today}"
    )


@pytest.mark.asyncio
async def test_dedupe_day_is_the_users_day_not_the_servers():
    """The notice window is scoped to the user's local day.

    A user in Kiritimati gets one reconnect nudge per KIRITIMATI day. If the
    runner stamped the server's date instead, that user's 24h window would be
    cut or doubled at an arbitrary hour — and every user outside UTC would be
    affected differently, which is the kind of bug that never reproduces for
    whoever is debugging it.

    This pins the behaviour `_write_nudge` already has (it resolves the user's
    IANA zone and calls `datetime.now(tz).date()`). Written after the
    date-filtered assertion above failed CI at 02:21 UTC and looked, for a
    few minutes, exactly like a broken dedupe.
    """
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, RoutineNotificationDedupe

    tz_name, user_today = _zone_whose_date_differs_from_the_server()
    assert user_today != date.today()  # the whole premise of the test

    user_id = await _make_user(timezone=tz_name)
    routine_id = await _make_routine(user_id, delivery_channels=["website"])

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)

    run_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run_id, routine_id=routine_id, user_id=user_id,
                scheduled_for_local_date=user_today, status="running",
                fire_instant=datetime.utcnow(),
            )
        )
        await db.commit()
    await rr._post_terminal(
        routine, run_id,
        RoutineResult(status="skipped_reauth", error_class="reauth_required"),
    )

    from zoneinfo import ZoneInfo

    if datetime.now(ZoneInfo(tz_name)).date() != user_today:
        pytest.skip(f"run straddled midnight in {tz_name}; premise void")

    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineNotificationDedupe).where(
                RoutineNotificationDedupe.routine_id == routine_id,
            )
        )
        rows = list(result.scalars().all())

    assert len(rows) == 1, f"expected one dedupe row, got {len(rows)}"
    assert rows[0].scope_date == user_today, (
        f"dedupe row for a {tz_name} user is stamped {rows[0].scope_date}; "
        f"that user's local date is {user_today} (the server's is "
        f"{date.today()}). The notice window must follow the user's day."
    )


# ── 3. Different notice kinds are independent ──────────────────────


@pytest.mark.asyncio
async def test_reauth_and_failure_nudges_are_independent_dedupe_scopes():
    """A reauth nudge AND a failure nudge can both land on the same day
    — the dedupe is scoped per-kind. (Unlikely in practice: a routine
    that hit reauth_required usually doesn't then exhaust retries on a
    different code path, but the test pins the dedupe semantics.)"""
    from app.agent.routines.base_handler import RoutineResult
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, RoutineNotificationDedupe

    user_id = await _make_user()
    routine_id = await _make_routine(
        user_id, delivery_channels=["website", "telegram"],
    )

    rr = RoutineRunner()
    async with async_session_maker() as db:
        routine = await db.get(Routine, routine_id)

    # Reauth path.
    run_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run_id, routine_id=routine_id, user_id=user_id,
                scheduled_for_local_date=date.today(), status="running",
                fire_instant=datetime.utcnow(),
            )
        )
        await db.commit()
    await rr._post_terminal(
        routine, run_id,
        RoutineResult(status="skipped_reauth", error_class="reauth_required"),
    )

    # Failure path (separate kind).
    from sqlalchemy import delete
    async with async_session_maker() as db:
        await db.execute(delete(RoutineRun).where(RoutineRun.id == run_id))
        await db.commit()

    run_id2 = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(
            RoutineRun(
                id=run_id2, routine_id=routine_id, user_id=user_id,
                scheduled_for_local_date=date.today(), status="running",
                fire_instant=datetime.utcnow(),
            )
        )
        await db.commit()
    await rr._post_terminal(
        routine, run_id2,
        RoutineResult(status="failed", error_class="provider_down"),
    )

    # Both dedupe rows must exist.
    async with async_session_maker() as db:
        result = await db.execute(
            select(RoutineNotificationDedupe).where(
                RoutineNotificationDedupe.routine_id == routine_id,
            )
        )
        rows = list(result.scalars().all())
    kinds = {r.kind for r in rows}
    assert kinds == {"reauth", "failure"}, (
        f"Expected both kinds in dedupe table; got {kinds!r}. "
        "Dedupe scope is (routine_id, kind, scope_date) — different "
        "kinds must coexist."
    )


# ── 4. Reconnect message text contains the deep link ───────────────


def test_reconnect_message_carries_actionable_link_and_connector_card():
    """The notice body must include:
      • the integrations page URL (Telegram/WhatsApp clickable),
      • the `[[connector_card:gmail|Reconnect]]` marker so the chat UI
        renders an inline one-tap Reconnect card."""
    from app.agent.routines.runner import RoutineRunner

    class _R:
        name = "Morning briefing"
        config_json = None

    reauth_msg = RoutineRunner._build_reconnect_message(_R(), kind="reauth")
    failure_msg = RoutineRunner._build_reconnect_message(_R(), kind="failure")

    assert "toup.ai/agent/integrations" in reauth_msg
    assert "reconnect=gmail" in reauth_msg
    # Inline connector-card marker.
    assert "[[connector_card:gmail|Reconnect]]" in reauth_msg
    assert "[[connector_card:gmail|Reconnect]]" in failure_msg


def test_reconnect_message_respects_custom_connector_from_config():
    """Future routine kinds may target a non-Gmail connector. The
    `config_json.reconnect_connector` field overrides the default."""
    from app.agent.routines.runner import RoutineRunner

    class _R:
        name = "Calendar briefing"
        config_json = {"reconnect_connector": "google_calendar"}

    msg = RoutineRunner._build_reconnect_message(_R(), kind="reauth")
    assert "[[connector_card:google_calendar|Reconnect]]" in msg
    assert "reconnect=google_calendar" in msg
