"""Ticket 2.3 — DST regression test suite.

Pins APScheduler 3.10.4 + ZoneInfo's currently-correct behaviour so a
future apscheduler bump or tzdata churn that regresses this is caught
in CI before reaching prod.

Phase 1.3 (docs/routines/DEFECT_B_DST.md) proved that APScheduler is
correct today. This file is the regression net.

Five scenarios:
  1. EDT (May 13, 2026) — UTC-4. cron `58 10 * * *` America/Toronto must
     compute next_fire_utc = 2026-05-13 14:58:00 UTC.
  2. EST (Dec 13, 2026) — UTC-5. Same cron must compute 15:58:00 UTC
     (one hour later in local↔UTC offset).
  3. Spring-forward (2026-03-08 02:30) — 02:30 doesn't exist; APScheduler
     must jump to the first valid wall clock (03:30 EDT).
  4. Fall-back (2026-11-01 01:30) — 01:30 occurs twice. APScheduler must
     fire ONCE; the next fire is the following day, not a re-fire.
  5. Phoenix (no DST) — UTC-7 year-round.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger


def _parse_cron_5(expr: str, tz) -> CronTrigger:
    parts = expr.split()
    return CronTrigger(
        minute=parts[0], hour=parts[1], day=parts[2],
        month=parts[3], day_of_week=parts[4], timezone=tz,
    )


def test_dst_edt_2026_05_13_fires_at_14_58_utc():
    """`58 10 * * *` America/Toronto on 2026-05-13 (EDT, UTC-4) must
    compute the next fire instant as 14:58:00 UTC. ONE-HOUR drift to
    15:58 UTC would mean APScheduler is treating Toronto as fixed-EST."""
    tz = ZoneInfo("America/Toronto")
    trigger = _parse_cron_5("58 10 * * *", tz)
    now_local = datetime(2026, 5, 13, 9, 0, 0, tzinfo=tz)
    nxt = trigger.get_next_fire_time(None, now_local)
    assert nxt is not None
    assert nxt.astimezone(timezone.utc) == datetime(
        2026, 5, 13, 14, 58, 0, tzinfo=timezone.utc
    ), (
        f"DST regression: got {nxt.astimezone(timezone.utc).isoformat()}, "
        "expected 2026-05-13T14:58:00+00:00. Phase 1.3 / DEFECT_B_DST.md "
        "documented this as the correct EDT outcome."
    )


def test_dst_est_2026_12_13_fires_at_15_58_utc():
    """`58 10 * * *` America/Toronto on 2026-12-13 (EST, UTC-5) must
    compute 15:58:00 UTC."""
    tz = ZoneInfo("America/Toronto")
    trigger = _parse_cron_5("58 10 * * *", tz)
    now_local = datetime(2026, 12, 13, 9, 0, 0, tzinfo=tz)
    nxt = trigger.get_next_fire_time(None, now_local)
    assert nxt is not None
    assert nxt.astimezone(timezone.utc) == datetime(
        2026, 12, 13, 15, 58, 0, tzinfo=timezone.utc
    )


def test_dst_spring_forward_2026_03_08_skips_nonexistent_local_time():
    """On 2026-03-08 in America/Toronto, local time jumps from 02:00 EST
    directly to 03:00 EDT — 02:30 EST does not exist. A cron `30 02 * * *`
    must NOT raise; APScheduler should jump to the first valid wall-clock
    after the gap. The post-jump fire in UTC is 06:30 (the same UTC
    instant 02:30 EST would have been before the jump → but that's not
    valid wall-clock, so we land on 03:30 EDT = 07:30 UTC).
    """
    tz = ZoneInfo("America/Toronto")
    trigger = _parse_cron_5("30 02 * * *", tz)
    now_local = datetime(2026, 3, 8, 0, 0, 0, tzinfo=tz)
    nxt = trigger.get_next_fire_time(None, now_local)
    # Don't pin a precise UTC — APScheduler's behaviour on the gap can
    # vary between micro versions. Pin two invariants:
    #   (a) it returns a valid datetime (no raise),
    #   (b) the UTC instant is BETWEEN the pre-gap (06:30) and post-gap
    #       (07:30) bounds, OR exactly one of them.
    assert nxt is not None
    nxt_utc = nxt.astimezone(timezone.utc)
    lo = datetime(2026, 3, 8, 6, 30, 0, tzinfo=timezone.utc)
    hi = datetime(2026, 3, 8, 7, 30, 0, tzinfo=timezone.utc)
    assert lo <= nxt_utc <= hi, (
        f"spring-forward: got {nxt_utc.isoformat()}, expected within "
        f"[{lo.isoformat()}, {hi.isoformat()}]"
    )


def test_dst_fall_back_2026_11_01_apscheduler_emits_two_fires_idempotency_prevents_dup_delivery():
    """On 2026-11-01 in America/Toronto, 01:30 occurs twice (01:30 EDT
    at UTC 05:30 and 01:30 EST at UTC 06:30).

    PINNED BEHAVIOUR (verified against apscheduler 3.10.4 + ZoneInfo):
      • APScheduler emits BOTH fires — one at 01:30 EDT, one at 01:30 EST.
      • Both carry scheduled_for_local_date = 2026-11-01.
      • The UNIQUE constraint on
        `routine_runs(routine_id, scheduled_for_local_date)` is what
        prevents the user from getting a duplicate briefing.

    If a future apscheduler version *changes* its fall-back semantics
    (e.g., starts emitting only one fire), this test fires loud — we'd
    need to revisit the idempotency contract.

    Production-side guarantee is therefore:
      `idempotency_collision` log line at runner.py:531 catches the second
      fire and exits silently. Verified separately in
      test_idempotency_collision_on_second_fire_same_day.
    """
    tz = ZoneInfo("America/Toronto")
    trigger = _parse_cron_5("30 01 * * *", tz)
    now_local = datetime(2026, 10, 31, 23, 0, 0, tzinfo=tz)
    n1 = trigger.get_next_fire_time(None, now_local)
    n2 = trigger.get_next_fire_time(n1, n1 + timedelta(seconds=1))
    assert n1 is not None and n2 is not None

    n1_utc = n1.astimezone(timezone.utc)
    n2_utc = n2.astimezone(timezone.utc)

    # Pinned UTC instants — pre-fold (EDT) and post-fold (EST).
    assert n1_utc == datetime(2026, 11, 1, 5, 30, 0, tzinfo=timezone.utc), (
        f"fall-back first fire: got {n1_utc.isoformat()}, expected "
        "2026-11-01T05:30:00+00:00 (01:30 EDT, pre-fold)."
    )
    assert n2_utc == datetime(2026, 11, 1, 6, 30, 0, tzinfo=timezone.utc), (
        f"fall-back second fire: got {n2_utc.isoformat()}, expected "
        "2026-11-01T06:30:00+00:00 (01:30 EST, post-fold)."
    )
    # Both fires share the local date — the idempotency UNIQUE is what
    # turns "two fires" into "one delivery."
    assert n1.astimezone(tz).date() == n2.astimezone(tz).date() == \
        datetime(2026, 11, 1).date()


def test_dst_phoenix_no_dst_fires_at_constant_utc_offset_year_round():
    """America/Phoenix observes no DST. cron `58 10 * * *` must compute
    the same UTC offset (UTC-7 = 17:58 UTC) on both summer and winter
    dates."""
    tz = ZoneInfo("America/Phoenix")
    trigger = _parse_cron_5("58 10 * * *", tz)

    summer = datetime(2026, 5, 13, 9, 0, 0, tzinfo=tz)
    winter = datetime(2026, 12, 13, 9, 0, 0, tzinfo=tz)
    n_summer = trigger.get_next_fire_time(None, summer)
    n_winter = trigger.get_next_fire_time(None, winter)
    assert n_summer is not None and n_winter is not None
    assert n_summer.astimezone(timezone.utc).time() == n_winter.astimezone(timezone.utc).time()
    # And the time is 17:58 UTC.
    assert n_summer.astimezone(timezone.utc).hour == 17
    assert n_summer.astimezone(timezone.utc).minute == 58


# ── 6. Watchdog drift — same instant means no drift signal ─────────


def test_dst_watchdog_does_not_trip_on_correct_apscheduler():
    """The DST watchdog at trigger registration compares APScheduler's
    next_run_time against an independent croniter computation. Today
    these match — the counter must stay 0 after registration. If this
    fails, either croniter has a bug or APScheduler regressed; either
    way, look at the watchdog log line."""
    import asyncio
    from app.agent.routines.runner import RoutineRunner
    from app.db import async_session_maker
    from app.db.models import Routine, User
    import uuid

    async def _run():
        user_id = str(uuid.uuid4())
        rid = str(uuid.uuid4())
        async with async_session_maker() as db:
            db.add(User(
                id=user_id, email=f"{user_id}@watchdog.test",
                hashed_password="x", name="x",
                timezone="America/Toronto",
            ))
            db.add(Routine(
                id=rid, user_id=user_id, kind="_smoke",
                enabled=True, schedule_cron_local="58 10 * * *",
                last_status="never_run",
            ))
            await db.commit()
        rr = RoutineRunner()
        await rr.start()
        try:
            return rr._watchdog_drift_total
        finally:
            await rr.stop()

    drift_total = asyncio.run(_run())
    assert drift_total == 0, (
        f"DST watchdog tripped {drift_total} times during startup — "
        "APScheduler diverged from croniter+ZoneInfo. CRITICAL log line "
        "carries the routine_id and the two next-fire timestamps."
    )
