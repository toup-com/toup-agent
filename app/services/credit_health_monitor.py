"""Credit-system invariant monitor — alarms on the ways the meter lies.

Why this exists
===============
The 2026-08-03 incident was not subtle in the data: the ledger held 274 rows
saying ``{"denied": true}`` next to a streamed answer, two accounts had 48 and
117 duplicate ``plan_grant`` rows, and $17.17 of provider spend had been handed
out free. Every one of those was queryable the whole time. Nothing looked, so
nothing knew, and the first report came from a founder noticing that a number
on a phone screen had not moved in four minutes.

Correctness fixes stop the specific bugs. This stops the *class*: it asserts,
hourly, the handful of invariants that must hold for the meter to mean
anything, and pages when one breaks.

The invariants
==============
1. **Nothing is served unbilled.** A ledger row with ``denied=true`` means the
   charge was refused. Provider cost attached to such a row is work we gave
   away. Steady state is zero.
2. **The one-time grant fires once.** More than one ``plan_grant`` per bucket
   per user means a re-grant loop, which silently resets wallets and erases
   spend — and is a free-credit farm.
3. **Revenue covers cost.** Credits charged should track the underlying
   provider cents they represent (1 credit ≈ 1¢ by design). A ratio far below
   1 means the meter is undercounting, not that pricing is generous.
4. **The meter is moving.** LLM proxy events with no corresponding charge rows
   means the charge path is broken again, whatever the reason.

Each is a cheap aggregate over an indexed window. Read-only; it never writes to
the credit tables.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from sqlalchemy import func, or_, select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    BUCKET_MESSAGE, CreditLedger, LEDGER_CHAT_MESSAGE, LEDGER_PLAN_GRANT,
    LLMProxyEvent,
)
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)


def _cfg(name: str, default):
    return getattr(settings, name, default)


async def check_credit_health() -> dict:
    """Run every invariant once. Returns the readings + which alarms fired."""
    window_h = int(_cfg("credit_health_window_h", 24))
    since = datetime.utcnow() - timedelta(hours=window_h)
    fired: list[str] = []
    readings: dict = {"window_h": window_h}

    async with async_session_maker() as db:
        # Expressed with SQLAlchemy rather than raw SQL so the same predicates
        # run on the SQLite test DB as on Postgres — a monitor nobody can test
        # is how you end up with an alarm that was silently broken.
        _denied = CreditLedger.metadata_json["denied"].as_boolean().is_(True)
        _is_admin = CreditLedger.metadata_json["admin_unlimited"].as_boolean().is_(True)

        # ── 1. Served but not billed ──────────────────────────────────
        # `denied` is stamped by try_charge when enforcement refuses. A row
        # carrying underlying_cost_cents is one we paid a provider for.
        row = (await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
            ).where(
                CreditLedger.created_at >= since,
                _denied,
                func.coalesce(CreditLedger.underlying_cost_cents, 0) > 0,
            )
        )).first()
        given_away_calls = int(row[0] or 0)
        given_away_usd = float(row[1] or 0) / 100.0
        readings["served_unbilled_calls"] = given_away_calls
        readings["served_unbilled_usd"] = round(given_away_usd, 2)

        if given_away_usd >= float(_cfg("credit_health_unbilled_usd_critical", 5.0)):
            fired.append("served_unbilled_critical")
            await send_infra_alert(
                "credit-served-unbilled", "critical",
                f"{given_away_calls} call(s) were DENIED but served anyway in the last "
                f"{window_h}h — ${given_away_usd:.2f} of provider spend billed to us and "
                f"not to anyone. A denial that does not stop the work is a discount. "
                f"Check `credit_ledger` where metadata->>'denied' is true.",
            )
        elif given_away_calls > 0:
            fired.append("served_unbilled_warning")
            await send_infra_alert(
                "credit-served-unbilled", "warning",
                f"{given_away_calls} denied-but-served call(s) in the last {window_h}h "
                f"(${given_away_usd:.2f}). Expected steady state is zero.",
            )

        # ── 2. Duplicate one-time grants ──────────────────────────────
        _dupe_users = (
            select(CreditLedger.user_id)
            .where(
                CreditLedger.event_type == LEDGER_PLAN_GRANT,
                CreditLedger.bucket == BUCKET_MESSAGE,
            )
            .group_by(CreditLedger.user_id)
            .having(func.count() > 1)
            .subquery()
        )
        dupes = (await db.execute(
            select(func.count()).select_from(_dupe_users)
        )).scalar() or 0
        readings["users_with_duplicate_grants"] = int(dupes)
        if int(dupes) > 0:
            fired.append("duplicate_grants")
            await send_infra_alert(
                "credit-duplicate-grants", "critical",
                f"{dupes} user(s) hold more than one one-time plan grant. The grant "
                f"ASSIGNS the monthly allotment, so each repeat silently resets the "
                f"wallet and erases that period's spend — and anyone who can trigger "
                f"it has an unlimited credit farm. "
                f"Repair: python -m app.scripts.reconcile_duplicate_grants --apply",
            )

        # ── 3. Charged credits vs the cost they represent ─────────────
        row = (await db.execute(
            select(
                func.coalesce(-func.sum(CreditLedger.amount), 0),
                func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
            ).where(
                CreditLedger.created_at >= since,
                CreditLedger.amount < 0,
                # NOT IN drops NULLs in SQL and most rows have no such key, so
                # the IS NULL arm is load-bearing, not defensive.
                or_(_is_admin.is_(None), ~_is_admin),
            )
        )).first()
        credits_charged = float(row[0] or 0)
        cost_cents = float(row[1] or 0)
        readings["credits_charged"] = round(credits_charged, 2)
        readings["provider_cost_usd"] = round(cost_cents / 100.0, 2)

        # Only meaningful once there is real volume; below the floor the ratio
        # is noise, not signal.
        floor = float(_cfg("credit_health_min_cost_cents", 200.0))
        if cost_cents >= floor:
            ratio = credits_charged / cost_cents if cost_cents else 0.0
            readings["charge_ratio"] = round(ratio, 3)
            if ratio < float(_cfg("credit_health_ratio_critical", 0.5)):
                fired.append("undercharging")
                await send_infra_alert(
                    "credit-undercharge", "critical",
                    f"Charged {credits_charged:.1f} credits against "
                    f"${cost_cents/100:.2f} of provider cost in {window_h}h "
                    f"(ratio {ratio:.2f}, 1.0 = break-even by design). The meter is "
                    f"undercounting or a charge path is broken.",
                )

        # ── 4. The meter is moving at all ─────────────────────────────
        events = (await db.execute(
            select(func.count()).select_from(LLMProxyEvent)
            .where(LLMProxyEvent.created_at >= since)
        )).scalar() or 0
        charges = (await db.execute(
            select(func.count()).select_from(CreditLedger).where(
                CreditLedger.created_at >= since,
                CreditLedger.event_type == LEDGER_CHAT_MESSAGE,
            )
        )).scalar() or 0
        readings["llm_events"] = int(events)
        readings["chat_charge_rows"] = int(charges)
        if int(events) >= int(_cfg("credit_health_min_events", 50)) and int(charges) == 0:
            fired.append("meter_stalled")
            await send_infra_alert(
                "credit-meter-stalled", "critical",
                f"{events} LLM proxy event(s) in {window_h}h produced ZERO charge "
                f"rows. The charge path is disconnected.",
            )

    logger.info("[credit-health] %s alarms=%s", readings, fired or "none")
    return {"readings": readings, "alerts": fired}


async def credit_health_monitor_loop() -> None:
    """Forever loop; start via asyncio.create_task in the lifespan."""
    interval = max(600, int(_cfg("credit_health_check_interval_s", 3600)))
    logger.info("[credit-health] monitor started (interval=%ss)", interval)
    while True:
        try:
            # Sleep first — a deploy is the worst moment to page, and at boot
            # the window holds whatever the previous replica left behind.
            await asyncio.sleep(interval)
            await check_credit_health()
        except asyncio.CancelledError:
            raise
        except Exception:
            # One bad query must not silence the monitor for the process
            # lifetime, but it is never swallowed quietly either.
            logger.exception("[credit-health] check failed; continuing")
