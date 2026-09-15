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
   1 means the meter is undercounting, not that pricing is generous. Measured
   on rows whose cost is NOT the 1-cent floor — see the query for why that
   exclusion is the difference between 0.428 (false alarm) and 1.087 (true).
4. **The meter is moving.** LLM proxy events with no corresponding charge rows
   means the charge path is broken again, whatever the reason.
5. **Unlimited is bounded.** An account that is never denied and never debited
   still costs money. Per user, per window, the provider cost behind rows
   marked ``unlimited`` has a warning bar and a critical bar. This is also the
   cross-replica arm of the anti-abuse ladder — the in-process counters in
   ``unlimited_abuse`` see one replica each; this SQL sees them all.
6. **A zero charge always says why.** A usage row with real provider cost,
   ``amount = 0`` and NO reason marker is a bug.
7. **A refusal is never wrong about who.** A ``denied`` row against a balance
   on the ``unlimited`` plan is critical on a SINGLE occurrence.

The rule invariants 5–7 share, stated once because everything else follows
from it:

    A zero-amount ledger row means nothing on its own. The reason is always a
    metadata marker. A zero-amount row with provider cost and no marker is an
    alarm.

Without invariant 6, "unlimited" becomes an unfalsifiable explanation for any
charge that stops landing: a bug and a feature would write byte-identical
rows. With it they are distinguishable by one boolean, which is the price of
admission for adding a second zero-amount path at all.

Free-tier denials are the system WORKING and must never be silenced by any of
this. They are counted every run in ``denials_by_plan`` — visible in the same
log line the operator already reads — and only a windowed SPIKE warns.

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
    BUCKET_MESSAGE, CreditBalance, CreditLedger, LEDGER_CHAT_MESSAGE,
    LEDGER_IMAGE_GEN, LEDGER_PLAN_GRANT, LEDGER_TOOL_CALL, LLMProxyEvent,
)
from app.db.plan_catalog import UNLIMITED_PLAN_ID
from app.services.abuse_metrics import uidp
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)

# Ledger event types that represent a user CONSUMING something. Grants,
# renewals, daily resets and plan changes all move the balance without being
# usage, so the revenue-vs-cost ratio must ignore them.
_USAGE_EVENT_TYPES = (LEDGER_CHAT_MESSAGE, LEDGER_TOOL_CALL, LEDGER_IMAGE_GEN)


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
        # `unlimited` is stamped by try_charge for EVERY served-without-debit
        # row (admins included), so one predicate covers what used to need two.
        _unlimited = CreditLedger.metadata_json["unlimited"].as_boolean().is_(True)
        _meter_only = CreditLedger.metadata_json["meter_only"].as_boolean().is_(True)
        # NOT IN drops NULLs in SQL and most rows carry no such key, so the
        # IS NULL arm is load-bearing, not defensive — the same reason it is
        # load-bearing on the admin predicate in invariant 3.
        _not_unlimited = or_(_unlimited.is_(None), ~_unlimited)
        _not_admin = or_(_is_admin.is_(None), ~_is_admin)
        _not_meter_only = or_(_meter_only.is_(None), ~_meter_only)
        _not_denied = or_(_denied.is_(None), ~_denied)

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
                # An unlimited account is never denied (try_charge forces
                # deny_reason to None for it), so this exclusion is already
                # true for every row it could see. It is written anyway so the
                # intent is explicit rather than incidental: if a future edit
                # ever lets the two markers co-occur, that row is invariant
                # 7's alarm, not this one's — and reading it here as $5 of
                # "denied but served" would page on the wrong cause.
                _not_unlimited,
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
        # Windowed, NOT all-time. A repaired account keeps its historical
        # duplicate rows forever, so an all-time count fires on every run from
        # now until the end of time — and an alarm that never clears is one
        # people learn to ignore. More than one one-time grant INSIDE the
        # window is a live loop; a legitimate signup contributes exactly one.
        _dupe_users = (
            select(CreditLedger.user_id)
            .where(
                CreditLedger.event_type == LEDGER_PLAN_GRANT,
                CreditLedger.bucket == BUCKET_MESSAGE,
                CreditLedger.created_at >= since,
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
                # USAGE only. A negative amount is not necessarily revenue —
                # `plan_change` carries admin corrections and the
                # duplicate-grant reconciliation, and counting those made a
                # clawback look like income: the live reading jumped from 12.1
                # to 73.7 credits the moment the repair script ran.
                CreditLedger.event_type.in_(_USAGE_EVENT_TYPES),
                # NOT IN drops NULLs in SQL and most rows have no such key, so
                # the IS NULL arm is load-bearing, not defensive.
                _not_admin,
                # Unlimited rows carry real `underlying_cost_cents` and
                # amount = 0, so they cannot enter the numerator TODAY — the
                # `amount < 0` filter above already sheds them. That is
                # exactly the reasoning that produced the admin exclusion on
                # the line above, after it turned out to be wrong about a
                # sibling filter. Excluded explicitly for the same reason: a
                # future edit to `amount < 0` would otherwise re-admit them
                # into the DENOMINATOR alone and collapse the ratio toward 0,
                # paging `credit-undercharge` critical, hourly, forever, from
                # the first month of Unlimited.
                _not_unlimited,
                # Exclude the 1-cent FLOOR era. Until 2026-08-10 (#559/084)
                # `_calc_cost_cents` ended in `max(1, int(cost_usd * 100))`,
                # so a 15-token embedding whose true cost is ~$0.0000003 was
                # recorded as a full cent — five orders of magnitude high.
                # Post-084 rows are exact (min(exact_4dp, legacy)); this
                # filter keeps excluding the historical floored mass, and
                # drops post-084 sub-cent rows as ratio noise, which is the
                # conservative direction for this alarm.
                # Measured over 30 days of pre-084 production:
                # 3507 floored rows contributing a fictitious $35.07 against
                # 1059 real rows worth $70.37, which dragged this ratio to 0.428
                # — under the 0.5 critical bar. The alarm would have paged
                # forever on a correctly-priced system the moment volume cleared
                # the min-cost floor. On real rows alone the ratio is 1.087,
                # i.e. break-even, which is what the pricing is designed for.
                # This is a denominator the column cannot support, not a
                # threshold that needs loosening.
                CreditLedger.underlying_cost_cents > 1,
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

        # ── 5. Unlimited cost is bounded ──────────────────────────────
        # Grouped PER USER and alarmed on the max, not on the fleet total: one
        # runaway account is the thing this catches, and a sum over a growing
        # subscriber base would mask it behind ordinary growth.
        #
        # `underlying_cost_cents` IS the charged-equivalent credit figure —
        # 1 credit = 1¢ by design — so no JSON arithmetic is needed here and
        # `metadata_json.credits_quoted` stays a reporting field rather than a
        # monitoring dependency.
        unl_rows = (await db.execute(
            select(
                CreditLedger.user_id,
                func.count(),
                func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
            ).where(
                CreditLedger.created_at >= since,
                CreditLedger.event_type.in_(_USAGE_EVENT_TYPES),
                _unlimited,
            ).group_by(CreditLedger.user_id)
        )).all()
        unl_total_usd = sum(float(r[2] or 0) for r in unl_rows) / 100.0
        readings["unlimited_users"] = len(unl_rows)
        readings["unlimited_calls"] = sum(int(r[1] or 0) for r in unl_rows)
        readings["unlimited_usd"] = round(unl_total_usd, 2)
        top = max(unl_rows, key=lambda r: float(r[2] or 0), default=None)
        if top is not None:
            top_usd = float(top[2] or 0) / 100.0
            readings["unlimited_top_user"] = uidp(top[0])
            readings["unlimited_top_usd"] = round(top_usd, 2)
            crit = float(_cfg("credit_health_unlimited_usd_critical", 100.0))
            warn = float(_cfg("credit_health_unlimited_usd_warning", 25.0))
            if top_usd >= crit or top_usd >= warn:
                level = "critical" if top_usd >= crit else "warning"
                fired.append(f"unlimited_cost_{level}")
                await send_infra_alert(
                    "credit-unlimited-cost", level,
                    f"Unlimited account {uidp(top[0])} consumed "
                    f"${top_usd:.2f} of provider spend in {window_h}h across "
                    f"{int(top[1] or 0)} call(s). Nothing was debited — that is "
                    f"the plan working — but this is above the "
                    f"${warn:.2f}/{crit:.2f} bar and worth a look. Fleet total "
                    f"across {len(unl_rows)} unlimited account(s): "
                    f"${unl_total_usd:.2f}.",
                    subject=uidp(top[0]),
                )

        # ── 6. A zero charge always says why ──────────────────────────
        # The price of admission for adding a second zero-amount path. A usage
        # row with real provider cost, amount 0, and NONE of the four reason
        # markers is the shape a BROKEN charge path writes — and, without this,
        # it is also the shape "unlimited" would be blamed for.
        row = (await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
            ).where(
                CreditLedger.created_at >= since,
                CreditLedger.event_type.in_(_USAGE_EVENT_TYPES),
                CreditLedger.amount == 0,
                func.coalesce(CreditLedger.underlying_cost_cents, 0) > 0,
                _not_denied,        # invariant 1 owns that shape
                _not_admin,
                _not_unlimited,
                _not_meter_only,
            )
        )).first()
        silent_calls = int(row[0] or 0)
        silent_usd = float(row[1] or 0) / 100.0
        readings["silent_zero_calls"] = silent_calls
        readings["silent_zero_usd"] = round(silent_usd, 2)
        if silent_calls > 0:
            crit = float(_cfg("credit_health_silent_zero_usd_critical", 5.0))
            level = "critical" if silent_usd >= crit else "warning"
            fired.append(f"silent_zero_{level}")
            await send_infra_alert(
                "credit-silent-zero", level,
                f"{silent_calls} usage row(s) in {window_h}h carry provider cost "
                f"(${silent_usd:.2f}) with amount=0 and NO reason marker — not "
                f"denied, not admin, not unlimited, not meter_only. A zero charge "
                f"must always say why; one that does not is a broken charge path "
                f"wearing the unlimited path's clothes. Check `credit_ledger` "
                f"where amount = 0 and underlying_cost_cents > 0.",
            )

        # ── 7. A refusal is never wrong about who ─────────────────────
        # Denials grouped by the DENIED USER'S plan. Always recorded, so a
        # free-tier denial — the system working — stays visible in the same
        # line the operator already reads and can never be silenced by
        # anything above.
        # Grouped on the plan the row itself RECORDS, falling back to the
        # user's current plan only for rows written before `try_charge` began
        # stamping it.
        #
        # The fallback was the whole grouping, and it made this alarm fire on
        # its own fix: `CreditBalance.plan_id` is the plan the user holds NOW,
        # so grandfathering an account that was refused yesterday
        # re-attributed yesterday's denials to 'unlimited' and paged
        # `credit-denied-unlimited` CRITICAL — "the user is locked out right
        # now" — about the customer the operator had just unblocked. On
        # cutover day, for every account in the window. An alarm whose value
        # rests on being true at one occurrence cannot also be the loudest
        # thing that happens when nothing is wrong.
        _row_plan = CreditLedger.metadata_json["plan_id"].as_string()
        denial_rows = (await db.execute(
            select(func.coalesce(_row_plan, CreditBalance.plan_id), func.count())
            .select_from(CreditLedger)
            .outerjoin(CreditBalance, CreditBalance.user_id == CreditLedger.user_id)
            .where(CreditLedger.created_at >= since, _denied)
            .group_by(func.coalesce(_row_plan, CreditBalance.plan_id))
        )).all()
        # OUTER join, so a denial whose user has no credit_balances row (or
        # whose account was since deleted) is still counted — `denials_total`
        # is a sum of this dict and would otherwise understate.
        denials_by_plan = {str(p): int(c or 0) for (p, c) in denial_rows}
        readings["denials_by_plan"] = denials_by_plan
        total_denials = sum(denials_by_plan.values())
        readings["denials_total"] = total_denials

        # An unlimited user was REFUSED. This is the single bug class the
        # entitlement must never produce, so it fires on one occurrence with
        # no threshold — a wall in front of the one account that is defined by
        # not having one is not a trend to watch.
        denied_unlimited = denials_by_plan.get(UNLIMITED_PLAN_ID, 0)
        if denied_unlimited > 0:
            fired.append("denied_unlimited")
            await send_infra_alert(
                "credit-denied-unlimited", "critical",
                f"{denied_unlimited} credit denial(s) in {window_h}h landed on an "
                f"account that held the `{UNLIMITED_PLAN_ID}` plan AT THE TIME "
                f"OF THE REFUSAL. An unlimited account is never supposed to be "
                f"refused — this is a gate that stopped reading the "
                f"entitlement, and the user is locked out right now. Check "
                f"`credit_ledger` where metadata->>'denied' is true and "
                f"metadata->>'plan_id' = '{UNLIMITED_PLAN_ID}'.",
            )

        spike = int(_cfg("credit_health_denial_spike", 50))
        if total_denials >= spike:
            fired.append("denial_spike")
            await send_infra_alert(
                "credit-denial-spike", "warning",
                f"{total_denials} credit denial(s) in {window_h}h "
                f"(by plan: {denials_by_plan}) — over the {spike} bar. Free-tier "
                f"denials are the system working; a spike is still worth a look "
                f"because the alternative reading is a gate refusing people it "
                f"should not.",
            )

    logger.info("[credit-health] %s alarms=%s", readings, fired or "none")
    return {"readings": readings, "alerts": fired}


async def credit_health_monitor_loop() -> None:
    """Forever loop; start via asyncio.create_task in the lifespan."""
    interval = max(600, int(_cfg("credit_health_check_interval_s", 3600)))
    logger.info("[credit-health] monitor started (interval=%ss)", interval)
    # LEADER-GATED (2026-09-12, L3-6): the alert window in `alerting` is
    # per-process, so an unelected second replica does not merely duplicate
    # work — it duplicates every page. One runner, one page.
    from app.services.infra_lease import acquire_lease, lease_ttl_for
    _ttl = lease_ttl_for(interval)
    while True:
        try:
            # Sleep first — a deploy is the worst moment to page, and at boot
            # the window holds whatever the previous replica left behind.
            await asyncio.sleep(interval)
            if not await acquire_lease("credit_health_monitor", ttl_s=_ttl):
                continue
            await check_credit_health()
        except asyncio.CancelledError:
            raise
        except Exception:
            # One bad query must not silence the monitor for the process
            # lifetime, but it is never swallowed quietly either.
            logger.exception("[credit-health] check failed; continuing")
