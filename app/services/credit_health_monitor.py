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
1. **A refusal holds.** A ledger row with ``denied=true`` means the charge was
   refused; provider cost attached to such a row is work we paid for and did
   not bill. Since 2026-09-18 that is no longer always a leak. The turn that
   CROSSES zero settles what the wallet holds, records the residual and the
   next pre-flight refuses — so every legitimate exhaustion leaves a short
   burst of these rows: the crossing turn, plus one for each proxy call it
   already had in flight (main model, utility, tools).

   *Expected:* those bursts. Seconds long, one account, counted in the
   readings and in the log line, never paged on their own.

   *Not expected:* an account still being served-and-denied more than
   ``credit_health_unbilled_loop_span_min`` minutes after its first refusal
   **with no refill in between**. That is the stop not holding — measured
   2026-09-15 as ONE account, 21 rows, 6h07m — and it pages one warning per
   account, naming the span, the counts, what the settlement recovered, the
   worst (reason, event, model, operation) group and where to look. The
   refill clause is not a loophole: exhaust → top up → exhaust again is two
   bounded crossings hours apart, and first→last would call a customer who
   has just paid a leak. A burst that runs long AFTER a refill is still the
   stop failing, and still pages. Above
   ``credit_health_unbilled_usd_critical`` the fleet-wide provider total
   pages critical whichever it is: enough crossings to cost real money is
   itself news, and the body says how much is which.
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
    LEDGER_DAILY_RESET, LEDGER_IMAGE_GEN, LEDGER_PLAN_GRANT, LEDGER_TOOL_CALL,
    LLMProxyEvent,
)
from app.db.plan_catalog import UNLIMITED_PLAN_ID
from app.services.abuse_metrics import uidp
from app.services.alerting import send_infra_alert

logger = logging.getLogger(__name__)

# Ledger event types that represent a user CONSUMING something. Grants,
# renewals, daily resets and plan changes all move the balance without being
# usage, so the revenue-vs-cost ratio must ignore them.
_USAGE_EVENT_TYPES = (LEDGER_CHAT_MESSAGE, LEDGER_TOOL_CALL, LEDGER_IMAGE_GEN)

# How long the SAME looping account stays quiet after it pages. This monitor
# runs hourly over a 24-hour window, so at `alerting.py`'s 600 s default an
# unchanged loop re-pages every hour for a day — the founder's feed shows
# exactly that, an identical "3 call(s), $0.19" an hour apart. 6 h is the
# fleet-watch convention. The trade, stated because it is not a bug: the
# window lives in module state, so a redeploy clears it and a loop that is
# still running can page once more than the interval implies.
_LOOP_ALERT_INTERVAL_S = 6 * 3600


def _cfg(name: str, default):
    return getattr(settings, name, default)


def _span_words(seconds: float) -> str:
    """A duration an operator reads at a glance. Short on purpose."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _group_words(subject, reason, event, model, op, calls=None, cents=None) -> str:
    """The whole identity of a leak: user × deny reason × event type × model ×
    operation type. Grouped on all five because the fix differs per
    combination — a ``system.*`` operation billed to a tenant user is a tagging
    bug, an ``image_generation`` row is a different charge path from
    ``chat_message``, and the reason says which gate refused.

    ``subject=None`` drops the prefix, for the per-account warning where the
    account is already the subject of the sentence AND of the alert key —
    naming it a third time reads as a formatting bug. ``calls``/``cents``
    omitted drop the counts for the same reason: the per-account warning has
    already said how many calls and how many dollars, and a single-group
    account — the common case, and the 2026-09-15 shape — printed both twice.
    """
    head = f"{subject} — " if subject else ""
    counts = (
        f"{int(calls or 0)} call(s), ${float(cents or 0)/100:.2f}, "
        if calls is not None or cents is not None else ""
    )
    return (
        f"{head}{counts}"
        f"reason={reason or '?'}, event={event}, model={model or '?'}, "
        f"op={op or 'user'}"
    )


async def _bursts_for_account(db, user_id, first, last, where, reason_expr, op_expr):
    """Split one account's denied-but-served rows into BURSTS at each credit
    refill, and describe each burst.

    Why this exists: first→last across a 24-hour window is not the span of a
    loop. A user who exhausts their allowance, tops up (or renews) and
    exhausts again leaves two bounded bursts hours apart — two legitimate
    crossings — and a first→last reading pages the founder that "the refusal
    is not stopping the work" about a customer who has just paid. A false
    accusation costs more than a missed page, so a refill ends a burst: the
    stop is only "not holding" while nothing has put credits back.

    A refill is any ledger row for that account with ``amount > 0``, OR a
    ``daily_reset``. The positive-amount half is deliberately NOT an
    event-type allowlist — ``plan_grant`` (initial grant), ``period_renewal``,
    ``iap_purchase`` (a StoreKit credit pack), ``manual_adjust`` (an admin
    comp), ``plan_change`` (an upgrade delta, also written by
    ``scripts/grandfather_unlimited`` and ``reconcile_duplicate_grants``
    outside ``credit_service``) and ``refund`` all add credits, so any new
    path that tops a BALANCE up is covered without being enumerated here.

    ``daily_reset`` has to be named explicitly because it is the counterexample
    to that reasoning: it restores spendable headroom by zeroing
    ``message_credits_used_today`` and writes ``amount = 0``, so on the CAP
    dimension a wallet moves with no positive amount anywhere. Left out, two
    cap-denied bursts either side of a day roll are one "loop" and the warning
    asserts "with no refill in between" about a user whose cap had in fact just
    re-opened — the false accusation this whole function exists to prevent.
    (Not reachable today: a ``daily_cap_exceeded`` row with provider cost needs
    a non-NULL cap AND ``credit_cap_admission_control`` off, and every cap has
    been NULL since 2026-08-29. It is exactly the 2026-08-03 configuration and
    returns the moment a cap is restored ahead of that flag.)

    Bucket is not filtered either: a grant writes both buckets in the same
    instant (so the pair is one boundary, not two), and being generous about
    what counts as a refill fails toward silence, which is the safer direction
    for this alarm.

    Runs ONLY for an account whose whole-window span already exceeds the
    limit, so a healthy window still costs the two aggregates it always did.
    Both queries ride ``ix_credit_ledger_user_created``.
    """
    rows = (await db.execute(
        select(
            CreditLedger.created_at, CreditLedger.underlying_cost_cents,
            CreditLedger.amount, reason_expr, CreditLedger.event_type,
            CreditLedger.model, op_expr,
        ).where(*where, CreditLedger.user_id == user_id)
        .order_by(CreditLedger.created_at)
    )).all()
    refills = (await db.execute(
        select(CreditLedger.created_at).where(
            CreditLedger.user_id == user_id,
            or_(CreditLedger.amount > 0,
                CreditLedger.event_type == LEDGER_DAILY_RESET),
            CreditLedger.created_at >= first,
            CreditLedger.created_at <= last,
        ).order_by(CreditLedger.created_at)
    )).scalars().all()

    bursts: list[dict] = []
    cur: dict | None = None
    groups: dict = {}
    ri = 0
    for (ts, cents, amount, reason, event, model, op) in rows:
        refilled = False
        # Every refill at or before this row ends the previous burst. A tie on
        # the timestamp treats the refill as landing first, which is the
        # reading that splits rather than accuses.
        while ri < len(refills) and refills[ri] <= ts:
            ri += 1
            refilled = True
        if cur is None or refilled:
            groups = {}
            cur = {"calls": 0, "cents": 0.0, "recovered": 0.0,
                   "first": ts, "last": ts, "groups": groups}
            bursts.append(cur)
        cur["calls"] += 1
        cur["cents"] += float(cents or 0)
        # The debit a settled refusal DID manage to take, negated in Python
        # for the same reason the span is: no dialect-specific SQL.
        cur["recovered"] += -float(amount or 0)
        cur["last"] = ts
        g = groups.setdefault((reason, event, model, op), 0.0)
        groups[(reason, event, model, op)] = g + float(cents or 0)
    for b in bursts:
        b["span_s"] = (b["last"] - b["first"]).total_seconds()
        b["dims"] = max(b["groups"].items(), key=lambda kv: kv[1])[0]
        del b["groups"]
    return bursts


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

        # ── 1. A refusal holds ────────────────────────────────────────
        # `denied` is stamped by try_charge when enforcement refuses. A row
        # carrying underlying_cost_cents is one we paid a provider for.
        #
        # This alarm used to page on the mere EXISTENCE of such a row, with the
        # words "Expected steady state is zero". Since the shortfall settlement
        # that sentence is false: the turn that crosses zero debits what the
        # wallet holds and is still recorded denied, so every legitimate
        # exhaustion writes one — plus one per proxy call that turn already had
        # in flight. With users reaching their allowance daily the hourly ⚠️
        # would be PERMANENT, which is new noise from the round whose purpose
        # was to remove it. What is still a defect is the stop NOT HOLDING, and
        # that has a measurable signature: a crossing burst spans seconds to a
        # couple of minutes (one turn), while 2026-09-15 was one account with
        # 21 rows over 6h07m. So the span decides — the span of one burst,
        # where a credit refill ends a burst, because exhaust → top up →
        # exhaust again is two crossings and not a loop.
        _unbilled_where = (
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
        row = (await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
                # What a refusal DID manage to take. Since 2026-09-18 a
                # settlement of an already-incurred cost debits what the wallet
                # holds and still records the denial, so "denied" no longer
                # implies "amount = 0" — and an alarm that reads the whole
                # provider cost as given away would overstate a system that is
                # now recovering most of it. Column arithmetic on purpose: this
                # file keeps `metadata_json.credits_*` a reporting field, never
                # a monitoring dependency.
                func.coalesce(-func.sum(CreditLedger.amount), 0),
            ).where(*_unbilled_where)
        )).first()
        given_away_calls = int(row[0] or 0)
        given_away_usd = float(row[1] or 0) / 100.0
        recovered_credits = float(row[2] or 0)
        readings["served_unbilled_calls"] = given_away_calls
        readings["served_unbilled_usd"] = round(given_away_usd, 2)
        readings["served_unbilled_recovered_credits"] = round(recovered_credits, 2)

        # WHO and FOR HOW LONG, not just how many. One grouped query carries
        # both: grouped on the identity of the leak for the message, with
        # min/max created_at per group so folding the groups per user yields
        # that ACCOUNT's first→last span over the same predicate. Run only when
        # the count is non-zero, so a healthy window still costs exactly the
        # one aggregate it always did.
        span_limit_s = float(_cfg("credit_health_unbilled_loop_span_min", 15)) * 60.0
        # The LIMIT, not a measurement: "…loop_span_min: 15.0" printed beside
        # "loop_calls: 21" read as "the loop spanned 15 minutes". The observed
        # worst is `served_unbilled_loop_worst_span_s`, below.
        readings["served_unbilled_loop_span_limit_min"] = round(span_limit_s / 60.0, 1)
        loops: list[dict] = []
        crossings: list[dict] = []
        loop_users: set[str] = set()
        worst_row = None
        if given_away_calls > 0:
            _reason = CreditLedger.metadata_json["reason"].as_string()
            _op = CreditLedger.metadata_json["operation_type"].as_string()
            offenders = (await db.execute(
                select(
                    CreditLedger.user_id, _reason, CreditLedger.event_type,
                    CreditLedger.model, _op,
                    func.count(),
                    func.coalesce(func.sum(CreditLedger.underlying_cost_cents), 0),
                    func.coalesce(-func.sum(CreditLedger.amount), 0),
                    # `func.min`/`func.max` take their type from the argument
                    # (ReturnTypeFromArgs), so these come back as datetimes on
                    # SQLite too — the same reason the whole file is
                    # SQLAlchemy rather than raw SQL.
                    func.min(CreditLedger.created_at),
                    func.max(CreditLedger.created_at),
                ).where(*_unbilled_where).group_by(
                    CreditLedger.user_id, _reason, CreditLedger.event_type,
                    CreditLedger.model, _op,
                )
            )).all()
            readings["served_unbilled_groups"] = len(offenders)

            per_user: dict[str, dict] = {}
            for g in offenders:
                acc = per_user.setdefault(g[0], {
                    "user_id": g[0], "calls": 0, "cents": 0.0, "recovered": 0.0,
                    "first": g[8], "last": g[9], "dims": None, "worst_cents": -1.0,
                })
                acc["calls"] += int(g[5] or 0)
                acc["cents"] += float(g[6] or 0)
                acc["recovered"] += float(g[7] or 0)
                if g[8] is not None and (acc["first"] is None or g[8] < acc["first"]):
                    acc["first"] = g[8]
                if g[9] is not None and (acc["last"] is None or g[9] > acc["last"]):
                    acc["last"] = g[9]
                if float(g[6] or 0) > acc["worst_cents"]:
                    acc["worst_cents"] = float(g[6] or 0)
                    acc["dims"] = (g[1], g[2], g[3], g[4])
            for acc in per_user.values():
                span_s = (
                    (acc["last"] - acc["first"]).total_seconds()
                    if acc["first"] is not None and acc["last"] is not None else 0.0
                )
                # INCLUSIVE at exactly the setting: the config's wording is the
                # contract ("spans MORE than this many minutes is paged as a
                # LOOP"), and a bound that pages AT the number makes a
                # 15-minute setting mean 14-and-a-bit.
                if span_s <= span_limit_s:
                    # One bounded burst across every group the account touched
                    # — the designed cost of crossing zero. No second query.
                    acc["span_s"] = span_s
                    crossings.append(acc)
                    continue
                # Over the limit on first→last. That is NOT yet a loop: it is
                # also what two legitimate crossings either side of a top-up
                # look like. Ask the refills. (Only here, so the common case
                # pays nothing for it.)
                bursts = await _bursts_for_account(
                    db, acc["user_id"], acc["first"], acc["last"],
                    _unbilled_where, _reason, _op,
                )
                if not bursts:
                    # Unreachable from one session — same predicate, same
                    # window, rows the grouped query just counted. Guarded
                    # anyway because an empty answer would drop this account's
                    # calls and dollars out of the loop/crossing breakdown
                    # entirely, and that breakdown has to add back up to the
                    # totals the critical body quotes.
                    acc["span_s"] = span_s
                    bursts = [acc]
                for burst in bursts:
                    burst["user_id"] = acc["user_id"]
                    if burst["span_s"] > span_limit_s:
                        loops.append(burst)
                        loop_users.add(acc["user_id"])
                    else:
                        crossings.append(burst)

            worst_row = max(offenders, key=lambda r: float(r[6] or 0), default=None)
            if worst_row is not None:
                readings["served_unbilled_top_user"] = uidp(worst_row[0])
                readings["served_unbilled_top_reason"] = worst_row[1] or "?"
                readings["served_unbilled_top_usd"] = round(
                    float(worst_row[6] or 0) / 100.0, 2)

        # Everything is now a BURST, filed as a loop or a crossing, so
        # loop_* + crossing_* reconciles to the totals above — an operator
        # reading "$9.00" can add the breakdown back up. The two *_accounts
        # counts partition the accounts as well: an account with any looping
        # burst is a looping account, and its own quiet bursts still count
        # their calls and dollars as crossings.
        loop_usd = sum((b["cents"] for b in loops), 0.0) / 100.0
        crossing_usd = sum((b["cents"] for b in crossings), 0.0) / 100.0
        crossing_users = {b["user_id"] for b in crossings} - loop_users
        readings["served_unbilled_loop_accounts"] = len(loop_users)
        readings["served_unbilled_loop_calls"] = sum(int(b["calls"]) for b in loops)
        readings["served_unbilled_loop_usd"] = round(loop_usd, 2)
        readings["served_unbilled_loop_recovered_credits"] = round(
            sum((b["recovered"] for b in loops), 0.0), 2)
        readings["served_unbilled_loop_worst_span_s"] = int(
            max((b["span_s"] for b in loops), default=0.0))
        readings["served_unbilled_crossing_accounts"] = len(crossing_users)
        readings["served_unbilled_crossings"] = len(crossings)
        readings["served_unbilled_crossing_calls"] = sum(
            int(b["calls"]) for b in crossings)
        readings["served_unbilled_crossing_usd"] = round(crossing_usd, 2)
        readings["served_unbilled_crossing_recovered_credits"] = round(
            sum((b["recovered"] for b in crossings), 0.0), 2)

        # The worst group is picked over the whole window, before the split, so
        # say WHICH it is: "loop" sends the operator to the runbook, "crossing"
        # tells them the biggest number in the window is the designed cost. The
        # label is per ACCOUNT (any looping burst makes the account a loop),
        # because chasing one is an account-level action.
        #
        # DIMENSIONS ONLY, no calls and no dollars — the same rule the warning
        # follows, for a sharper reason here. The group is aggregated over the
        # whole window, so on an account that both loops and crosses its
        # figures are the PRE-SPLIT total: "Worst (loop): … 5 call(s), $6.00"
        # overstated the loop by the crossing burst's $0.60, in the same
        # message whose previous sentence reported the loop total correctly as
        # $5.40. Two numbers for one thing, one of them wrong.
        _worst_clause = ""
        if worst_row is not None:
            _bucket = "loop" if worst_row[0] in loop_users else "crossing"
            _worst_clause = (
                f" Worst ({_bucket}): "
                + _group_words(
                    uidp(worst_row[0]), worst_row[1], worst_row[2], worst_row[3],
                    worst_row[4],
                )
                + "."
            )
        _recovered_clause = (
            f" {recovered_credits:.1f} credit(s) were recovered from those "
            f"rows by the shortfall settlement."
            if recovered_credits > 0 else ""
        )
        # CRITICAL is the fleet-wide safety net on provider dollars, crossings
        # INCLUDED: a flood of crossings large enough to cost real money is
        # itself news. It says how much of the total is which, because "$9 was
        # denied and served" sends an operator hunting for a loop that may not
        # exist. It fires independently of the per-account warnings below —
        # they answer different questions ("we are losing money" vs "these
        # accounts are stuck") and only the warning can name every account.
        #
        # The breakdown is stated in ONE scope, BURSTS, because that is the
        # scope the dollars are summed in. Pairing an account count with a
        # burst total let a single account that loops AND crosses render "0
        # crossing account(s) ($0.60)" — a zero carrying money, which reads as
        # a formatting bug in a critical page. Accounts are still named, as the
        # count the looping bursts are spread across, which is the number that
        # says how many people to chase.
        if given_away_usd >= float(_cfg("credit_health_unbilled_usd_critical", 5.0)):
            fired.append("served_unbilled_critical")
            await send_infra_alert(
                "credit-served-unbilled", "critical",
                f"${given_away_usd:.2f} of provider spend across {given_away_calls} "
                f"denied-but-served call(s) in the last {window_h}h: "
                f"{len(loops)} looping burst(s) (${loop_usd:.2f}) across "
                f"{len(loop_users)} account(s), and {len(crossings)} crossing "
                f"burst(s) (${crossing_usd:.2f}). A "
                f"crossing is the bounded cost of settling a turn that was already "
                f"incurred; a loop is a refusal that is not stopping the work. "
                f"Either way this is real money."
                f"{_worst_clause}{_recovered_clause} "
                f"Check `credit_ledger` where metadata->>'denied' is true.",
                # subject stays None DELIBERATELY on the critical arm: a
                # subject-keyed alert is what `infra_alert_category_subject_cap`
                # counts, and a critical that can be collected into a digest is
                # a critical that can arrive late. The account is named in the
                # body instead.
            )

        # One WARNING per looping account, WORST FIRST. `alerting.py` collects
        # past `infra_alert_category_subject_cap` (5) distinct subjects in a
        # window into the next message's digest line rather than paging them,
        # so the order decides which accounts page and which are merely named:
        # sorted by provider cost, the expensive loops are the ones that ring.
        # The critical arm above shares this category and is `subject=None`, so
        # it consumes no cap slot — but it does re-roll the category window at
        # its own 600 s, which frees the cap early. Harmless: a run loud enough
        # to page critical is one where more named accounts is the right answer.
        #
        # ONE warning per account, describing its WORST burst — an account can
        # loop twice in a window, and two messages about one account is the
        # noise this round removes.
        per_loop_user: dict[str, dict] = {}
        for burst in loops:
            e = per_loop_user.setdefault(
                burst["user_id"], {"cents": 0.0, "worst": burst})
            e["cents"] += burst["cents"]
            w = e["worst"]
            if (burst["cents"], burst["span_s"]) > (w["cents"], w["span_s"]):
                e["worst"] = burst
        # The name is appended ONCE, not once per account: `alerts` and the
        # `[credit-health] … alarms=` log line are the operator's record of a
        # run, and three looping accounts wrote the same word three times,
        # which reads as three kinds of problem. How many accounts is
        # `served_unbilled_loop_accounts`, in the readings beside it.
        if per_loop_user:
            fired.append("served_unbilled_loop")
        for e in sorted(per_loop_user.values(), key=lambda e: e["cents"], reverse=True):
            b = e["worst"]
            subj = uidp(b["user_id"])
            span = _span_words(b["span_s"])
            await send_infra_alert(
                "credit-served-unbilled", "warning",
                # Read on a phone, so every clause has to earn its width — the
                # ceiling is pinned in the tests. What a crossing is gets a
                # clause rather than the sentence it used to have; the span is
                # printed once rather than twice.
                f"Account {subj}: denied-but-served {int(b['calls'])} time(s) "
                f"over {span}, no refill in between (${b['cents']/100:.2f}"
                + (f", {b['recovered']:.1f} credit(s) recovered"
                   if b["recovered"] > 0 else "")
                + ") — a crossing spans seconds; the stop is not holding."
                + (f" Worst group: {_group_words(None, *b['dims'])}."
                   if b["dims"] is not None else "")
                # The next step, because naming a problem is half an alert. It
                # carries the burst's START in compact UTC: the message used to
                # say "in that window" while containing no absolute time at
                # all, only a duration, and the burst can have ended hours
                # before the page — so the arrival time does not recover it and
                # the operator could not write the query they were told to
                # write. (The account prefix is not an id, so the query still
                # cannot be scoped to the account here.) This exact shape is
                # also what the revert switch produces, which is the one check
                # worth doing first.
                + f" Check credit_ledger where metadata->>'denied' = 'true' "
                  f"since {b['first']:%m-%d %H:%MZ}; if all show amount = 0, "
                  "credit_settle_incurred_shortfall is off.",
                # Keyed on the account so a SECOND looping account is not
                # suppressed by the first one's window.
                subject=subj,
                min_interval_s=_LOOP_ALERT_INTERVAL_S,
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
