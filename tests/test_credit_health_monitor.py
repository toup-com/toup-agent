"""Credit-health invariants under the UNLIMITED entitlement (design §6).

The rule the whole file turns on:

    A zero-amount ledger row means nothing on its own. The REASON is always a
    metadata marker, and a zero-amount row with provider cost and no marker is
    an alarm.

So the tests come in matched pairs. For each shape, one test proves the
monitor stays QUIET when the reason is present, and one proves it FIRES when
the same row loses its marker. A suppression that cannot be shown to still
alarm on the bug it resembles is just a muted alert.

Free-tier denials get their own pair too: they are the system WORKING and must
survive every exclusion added here, which is asserted by execution rather than
by reading the diff.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


# ── helpers ──────────────────────────────────────────────────────────


async def _mk_user(email: str | None = None) -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid,
            email=email or f"h-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Health Test",
        ))
        await db.commit()
    return uid


async def _mk_balance(user_id: str, plan_id: str = "free"):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        bal = await credit_service.get_or_create_balance(db, user_id)
        bal.plan_id = plan_id
        await db.commit()


async def _set_plan(user_id: str, plan_id: str):
    """Move an existing balance to another plan — what the grandfather does."""
    await _mk_balance(user_id, plan_id)


async def _row(
    user_id: str, *, amount="0", cost="300", metadata=None,
    event_type="chat_message", bucket="message", age_h: float = 1.0,
    model=None, at: datetime | None = None,
):
    """One ledger row, placed inside the monitor's window.

    `at` pins `created_at` exactly. `age_h` reads its own `utcnow()` per call,
    so two rows written "0.25h apart" are 900 s plus however long the first
    INSERT took — unusable for the span tests, where the whole question is
    which side of the boundary a pair lands on.
    """
    from app.db import async_session_maker
    from app.db.models import CreditLedger

    async with async_session_maker() as db:
        db.add(CreditLedger(
            user_id=user_id,
            event_type=event_type,
            bucket=bucket,
            amount=Decimal(amount),
            balance_after=Decimal("0"),
            underlying_cost_cents=(Decimal(cost) if cost is not None else None),
            model=model,
            metadata_json=metadata,
            created_at=at if at is not None
            else datetime.utcnow() - timedelta(hours=age_h),
        ))
        await db.commit()


# The denial shape invariant 1 reads, written once so the span — not the
# metadata — is what varies between the crossing and loop tests.
_DENY = {"denied": True, "reason": "insufficient_message_credits"}


async def _refill(user_id: str, at: datetime, *, amount="500",
                  event_type="iap_purchase"):
    """Credits going back INTO the wallet — a top-up, a renewal, a comp.

    The monitor recognises a refill by ``amount > 0`` and nothing else, so the
    event type here is flavour. `iap_purchase` is the shape that matters most:
    a customer who just paid is the one a false loop page would accuse. NOT
    `plan_grant`, which would also trip invariant 2 (`_mk_balance` already
    writes one).
    """
    await _row(user_id, amount=amount, cost=None, event_type=event_type,
               metadata={"reason": "credit_pack"}, at=at)


async def _daily_reset(user_id: str, at: datetime):
    """The day roll, written exactly as `_reset_daily_if_needed` writes it.

    The one refill that moves a wallet with ``amount = 0``: it zeroes
    ``message_credits_used_today``, so on the CAP dimension spendable headroom
    is restored with no positive amount anywhere in the ledger. The monitor
    therefore has to name this event type explicitly — `amount > 0` alone
    cannot see it — which is what the test below proves.
    """
    from app.db.models import LEDGER_DAILY_RESET

    await _row(user_id, amount="0", cost=None, event_type=LEDGER_DAILY_RESET,
               metadata={"prior_used_today": "100", "new_anchor": "2026-09-16"},
               at=at)


@pytest_asyncio.fixture
async def quiet_alerts(monkeypatch):
    """Capture alerts instead of posting them, and assert on the capture.

    `send_infra_alert` is rate-limited per (category, subject) at MODULE
    scope, so a real call in one test would silence the same alert in the
    next. Capturing is not only about the network.
    """
    sent: list[tuple] = []

    async def _fake(category, level, message, *, subject=None, min_interval_s=600):
        # `min_interval_s` is captured because it is now load-bearing, not a
        # default: a looping account that re-pages hourly for a day is the
        # noise this round removes, so the interval is asserted.
        sent.append((category, level, message, subject, min_interval_s))
        return True

    import app.services.credit_health_monitor as mon
    monkeypatch.setattr(mon, "send_infra_alert", _fake)
    return sent


async def _run():
    from app.services.credit_health_monitor import check_credit_health
    return await check_credit_health()


def _cats(sent) -> set[str]:
    return {s[0] for s in sent}


def _unbilled(sent) -> list[tuple]:
    return [s for s in sent if s[0] == "credit-served-unbilled"]


# ── invariant 5: unlimited is served, metered, and BOUNDED ───────────


async def test_unlimited_traffic_does_not_page_the_undercharge_alarm(quiet_alerts):
    """200 unlimited rows worth $50 of provider cost and zero credits charged.

    Invariant 3 is the ratio charged-credits ÷ provider-cents; feeding it a
    large denominator with an empty numerator would collapse it toward 0 and
    page `credit-undercharge` CRITICAL, hourly, forever, from the first month
    of Unlimited. This is the false alarm the exclusion exists to prevent.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "unlimited")
    for _ in range(40):
        await _row(uid, amount="0", cost="25", metadata={
            "unlimited": True, "unlimited_reason": "plan", "credits_quoted": "0.25",
        })
    out = await _run()
    assert "undercharging" not in out["alerts"]
    assert "credit-undercharge" not in _cats(quiet_alerts)
    # It IS counted as unlimited spend — suppressed from the ratio, never
    # from the books.
    assert out["readings"]["unlimited_calls"] == 40
    assert out["readings"]["unlimited_usd"] == pytest.approx(10.0)


async def test_unlimited_spend_over_the_bar_alerts(quiet_alerts):
    """Invariant 5. Served without a debit is expected; served without a
    debit at $100/day is a business fact the operator has to see."""
    from app.config import settings

    uid = await _mk_user()
    await _mk_balance(uid, "unlimited")
    # $30 of provider cost in the window — over the $25 warning bar.
    for _ in range(10):
        await _row(uid, amount="0", cost="300", metadata={"unlimited": True})
    out = await _run()
    assert "unlimited_cost_warning" in out["alerts"]
    cat = [s for s in quiet_alerts if s[0] == "credit-unlimited-cost"]
    assert cat and cat[0][1] == "warning"
    # Subject-scoped, so one heavy account cannot suppress an alert about
    # another (alerting.py keys the window on (category, subject)).
    assert cat[0][3] == uid[:8]
    assert settings.credit_health_unlimited_usd_warning == 25.0


async def test_unlimited_alarm_is_per_user_not_fleet_total(quiet_alerts):
    """Ten unlimited accounts at $5 each is $50 fleet-wide and NOT an alarm.

    A sum over a growing subscriber base would page on ordinary growth and
    mask the one runaway this invariant exists to catch.
    """
    for _ in range(10):
        uid = await _mk_user()
        await _mk_balance(uid, "unlimited")
        await _row(uid, amount="0", cost="500", metadata={"unlimited": True})
    out = await _run()
    assert out["readings"]["unlimited_usd"] == pytest.approx(50.0)
    assert not [a for a in out["alerts"] if a.startswith("unlimited_cost")]


# ── invariant 6: a zero charge always says why ───────────────────────


async def test_silent_zero_charge_alarms(quiet_alerts):
    """The price of admission for a second zero-amount path.

    Identical row shape to the unlimited one above, minus the marker. If this
    stayed quiet, "unlimited" would be an unfalsifiable explanation for any
    charge that stopped landing.
    """
    uid = await _mk_user()
    await _mk_balance(uid)
    await _row(uid, amount="0", cost="300", metadata=None)
    out = await _run()
    assert "silent_zero_warning" in out["alerts"]
    assert out["readings"]["silent_zero_calls"] == 1
    assert out["readings"]["silent_zero_usd"] == pytest.approx(3.0)
    assert "credit-silent-zero" in _cats(quiet_alerts)


async def test_silent_zero_goes_critical_past_the_dollar_bar(quiet_alerts):
    uid = await _mk_user()
    await _mk_balance(uid)
    for _ in range(3):
        await _row(uid, amount="0", cost="300", metadata=None)   # $9 > $5
    out = await _run()
    assert "silent_zero_critical" in out["alerts"]
    lvl = [s[1] for s in quiet_alerts if s[0] == "credit-silent-zero"]
    assert lvl == ["critical"]


@pytest.mark.parametrize("marker", [
    {"unlimited": True},
    {"admin_unlimited": True, "unlimited": True},
    {"meter_only": True, "credits_quoted": "3"},
    {"denied": True, "reason": "insufficient_message_credits"},
])
async def test_every_declared_reason_silences_invariant_6(quiet_alerts, marker):
    """Each of the four markers is a complete answer to "why zero?".

    `denied` is included because invariant 1 already owns that shape — two
    alarms for one row is how an operator learns to ignore both.
    """
    uid = await _mk_user()
    await _mk_balance(uid)
    await _row(uid, amount="0", cost="300", metadata=marker)
    out = await _run()
    assert out["readings"]["silent_zero_calls"] == 0
    assert "credit-silent-zero" not in _cats(quiet_alerts)


# ── invariant 7: a refusal is never wrong about who ──────────────────


async def test_denied_unlimited_alarms_critical_on_one_occurrence(quiet_alerts):
    """The single bug class this design must never produce.

    No threshold: a wall in front of the one account defined by not having
    one is not a trend to watch. It means a gate stopped reading the
    entitlement and a paying customer is locked out right now.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "unlimited")
    await _row(uid, amount="0", cost="300", metadata={
        "denied": True, "reason": "insufficient_message_credits",
    })
    out = await _run()
    assert "denied_unlimited" in out["alerts"]
    hit = [s for s in quiet_alerts if s[0] == "credit-denied-unlimited"]
    assert hit and hit[0][1] == "critical"
    assert out["readings"]["denials_by_plan"]["unlimited"] == 1


async def test_free_denials_are_counted_never_silenced(quiet_alerts):
    """A free user hitting their limit and being refused is the system
    working. Nothing added for Unlimited may hide it."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    for _ in range(3):
        await _row(uid, amount="0", cost=None, metadata={
            "denied": True, "reason": "insufficient_message_credits",
        })
    out = await _run()
    assert out["readings"]["denials_by_plan"] == {"free": 3}
    assert out["readings"]["denials_total"] == 3
    # Below the spike bar → visible, not alarming. And emphatically NOT the
    # unlimited alarm.
    assert "denial_spike" not in out["alerts"]
    assert "denied_unlimited" not in out["alerts"]


async def test_denial_spike_warns(quiet_alerts):
    """88 denials all-time makes 50 in 24h a change, not a busy day."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    for _ in range(50):
        await _row(uid, amount="0", cost=None, metadata={"denied": True})
    out = await _run()
    assert "denial_spike" in out["alerts"]
    lvl = [s[1] for s in quiet_alerts if s[0] == "credit-denial-spike"]
    assert lvl == ["warning"]


async def test_denied_and_served_still_pages(quiet_alerts):
    """Invariant 1's CRITICAL arm is untouched. This is the 2026-08-03 alarm —
    274 calls denied and served anyway — and nothing in this change may soften
    it.

    Updated 2026-09-18: these three rows are milliseconds apart, so they are a
    CROSSING, and the warning arm no longer fires on "any row". They page
    anyway because $9 clears the fleet-wide dollar bar — which is the point of
    keeping that bar: enough crossings to cost real money is still news.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    for _ in range(3):
        await _row(uid, amount="0", cost="300", metadata={
            "denied": True, "reason": "daily_cap_exceeded",
        })
    out = await _run()
    assert "served_unbilled_critical" in out["alerts"]   # $9 ≥ $5
    assert out["readings"]["served_unbilled_calls"] == 3
    assert out["readings"]["served_unbilled_crossing_accounts"] == 1
    assert out["readings"]["served_unbilled_loop_accounts"] == 0


# ── invariant 1: a crossing is not a loop ────────────────────────────
#
# Since the shortfall settlement (2026-09-18) a denied-but-served row is no
# longer always a leak. The turn that CROSSES zero debits what the wallet
# holds, records the residual, and the next pre-flight refuses — so EVERY
# legitimate exhaustion leaves such a row, plus one for each proxy call that
# turn already had in flight (main model, utility, tools). That burst spans
# seconds. The defect invariant 1 exists for is the stop NOT holding: the
# same account still being served-and-denied long after its first refusal,
# measured 2026-09-15 as ONE account, 21 rows, 6h07m.
#
# So the tests below come in the same matched-pair shape as the rest of the
# file: for each span, one proving the monitor stays quiet where quiet is
# correct, and one proving it still pages on the shape it exists to catch.


async def test_a_single_crossing_burst_does_not_page(quiet_alerts):
    """A user hitting their allowance: three denied-but-served rows 40 s apart.

    Before this change every one of these paged ⚠️ "Expected steady state is
    zero" — hourly, for 24 h, per account, on the founder's Telegram. With
    users exhausting an allowance daily that warning was permanent: new noise
    introduced by the settlement whose purpose was to stop the give-away.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=3)
    for i in range(3):
        # The real settlement shape: ~100¢ of cost, 95 credits in the wallet,
        # the 5-credit residual eaten and the row recorded denied.
        await _row(uid, amount="-95", cost="100", metadata=_DENY,
                   at=t0 + timedelta(seconds=40 * i))

    out = await _run()

    assert out["alerts"] == []
    assert quiet_alerts == []
    # Recorded, not paged. The money is still in the books.
    r = out["readings"]
    assert r["served_unbilled_calls"] == 3
    assert r["served_unbilled_crossing_accounts"] == 1
    assert r["served_unbilled_crossing_calls"] == 3
    assert r["served_unbilled_crossing_usd"] == pytest.approx(3.0)
    assert r["served_unbilled_crossing_recovered_credits"] == pytest.approx(285.0)
    assert r["served_unbilled_loop_accounts"] == 0
    assert r["served_unbilled_loop_calls"] == 0
    assert r["served_unbilled_loop_worst_span_s"] == 0
    assert r["served_unbilled_crossings"] == 1


async def test_the_2026_09_15_shape_pages_as_a_loop(quiet_alerts):
    """The measured signature of the stop not holding: one account, 21 rows,
    6h07m. This is the only thing invariant 1's warning arm still pages for,
    and it has to name the account, the span and the counts — the founder's
    feed showed "3 call(s), $0.19" with no span and no account, which is
    unactionable either way you read it.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=7)
    for i in range(21):
        await _row(uid, amount="0", cost="19", model="gpt-5.5", metadata={
            "denied": True, "reason": "insufficient_message_credits",
            "operation_type": "system.day_summary",
        }, at=t0 + timedelta(seconds=i * 1101))   # 20 gaps → 22020 s = 6h07m

    out = await _run()

    assert out["alerts"] == ["served_unbilled_loop"]
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    (_cat, level, msg, subject, interval) = hits[0]
    assert level == "warning"
    assert subject == uid[:8], "keyed on the ACCOUNT, so a second loop is not suppressed"
    assert uid not in msg, "the raw user id must never reach a Telegram body"
    assert "21 time(s)" in msg
    assert "6h07m" in msg
    assert "$3.99" in msg
    # …and the worst group for that account, because the fix differs per
    # (reason, event, model, operation).
    assert "insufficient_message_credits" in msg
    assert "chat_message" in msg
    assert "gpt-5.5" in msg
    assert "system.day_summary" in msg
    # DIMENSIONS ONLY in the group clause. This account has one group — the
    # common case and the measured 2026-09-15 shape — so repeating its counts
    # there printed "21" and "$3.99" twice in a message read on a phone.
    assert "Worst group: reason=" in msg
    assert msg.count("$3.99") == 1
    assert "call(s)" not in msg, "the account line already says 'time(s)'"
    # A next step, like invariant 2's "Repair:" line. The prefix is not an id,
    # so the query cannot be scoped to it — it has to say how to find the rows.
    assert "metadata->>'denied' = 'true'" in msg
    assert "credit_settle_incurred_shortfall is off" in msg
    # …and WHEN. The message used to say "in that window" while carrying no
    # absolute time at all, only a duration — and this burst can have ended
    # hours before the page, so the arrival time does not recover it either.
    # An operator could not write the query they were being told to write.
    assert f"since {t0:%m-%d %H:%M}Z" in msg
    assert "in that window" not in msg
    assert len(msg) <= 400, f"read on a phone; {len(msg)} chars is too long"
    # The monitor runs hourly over a 24 h window: at alerting.py's 600 s
    # default this same unchanged loop re-pages every hour for a day.
    assert interval == 6 * 3600
    r = out["readings"]
    assert r["served_unbilled_loop_accounts"] == 1
    assert r["served_unbilled_loop_calls"] == 21
    assert r["served_unbilled_loop_usd"] == pytest.approx(3.99)
    assert r["served_unbilled_crossing_accounts"] == 0
    # The limit is a SETTING; the observed worst span is the measurement. The
    # log line printing only the former read as "the loop spanned 15 minutes".
    assert r["served_unbilled_loop_span_limit_min"] == pytest.approx(15.0)
    assert r["served_unbilled_loop_worst_span_s"] == 22020


async def test_two_loops_page_worst_first_and_a_crossing_is_not_paged(quiet_alerts):
    """One alert per looping account, WORST FIRST.

    alerting.py collects past `infra_alert_category_subject_cap` (5) distinct
    subjects in a window into the next message's digest line, so the order is
    the difference between "the expensive loop paged and the cheap ones are
    named in a digest" and the reverse.
    """
    worst = await _mk_user()
    lesser = await _mk_user()
    crossing = await _mk_user()
    for u in (worst, lesser, crossing):
        await _mk_balance(u, "free")
    t0 = datetime.utcnow() - timedelta(hours=7)
    for i in range(4):                                   # $2.40 over 5h
        await _row(worst, amount="0", cost="60", metadata=_DENY,
                   at=t0 + timedelta(hours=i * 5 / 3))
    for i in range(3):                                   # $0.90 over 2h
        await _row(lesser, amount="0", cost="30", metadata=_DENY,
                   at=t0 + timedelta(hours=i))
    for i in range(2):                                   # $0.80 in 30s
        await _row(crossing, amount="0", cost="40", metadata=_DENY,
                   at=t0 + timedelta(seconds=30 * i))

    out = await _run()

    hits = _unbilled(quiet_alerts)
    assert [s[3] for s in hits] == [worst[:8], lesser[:8]]
    assert all(s[1] == "warning" for s in hits)
    # TWO warnings, ONE alarm name. `alerts` — and the `[credit-health] …
    # alarms=` log line built from it — is the operator's record of the run,
    # and the same word repeated per account reads as several kinds of
    # problem. How many accounts is a reading, asserted below.
    assert out["alerts"] == ["served_unbilled_loop"]
    assert out["alerts"].count("served_unbilled_loop") == 1
    assert crossing[:8] not in hits[0][2] and crossing[:8] not in hits[1][2]
    r = out["readings"]
    assert r["served_unbilled_loop_accounts"] == 2
    assert r["served_unbilled_crossing_accounts"] == 1
    assert r["served_unbilled_crossing_usd"] == pytest.approx(0.80)


async def test_the_span_boundary_is_inclusive_at_exactly_the_setting(quiet_alerts):
    """Pinned: rows exactly `credit_health_unbilled_loop_span_min` apart are a
    CROSSING; one second more is a LOOP.

    Inclusive because the setting's own wording is the contract — "an account
    whose rows span MORE than this many minutes is paged as a LOOP" — and
    because a bound that pages AT the number makes a 15-minute setting mean
    14-and-a-bit, which is the kind of off-by-one nobody re-derives later.
    """
    from app.config import settings

    span = int(settings.credit_health_unbilled_loop_span_min)
    assert span == 15, "the boundary tests below are written against 15 minutes"
    at_edge = await _mk_user()
    over_edge = await _mk_user()
    await _mk_balance(at_edge, "free")
    await _mk_balance(over_edge, "free")
    t0 = datetime.utcnow() - timedelta(hours=3)
    await _row(at_edge, amount="0", cost="40", metadata=_DENY, at=t0)
    await _row(at_edge, amount="0", cost="40", metadata=_DENY,
               at=t0 + timedelta(minutes=span))
    await _row(over_edge, amount="0", cost="40", metadata=_DENY, at=t0)
    await _row(over_edge, amount="0", cost="40", metadata=_DENY,
               at=t0 + timedelta(minutes=span, seconds=1))

    out = await _run()

    assert [s[3] for s in _unbilled(quiet_alerts)] == [over_edge[:8]]
    assert out["readings"]["served_unbilled_loop_accounts"] == 1
    assert out["readings"]["served_unbilled_crossing_accounts"] == 1


async def test_crossings_past_the_dollar_bar_go_critical_and_say_so(quiet_alerts):
    """The critical arm is a fleet-wide safety net on provider dollars and
    crossings COUNT toward it: enough of them to cost real money is itself
    news. But the body has to say which it is, or the operator reads $9 and
    goes hunting for a loop that is not there.
    """
    t0 = datetime.utcnow() - timedelta(hours=3)
    for _ in range(3):
        uid = await _mk_user()
        await _mk_balance(uid, "free")
        for i in range(2):
            await _row(uid, amount="0", cost="150", metadata=_DENY,
                       at=t0 + timedelta(seconds=20 * i))

    out = await _run()

    assert "served_unbilled_critical" in out["alerts"]
    assert "served_unbilled_loop" not in out["alerts"]
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    (_cat, level, msg, subject, _interval) = hits[0]
    assert level == "critical"
    assert subject is None, (
        "a subject-keyed alert is what the per-category cap counts, and a "
        "critical that can be collected into a digest can arrive late"
    )
    assert "$9.00" in msg
    # The breakdown is in ONE scope, bursts, because that is the scope the
    # dollars are summed in.
    assert "0 looping burst(s) ($0.00) across 0 account(s)" in msg
    assert "3 crossing burst(s) ($9.00)" in msg
    # The worst group is picked over the whole window, so it has to say WHICH
    # bucket its account is in: unlabelled, this sentence reads as the worst
    # LOOP on a window that holds none.
    assert "Worst (crossing):" in msg
    # The critical arm's diagnostic query is load-bearing and stays.
    assert "metadata->>'denied' is true" in msg


async def test_the_unbilled_critical_names_the_worst_account(quiet_alerts):
    """Updated 2026-09-18 (was `…_but_stays_uncapped`): the account is still
    named in the BODY with `subject=None`, whether the worst group is a loop or
    a crossing — a count and a dollar figure name nothing an operator can act
    on."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    for _ in range(3):
        await _row(uid, amount="0", cost="300", model="gpt-5.5", metadata={
            "denied": True, "reason": "insufficient_message_credits",
        })

    out = await _run()

    assert "served_unbilled_critical" in out["alerts"]
    (_cat, level, msg, subject, _interval) = _unbilled(quiet_alerts)[0]
    assert level == "critical"
    assert subject is None
    assert uid[:8] in msg
    assert "across 3 denied-but-served call(s)" in msg
    # The labelled worst-group clause carries DIMENSIONS only. Its figures are
    # aggregated over the whole window, before the loop/crossing split, so on
    # an account that does both they overstate whichever bucket the label
    # names — see the mixed-account test below.
    assert "3 call(s)" not in msg
    assert "Worst (crossing): " f"{uid[:8]} — reason=" in msg


async def test_the_named_account_is_the_worst_one_not_the_first(quiet_alerts):
    """`max` over the grouped cost, like invariant 5 — otherwise the alert
    names whichever row the planner happened to return first.

    Updated 2026-09-18: $5.30 total clears the dollar bar, so the arm under
    test is the critical one; the old warning arm no longer fires on "any row".
    """
    small = await _mk_user()
    big = await _mk_user()
    await _mk_balance(small, "free")
    await _mk_balance(big, "free")
    await _row(small, amount="0", cost="50", metadata=_DENY)
    for _ in range(4):
        await _row(big, amount="0", cost="120", metadata=_DENY)

    out = await _run()

    assert out["readings"]["served_unbilled_top_user"] == big[:8]
    assert out["readings"]["served_unbilled_groups"] == 2
    msg = _unbilled(quiet_alerts)[0][2]
    assert big[:8] in msg
    assert small[:8] not in msg


async def test_a_partially_settled_loop_reports_what_was_recovered(quiet_alerts):
    """Since the shortfall settlement, `denied` no longer implies `amount = 0`:
    such a row carries the debit the wallet COULD cover. Reading its whole
    provider cost as given away overstates a system that is recovering most of
    it, so the alarm says how much came back.

    Updated 2026-09-18: two rows 30 min apart, because ONE settled row is a
    crossing and no longer pages at all — the recovered figure now has to
    survive into the loop message, which is the only warning left.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=3)
    for i in range(2):
        await _row(uid, amount="-190", cost="200", metadata={
            "denied": True, "reason": "insufficient_message_credits",
            "settled_incurred": True, "credits_quoted": "200.0000",
            "credits_shortfall": "10.0000",
        }, at=t0 + timedelta(minutes=30 * i))

    out = await _run()

    assert out["readings"]["served_unbilled_calls"] == 2
    assert out["readings"]["served_unbilled_recovered_credits"] == pytest.approx(380.0)
    assert out["readings"]["served_unbilled_loop_recovered_credits"] == pytest.approx(380.0)
    msg = _unbilled(quiet_alerts)[0][2]
    assert "380.0 credit(s) recovered" in msg
    # The recovered clause is the widest optional piece of the message, so the
    # phone ceiling is pinned on THIS shape too — the 400 bar has to hold with
    # it present, not only on the shape that omits it.
    assert len(msg) <= 400, f"read on a phone; {len(msg)} chars is too long"
    # And it is still an alarm: the residual really was eaten, twice, 30
    # minutes apart, which is the stop not holding.
    assert out["alerts"] == ["served_unbilled_loop"]


async def test_with_the_settlement_switch_off_the_old_loop_still_pages(quiet_alerts):
    """`credit_settle_incurred_shortfall` is the revert switch, and reverting it
    brings the give-away back: an infeasible settlement debits nothing, the
    wallet freezes above zero and the flat pre-flight admits the next turn.

    The alarm must not depend on the switch — a monitor that only works while
    the fix is on cannot tell you the fix was turned off. So the rows here are
    written by the real `try_charge` with the switch OFF (amount = 0, the
    pre-fix shape) and then spread across 6h07m.
    """
    from sqlalchemy import select

    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import CreditLedger
    from app.services.credit_service import credit_service

    prev_enf = settings.credit_enforcement_enabled
    prev_settle = settings.credit_settle_incurred_shortfall
    settings.credit_enforcement_enabled = True
    settings.credit_settle_incurred_shortfall = False
    try:
        uid = await _mk_user()
        await _mk_balance(uid, "free")
        async with async_session_maker() as db:
            bal = await credit_service.get_or_create_balance(db, uid)
            bal.message_credits_remaining = Decimal("3")
            await db.commit()
            for i in range(2):
                res = await credit_service.try_charge(
                    db, uid, amount=Decimal("40"), bucket="message",
                    event_type="chat_message", model="gpt-5.5",
                    underlying_cost_cents=Decimal("40"),
                    idempotency_key=f"revert-loop-{i}",
                    already_incurred=True,
                )
                assert res.success is False
            await db.commit()

        async with async_session_maker() as db:
            rows = (await db.execute(
                select(CreditLedger).where(CreditLedger.user_id == uid)
            )).scalars().all()
            denied = [r for r in rows if (r.metadata_json or {}).get("denied")]
            assert len(denied) == 2
            assert all(Decimal(r.amount) == 0 for r in denied), (
                "with the switch off the settlement must debit nothing — that "
                "is the give-away this alarm has to keep seeing"
            )
            base = datetime.utcnow() - timedelta(hours=7)
            denied[0].created_at = base
            denied[1].created_at = base + timedelta(hours=6, minutes=7)
            await db.commit()

        out = await _run()
    finally:
        settings.credit_enforcement_enabled = prev_enf
        settings.credit_settle_incurred_shortfall = prev_settle

    assert [a for a in out["alerts"] if a.startswith("served_unbilled")] == [
        "served_unbilled_loop"]
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    assert hits[0][1] == "warning"
    assert hits[0][3] == uid[:8]
    assert "6h07m" in hits[0][2]
    assert "2 time(s)" in hits[0][2]
    # Nothing came back, so the alarm must not invent a recovery line.
    assert "recovered" not in hits[0][2]
    assert out["readings"]["served_unbilled_loop_recovered_credits"] == 0.0


async def test_the_span_is_per_account_not_per_group(quiet_alerts):
    """One account, two model groups, each burst seconds long, six hours
    apart. That is a LOOP and the account must page for it.

    A turn is several (reason, event, model, operation) groups — main model,
    utility model, tools — so a span taken per GROUP asks "did this one model
    keep being served", which no loop has to satisfy. Here each group's own
    rows are 20 s apart: per-group spans are 20 s and 20 s, and the account's
    is 6h00m20s. This test is the difference.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=7)
    for i in range(2):
        await _row(uid, amount="0", cost="60", model="gpt-5.5", metadata=_DENY,
                   at=t0 + timedelta(seconds=20 * i))
    for i in range(2):
        await _row(uid, amount="0", cost="60", model="haiku-utility",
                   metadata=_DENY,
                   at=t0 + timedelta(hours=6, seconds=20 * i))

    out = await _run()

    assert out["alerts"] == ["served_unbilled_loop"]
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1, "one warning for the ACCOUNT, not one per group"
    assert hits[0][3] == uid[:8]
    r = out["readings"]
    assert r["served_unbilled_groups"] == 2
    assert r["served_unbilled_loop_accounts"] == 1
    assert r["served_unbilled_loop_calls"] == 4
    assert r["served_unbilled_loop_worst_span_s"] == 6 * 3600 + 20


# ── invariant 1: a refill ends a burst ───────────────────────────────
#
# first→last across a 24 h window is not the span of a loop. Exhaust → top up
# → exhaust again is TWO bounded crossings hours apart, and a first→last
# reading pages "the refusal is not stopping the work" about a customer who
# has just paid. A false accusation costs more than a missed page, so the
# boundary is the refill (`amount > 0`), and the question becomes "did one
# burst run long with nothing putting credits back".


async def test_a_topped_up_account_that_re_exhausts_is_two_crossings(quiet_alerts):
    """The legitimate shape that first→last would have accused: two bursts of
    two rows, 30 s wide each, four hours apart, with a credit-pack purchase in
    between. Both bursts are crossings; the account pages NOTHING.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=8)
    for i in range(2):                                   # burst 1: exhausted
        await _row(uid, amount="-95", cost="100", metadata=_DENY,
                   at=t0 + timedelta(seconds=30 * i))
    await _refill(uid, t0 + timedelta(hours=4))          # …they paid…
    for i in range(2):                                   # burst 2: again
        await _row(uid, amount="-95", cost="100", metadata=_DENY,
                   at=t0 + timedelta(hours=4, minutes=1, seconds=30 * i))

    out = await _run()

    assert out["alerts"] == []
    assert quiet_alerts == []
    r = out["readings"]
    # first→last is 4h01m30s — over the limit, and not a loop.
    assert r["served_unbilled_loop_accounts"] == 0
    assert r["served_unbilled_crossing_accounts"] == 1
    assert r["served_unbilled_crossings"] == 2, "each burst is its own crossing"
    assert r["served_unbilled_crossing_calls"] == 4
    assert r["served_unbilled_crossing_usd"] == pytest.approx(4.0)


async def test_a_loop_that_contains_a_top_up_still_pages_the_worst_burst(quiet_alerts):
    """The stop failing AFTER a refill is still the stop failing, so a refill
    is a boundary and never an exemption.

    Two bursts: a 10-minute crossing, a top-up, then 20 minutes of being
    served-and-denied. Only the second is a loop, and the message must carry
    THAT burst's numbers — the account total would overstate the loop and the
    operator would go looking for money that is accounted for.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=8)
    for i in range(2):                                   # crossing: $0.20
        await _row(uid, amount="0", cost="10", metadata=_DENY,
                   at=t0 + timedelta(minutes=10 * i))
    await _refill(uid, t0 + timedelta(hours=2))
    for i in range(2):                                   # loop: $0.80 / 20m
        await _row(uid, amount="0", cost="40", metadata=_DENY,
                   at=t0 + timedelta(hours=2, minutes=1 + 20 * i))

    out = await _run()

    assert out["alerts"] == ["served_unbilled_loop"]
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    msg = hits[0][2]
    assert hits[0][3] == uid[:8]
    assert "2 time(s)" in msg
    assert "20m00s" in msg
    assert "$0.80" in msg, "the LOOPING burst's dollars"
    assert "$1.00" not in msg, "not the account's whole-window total"
    assert "no refill in between" in msg
    r = out["readings"]
    assert r["served_unbilled_loop_accounts"] == 1
    assert r["served_unbilled_loop_calls"] == 2
    assert r["served_unbilled_loop_usd"] == pytest.approx(0.80)
    assert r["served_unbilled_loop_worst_span_s"] == 1200
    # Its own quiet burst is still counted — as a crossing, which is what it
    # is — so the loop and crossing readings add back up to the totals.
    assert r["served_unbilled_crossing_accounts"] == 0, "this account loops"
    assert r["served_unbilled_crossings"] == 1
    assert r["served_unbilled_crossing_calls"] == 2
    assert r["served_unbilled_crossing_usd"] == pytest.approx(0.20)
    assert r["served_unbilled_calls"] == 4
    assert (r["served_unbilled_loop_calls"] + r["served_unbilled_crossing_calls"]
            == r["served_unbilled_calls"])
    assert (r["served_unbilled_loop_usd"] + r["served_unbilled_crossing_usd"]
            == pytest.approx(r["served_unbilled_usd"]))


async def test_two_loops_on_one_account_page_once_naming_the_worst(quiet_alerts):
    """An account can loop, be topped up, and loop again. That is ONE account
    to chase, so it gets ONE warning — the worst burst by provider cost.

    Two messages about one account is the noise this round removes; the second
    would also be suppressed by the 6 h window and arrive a redeploy later.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=9)
    for i in range(3):                                   # $0.60 over 40m
        await _row(uid, amount="0", cost="20", metadata=_DENY,
                   at=t0 + timedelta(minutes=20 * i))
    await _refill(uid, t0 + timedelta(hours=3))
    for i in range(3):                                   # $3.00 over 1h
        await _row(uid, amount="0", cost="100", metadata=_DENY,
                   at=t0 + timedelta(hours=3, minutes=1 + 30 * i))

    out = await _run()

    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    msg = hits[0][2]
    assert out["alerts"] == ["served_unbilled_loop"]
    assert "1h00m" in msg and "$3.00" in msg
    assert "40m00s" not in msg
    r = out["readings"]
    assert r["served_unbilled_loop_accounts"] == 1
    assert r["served_unbilled_loop_calls"] == 6, "both bursts are loops"
    assert r["served_unbilled_loop_usd"] == pytest.approx(3.60)
    assert r["served_unbilled_loop_worst_span_s"] == 3600


async def test_the_critical_says_whether_the_worst_account_loops(quiet_alerts):
    """The mixed window, which is the one an operator will actually get: one
    looping account and one crossing account, over the dollar bar.

    "Worst: 3bf753ea — 4 call(s), $6.00" is the same sentence whether that
    account is stuck or has simply had an expensive day, and only one of those
    is worth opening the runbook for.
    """
    looper = await _mk_user()
    crosser = await _mk_user()
    for u in (looper, crosser):
        await _mk_balance(u, "free")
    t0 = datetime.utcnow() - timedelta(hours=6)
    for i in range(2):                                   # $6.00 over 2h
        await _row(looper, amount="0", cost="300", metadata=_DENY,
                   at=t0 + timedelta(hours=2 * i))
    for i in range(2):                                   # $1.00 in 20s
        await _row(crosser, amount="0", cost="50", metadata=_DENY,
                   at=t0 + timedelta(seconds=20 * i))

    out = await _run()

    assert "served_unbilled_critical" in out["alerts"]
    assert "served_unbilled_loop" in out["alerts"]
    crit = [h for h in _unbilled(quiet_alerts) if h[1] == "critical"]
    assert len(crit) == 1
    msg = crit[0][2]
    assert f"Worst (loop): {looper[:8]}" in msg
    assert "1 looping burst(s) ($6.00) across 1 account(s)" in msg
    assert "1 crossing burst(s) ($1.00)" in msg


async def test_a_daily_reset_ends_a_burst_like_any_other_refill(quiet_alerts):
    """The day roll restores spendable headroom while writing ``amount = 0``,
    so `amount > 0` alone cannot see it — and two cap-denied bursts either
    side of a day roll were filed as ONE loop, with the warning asserting
    "with no refill in between" about a user whose cap had just re-opened.
    That is the false accusation the segmentation exists to prevent.

    TWO accounts, identical rows, so what is under test is the BOUNDARY and
    not the fixture: only the second has the reset row between its bursts.
    """
    reset = await _mk_user()
    no_reset = await _mk_user()
    for u in (reset, no_reset):
        await _mk_balance(u, "free")
    _CAP = {"denied": True, "reason": "daily_cap_exceeded"}
    t0 = datetime.utcnow() - timedelta(hours=8)
    for u in (reset, no_reset):
        for i in range(2):                               # burst 1: capped
            await _row(u, amount="0", cost="40", metadata=_CAP,
                       at=t0 + timedelta(seconds=30 * i))
        for i in range(2):                               # burst 2: capped again
            await _row(u, amount="0", cost="40", metadata=_CAP,
                       at=t0 + timedelta(hours=4, minutes=1, seconds=30 * i))
    await _daily_reset(reset, t0 + timedelta(hours=4))   # …the day rolled…

    out = await _run()

    # One page, for the account whose cap never re-opened.
    hits = _unbilled(quiet_alerts)
    assert len(hits) == 1
    assert hits[0][3] == no_reset[:8]
    assert reset[:8] not in hits[0][2], "its cap re-opened; it is two crossings"
    assert out["alerts"] == ["served_unbilled_loop"]
    r = out["readings"]
    assert r["served_unbilled_loop_accounts"] == 1
    assert r["served_unbilled_loop_calls"] == 4, "no boundary → one long burst"
    assert r["served_unbilled_crossing_accounts"] == 1
    assert r["served_unbilled_crossings"] == 2, "the reset split them in two"
    assert r["served_unbilled_crossing_calls"] == 4


async def test_one_account_that_loops_and_crosses_reports_both_in_burst_scope(
        quiet_alerts):
    """The mixed SINGLE account, which is where an account count paired with a
    burst dollar total renders a zero that carries money.

    A 20-second crossing, a top-up, then an hour of being served-and-denied.
    The account is a looping account, so its quiet burst contributed
    `crossing_usd` while adding nothing to `crossing_accounts` — and the
    critical body said "0 crossing account(s) ($0.60)", which reads as a
    formatting bug in a critical page. Both halves are bursts now.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    t0 = datetime.utcnow() - timedelta(hours=9)
    for i in range(2):                                   # crossing: $0.60
        await _row(uid, amount="0", cost="30", metadata=_DENY,
                   at=t0 + timedelta(seconds=20 * i))
    await _refill(uid, t0 + timedelta(hours=2))
    for i in range(3):                                   # loop: $5.40 / 1h00m
        await _row(uid, amount="0", cost="180", metadata=_DENY,
                   at=t0 + timedelta(hours=2, minutes=1 + 30 * i))

    out = await _run()

    assert "served_unbilled_critical" in out["alerts"]
    crit = [h for h in _unbilled(quiet_alerts) if h[1] == "critical"]
    assert len(crit) == 1
    msg = crit[0][2]
    assert ("1 looping burst(s) ($5.40) across 1 account(s), and 1 crossing "
            "burst(s) ($0.60)") in msg
    assert "account(s) ($" not in msg, (
        "an account count must never carry a burst dollar total — that is how "
        "a zero came to be quoted with money beside it"
    )
    # The labelled clause keeps the label and the account, and carries the
    # DIMENSIONS only: the worst group is aggregated before the split, so its
    # $6.00 / 5 calls are the account's whole window and would overstate the
    # loop in the very message that reports the loop total as $5.40.
    assert f"Worst (loop): {uid[:8]} — reason=" in msg
    assert msg.count("$6.00") == 1, "only the fleet total, not the worst group"
    assert "5 call(s)" not in msg
    r = out["readings"]
    assert r["served_unbilled_loop_usd"] == pytest.approx(5.40)
    assert r["served_unbilled_crossing_usd"] == pytest.approx(0.60)
    assert r["served_unbilled_crossing_accounts"] == 0, "this account loops"


# ── the readings themselves ──────────────────────────────────────────


async def test_a_clean_window_fires_nothing(quiet_alerts):
    """The whole point of tight thresholds is that steady state is silence."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    # A break-even row: 1 credit = 1¢ by design, so 250 cents of provider
    # cost against 250 charged credits is ratio 1.0.
    await _row(uid, amount="-250", cost="250", metadata=None)
    out = await _run()
    assert out["readings"]["charge_ratio"] == pytest.approx(1.0)
    assert out["alerts"] == []
    assert quiet_alerts == []


async def test_grandfathering_a_refused_customer_does_not_page(quiet_alerts):
    """The alarm must not fire on its own fix.

    Parmida has been refused since 2026-09-11. The grandfather sets her
    `credit_balances.plan_id` to 'unlimited'. Grouping denials by the CURRENT
    balance then re-attributes every one of yesterday's refusals to
    'unlimited' and pages `credit-denied-unlimited` CRITICAL — "the user is
    locked out right now" — about the customer the operator has just
    unblocked, for the rest of the window, on cutover day, for every account
    in it. An alarm whose whole value is being true on a single occurrence
    cannot also be the loudest thing that happens when nothing is wrong.
    """
    uid = await _mk_user()
    await _mk_balance(uid, "builder")
    for _ in range(3):
        await _row(uid, amount="0", cost="240", metadata={
            "denied": True, "reason": "insufficient_message_credits",
            "plan_id": "builder",
        })
    # …and now the grandfather runs.
    await _set_plan(uid, "unlimited")

    out = await _run()

    assert "denied_unlimited" not in out["alerts"], (
        "grandfathering a refused customer paged the refused-unlimited alarm"
    )
    assert out["readings"]["denials_by_plan"] == {"builder": 3}


async def test_a_refusal_recorded_while_unlimited_still_pages(quiet_alerts):
    """Anti-vacuity for the test above: the alarm is about WHEN, not about
    being harder to fire. A row stamped 'unlimited' still pages on one."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")          # moved off since — irrelevant
    await _row(uid, amount="0", cost="300", metadata={
        "denied": True, "reason": "insufficient_message_credits",
        "plan_id": "unlimited",
    })
    out = await _run()
    assert "denied_unlimited" in out["alerts"]
    assert out["readings"]["denials_by_plan"]["unlimited"] == 1


async def test_try_charge_stamps_the_plan_on_every_denial():
    """The monitor's grouping is only as good as the stamp. A denial written
    without `plan_id` falls back to the current balance, which is the
    behaviour being removed."""
    from decimal import Decimal

    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import CreditLedger
    from app.services.credit_service import credit_service
    from sqlalchemy import select

    prev = settings.credit_enforcement_enabled
    settings.credit_enforcement_enabled = True
    try:
        uid = await _mk_user()
        await _mk_balance(uid, "free")
        async with async_session_maker() as db:
            bal = await credit_service.get_or_create_balance(db, uid)
            bal.message_credits_remaining = Decimal("0")
            await db.commit()
            res = await credit_service.try_charge(
                db, uid, amount=Decimal("5"), bucket="message",
                event_type="chat_message", underlying_cost_cents=Decimal("240"),
            )
            await db.commit()
        assert res.success is False

        async with async_session_maker() as db:
            rows = (await db.execute(
                select(CreditLedger).where(CreditLedger.user_id == uid)
            )).scalars().all()
        denied = [r for r in rows if (r.metadata_json or {}).get("denied")]
        assert denied, "no denial row was written"
        assert denied[0].metadata_json.get("plan_id") == "free"
    finally:
        settings.credit_enforcement_enabled = prev
