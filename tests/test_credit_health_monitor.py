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
):
    """One ledger row, placed inside the monitor's window."""
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
            metadata_json=metadata,
            created_at=datetime.utcnow() - timedelta(hours=age_h),
        ))
        await db.commit()


@pytest_asyncio.fixture
async def quiet_alerts(monkeypatch):
    """Capture alerts instead of posting them, and assert on the capture.

    `send_infra_alert` is rate-limited per (category, subject) at MODULE
    scope, so a real call in one test would silence the same alert in the
    next. Capturing is not only about the network.
    """
    sent: list[tuple] = []

    async def _fake(category, level, message, *, subject=None, min_interval_s=600):
        sent.append((category, level, message, subject))
        return True

    import app.services.credit_health_monitor as mon
    monkeypatch.setattr(mon, "send_infra_alert", _fake)
    return sent


async def _run():
    from app.services.credit_health_monitor import check_credit_health
    return await check_credit_health()


def _cats(sent) -> set[str]:
    return {c for (c, _l, _m, _s) in sent}


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
    """Invariant 1 is untouched. This is the 2026-08-03 alarm — 274 calls
    denied and served anyway — and nothing in this change may soften it."""
    uid = await _mk_user()
    await _mk_balance(uid, "free")
    for _ in range(3):
        await _row(uid, amount="0", cost="300", metadata={
            "denied": True, "reason": "daily_cap_exceeded",
        })
    out = await _run()
    assert "served_unbilled_critical" in out["alerts"]   # $9 ≥ $5
    assert out["readings"]["served_unbilled_calls"] == 3


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
