"""The UNLIMITED rate ladder (design §7): log → alert → pace → refuse.

Two properties matter more than any individual threshold, and both are tested
by execution rather than by reading the numbers:

* **A human cannot reach a tier.** The heaviest real day on record is replayed
  through the module and must produce total silence. If a bar can be reached
  by a person using the product, it is not a bar, it is a bug.
* **A refusal never speaks billing.** The account this can fire on is one that
  is never denied and never charged, so a price, a plan name, a credit count
  or an upgrade link would each be a lie — delivered at the worst possible
  moment, to the only people who pay us.

The ladder also ships OFF, and "off" has to mean off: nothing slowed, nothing
refused, but the shadow series still recorded so the false-positive rate is
measurable BEFORE the flag is ever flipped.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Fresh in-process counters and no Telegram, for every test.

    The counters are module state by design (they must survive a request) and
    `send_infra_alert` rate-limits at module scope, so without this a passing
    test could be one that inherited another's suppression window.
    """
    from app.services import unlimited_abuse

    unlimited_abuse.reset_for_tests()
    sent: list[tuple] = []

    async def _fake(category, level, message, *, subject=None, min_interval_s=600):
        sent.append((category, level, message, subject))
        return True

    monkeypatch.setattr(unlimited_abuse, "send_infra_alert", _fake)
    emitted: list[tuple] = []
    monkeypatch.setattr(
        unlimited_abuse.abuse_metrics, "emit",
        lambda event, **f: emitted.append((event, f)),
    )
    unlimited_abuse._sent_for_tests = sent          # type: ignore[attr-defined]
    unlimited_abuse._emitted_for_tests = emitted    # type: ignore[attr-defined]
    yield
    unlimited_abuse.reset_for_tests()


def _sent():
    from app.services import unlimited_abuse
    return unlimited_abuse._sent_for_tests          # type: ignore[attr-defined]


def _emitted():
    from app.services import unlimited_abuse
    return unlimited_abuse._emitted_for_tests       # type: ignore[attr-defined]


def _uid() -> str:
    return str(uuid.uuid4())


# ── the human bar ────────────────────────────────────────────────────


async def test_human_scale_traffic_is_completely_silent():
    """Replay the heaviest recorded real usage: 140 chat messages at the
    measured mean of 1.975 credits, all inside one hour — which is already a
    fortnight of that user's real activity compressed into 60 minutes.

    No notice, no alert, no shadow event, no pace. If this ever fails, the
    bars are wrong and must be RAISED; a person using the app by hand must
    never be able to reach any tier.
    """
    from app.services import unlimited_abuse

    uid = _uid()
    for _ in range(140):
        tier = await unlimited_abuse.observe(uid, Decimal("1.975"))
        assert tier == unlimited_abuse.TIER_OK
    assert _sent() == []
    assert _emitted() == []
    assert unlimited_abuse.admission_verdict(uid) == (unlimited_abuse.TIER_OK, 0.0)


async def test_the_heaviest_single_turn_ever_recorded_is_nothing():
    """23 credits is the largest single turn in the ledger's history. Even
    a hundred of them back to back stays under the notice bar."""
    from app.services import unlimited_abuse

    uid = _uid()
    for _ in range(60):
        assert await unlimited_abuse.observe(uid, Decimal("23")) == unlimited_abuse.TIER_OK
    assert _sent() == []


# ── the ladder ───────────────────────────────────────────────────────


async def test_notice_is_log_only():
    """The first rung tells nobody and blocks nothing."""
    from app.services import unlimited_abuse

    uid = _uid()
    tier = None
    for _ in range(300):
        tier = await unlimited_abuse.observe(uid, Decimal("0.1"))
    assert tier == unlimited_abuse.TIER_NOTICE
    assert _sent() == [], "notice must not page anyone"
    assert any(e[0] == "unlimited_throttle_would" for e in _emitted())


async def test_alert_pages_the_operator_and_still_blocks_nothing():
    from app.services import unlimited_abuse

    uid = _uid()
    for _ in range(900):
        tier = await unlimited_abuse.observe(uid, Decimal("0.1"))
    assert tier == unlimited_abuse.TIER_PACE
    cats = [s[0] for s in _sent()]
    assert "unlimited-abuse" in cats
    # Subject-scoped: one heavy account must not suppress an alert about
    # another (alerting.py keys its window on (category, subject)).
    assert _sent()[0][3] == uid[:8]
    # Throttle is OFF by default, so nothing is actually slowed.
    assert unlimited_abuse.admission_verdict(uid) == (unlimited_abuse.TIER_OK, 0.0)


async def test_credits_bar_fires_independently_of_the_call_bar():
    """Ten very expensive turns beat 900 cheap ones to the alert tier —
    the ladder is about spend as much as about rate."""
    from app.services import unlimited_abuse

    uid = _uid()
    for _ in range(10):
        tier = await unlimited_abuse.observe(uid, Decimal("600"))   # 6,000 > 5,000
    assert tier == unlimited_abuse.TIER_PACE
    assert "unlimited-abuse" in [s[0] for s in _sent()]


# ── ships OFF ────────────────────────────────────────────────────────


async def test_disabled_by_default_emits_shadow_and_refuses_nothing():
    """A 3,000-call hour — past every bar including refuse — blocks nothing
    while the flag is off, and records the tier it WOULD have applied."""
    from app.config import settings
    from app.services import unlimited_abuse

    assert settings.unlimited_throttle_enabled is False, "must ship OFF"
    uid = _uid()
    for _ in range(3000):
        await unlimited_abuse.observe(uid, Decimal("0.1"))

    shadow = [e for e in _emitted() if e[0] == "unlimited_throttle_would"]
    assert shadow, "shadow series is what makes the flag flippable"
    assert shadow[-1][1]["tier"] == unlimited_abuse.TIER_REFUSE
    assert shadow[-1][1]["enforcing"] is False
    # Nothing real, and no delay.
    assert not [e for e in _emitted() if e[0] == "unlimited_usage_notice"]
    assert unlimited_abuse.admission_verdict(uid) == (unlimited_abuse.TIER_OK, 0.0)
    assert await unlimited_abuse.apply_admission(uid) is True


# ── enabled: pace before refuse, always ──────────────────────────────


async def test_pace_comes_first_and_refuse_is_the_last_resort(monkeypatch):
    """Past the REFUSE bar the account is still only PACED until the grace
    is exhausted. A delay is recoverable and self-limiting; a denial is not,
    and that ordering is what makes "last resort" true rather than a label."""
    from app.config import settings
    from app.services import unlimited_abuse

    monkeypatch.setattr(settings, "unlimited_throttle_enabled", True)
    monkeypatch.setattr(settings, "unlimited_throttle_grace_turns", 5)
    uid = _uid()
    for _ in range(2000):
        await unlimited_abuse.observe(uid, Decimal("0.1"))

    tier, delay = unlimited_abuse.admission_verdict(uid)
    assert tier == unlimited_abuse.TIER_PACE
    assert 0 < delay <= settings.unlimited_pace_max_delay_s

    # Past the grace the ladder finally refuses.
    for _ in range(10):
        await unlimited_abuse.observe(uid, Decimal("0.1"))
    tier, delay = unlimited_abuse.admission_verdict(uid)
    assert tier == unlimited_abuse.TIER_REFUSE
    assert delay == 0.0
    assert await unlimited_abuse.apply_admission(uid) is False
    crit = [s for s in _sent() if s[0] == "unlimited-abuse-refusal"]
    assert crit and crit[0][1] == "critical"


async def test_pace_delay_is_capped(monkeypatch):
    from app.config import settings
    from app.services import unlimited_abuse

    monkeypatch.setattr(settings, "unlimited_pace_max_delay_s", 20.0)
    assert unlimited_abuse.pace_delay_s(unlimited_abuse.TIER_PACE, 10_000) == 20.0
    assert unlimited_abuse.pace_delay_s(unlimited_abuse.TIER_NOTICE, 10_000) == 0.0
    assert unlimited_abuse.pace_delay_s(unlimited_abuse.TIER_OK, 10_000) == 0.0


async def test_a_user_the_module_never_saw_is_always_ok(monkeypatch):
    """The inertness proof for every non-unlimited account.

    `observe` is called ONLY from try_charge's unlimited branch, so a free,
    legacy-paid or admin user never gets an entry — and `admission_verdict`
    answers TIER_OK for anyone with no entry, even with the flag ON.
    """
    from app.config import settings
    from app.services import unlimited_abuse

    monkeypatch.setattr(settings, "unlimited_throttle_enabled", True)
    assert unlimited_abuse.admission_verdict("never-seen") == (
        unlimited_abuse.TIER_OK, 0.0,
    )
    assert await unlimited_abuse.apply_admission("never-seen") is True


async def test_counters_are_bounded():
    """An unbounded per-user dict on a hot path is a leak wearing a
    counter's clothes."""
    from app.services import unlimited_abuse

    for i in range(unlimited_abuse._MAX_TRACKED_USERS + 50):
        await unlimited_abuse.observe(f"u{i}", Decimal("0.1"))
    assert len(unlimited_abuse._users) == unlimited_abuse._MAX_TRACKED_USERS


async def test_the_hour_window_actually_rolls(monkeypatch):
    """A tripwire that never forgets is a tripwire that fires on everyone,
    eventually."""
    import time as _time
    from app.services import unlimited_abuse

    uid = _uid()
    base = _time.time()
    monkeypatch.setattr(unlimited_abuse.time, "time", lambda: base)
    for _ in range(400):
        await unlimited_abuse.observe(uid, Decimal("0.1"))
    assert len(unlimited_abuse._users[uid].calls) == 400

    # Two hours later those calls are not evidence of anything.
    monkeypatch.setattr(unlimited_abuse.time, "time", lambda: base + 7200)
    assert await unlimited_abuse.observe(uid, Decimal("0.1")) == unlimited_abuse.TIER_OK
    assert len(unlimited_abuse._users[uid].calls) == 1


async def test_observe_never_raises_on_a_junk_amount():
    """It runs inside try_charge, after the flush. A measurement that can
    break a charge is worse than no measurement."""
    from app.services import unlimited_abuse

    uid = _uid()
    assert await unlimited_abuse.observe(uid, None) == unlimited_abuse.TIER_OK  # type: ignore[arg-type]
    assert await unlimited_abuse.observe(uid, "not-a-number") == unlimited_abuse.TIER_OK  # type: ignore[arg-type]


async def test_the_window_dies_with_the_entitlement(monkeypatch):
    """`credit_service.check_balance`'s comment says "no free, legacy-paid or
    admin account can ever have an entry". That was true only at WRITE time.

    Nothing evicted before the 24h cutoff, so a lapsed or revoked account kept
    its counters — and with the throttle enabled its next `check_balance` could
    still be paced or refused with REASON_RATE_LIMITED off traffic it ran while
    it was unlimited, before any balance was consulted. On the shipped iOS
    client that refusal renders as a paywall.
    """
    from decimal import Decimal as _D

    from app.config import settings
    from app.db import async_session_maker
    from app.services import unlimited_abuse
    from app.services.credit_service import credit_service

    monkeypatch.setattr(settings, "unlimited_throttle_enabled", True, raising=False)

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        from app.db.models import User
        db.add(User(id=uid, email=f"{uid[:8]}@example.com", hashed_password="x"))
        await db.commit()
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, uid)
        await credit_service.activate_subscription(
            db, uid, "unlimited", "apple",
            datetime.utcnow() + timedelta(days=30),
        )
        await db.commit()

    for _ in range(60):
        await unlimited_abuse.observe(uid, _D("400"))
    assert unlimited_abuse.admission_verdict(uid)[0] != unlimited_abuse.TIER_OK, (
        "the fixture did not reach a throttled tier — nothing is under test"
    )

    async with async_session_maker() as db:
        await credit_service.downgrade_to_free(db, uid, "apple:expired")
        await db.commit()

    tier, delay = unlimited_abuse.admission_verdict(uid)
    assert tier == unlimited_abuse.TIER_OK, tier
    assert delay == 0
