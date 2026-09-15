"""Proof by execution that the free tier is untouched (design §8, brief D).

Every branch this round added is gated, and each gate is checked here against
a real free user in a database that ALSO contains the `unlimited` plan row and
a live unlimited account — because "inert in isolation" is not the claim. The
claim is that a free user's behaviour is identical while the entitlement is
running beside them.

The branches, and the gate each one hangs on:

======================================================  ==========================================
branch                                                  why a free user never takes it
======================================================  ==========================================
`check_balance` rate ladder                             `unlimited_throttle_enabled` is False, and
                                                        with it True `admission_verdict` answers
                                                        TIER_OK for any user with no counter entry
`try_charge` → `unlimited_abuse.observe`                inside `if is_unlimited:`
`credit_health_monitor` `_not_unlimited` exclusions     always-true for a row with no marker
`credit_health_monitor` invariants 5/6/7                keyed on markers/plan a free row lacks
`credit_exhausted` REASON_RATE_LIMITED branch           a free denial never carries that reason
`connector_dispatcher` rate-limited copy                same
`credit_reporter` unlimited/TTL latch guards            plan_id is 'free'; the TTL arm is the one
                                                        deliberate behaviour change, tested below
`ws_realtime` plan_source anti-steering                 free accounts are not 'iap'
======================================================  ==========================================
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


async def _mk_user() -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"f-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Free Test",
        ))
        await db.commit()
    return uid


async def _balance(uid):
    from app.db import CreditBalance, async_session_maker
    async with async_session_maker() as db:
        return await db.get(CreditBalance, uid)


async def _set_plan(uid, plan_id, *, message=None):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        b = await credit_service.get_or_create_balance(db, uid)
        b.plan_id = plan_id
        if message is not None:
            b.message_credits_remaining = Decimal(message)
        await db.commit()


async def _charge(uid, amount="10", **kw):
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    prev = (settings.credit_enforcement_enabled,
            settings.credit_cap_admission_control)
    settings.credit_enforcement_enabled = True
    settings.credit_cap_admission_control = True
    try:
        async with async_session_maker() as db:
            res = await credit_service.try_charge(
                db, uid, event_type="chat_message", bucket="message",
                amount=Decimal(amount), underlying_cost_cents=Decimal(amount),
                **kw,
            )
            await db.commit()
            return res
    finally:
        settings.credit_enforcement_enabled = prev[0]
        settings.credit_cap_admission_control = prev[1]


async def _preflight(uid, required="0.1", bucket="message"):
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    prev = settings.credit_enforcement_enabled
    settings.credit_enforcement_enabled = True
    try:
        async with async_session_maker() as db:
            return await credit_service.check_balance(
                db, uid, bucket, Decimal(required),
            )
    finally:
        settings.credit_enforcement_enabled = prev


@pytest.fixture(autouse=True)
def _clean_abuse():
    from app.services import unlimited_abuse
    unlimited_abuse.reset_for_tests()
    yield
    unlimited_abuse.reset_for_tests()


@pytest.fixture
async def neighbour():
    """A live unlimited account in the same database, so nothing below is
    tested against a world where the entitlement does not exist."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    uid = await _mk_user()
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, uid)
        # The REAL materialisation path, not a hand-set column: it is what
        # assigns BOTH wallets, and the integration one is what the connector
        # pre-flight below reads.
        await credit_service.apply_plan_change(
            db, uid, "unlimited", reason="test:grandfather",
        )
        await db.commit()
    await _charge(uid, "500")
    return uid


# ── the charge gate ──────────────────────────────────────────────────


async def test_free_user_is_charged_normally(neighbour):
    uid = await _mk_user()
    await _set_plan(uid, "free", message="100")
    res = await _charge(uid, "10")
    assert res.success is True
    bal = await _balance(uid)
    assert Decimal(bal.message_credits_remaining) == Decimal("90")
    assert Decimal(bal.message_credits_used_today) == Decimal("10")


async def test_free_user_is_still_refused_at_zero(neighbour):
    """The wall is the free tier's whole shape. Nothing here softens it."""
    uid = await _mk_user()
    await _set_plan(uid, "free", message="1")
    res = await _charge(uid, "50")
    assert res.success is False
    assert res.reason == "insufficient_message_credits"
    # …and NOT the new reason, which would have suppressed the CTA and told
    # a free user out of credits to "try again in a few seconds".
    assert res.reason != "rate_limited"


async def test_free_denial_still_carries_the_upgrade_path(neighbour):
    from app.services.credit_exhausted import build_exhausted_response

    resp = build_exhausted_response(
        reason="insufficient_message_credits", bucket="message",
        balance_after=Decimal("0"), plan_id="free", plan_display_name="Free",
        period_end=datetime.now(timezone.utc) + timedelta(days=10),
    )
    assert resp.cta_hidden is False
    assert resp.cta_url == "/pricing"
    assert "/pricing" in resp.message


async def test_the_rate_ladder_never_sees_a_free_user(neighbour):
    """`observe` is called only from try_charge's unlimited branch, so the
    counters cannot contain a free user — which is what makes
    `admission_verdict`'s TIER_OK-for-unknown answer a proof and not a
    coincidence."""
    from app.services import unlimited_abuse

    uid = await _mk_user()
    await _set_plan(uid, "free", message="100")
    for _ in range(5):
        await _charge(uid, "1")
    assert uid not in unlimited_abuse._users
    assert neighbour in unlimited_abuse._users, "the unlimited neighbour IS tracked"


async def test_free_preflight_is_identical_with_the_throttle_forced_on(
    neighbour, monkeypatch,
):
    """The gate that could conceivably reach a free user, with its flag
    flipped ON: still TIER_OK, still the same verdict, still no delay."""
    from app.config import settings
    from app.services import unlimited_abuse

    monkeypatch.setattr(settings, "unlimited_throttle_enabled", True)
    uid = await _mk_user()
    await _set_plan(uid, "free", message="100")
    assert unlimited_abuse.admission_verdict(uid) == (unlimited_abuse.TIER_OK, 0.0)
    ok = await _preflight(uid, "0.1")
    assert ok.success is True and ok.reason is None
    await _set_plan(uid, "free", message="0")
    broke = await _preflight(uid, "50")
    assert broke.success is False
    assert broke.reason == "insufficient_message_credits"


async def test_free_ledger_rows_carry_no_unlimited_marker(neighbour):
    """Invariant 6 alarms on a zero-amount row with no marker, so a free
    row must never grow one by accident — and a free CHARGE row is never
    zero-amount in the first place."""
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import CreditLedger

    uid = await _mk_user()
    await _set_plan(uid, "free", message="100")
    await _charge(uid, "10")
    async with async_session_maker() as db:
        row = (await db.execute(
            select(CreditLedger).where(CreditLedger.user_id == uid)
            .order_by(CreditLedger.created_at.desc())
        )).scalars().first()
    assert Decimal(row.amount) == Decimal("-10")
    assert not (row.metadata_json or {}).get("unlimited")
    assert not (row.metadata_json or {}).get("admin_unlimited")


# ── the connector pre-flight ─────────────────────────────────────────


async def test_connector_preflight_never_refuses_unlimited(neighbour):
    """Design §8 site 8: `check_balance` is entitlement-blind, and that is
    CORRECT rather than merely lucky — a million remaining means no pre-flight
    in the system can refuse an unlimited account, whether or not it knows the
    concept exists. This pins it for the integration bucket, whose copy is the
    one that says "upgrade your plan"."""
    res = await _preflight(neighbour, "5000", bucket="integration")
    assert res.success is True


async def test_connector_preflight_still_refuses_a_broke_free_user(neighbour):
    uid = await _mk_user()
    await _set_plan(uid, "free")
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        b = await credit_service.get_or_create_balance(db, uid)
        b.integration_credits_remaining = Decimal("0")
        await db.commit()
    res = await _preflight(uid, "5", bucket="integration")
    assert res.success is False
    assert res.reason == "insufficient_integration_credits"


# ── the agent's in-process exhausted latch ───────────────────────────


def _latched(plan_id="free", age_s=0.0):
    """A CreditState latched exhausted `age_s` ago on `plan_id`."""
    from app.services.credit_reporter import CreditState, DeductOutcome

    st = CreditState()
    st.record_deduct(
        DeductOutcome(network_ok=True, success=False, enforcement_enabled=True,
                      balance_after=0.0, reason="insufficient_message_credits"),
        plan_id=plan_id, plan_display_name=plan_id.title(),
        period_end=datetime.now(timezone.utc) + timedelta(days=10),
    )
    if age_s:
        st.last_updated_at = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    return st


async def test_a_fresh_free_latch_still_blocks():
    """The unchanged case, and the one that matters most: a free user who
    just hit zero stays blocked."""
    st = _latched("free", age_s=0)
    assert st.is_exhausted() is True
    assert st.build_exhausted_response() is not None


async def test_an_unlimited_latch_never_blocks():
    """The live case this fixes. Parmida's agent container held an exhausted
    latch from before the grandfather; nothing in that process could clear it
    (the latch is set by a deduct RESPONSE, and the deduct only happens after
    an LLM call, which the latch is blocking). She would have kept seeing an
    upgrade card on an unlimited plan until the container restarted."""
    st = _latched("unlimited", age_s=0)
    assert st.is_exhausted() is False
    assert st.build_exhausted_response() is None, (
        "ws_chat's layer-2 fallback calls this directly — the guard has to "
        "live in BOTH or the card comes back by that route"
    )


async def test_a_stale_free_latch_still_blocks(monkeypatch):
    """Staleness is not an expiry.

    For one revision it was, and that let a full LLM call through per
    exhausted account per TTL — 4/hour, 96/day, because `record_deduct`
    re-arms the timer on each let-through. For a FREE user that is provider
    spend billed to nobody plus a `denied: true` ledger row carrying real
    `underlying_cost_cents`: the exact denied-but-served shape credit-health
    invariant 1 pages on. The refusal has to stand until a platform read says
    otherwise.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 900)
    fresh, stale = _latched("free", age_s=100), _latched("free", age_s=1000)
    assert fresh.is_exhausted() is True
    assert stale.is_exhausted() is True
    # …but the stale one is KNOWN to be stale, which is what lets the async
    # gate go and ask.
    assert fresh.latch_is_stale() is False
    assert stale.latch_is_stale() is True


async def test_the_ttl_can_be_switched_off(monkeypatch):
    """0 disables staleness — nothing ever re-reads. The rollback lever."""
    from app.config import settings

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 0)
    st = _latched("free", age_s=10_000)
    assert st.is_exhausted() is True
    assert st.latch_is_stale() is False


async def test_a_stale_latch_is_reopened_by_a_PLATFORM_read_not_a_paid_call(
    monkeypatch,
):
    """The reopener costs one authenticated GET, never provider spend.

    This is the mechanism that unblocks a container whose account was
    grandfathered after its latch closed. Guard 1 alone cannot: it reads
    `last_known_plan_id`, which is whatever the last deduct said, and no
    deduct can happen while the latch is shut.
    """
    from app.config import settings
    from app.services import credit_reporter

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 900)
    st = _latched("builder", age_s=1000)
    st.last_user_id = "u-1"
    monkeypatch.setattr(credit_reporter, "_state", st)
    monkeypatch.setattr(credit_reporter, "_platform_endpoint",
                        lambda path: "https://platform.test" + path)
    monkeypatch.setattr(credit_reporter, "_agent_key", lambda: "k")

    calls = []

    async def fake_remote(*, user_id, bucket="message", required=0.5):
        calls.append(user_id)
        # What the platform now says: grandfathered onto Unlimited.
        st.record_deduct(
            credit_reporter.DeductOutcome(
                network_ok=True, success=True, enforcement_enabled=True,
                balance_after=1_000_000.0,
            ),
            plan_id="unlimited", plan_display_name="Unlimited", user_id=user_id,
        )
        return None

    monkeypatch.setattr(credit_reporter, "check_balance_remote", fake_remote)

    await credit_reporter.raise_if_exhausted_async()      # must NOT raise
    assert calls == ["u-1"], "the stale latch was not re-read"
    assert st.is_exhausted() is False


async def test_an_unreachable_platform_keeps_a_stale_latch_shut(monkeypatch):
    """Fail CLOSED. Letting a paid call through on exactly the evidence we
    just failed to obtain is the behaviour being removed, not preserved."""
    from app.config import settings
    from app.services import credit_reporter
    from app.services.credit_exhausted import OutOfCreditsError

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 900)
    st = _latched("free", age_s=1000)
    st.last_user_id = "u-2"
    monkeypatch.setattr(credit_reporter, "_state", st)
    monkeypatch.setattr(credit_reporter, "_platform_endpoint",
                        lambda path: "https://platform.test" + path)
    monkeypatch.setattr(credit_reporter, "_agent_key", lambda: "k")

    async def unreachable(*, user_id, bucket="message", required=0.5):
        return None                      # what check_balance_remote does

    monkeypatch.setattr(credit_reporter, "check_balance_remote", unreachable)

    with pytest.raises(OutOfCreditsError):
        await credit_reporter.raise_if_exhausted_async()


async def test_a_healthy_state_is_never_exhausted():
    from app.services.credit_reporter import CreditState, DeductOutcome

    st = CreditState()
    st.record_deduct(DeductOutcome(network_ok=True, success=True,
                                   enforcement_enabled=True, balance_after=90.0),
                     plan_id="free")
    assert st.is_exhausted() is False
    # A network failure is "no information", not "blocked" — fail-open.
    st2 = CreditState()
    st2.record_deduct(DeductOutcome(network_ok=False), plan_id="free")
    assert st2.is_exhausted() is False


async def test_a_scheduled_routine_still_SKIPS_a_broke_free_user(monkeypatch):
    """The clean-skip gate, end to end, through the real handler.

    A scheduled routine's interval always exceeds the latch TTL, so this gate
    is ALWAYS looking at a stale latch. An expiry that simply opened it made
    the gate unreachable for every plan, free included: the handler ran on,
    the Gmail/tool call debited the INTEGRATION bucket the free user still
    had, the LLM step was then refused 402, and the run ended 'failed' with no
    'insufficient_credits' class instead of 'skipped'.
    """
    from app.config import settings
    from app.services import credit_reporter
    from app.agent.routines.agent_task_handler import AgentTaskHandler

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 900)
    st = _latched("free", age_s=86_400)          # a day old: a daily routine
    st.last_user_id = "u-3"
    monkeypatch.setattr(credit_reporter, "_state", st)
    monkeypatch.setattr(credit_reporter, "_platform_endpoint",
                        lambda path: "https://platform.test" + path)
    monkeypatch.setattr(credit_reporter, "_agent_key", lambda: "k")

    async def still_broke(*, user_id, bucket="message", required=0.5):
        st.record_deduct(
            credit_reporter.DeductOutcome(
                network_ok=True, success=False, enforcement_enabled=True,
                balance_after=0.0, reason="insufficient_message_credits",
            ),
            plan_id="free", plan_display_name="Free", user_id=user_id,
        )
        return None

    monkeypatch.setattr(credit_reporter, "check_balance_remote", still_broke)

    ran = []

    class _Runner:
        async def run(self, *a, **kw):       # pragma: no cover - must not run
            ran.append(1)
            raise AssertionError("the handler ran past its own credit gate")

    handler = AgentTaskHandler(agent_runner=_Runner())
    routine = SimpleNamespace(
        id="r1", user_id="u-3", name="daily", prompt_text="do the thing",
        config_json={}, metadata_json={},
    )
    result = await handler.execute(routine, SimpleNamespace(id="run1"), None)

    assert result.status == "skipped", result
    assert result.error_class == "insufficient_credits"
    assert not ran


async def test_a_down_platform_is_asked_once_per_ttl_not_once_per_turn(monkeypatch):
    """A credit gate must not become a retry storm against the thing it
    depends on.

    A successful read re-arms `last_updated_at` through `record_deduct` and so
    un-stales the latch on its own. A read that returns None — unreachable, or
    the platform not configured — does not, so without a back-off every gate
    call in a hot chat loop fires another GET at a platform that is already
    down.
    """
    from app.config import settings
    from app.services import credit_reporter
    from app.services.credit_exhausted import OutOfCreditsError

    monkeypatch.setattr(settings, "credit_exhausted_latch_ttl_s", 900)
    monkeypatch.setattr(credit_reporter, "_last_refresh_at", float("-inf"))
    st = _latched("free", age_s=5000)
    st.last_user_id = "u-9"
    monkeypatch.setattr(credit_reporter, "_state", st)
    monkeypatch.setattr(credit_reporter, "_platform_endpoint",
                        lambda path: "https://platform.test" + path)
    monkeypatch.setattr(credit_reporter, "_agent_key", lambda: "k")

    attempts = []

    async def down(*, user_id, bucket="message", required=0.5):
        attempts.append(user_id)
        return None                       # what check_balance_remote does

    monkeypatch.setattr(credit_reporter, "check_balance_remote", down)

    for _ in range(25):
        with pytest.raises(OutOfCreditsError):
            await credit_reporter.raise_if_exhausted_async()

    assert len(attempts) == 1, f"asked the platform {len(attempts)} times"
