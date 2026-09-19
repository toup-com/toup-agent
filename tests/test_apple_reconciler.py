"""The Apple reconciliation sweep, and the alert on the notification we lose.

Design §5 and §5.6. What is under test is not "does it call Apple" — it is the
three properties the loop is worthless without:

1. **It reconciles against ``expires_date``, never ``status``.** The mirror's
   status column is the thing being doubted: the Sandbox Elite has read
   ``status='active'`` for 2.7 months past its 2026-06-24 expiry with
   ``last_notification_uuid`` NULL. Every classification test here drives the
   two apart on purpose so a regression that starts trusting ``status`` fails.
   The one place status IS read is the opposite direction — a mirror that
   records a REVOKE must not have the plan handed back by clock alone — and
   that is tested too.
2. **It is idempotent.** A converged database produces a pass with zero
   writes and zero Apple calls, because every correction is a convergence to
   Apple's answer rather than a delta.
3. **Sandbox never moves a production balance.** A sandbox subscription's
   lifetime is minutes; letting Apple's test environment decide a real
   ``credit_balances`` row trades a visible drift for an invisible one. The
   drift is REPORTED every pass instead, and the opt-in is proven to be the
   only thing standing between the two behaviours.

Plus §5.6: the endpoint that acks 200 on any handler exception must now say
so out loud, because Apple never retries and, before this loop existed,
nothing in the system ever looked at that subscription again.

Apple's library is not installed in this lane (and needs signing keys it must
never have here), so ``fetch_subscription_status`` is patched with the exact
dataclass it returns. The seam is deliberate: everything below the patch is
the real reconciler, the real credit_service and the real database.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


LEGACY_BUILDER = "ai.toup.app.sub.builder"
LEGACY_ELITE = "ai.toup.app.sub.elite"
UNLIMITED_PRODUCT = "ai.toup.app.sub.unlimited"


# ── helpers ──────────────────────────────────────────────────────────


async def _mk_user(email: str | None = None, *, role: str | None = None) -> str:
    from app.db import User, async_session_maker
    from app.services.auth_service import get_password_hash

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        u = User(
            id=uid,
            email=email or f"r-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Reconcile Test",
        )
        if role:
            u.role = role
        db.add(u)
        await db.commit()
    return uid


async def _mk_balance(user_id: str, *, plan_id: str | None = None,
                      plan_source: str | None = None):
    """A balance, optionally already on a paid plan through the real path."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, user_id)
        if plan_id:
            await credit_service.activate_subscription(
                db, user_id, plan_id, plan_source or "apple",
                datetime.utcnow() + timedelta(days=30),
            )
        await db.commit()


async def _mk_sub(user_id: str, *, txn: str | None = None,
                  product: str = LEGACY_BUILDER, plan_id: str = "builder",
                  status: str = "active", days: float = 20,
                  environment: str = "Production",
                  grandfathered: bool = False,
                  updated_days_ago: float = 0) -> str:
    from app.db import async_session_maker
    from app.db.models import AppleSubscription

    txn = txn or f"txn-{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow()
    async with async_session_maker() as db:
        sub = AppleSubscription(
            user_id=user_id, original_transaction_id=txn, product_id=product,
            plan_id=plan_id, status=status,
            expires_date=now + timedelta(days=days),
            auto_renew_status=True, environment=environment,
        )
        if grandfathered:
            sub.grandfathered_at = now
            sub.grandfathered_product_id = product
        db.add(sub)
        await db.flush()
        if updated_days_ago:
            sub.updated_at = now - timedelta(days=updated_days_ago)
        await db.commit()
    return txn


async def _sub_row(txn: str):
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    async with async_session_maker() as db:
        return (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()


async def _balance(user_id: str):
    from app.db import CreditBalance, async_session_maker
    async with async_session_maker() as db:
        return await db.get(CreditBalance, user_id)


async def _ledger_count(user_id: str) -> int:
    from sqlalchemy import func, select
    from app.db import CreditLedger, async_session_maker
    async with async_session_maker() as db:
        return int((await db.execute(
            select(func.count(CreditLedger.id))
            .where(CreditLedger.user_id == user_id)
        )).scalar() or 0)


def _apple(status: int, *, product=None, expires=None, auto_renew=None,
           arp=None, txn="?", environment="Production"):
    """Exactly the dataclass fetch_subscription_status returns."""
    from app.services.apple_iap_service import AppleSubscriptionStatus
    return AppleSubscriptionStatus(
        original_transaction_id=txn, environment=environment, status=status,
        product_id=product, expires_date=expires,
        auto_renew_status=auto_renew, auto_renew_product_id=arp,
    )


@pytest.fixture
def apple_says(monkeypatch):
    """Install Apple's answers, keyed by original_transaction_id.

    Returns a recorder so a test can assert HOW MANY calls a pass made —
    "the second pass issues zero Apple calls" is the idempotency claim and it
    is only checkable from here.
    """
    from app.services import apple_iap_service

    state: dict[str, object] = {}
    calls: list[tuple[str, str]] = []

    async def fake(original_transaction_id: str, environment_hint: str):
        calls.append((original_transaction_id, environment_hint))
        answer = state.get(original_transaction_id, "__missing__")
        if answer == "__missing__":
            return None
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(apple_iap_service, "fetch_subscription_status", fake)
    return SimpleNamespace(set=state.__setitem__, calls=calls)


@pytest.fixture
def reconcile_settings():
    """Deterministic reconciler settings, restored after the test."""
    from app.config import settings

    names = (
        "apple_reconcile_apply", "apple_reconcile_max_api_calls",
        "apple_reconcile_stale_days", "apple_reconcile_scan_sandbox",
        "apple_reconcile_sandbox_apply",
    )
    prev = {n: getattr(settings, n) for n in names}
    settings.apple_reconcile_apply = True
    settings.apple_reconcile_max_api_calls = 100
    settings.apple_reconcile_stale_days = 40
    settings.apple_reconcile_scan_sandbox = True
    settings.apple_reconcile_sandbox_apply = False
    yield settings
    for n, v in prev.items():
        setattr(settings, n, v)


async def _run(**kwargs):
    """One pass in its own session, committed — the shape the loop uses."""
    from app.db import async_session_maker
    from app.services.apple_reconciler import reconcile_once

    async with async_session_maker() as db:
        result = await reconcile_once(db, **kwargs)
        await db.commit()
    return result


def _classes(result):
    return result.classes


def _alert_categories(result):
    return [a[0] for a in result.alerts]


# ── 1. expires_date is the oracle, status is the suspect ─────────────


async def test_overdue_is_classified_from_expires_date_not_status(
    reconcile_settings, apple_says,
):
    """The Sandbox Elite's exact shape, in Production: status says active,
    Apple's own clock says it ended months ago."""
    from app.services.apple_reconciler import CLASS_OVERDUE, scan
    from app.db import async_session_maker

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="builder")
    txn = await _mk_sub(uid, status="active", days=-90)

    async with async_session_maker() as db:
        findings = await scan(db)

    assert [f.cls for f in findings] == [CLASS_OVERDUE]
    assert findings[0].original_txn == txn
    # The classification must name the DATE, not the status, as its evidence.
    assert "expires_date" in findings[0].detail


async def test_resurrected_is_classified_when_status_says_dead(
    reconcile_settings, apple_says,
):
    """The other direction: the mirror recorded a terminal event, the clock
    disagrees. Neither is acted on without asking Apple."""
    from app.services.apple_reconciler import CLASS_RESURRECTED, scan
    from app.db import async_session_maker

    uid = await _mk_user()
    await _mk_balance(uid)
    await _mk_sub(uid, status="expired", days=+20)

    async with async_session_maker() as db:
        findings = await scan(db)

    assert [f.cls for f in findings] == [CLASS_RESURRECTED]


async def test_a_revoked_row_with_a_future_expiry_is_never_re_entitled(
    reconcile_settings, apple_says,
):
    """The one place `status` is allowed to speak, and the reason it must be.

    A REVOKE (family-sharing removal, a refund) ends the entitlement while
    leaving expires_date in the FUTURE. Reading the clock alone would make
    this loop hand back the plan the notification handler correctly took away
    — so a terminal mirror status keeps the row out of the
    missing-entitlement class, and Apple is asked instead.
    """
    from app.services.apple_reconciler import (
        APPLE_STATUS_REVOKED, CLASS_MISSING, CLASS_RESURRECTED,
    )

    uid = await _mk_user()
    await _mk_balance(uid)  # already downgraded to free by the REVOKE
    txn = await _mk_sub(uid, status="revoked", days=+20)
    apple_says.set(txn, _apple(APPLE_STATUS_REVOKED, product=LEGACY_BUILDER, txn=txn))

    result = await _run()

    assert CLASS_MISSING not in _classes(result)
    assert _classes(result) == {CLASS_RESURRECTED: 1}
    assert (await _balance(uid)).plan_id == "free"


# ── 2. the corrections, both directions ──────────────────────────────


async def test_missed_expired_notification_downgrades_and_pages(
    reconcile_settings, apple_says,
):
    """The failure this loop exists for: Apple ended the subscription, the
    notification never arrived, the account kept an entitlement it stopped
    paying for."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, product=LEGACY_BUILDER, status="active", days=-3,
                        grandfathered=True)
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER,
        expires=datetime.utcnow() - timedelta(days=3), txn=txn,
    ))

    result = await _run()

    bal = await _balance(uid)
    assert bal.plan_id == "free"
    assert bal.plan_source is None
    assert float(bal.message_credits_remaining) == 100.0
    # The mirror converged too, and to Apple's status — not to a guess.
    assert (await _sub_row(txn)).status == "expired"
    assert "apple-reconcile-downgrade" in _alert_categories(result)
    crit = [a for a in result.alerts if a[0] == "apple-reconcile-downgrade"]
    assert crit[0][1] == "critical"


async def test_missed_renewal_restores_the_entitlement(
    reconcile_settings, apple_says,
):
    """DID_RENEW has never executed in production. When it does not arrive,
    Apple still knows — and a paying customer must not stay locked out."""
    from app.services.apple_reconciler import APPLE_STATUS_ACTIVE

    uid = await _mk_user()
    await _mk_balance(uid)  # free: the lapse was processed, the renewal was not
    txn = await _mk_sub(uid, product=LEGACY_BUILDER, status="active", days=-2)
    new_expiry = datetime.utcnow() + timedelta(days=28)
    apple_says.set(txn, _apple(
        APPLE_STATUS_ACTIVE, product=LEGACY_BUILDER, expires=new_expiry, txn=txn,
    ))

    result = await _run()

    bal = await _balance(uid)
    assert bal.plan_id == "builder"
    assert bal.plan_source == "apple"
    row = await _sub_row(txn)
    assert row.status == "active"
    assert row.expires_date == new_expiry
    assert "apple-reconcile-drift" in _alert_categories(result)


async def test_a_live_payer_who_is_not_entitled_pages_critical(
    reconcile_settings, apple_says,
):
    """The Parmida class — paying right now, locked out right now. Needs no
    Apple call: a Production row whose own expiry is in the future is already
    the strongest statement the database can make."""
    from app.services.apple_reconciler import CLASS_MISSING

    uid = await _mk_user()
    await _mk_balance(uid)  # free
    txn = await _mk_sub(uid, product=LEGACY_BUILDER, status="active", days=+18)

    result = await _run()

    assert _classes(result) == {CLASS_MISSING: 1}
    assert apple_says.calls == []          # answered from the mirror alone
    bal = await _balance(uid)
    assert bal.plan_id == "builder"
    assert bal.plan_source == "apple"
    assert "apple-reconcile-missing-entitlement" in _alert_categories(result)
    assert [a[1] for a in result.alerts
            if a[0] == "apple-reconcile-missing-entitlement"] == ["critical"]


async def test_a_grandfathered_live_payer_is_entitled_to_unlimited(
    reconcile_settings, apple_says,
):
    """The resolver, not the product map, decides — so the sweep restores
    Unlimited for a grandfathered legacy subscription rather than Builder."""
    uid = await _mk_user()
    await _mk_balance(uid)
    await _mk_sub(uid, product=LEGACY_BUILDER, status="active", days=+18,
                  grandfathered=True)

    await _run()

    bal = await _balance(uid)
    assert bal.plan_id == "unlimited"
    assert float(bal.message_credits_remaining) == 1_000_000.0
    assert bal.plan_source == "apple"


async def test_a_stripe_sourced_plan_is_left_to_its_own_source(
    reconcile_settings, apple_says,
):
    """An Apple subscription ending is not evidence about a Stripe plan.
    Downgrading one here would be this loop inventing a lapse nobody
    reported."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="builder", plan_source="stripe")
    txn = await _mk_sub(uid, status="active", days=-5)
    apple_says.set(txn, _apple(APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER, txn=txn))

    result = await _run()

    assert (await _balance(uid)).plan_id == "builder"
    assert "apple-reconcile-downgrade" not in _alert_categories(result)


async def test_billing_retry_does_not_downgrade(reconcile_settings, apple_says):
    """Apple status 3 means the card failed and Apple has NOT ended the
    subscription. _SUB_DOWNGRADE_TYPES deliberately waits for EXPIRED; a
    reconciler that read 'not active' as 'dead' would cut off customers the
    notification path is deliberately keeping."""
    from app.services.apple_reconciler import APPLE_STATUS_BILLING_RETRY

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="builder")
    txn = await _mk_sub(uid, status="active", days=-2)
    apple_says.set(txn, _apple(
        APPLE_STATUS_BILLING_RETRY, product=LEGACY_BUILDER, txn=txn,
    ))

    result = await _run()

    assert (await _balance(uid)).plan_id == "builder"
    assert (await _sub_row(txn)).status == "billing_retry"
    assert "apple-reconcile-downgrade" not in _alert_categories(result)


# ── 3. idempotency ───────────────────────────────────────────────────


async def test_second_pass_writes_nothing_and_asks_apple_nothing(
    reconcile_settings, apple_says,
):
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=-3, grandfathered=True)
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER,
        expires=datetime.utcnow() - timedelta(days=3), txn=txn,
    ))

    first = await _run()
    assert first.applied >= 1
    ledger_after_first = await _ledger_count(uid)
    calls_after_first = len(apple_says.calls)

    second = await _run()

    assert second.findings == []
    assert second.applied == 0
    assert len(apple_says.calls) == calls_after_first   # zero new Apple calls
    assert await _ledger_count(uid) == ledger_after_first


async def test_observe_only_writes_absolutely_nothing(apple_says):
    """apple_reconcile_apply=False is the mode this ships in. A fully drifted
    fixture must come out byte-identical, with every action recorded."""
    from app.config import settings
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    prev = settings.apple_reconcile_apply
    settings.apple_reconcile_apply = False
    try:
        uid = await _mk_user()
        await _mk_balance(uid, plan_id="unlimited")
        txn = await _mk_sub(uid, status="active", days=-3, grandfathered=True)
        apple_says.set(txn, _apple(
            APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER,
            expires=datetime.utcnow() - timedelta(days=3), txn=txn,
        ))
        before_ledger = await _ledger_count(uid)

        result = await _run()   # apply comes from the setting

        assert result.apply is False
        assert result.would >= 1
        assert result.applied == 0
        bal = await _balance(uid)
        assert bal.plan_id == "unlimited"
        assert (await _sub_row(txn)).status == "active"
        assert await _ledger_count(uid) == before_ledger
        # …and it SAID what it would have done, or the mode is useless.
        acted = [a for f in result.findings for a in f.actions]
        assert any(a.startswith("downgrade ") for a in acted), acted
        assert any("observe-only" in a[2] for a in result.alerts)
    finally:
        settings.apple_reconcile_apply = prev


# ── 4. Sandbox never moves a production balance ──────────────────────


async def test_sandbox_drift_is_reported_but_never_applied(
    reconcile_settings, apple_says,
):
    """The Sandbox Elite, exactly: environment=Sandbox, status='active',
    expired 2026-06-24, and the balance still on the paid plan."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED, CLASS_OVERDUE

    uid = await _mk_user("b5fm6n85mt@privaterelay.appleid.com")
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    result = await _run()

    assert CLASS_OVERDUE in _classes(result)
    # Seen, reported, NOT touched.
    assert (await _balance(uid)).plan_id == "elite"
    assert (await _sub_row(txn)).status == "active"
    sandbox_notes = [a for f in result.findings for a in f.actions
                     if a == "sandbox:not_applied"]
    assert sandbox_notes, [f.actions for f in result.findings]
    assert any("Sandbox" in a[2] for a in result.alerts)
    # …and the summary must not claim it changed something it did not.
    assert result.applied == 0
    assert result.would >= 1
    # A CRITICAL "had to downgrade a payer" for a downgrade that deliberately
    # did not happen is an alarm describing an action rather than a fact.
    assert "apple-reconcile-downgrade" not in _alert_categories(result)


async def test_sandbox_drift_in_observe_only_warns_and_never_pages_critical(
    reconcile_settings, apple_says,
):
    """The same fixture in the mode production actually runs — which had no
    coverage at all, and was the one mode the fold above did not work in.

    `apple_reconcile_apply` is False in production, so `_apply_finding` never
    reaches the `apply and not may` branch that writes the
    "sandbox:not_applied" marker. The fold keyed on that marker was therefore
    unreachable, and every 6 h pass took the `apple-reconcile-downgrade`
    CRITICAL arm for the Sandbox Elite — paging that the reconciler "had to
    downgrade" a row it is structurally forbidden to touch. The fold must key
    on the FACT (this environment is not mutatable), which holds in both
    modes.
    """
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED, CLASS_OVERDUE

    reconcile_settings.apple_reconcile_apply = False

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    result = await _run()      # apply comes from the setting, as in production

    assert result.apply is False
    assert CLASS_OVERDUE in _classes(result)
    assert (await _balance(uid)).plan_id == "elite"
    assert (await _sub_row(txn)).status == "active"
    assert result.applied == 0 and result.would >= 1

    assert "apple-reconcile-downgrade" not in _alert_categories(result)
    assert [a[1] for a in result.alerts] == ["warning"], result.alerts
    body = result.alerts[0][2]
    assert "Sandbox" in body and txn in body
    # Nothing was written, so no body may assert a completed action.
    assert not any("had to downgrade" in a[2] for a in result.alerts)
    # …and in observe-only the opt-in flag alone would not act either, so the
    # remedy must name both switches rather than send an operator to flip one
    # and watch nothing happen.
    assert "apple_reconcile_apply=true" in body
    assert "apple_reconcile_sandbox_apply=true" in body


async def test_the_aggregate_warning_names_the_opt_in_in_apply_mode(
    reconcile_settings, apple_says,
):
    """Apply mode is unchanged: one aggregate warning, no critical, and it
    names the single flag that WOULD let the loop act."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    result = await _run()

    assert [a[1] for a in result.alerts] == ["warning"], result.alerts
    assert "Set apple_reconcile_sandbox_apply=true to" in result.alerts[0][2]


async def test_a_sandbox_stale_row_folds_too_and_is_still_named(
    reconcile_settings, apple_says,
):
    """Every Sandbox class folds, not only the ones proposing a downgrade.

    A Sandbox `stale` finding raised `apple-reconcile-stale` in observe-only
    before the fold was keyed on mutability. It folds like the rest now — and
    the aggregate still names the transaction, so the operator loses the
    second category and no signal.
    """
    from app.services.apple_reconciler import CLASS_STALE

    reconcile_settings.apple_reconcile_apply = False

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=+20, updated_days_ago=60,
                        environment="Sandbox")
    apple_says.set(txn, None)      # Apple does not know it either

    result = await _run()

    assert CLASS_STALE in _classes(result)
    assert _alert_categories(result) == ["apple-reconcile-drift"], result.alerts
    assert txn in result.alerts[0][2]


async def test_the_aggregate_does_not_blame_the_policy_for_an_unread_row(
    reconcile_settings, apple_says,
):
    """The fold is now the only reporter of a Sandbox row, so its one sentence
    must not assert a cause it does not know.

    This row was not corrected because Apple returned no answer for it — the
    sandbox policy never got as far as refusing anything. The aggregate used
    to say "REPORTED, not corrected — a sandbox subscription must never move a
    production balance. Set …=true to let this loop clean them": the wrong
    reason, plus a remedy that cannot change the outcome. (Apple answering
    nothing for one transaction is not an outage, so no
    `apple-reconcile-unreachable` fires here to contradict it — which is
    exactly why this body has to be right on its own.)
    """
    reconcile_settings.apple_reconcile_apply = False

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=+20, updated_days_ago=60,
                        environment="Sandbox")
    apple_says.set(txn, None)      # Apple has no answer for this transaction

    result = await _run()

    assert _alert_categories(result) == ["apple-reconcile-drift"], result.alerts
    body = result.alerts[0][2]
    assert txn in body
    assert "could not be evaluated" in body, body
    assert "must never move a production balance" not in body, body
    # A flag that cannot change this row's outcome must not be offered for it.
    assert "apple_reconcile_sandbox_apply" not in body, body


async def test_a_sandbox_row_apple_agrees_with_is_not_called_refused_either(
    reconcile_settings, apple_says,
):
    """The third state: Apple answered, and there was nothing to correct.

    `_entitle` returns without appending an action when the balance already
    holds the target plan, so this row carries no action for the sandbox
    policy to have refused. Calling it "refused" would invent a correction,
    and calling it "could not be evaluated" would contradict the Apple read
    this pass paid for.
    """
    from app.services.apple_reconciler import APPLE_STATUS_ACTIVE, CLASS_STALE

    reconcile_settings.apple_reconcile_apply = False

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=+20, updated_days_ago=60,
                        environment="Sandbox")
    # Every field Apple leaves None is a field `_converge_mirror` skips, so
    # this is Apple agreeing with the mirror without pinning its clock.
    apple_says.set(txn, _apple(
        APPLE_STATUS_ACTIVE, product=LEGACY_ELITE, txn=txn,
        environment="Sandbox",
    ))

    result = await _run()

    assert CLASS_STALE in _classes(result)
    assert [f.actions for f in result.findings] == [[]], result.findings
    assert _alert_categories(result) == ["apple-reconcile-drift"], result.alerts
    body = result.alerts[0][2]
    assert txn in body
    assert "produced no correction to make" in body, body
    assert "must never move a production balance" not in body, body
    assert "could not be evaluated" not in body, body


async def test_the_aggregate_keeps_the_two_causes_on_their_own_rows(
    reconcile_settings, apple_says,
):
    """One pass, both states, one warning — and neither cause on the other's
    transaction. A single flat list cannot say this."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    reconcile_settings.apple_reconcile_apply = False

    uid_a = await _mk_user()
    await _mk_balance(uid_a, plan_id="elite")
    refused_txn = await _mk_sub(uid_a, product=LEGACY_ELITE, plan_id="elite",
                                status="active", days=-80,
                                environment="Sandbox")
    apple_says.set(refused_txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=refused_txn,
        environment="Sandbox",
    ))

    uid_b = await _mk_user()
    await _mk_balance(uid_b, plan_id="elite")
    unread_txn = await _mk_sub(uid_b, product=LEGACY_ELITE, plan_id="elite",
                               status="active", days=+20, updated_days_ago=60,
                               environment="Sandbox")
    apple_says.set(unread_txn, None)

    result = await _run()

    assert _alert_categories(result) == ["apple-reconcile-drift"], result.alerts
    body = result.alerts[0][2]
    assert "2 drifted Sandbox row(s)" in body, body
    # The policy sentence and the opt-in belong to the row a correction was
    # actually determined for; the unread row is named under its own cause.
    split = body.index("could not be evaluated")
    assert body.index(refused_txn) < split, body
    assert body.index(unread_txn) > split, body
    assert body.index("must never move a production balance") < split, body
    assert body.index("apple_reconcile_apply=true and "
                      "apple_reconcile_sandbox_apply=true") < split, body


async def test_the_fold_keys_on_mutability_not_on_the_word_sandbox(
    reconcile_settings, apple_says,
):
    """Anti-vacuity: the fold must RELEASE the moment an operator opts in.

    A fold keyed on `environment != 'Production'` would look identical on
    every test above and would silence the page for a downgrade this loop
    really did write.
    """
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    reconcile_settings.apple_reconcile_sandbox_apply = True

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    result = await _run()

    assert (await _balance(uid)).plan_id == "free"
    crit = [a for a in result.alerts if a[0] == "apple-reconcile-downgrade"]
    assert crit and crit[0][1] == "critical"
    assert "had to downgrade a Sandbox payer" in crit[0][2]


async def test_an_observe_only_alert_is_written_in_the_conditional(
    reconcile_settings, apple_says,
):
    """A Production row keeps its CRITICAL in observe-only — but the body may
    not assert an action nothing took.

    The shipped wording was "The reconciler had to downgrade a Production
    payer … Actions: downgrade 'unlimited'→'free' … [observe-only: nothing was
    written]": a completed action in the first clause, denied by a tag at the
    end. An operator who has to read an alert twice to learn whether anything
    happened is an operator who stops reading it.
    """
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    reconcile_settings.apple_reconcile_apply = False

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=-3, grandfathered=True)
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER,
        expires=datetime.utcnow() - timedelta(days=3), txn=txn,
    ))

    result = await _run()

    assert (await _balance(uid)).plan_id == "unlimited"     # nothing written
    crit = [a for a in result.alerts if a[0] == "apple-reconcile-downgrade"]
    assert crit and crit[0][1] == "critical", result.alerts
    body = crit[0][2]
    assert "had to downgrade" not in body
    assert "would have to downgrade" in body
    assert "nothing was written" in body
    assert "Proposed actions" in body
    assert "Actions:" not in body


async def test_the_sandbox_opt_in_is_the_only_thing_stopping_it(
    reconcile_settings, apple_says,
):
    """Mutation-style proof that the guard is load-bearing rather than
    incidental: flip the one setting and the same fixture converges."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    reconcile_settings.apple_reconcile_sandbox_apply = True

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    await _run()

    assert (await _balance(uid)).plan_id == "free"
    assert (await _sub_row(txn)).status == "expired"


async def test_sandbox_scan_can_be_switched_off_entirely(
    reconcile_settings, apple_says,
):
    reconcile_settings.apple_reconcile_scan_sandbox = False
    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                  status="active", days=-80, environment="Sandbox")

    result = await _run()

    assert result.findings == []


# ── 4b. the Sandbox aggregate's CADENCE ──────────────────────────────


class _FakeClock:
    """A clock the test advances by whole reconciler intervals.

    `send_infra_alert`'s window is real wall-clock arithmetic and two passes in
    one test are milliseconds apart, so without this every second send would be
    suppressed by the 600 s default and the 24 h claim would be vacuous — it
    would pass identically with no per-alert interval at all.
    """

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = float(start)

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


# `apple_reconcile_interval_s`'s default: what one pass actually costs in time.
RECONCILE_INTERVAL_S = 21600


@pytest.fixture
def telegram(monkeypatch):
    """Drive the REAL `send_infra_alert` with only the Telegram POST stubbed
    (the way `tests/test_alerting.py` does it), on a clock the test controls.

    The point is to execute `alerting.py`'s actual (category, subject) window
    rather than re-implement it: "delivered once" is a claim about that window,
    and a hand-rolled model of it could not be wrong in the way the bug was.
    """
    from app.services import alerting

    monkeypatch.setattr(alerting.settings, "infra_alert_telegram_token", "T",
                        raising=False)
    monkeypatch.setattr(alerting.settings, "infra_alert_telegram_chat_id", "C",
                        raising=False)
    clock = _FakeClock()
    monkeypatch.setattr(alerting, "time", clock)

    posts: list[dict] = []

    class _Client:
        def __init__(self, *a, **k):
            ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            posts.append(kw.get("json", {}))
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(alerting.httpx, "AsyncClient", _Client)
    alerting.reset_for_tests()
    yield SimpleNamespace(posts=posts, clock=clock)
    alerting.reset_for_tests()


def _delivered(telegram, marker: str) -> list[str]:
    return [p.get("text", "") for p in telegram.posts if marker in p.get("text", "")]


async def test_the_sandbox_aggregate_pages_at_most_once_a_day(
    reconcile_settings, apple_says, telegram,
):
    """Two passes, one message.

    The Sandbox Elite's condition is STATIC — the row has read
    `status='active'` since 2026-06-24 and this loop is forbidden to write it —
    so every 6 h pass re-derives a byte-identical aggregate. At `alerting.py`'s
    600 s default that is four identical Telegram messages a day, forever,
    about a TestFlight row nobody needs to act on within hours.
    """
    from app.services.apple_reconciler import (
        APPLE_STATUS_EXPIRED, SANDBOX_AGGREGATE_MIN_INTERVAL_S, _flush_alerts,
    )

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="elite")
    txn = await _mk_sub(uid, product=LEGACY_ELITE, plan_id="elite",
                        status="active", days=-80, environment="Sandbox")
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_ELITE, txn=txn, environment="Sandbox",
    ))

    first = await _run()
    await _flush_alerts(first)
    telegram.clock.advance(RECONCILE_INTERVAL_S)
    second = await _run()
    await _flush_alerts(second)

    # Nothing was written either pass, so the second pass really does raise the
    # same aggregate — the suppression is the only reason it is not delivered.
    assert _alert_categories(first) == ["apple-reconcile-drift"], first.alerts
    assert _alert_categories(second) == ["apple-reconcile-drift"], second.alerts

    sent = _delivered(telegram, "[apple-reconcile-drift sandbox]")
    assert len(sent) == 1, telegram.posts
    assert "drifted Sandbox row(s) were REPORTED" in sent[0]

    # …and it is the per-alert interval that did it, not a lucky window.
    assert [a[4] for a in first.alerts] == [SANDBOX_AGGREGATE_MIN_INTERVAL_S]
    assert [a[4] for a in second.alerts] == [SANDBOX_AGGREGATE_MIN_INTERVAL_S]
    assert SANDBOX_AGGREGATE_MIN_INTERVAL_S == 86400


async def test_a_production_downgrade_keeps_its_every_pass_cadence(
    reconcile_settings, apple_says, telegram,
):
    """The 24 h window belongs to the Sandbox aggregate ALONE.

    A Production payer this loop would have to downgrade is a dropped App Store
    notification — the failure the reconciler exists to catch — so it is queued
    with NO widened interval and two passes deliver two pages. Anti-vacuity for
    the test above: a module-wide widening would look identical there and would
    mute this.
    """
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED, _flush_alerts

    reconcile_settings.apple_reconcile_apply = False      # the mode production runs

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=-3, grandfathered=True)
    apple_says.set(txn, _apple(
        APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER,
        expires=datetime.utcnow() - timedelta(days=3), txn=txn,
    ))

    first = await _run()
    await _flush_alerts(first)
    telegram.clock.advance(RECONCILE_INTERVAL_S)
    second = await _run()
    await _flush_alerts(second)

    for r in (first, second):
        crit = [a for a in r.alerts if a[0] == "apple-reconcile-downgrade"]
        assert crit and crit[0][1] == "critical", r.alerts
        # Queued with no interval of its own: the cadence stays alerting.py's.
        assert crit[0][4] is None, crit

    assert len(_delivered(telegram, "[apple-reconcile-downgrade")) == 2, telegram.posts


# ── 5. budget, reachability, and the operator's proof of life ────────


async def test_the_api_budget_is_bounded_and_oldest_first(
    reconcile_settings, apple_says,
):
    """A cap that starved the oldest rows would let one permanently-broken
    subscription hide behind newer ones forever."""
    reconcile_settings.apple_reconcile_max_api_calls = 2

    txns = []
    for age in (30, 10, 20):
        uid = await _mk_user()
        await _mk_balance(uid, plan_id="builder")
        txns.append((age, await _mk_sub(
            uid, status="active", days=-5, updated_days_ago=age,
        )))

    result = await _run()

    assert result.api_deferred == 1
    asked = [c[0] for c in apple_says.calls]
    oldest_two = [t for _, t in sorted(txns, reverse=True)[:2]]
    assert sorted(asked) == sorted(oldest_two)


async def test_unreachable_apple_changes_nothing_and_says_so(
    reconcile_settings, apple_says,
):
    """"Every call failed" is a different fact from "nothing to fix", and a
    reconciler that cannot tell them apart reports health it never checked."""
    from app.services.apple_iap_service import IapVerificationError

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=-3, grandfathered=True)
    apple_says.set(txn, IapVerificationError("App Store Server API call failed"))

    result = await _run()

    assert result.api_failures == 1
    assert (await _balance(uid)).plan_id == "unlimited"   # untouched
    assert "apple-reconcile-unreachable" in _alert_categories(result)


async def test_stale_rows_warn(reconcile_settings, apple_says):
    from app.services.apple_reconciler import CLASS_STALE

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="builder")
    txn = await _mk_sub(uid, status="active", days=+20, updated_days_ago=60)
    apple_says.set(txn, None)      # Apple does not know it either

    result = await _run()

    assert CLASS_STALE in _classes(result)
    assert "apple-reconcile-stale" in _alert_categories(result)


async def test_the_pass_records_its_own_heartbeat(reconcile_settings, apple_says):
    """A `ran_at` that stopped moving is the failure a monitor cannot infer
    from the absence of alerts."""
    from app.db import async_session_maker
    from app.db.models import PlatformSetting
    from app.services.apple_reconciler import (
        LAST_RUN_SETTING_KEY, _store_summary, reconcile_once,
    )

    uid = await _mk_user()
    await _mk_balance(uid)
    await _mk_sub(uid, status="active", days=+18)

    async with async_session_maker() as db:
        result = await reconcile_once(db)
        await _store_summary(db, result)
        await db.commit()

    async with async_session_maker() as db:
        row = await db.get(PlatformSetting, LAST_RUN_SETTING_KEY)
    assert row is not None
    blob = json.loads(row.value)
    assert blob["apply"] is True
    assert blob["classes"] == {"missing_entitlement": 1}
    assert "ran_at" in blob


# ── 6. grants and cross-grades ───────────────────────────────────────


async def test_a_downgrade_lapses_the_grants_that_subscription_sponsored(
    reconcile_settings, apple_says,
):
    """The notification handler does this in the same transaction. When the
    notification never arrives, the sweep must do it too — otherwise a comp
    outlives the subscription paying for it, silently and forever."""
    from app.services import entitlement
    from app.db import async_session_maker
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    sponsor = await _mk_user("sponsor@example.com")
    grantee = await _mk_user("second@example.com")
    await _mk_balance(sponsor, plan_id="unlimited")
    await _mk_balance(grantee)
    # The grant is created while the sponsor is LIVE — create_grant refuses a
    # dead sponsor by design — and only then does the subscription lapse
    # without anybody telling us. That ordering is the scenario.
    txn = await _mk_sub(sponsor, status="active", days=+20, grandfathered=True)

    async with async_session_maker() as db:
        grant = await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )
        gid = grant.id
        await db.commit()
    assert (await _balance(grantee)).plan_id == "unlimited"

    from app.db.models import AppleSubscription
    from sqlalchemy import select
    async with async_session_maker() as db:
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()
        row.expires_date = datetime.utcnow() - timedelta(days=3)
        await db.commit()

    apple_says.set(txn, _apple(APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER, txn=txn))
    await _run()

    from app.db.models import UnlimitedGrant
    async with async_session_maker() as db:
        row = await db.get(UnlimitedGrant, gid)
    assert row.lapsed_at is not None
    assert (await _balance(grantee)).plan_id == "free"
    assert (await _balance(sponsor)).plan_id == "free"


async def test_expired_grants_are_swept_in_the_same_pass(
    reconcile_settings, apple_says,
):
    from app.db import async_session_maker
    from app.services import entitlement
    from app.services.apple_reconciler import CLASS_GRANT_LAPSED

    sponsor = await _mk_user()
    grantee = await _mk_user()
    await _mk_balance(sponsor, plan_id="builder")
    await _mk_balance(grantee)
    txn = await _mk_sub(sponsor, status="active", days=+40)

    async with async_session_maker() as db:
        grant = await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="trial comp",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        gid = grant.id
        await db.commit()

    result = await _run()

    assert CLASS_GRANT_LAPSED in _classes(result)
    from app.db.models import UnlimitedGrant
    async with async_session_maker() as db:
        assert (await db.get(UnlimitedGrant, gid)).lapsed_reason == "expires_at"
    assert (await _balance(grantee)).plan_id == "free"


async def test_crossgrade_is_silent_until_the_grandfather_has_run(
    reconcile_settings, apple_says,
):
    """Before the cutover every legacy Production row is legitimately
    unmarked. An alarm that fired here would page critical on the three real
    payers on the day it deployed."""
    from app.services.apple_reconciler import CLASS_CROSSGRADE

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="builder")
    await _mk_sub(uid, product=LEGACY_BUILDER, status="active", days=+18)

    result = await _run()

    assert CLASS_CROSSGRADE not in _classes(result)
    assert "apple-legacy-crossgrade" not in _alert_categories(result)


async def test_crossgrade_pages_once_the_receipt_names_the_allowlist(
    reconcile_settings, apple_says,
):
    """After the cutover, a Production row on a legacy product that is not in
    the grandfather receipt is either a new legacy purchase or an Unlimited
    subscriber cross-grading DOWN to $9.90 from iOS Settings — the route
    design §3.4 could not close in App Store Connect."""
    from app.db import async_session_maker
    from app.db.models import PlatformSetting
    from app.services.apple_reconciler import (
        CLASS_CROSSGRADE, GRANDFATHER_RECEIPT_KEY,
    )

    kept = await _mk_user("payer@example.com")
    await _mk_balance(kept, plan_id="unlimited")
    kept_txn = await _mk_sub(kept, status="active", days=+18, grandfathered=True)

    stranger = await _mk_user("stranger@example.com")
    await _mk_balance(stranger, plan_id="starter")
    stranger_txn = await _mk_sub(
        stranger, product="ai.toup.app.sub.starter", plan_id="starter",
        status="active", days=+18,
    )

    async with async_session_maker() as db:
        db.add(PlatformSetting(
            key=GRANDFATHER_RECEIPT_KEY,
            value=json.dumps({"entries": [{"original_transaction_id": kept_txn}]}),
        ))
        await db.commit()

    result = await _run()

    cross = [f for f in result.findings if f.cls == CLASS_CROSSGRADE]
    assert [f.original_txn for f in cross] == [stranger_txn]
    assert "apple-legacy-crossgrade" in _alert_categories(result)
    # An ALERT, never a mutation: a cross-grade is the customer's own choice
    # at Apple, and the only correct response is to tell a human.
    assert (await _balance(stranger)).plan_id == "starter"


# ── 7. orphaned stamps ───────────────────────────────────────────────


async def test_an_unlimited_stamp_with_no_subscription_is_reported_not_stripped(
    reconcile_settings, apple_says,
):
    """There is no transaction to ask Apple about, so the only evidence is an
    ABSENCE — and stripping a paid-looking entitlement on the strength of an
    absence is exactly the destructive inference this loop must not make. It
    is surfaced for a human instead, and surfaced LOUDLY: a silent orphan is
    indistinguishable from a healthy account."""
    from app.services.apple_reconciler import CLASS_ORPHAN

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")   # plan_source='apple', no sub row

    result = await _run()

    assert CLASS_ORPHAN in _classes(result)
    assert apple_says.calls == []
    assert (await _balance(uid)).plan_id == "unlimited"
    drift = [a for a in result.alerts if a[0] == "apple-reconcile-drift"]
    assert len(drift) == 1, result.alerts
    assert "nothing behind it" in drift[0][2]
    assert "NOT corrected" in drift[0][2]


async def test_an_orphan_with_a_dead_subscription_behind_it_IS_corrected(
    reconcile_settings, apple_says,
):
    """The other half: when there IS a transaction, Apple decides, and an
    entitlement Apple says nobody is paying for comes down."""
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    uid = await _mk_user()
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="expired", days=-40, grandfathered=True)
    apple_says.set(txn, _apple(APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER, txn=txn))

    result = await _run()

    assert (await _balance(uid)).plan_id == "free"
    assert "apple-reconcile-downgrade" in _alert_categories(result)


async def test_an_admin_stamp_is_not_an_orphan(reconcile_settings, apple_says):
    """role == 'admin' is already unlimited via _is_unlimited_user; the stamp
    is redundant, not unexplained."""
    from app.services.apple_reconciler import CLASS_ORPHAN

    uid = await _mk_user(role="admin")
    await _mk_balance(uid, plan_id="unlimited")

    result = await _run()

    assert CLASS_ORPHAN not in _classes(result)


async def test_a_grantee_is_not_an_orphan(reconcile_settings, apple_says):
    from app.db import async_session_maker
    from app.services import entitlement
    from app.services.apple_reconciler import CLASS_ORPHAN

    sponsor = await _mk_user()
    grantee = await _mk_user()
    await _mk_balance(sponsor, plan_id="builder")
    await _mk_balance(grantee)
    txn = await _mk_sub(sponsor, status="active", days=+40)
    async with async_session_maker() as db:
        await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )
        await db.commit()

    result = await _run()

    assert CLASS_ORPHAN not in _classes(result)
    assert (await _balance(grantee)).plan_id == "unlimited"


# ── 8. the notification we lose (design §5.6) ────────────────────────


async def test_handler_exception_alerts_critical_and_still_acks_200(
    monkeypatch, test_user_id,
):
    """The 200 is deliberate — Apple's retry would replay a non-idempotent
    apply_plan_change. What was missing was anybody KNOWING.
    """
    import app.api.iap as iap
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.config import settings

    sent: list[dict] = []

    async def fake_alert(category, level, message, *, subject=None, **kw):
        sent.append({"category": category, "level": level,
                     "message": message, "subject": subject})
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", fake_alert)
    monkeypatch.setattr(iap.apple_iap_service, "iap_configured", lambda: True)

    decoded = SimpleNamespace(
        notificationType=SimpleNamespace(value="DID_RENEW"),
        subtype=None,
        notificationUUID="uuid-lost-forever",
        data=SimpleNamespace(signedTransactionInfo="x", signedRenewalInfo=None),
    )

    async def fake_verify(_payload):
        return decoded

    monkeypatch.setattr(
        iap.apple_iap_service, "verify_and_decode_notification", fake_verify,
    )
    monkeypatch.setattr(iap, "_decode_inner_transaction", lambda signed: SimpleNamespace(
        productId="ai.toup.app.sub.builder",
        originalTransactionId="2000000912345678",
    ))
    monkeypatch.setattr(iap, "_decode_inner_renewal_info", lambda signed: None)

    boom = RuntimeError("kaboom in the renewal path")

    async def exploding(db, decoded_txn, renewal_info=None, *a, **kw):
        raise boom

    monkeypatch.setattr(iap, "_handle_subscription_notification",
                        lambda *a, **kw: exploding(*a, **kw))

    app = FastAPI()
    app.include_router(iap.router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"{settings.api_prefix}/iap/apple/notifications",
            json={"signedPayload": "SIGNED.JWS.PAYLOAD"},
        )

    assert resp.status_code == 200, "the 200 ack is deliberate and must not regress"

    alerts = [a for a in sent if a["category"] == "apple-notification-handler"]
    assert len(alerts) == 1, sent
    msg = alerts[0]["message"]
    assert alerts[0]["level"] == "critical"
    # It must NAME the notification, or the page says only "something raised".
    assert "DID_RENEW" in msg
    assert "uuid-lost-forever" in msg
    assert "2000000912345678" in msg
    assert "RuntimeError" in msg
    # Per-subscription subject: one broken subscription must not suppress an
    # alert about another (alerting.py's (category, subject) window).
    assert alerts[0]["subject"] == "2000000912345678"
    # NEVER the signed payload — it is a JWS of customer transaction data.
    assert "SIGNED.JWS.PAYLOAD" not in msg


async def test_a_handler_that_raises_before_it_knows_the_txn_still_names_itself(
    monkeypatch, test_user_id,
):
    """A raise on line 1 leaves `trace` empty. The alert must still fire and
    still carry the notification's outer identity."""
    import app.api.iap as iap
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.config import settings

    sent: list[dict] = []

    async def fake_alert(category, level, message, *, subject=None, **kw):
        sent.append({"category": category, "message": message, "subject": subject})
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", fake_alert)
    monkeypatch.setattr(iap.apple_iap_service, "iap_configured", lambda: True)

    decoded = SimpleNamespace(
        notificationType=SimpleNamespace(value="EXPIRED"),
        subtype=SimpleNamespace(value="VOLUNTARY"),
        notificationUUID="uuid-early-raise",
        data=SimpleNamespace(signedTransactionInfo="x", signedRenewalInfo=None),
    )

    async def fake_verify(_payload):
        return decoded

    async def explode(db, decoded_, *, trace=None):
        raise ValueError("died before decoding anything")

    monkeypatch.setattr(
        iap.apple_iap_service, "verify_and_decode_notification", fake_verify,
    )
    monkeypatch.setattr(iap, "_handle_notification", explode)

    app = FastAPI()
    app.include_router(iap.router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"{settings.api_prefix}/iap/apple/notifications",
            json={"signedPayload": "SIGNED.JWS.PAYLOAD"},
        )

    assert resp.status_code == 200
    alerts = [a for a in sent if a["category"] == "apple-notification-handler"]
    assert len(alerts) == 1
    assert "EXPIRED" in alerts[0]["message"]
    assert "VOLUNTARY" in alerts[0]["message"]
    assert alerts[0]["subject"] == "uuid-early-raise"


async def test_a_successful_notification_alerts_nothing(monkeypatch, test_user_id):
    """The alarm must be about failure, not about traffic."""
    import app.api.iap as iap
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.config import settings

    sent: list[str] = []

    async def fake_alert(category, level, message, *, subject=None, **kw):
        sent.append(category)
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", fake_alert)
    monkeypatch.setattr(iap.apple_iap_service, "iap_configured", lambda: True)

    decoded = SimpleNamespace(
        notificationType=SimpleNamespace(value="TEST"), subtype=None,
        notificationUUID="uuid-ok",
        data=SimpleNamespace(signedTransactionInfo=None, signedRenewalInfo=None),
    )

    async def fake_verify(_payload):
        return decoded

    monkeypatch.setattr(
        iap.apple_iap_service, "verify_and_decode_notification", fake_verify,
    )

    app = FastAPI()
    app.include_router(iap.router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"{settings.api_prefix}/iap/apple/notifications",
            json={"signedPayload": "SIGNED.JWS.PAYLOAD"},
        )

    assert resp.status_code == 200
    assert "apple-notification-handler" not in sent


async def test_a_dropped_renewal_must_not_lapse_a_live_sponsors_grant(
    reconcile_settings, apple_says,
):
    """The inverse of the test above, and the reason the sweep runs LAST.

    The mirror lying is the whole reason this loop exists: a dropped DID_RENEW
    leaves a live subscription reading `expires_date` in the past with
    `status='active'` — class `overdue`. `_sponsor_is_live` reads that same
    mirror. Sweeping before pass 2 therefore judged the sponsor against the
    very row this pass was about to correct: the grant was permanently lapsed
    and the grantee dropped to free in the SAME pass that correctly restored
    the sponsor, and nothing re-opens a lapsed grant.
    """
    from app.db import async_session_maker
    from app.db.models import AppleSubscription, UnlimitedGrant
    from app.services import entitlement
    from app.services.apple_reconciler import APPLE_STATUS_ACTIVE
    from sqlalchemy import select

    sponsor = await _mk_user("live-sponsor@example.com")
    grantee = await _mk_user("their-second@example.com")
    await _mk_balance(sponsor, plan_id="unlimited")
    await _mk_balance(grantee)
    txn = await _mk_sub(sponsor, status="active", days=+20, grandfathered=True)

    async with async_session_maker() as db:
        gid = (await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )).id
        await db.commit()
    assert (await _balance(grantee)).plan_id == "unlimited"

    # The renewal happened at Apple and the notification was lost.
    async with async_session_maker() as db:
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()
        row.expires_date = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    renewed = datetime.utcnow() + timedelta(days=29)
    apple_says.set(txn, _apple(
        APPLE_STATUS_ACTIVE, product=LEGACY_BUILDER, expires=renewed, txn=txn,
    ))

    await _run()

    async with async_session_maker() as db:
        grant = await db.get(UnlimitedGrant, gid)
    assert grant.lapsed_at is None, (
        "the sweep judged a live sponsor against the stale mirror this same "
        "pass corrected"
    )
    assert (await _balance(grantee)).plan_id == "unlimited"
    assert (await _balance(sponsor)).plan_id == "unlimited"


async def test_a_sponsor_apple_could_not_be_reached_for_defers_the_grant(
    reconcile_settings, apple_says,
):
    """Unreadable is not dead. With no answer from Apple the mirror stays
    stale, so the liveness judgement is deferred to the next pass rather than
    resolved against a row we know we could not confirm."""
    from app.db import async_session_maker
    from app.db.models import AppleSubscription, UnlimitedGrant
    from app.services import entitlement
    from app.services.apple_iap_service import IapVerificationError
    from sqlalchemy import select

    sponsor = await _mk_user("unreachable@example.com")
    grantee = await _mk_user("their-comp@example.com")
    await _mk_balance(sponsor, plan_id="unlimited")
    await _mk_balance(grantee)
    txn = await _mk_sub(sponsor, status="active", days=+20, grandfathered=True)

    async with async_session_maker() as db:
        gid = (await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )).id
        await db.commit()

    async with async_session_maker() as db:
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()
        row.expires_date = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    apple_says.set(txn, IapVerificationError("App Store Server API 503"))

    await _run()

    async with async_session_maker() as db:
        assert (await db.get(UnlimitedGrant, gid)).lapsed_at is None
    assert (await _balance(grantee)).plan_id == "unlimited"


async def test_observe_only_never_lapses_a_grant_off_a_stale_mirror(
    reconcile_settings, apple_says,
):
    """In observe-only nothing is written, so the mirror stays stale for the
    whole pass — and a `would` line claiming a live customer's comp is dead is
    exactly the finding an operator reads before flipping apply on."""
    from app.db import async_session_maker
    from app.db.models import AppleSubscription, UnlimitedGrant
    from app.services import entitlement
    from app.services.apple_reconciler import APPLE_STATUS_ACTIVE, CLASS_GRANT_LAPSED
    from sqlalchemy import select

    sponsor = await _mk_user("observed@example.com")
    grantee = await _mk_user("observed-comp@example.com")
    await _mk_balance(sponsor, plan_id="unlimited")
    await _mk_balance(grantee)
    txn = await _mk_sub(sponsor, status="active", days=+20, grandfathered=True)

    async with async_session_maker() as db:
        gid = (await entitlement.create_grant(
            db, granted_to_user_id=grantee, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )).id
        await db.commit()

    async with async_session_maker() as db:
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()
        row.expires_date = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    apple_says.set(txn, _apple(
        APPLE_STATUS_ACTIVE, product=LEGACY_BUILDER,
        expires=datetime.utcnow() + timedelta(days=29), txn=txn,
    ))

    result = await _run(apply=False)

    assert CLASS_GRANT_LAPSED not in _classes(result)
    async with async_session_maker() as db:
        assert (await db.get(UnlimitedGrant, gid)).lapsed_at is None


async def test_one_failed_correction_does_not_roll_back_the_others(
    reconcile_settings, apple_says, monkeypatch,
):
    """Each finding gets its own SAVEPOINT.

    Without one the per-finding try/except is a lie: a failed FLUSH marks the
    session inactive, so every later finding raises PendingRollbackError on
    its first statement and the pass rolls back wholesale — including the
    corrections that had already succeeded, the summary heartbeat, and the
    CRITICAL "a paying customer is locked out" alert this loop exists to
    raise. The failure is therefore injected as a real duplicate-key flush,
    not as a bare `raise`: a bare raise never poisons the session and would
    pass with or without the savepoint.
    """
    from app.db.models import CreditBalance
    from app.services import apple_reconciler
    from app.services.apple_reconciler import APPLE_STATUS_EXPIRED

    users = {}
    for name in ("one", "two"):
        uid = await _mk_user(f"{name}@example.com")
        await _mk_balance(uid, plan_id="unlimited")
        txn = await _mk_sub(
            uid, status="active", days=-3, grandfathered=True, txn=f"TXN-{name}",
        )
        apple_says.set(txn, _apple(
            APPLE_STATUS_EXPIRED, product=LEGACY_BUILDER, txn=txn,
        ))
        users[txn] = uid

    real = apple_reconciler._apply_finding
    poisoned: list[str] = []

    async def flaky(db, finding, subs_by_txn, **kw):
        if not poisoned and finding.original_txn in users:
            poisoned.append(finding.original_txn)
            # A duplicate primary key: the flush fails and the session is
            # inactive from here on unless something scoped the failure.
            db.add(CreditBalance(
                user_id=users[finding.original_txn], plan_id="free",
                message_credits_remaining=1, integration_credits_remaining=1,
            ))
            await db.flush()
            return True
        return await real(db, finding, subs_by_txn, **kw)

    monkeypatch.setattr(apple_reconciler, "_apply_finding", flaky)

    result = await _run()

    assert poisoned, "the harness never reached a finding to poison"
    survivor = next(u for t, u in users.items() if t != poisoned[0])
    assert result.errors, "the failed correction was not recorded"
    assert (await _balance(survivor)).plan_id == "free", (
        "a failed correction poisoned the transaction and rolled back a "
        "correction that had already succeeded"
    )


async def test_a_grandfathered_row_that_changed_product_pages(
    reconcile_settings, apple_says,
):
    """The cross-grade the receipt allowlist structurally cannot see.

    A cross-grade keeps the SAME original_transaction_id, so the row stays in
    the grandfather receipt and `_converge_mirror` rewrites product_id to the
    new one — nothing is left disagreeing. The only evidence is the product
    they hold versus the product they were grandfathered on, and that is a
    comparison this loop already has in hand.
    """
    from app.db import async_session_maker
    from app.db.models import AppleSubscription, PlatformSetting
    from app.services.apple_reconciler import (
        CLASS_CROSSGRADE, GRANDFATHER_RECEIPT_KEY,
    )
    from sqlalchemy import select

    uid = await _mk_user("crossgrader@example.com")
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=+18, grandfathered=True)

    async with async_session_maker() as db:
        db.add(PlatformSetting(
            key=GRANDFATHER_RECEIPT_KEY,
            value=json.dumps({"entries": [{"original_transaction_id": txn}]}),
        ))
        # …and they switch to Starter from iOS Settings.
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == txn))).scalars().first()
        row.product_id = "ai.toup.app.sub.starter"
        await db.commit()

    result = await _run()

    assert CLASS_CROSSGRADE in _classes(result)
    assert "apple-legacy-crossgrade" in _alert_categories(result)


async def test_a_grandfathered_row_on_its_own_product_is_silent(
    reconcile_settings, apple_says,
):
    """Anti-vacuity: the new arm must fire on the CHANGE, not on the stamp."""
    from app.db import async_session_maker
    from app.db.models import PlatformSetting
    from app.services.apple_reconciler import (
        CLASS_CROSSGRADE, GRANDFATHER_RECEIPT_KEY,
    )

    uid = await _mk_user("loyal@example.com")
    await _mk_balance(uid, plan_id="unlimited")
    txn = await _mk_sub(uid, status="active", days=+18, grandfathered=True)

    async with async_session_maker() as db:
        db.add(PlatformSetting(
            key=GRANDFATHER_RECEIPT_KEY,
            value=json.dumps({"entries": [{"original_transaction_id": txn}]}),
        ))
        await db.commit()

    result = await _run()

    assert CLASS_CROSSGRADE not in _classes(result)
    assert "apple-legacy-crossgrade" not in _alert_categories(result)
