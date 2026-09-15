"""Unlimited's two ends: the drop back to Free, and coming back.

The founder's rules 2 and 5, proven by execution rather than by reading the
notification table:

* **"If they cancel or the subscription expires, they drop back to Free at the
  end of the paid period — automatically, on every channel, with no manual
  step."** Three separate claims and each is tested on its own, because they
  fail independently:
    - *at the END of the paid period*, not before — AUTO_RENEW_DISABLED and
      DID_FAIL_TO_RENEW/GRACE_PERIOD must leave the entitlement standing;
    - *automatically* — the notification handler does it, and when the
      notification never arrives (the channel has already dropped one) the
      reconciler does it on its own schedule;
    - *on every channel* — after the drop, the llm-proxy pre-flight, the
      agent's own ``/credits/preflight``, the ``/agent-deduct`` charge path and
      the voice pre-flight all refuse the same way, because they all read the
      one balance row.
* **"If they re-subscribe, unlimited comes back."** Immediately, through the
  ordinary SUBSCRIBED path, with no reconciliation step in between.

The shape of the drop is asserted field by field, not just ``plan_id ==
'free'``: a downgrade that left ``used_today`` at six figures, or
``purchased_credits_remaining`` negative, or ``plan_source`` stuck at 'apple'
(which blocks the account from ever buying on iOS again and freezes the hourly
clock renewal) would pass a plan_id check and brick the account.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


LEGACY_BUILDER = "ai.toup.app.sub.builder"
UNLIMITED_PRODUCT = "ai.toup.app.sub.unlimited"


# ── helpers ──────────────────────────────────────────────────────────


def _ms(dt: datetime) -> int:
    """Apple sends epoch MILLISECONDS in UTC. calendar.timegm, never
    dt.timestamp() — the latter reads a naive datetime as LOCAL time and
    injects the machine's UTC offset, which is a silent multi-hour skew."""
    import calendar
    return int(calendar.timegm(dt.utctimetuple()) * 1000 + dt.microsecond // 1000)


def _enum(value):
    return SimpleNamespace(value=value)


def _fake_txn(*, product_id, original_txn, expires_dt, environment="Production"):
    return SimpleNamespace(
        productId=product_id, originalTransactionId=original_txn,
        transactionId=f"txn-{original_txn}",
        expiresDate=_ms(expires_dt) if expires_dt else None,
        environment=_enum(environment), isUpgraded=False,
    )


def _fake_renewal(*, auto_renew_status=1, auto_renew_product_id=None, grace_dt=None):
    return SimpleNamespace(
        autoRenewStatus=_enum(auto_renew_status),
        autoRenewProductId=auto_renew_product_id,
        gracePeriodExpiresDate=_ms(grace_dt) if grace_dt else None,
        expirationIntent=None,
    )


def _fake_notification(*, ntype, subtype=None, uuid_, renewal=None):
    data = SimpleNamespace(
        signedTransactionInfo="signed-txn-placeholder",
        signedRenewalInfo="signed-renewal-placeholder" if renewal is not None else None,
        status=None,
    )
    return SimpleNamespace(
        notificationType=_enum(ntype),
        subtype=_enum(subtype) if subtype else None,
        notificationUUID=uuid_, data=data,
    )


@pytest.fixture
def patch_notification_decode(monkeypatch):
    import app.api.iap as iap
    state = {"txn": None, "renewal": None}

    def set_payload(txn, renewal=None):
        state["txn"] = txn
        state["renewal"] = renewal

    monkeypatch.setattr(iap, "_decode_inner_transaction", lambda signed: state["txn"])
    monkeypatch.setattr(iap, "_decode_inner_renewal_info", lambda signed: state["renewal"])
    return set_payload


async def _deliver(decoded):
    """Run one notification through the real handler + commit."""
    import app.api.iap as iap
    from app.db import async_session_maker
    async with async_session_maker() as db:
        await iap._handle_notification(db, decoded)
        await db.commit()


async def _balance(user_id: str):
    from app.db import CreditBalance, async_session_maker
    async with async_session_maker() as db:
        return await db.get(CreditBalance, user_id)


@pytest_asyncio.fixture
async def grandfathered(test_user_id: str):
    """A legacy Builder payer, grandfathered onto Unlimited — the exact shape
    the 2026-10-01 renewals will land on."""
    from app.db import async_session_maker
    from app.db.models import APPLE_SUB_ACTIVE, AppleSubscription
    from app.services.credit_service import credit_service

    txn = f"orig-{uuid.uuid4().hex[:12]}"
    expires = datetime.utcnow() + timedelta(days=17)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, test_user_id)
        db.add(AppleSubscription(
            user_id=test_user_id, original_transaction_id=txn,
            product_id=LEGACY_BUILDER,
            # The MIRROR keeps the catalogue plan; the entitlement lives on
            # credit_balances and grandfathered_at says how the two differ.
            plan_id="builder", status=APPLE_SUB_ACTIVE, expires_date=expires,
            auto_renew_status=True, environment="Production",
            grandfathered_at=now, grandfathered_product_id=LEGACY_BUILDER,
        ))
        await credit_service.activate_subscription(
            db, test_user_id, "unlimited", "apple", expires,
        )
        await db.commit()
    bal = await _balance(test_user_id)
    assert bal.plan_id == "unlimited"
    assert float(bal.message_credits_remaining) == 1_000_000.0
    return SimpleNamespace(user_id=test_user_id, txn=txn, expires=expires)


async def _charge(user_id: str, amount: str = "500"):
    """A real charge through the real gate, with enforcement fully ON."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    prev = (settings.credit_enforcement_enabled, settings.credit_cap_admission_control)
    settings.credit_enforcement_enabled = True
    settings.credit_cap_admission_control = True
    try:
        async with async_session_maker() as db:
            res = await credit_service.try_charge(
                db, user_id, bucket="message", amount=Decimal(amount),
                event_type="chat_message", underlying_cost_cents=Decimal(amount),
            )
            await db.commit()
            return res
    finally:
        settings.credit_enforcement_enabled = prev[0]
        settings.credit_cap_admission_control = prev[1]


async def _check(user_id: str, bucket: str = "message", amount: str = "500"):
    """`check_balance` — the ONE pre-flight the llm proxy, the voice socket,
    the connector dispatcher and /agent-deduct all share."""
    from app.config import settings
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    prev = settings.credit_enforcement_enabled
    settings.credit_enforcement_enabled = True
    try:
        async with async_session_maker() as db:
            return await credit_service.check_balance(
                db, user_id, bucket, Decimal(amount),
            )
    finally:
        settings.credit_enforcement_enabled = prev


# ── 1. NOT before the end of the paid period ─────────────────────────


async def test_auto_renew_off_does_not_downgrade(grandfathered, patch_notification_decode):
    """Turning auto-renew off is a statement about the NEXT period. The user
    paid for this one."""
    patch_notification_decode(
        _fake_txn(product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
                  expires_dt=grandfathered.expires),
        _fake_renewal(auto_renew_status=0),
    )
    await _deliver(_fake_notification(
        ntype="DID_CHANGE_RENEWAL_STATUS", subtype="AUTO_RENEW_DISABLED",
        uuid_="n-arn-off", renewal=True,
    ))

    bal = await _balance(grandfathered.user_id)
    assert bal.plan_id == "unlimited"
    assert float(bal.message_credits_remaining) == 1_000_000.0
    assert (await _charge(grandfathered.user_id)).success is True


async def test_billing_grace_does_not_downgrade(grandfathered, patch_notification_decode):
    """The card failed; Apple has not ended anything. We wait for EXPIRED /
    GRACE_PERIOD_EXPIRED."""
    patch_notification_decode(
        _fake_txn(product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
                  expires_dt=grandfathered.expires),
        _fake_renewal(grace_dt=datetime.utcnow() + timedelta(days=10)),
    )
    await _deliver(_fake_notification(
        ntype="DID_FAIL_TO_RENEW", subtype="GRACE_PERIOD",
        uuid_="n-grace", renewal=True,
    ))

    assert (await _balance(grandfathered.user_id)).plan_id == "unlimited"


# ── 2. AT the end, automatically ─────────────────────────────────────


async def test_expired_drops_to_a_byte_for_byte_fresh_free_wallet(
    grandfathered, patch_notification_decode,
):
    """Not just plan_id — every field the absolute branch touches.

    A downgrade crossing the ~1,000,000-credit gap through the PRORATED body
    would drive purchased_credits_remaining to roughly −670,000 for a user who
    never bought a pack (design §1.6). That account would then read as deeply
    negative at every gate and be bricked, while a plan_id assertion passed.
    """
    from app.db import CreditBalance, async_session_maker

    # Give them a bought credit pack first: purchased credits never expire and
    # must survive the drop untouched, in either direction.
    async with async_session_maker() as db:
        bal = await db.get(CreditBalance, grandfathered.user_id)
        bal.purchased_credits_remaining = Decimal("50")
        period_start, period_end = bal.period_start, bal.period_end
        await db.commit()

    patch_notification_decode(
        _fake_txn(product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
                  expires_dt=datetime.utcnow() - timedelta(minutes=1)),
        _fake_renewal(auto_renew_status=0),
    )
    await _deliver(_fake_notification(
        ntype="EXPIRED", subtype="VOLUNTARY", uuid_="n-expired", renewal=True,
    ))

    bal = await _balance(grandfathered.user_id)
    assert bal.plan_id == "free"
    # The ROW carries the plan wallet; the purchased wallet is its own column
    # and is added only in the spendable view.
    assert float(bal.message_credits_remaining) == 100.0
    assert float(bal.integration_credits_remaining) == 500.0
    async with async_session_maker() as db:
        from app.services.credit_service import credit_service
        view = await credit_service.get_balance_view(db, grandfathered.user_id)
    assert float(view.message_credits_remaining) == 150.0  # 100 plan + 50 bought
    assert float(bal.purchased_credits_remaining) == 50.0  # untouched
    assert float(bal.message_credits_used_today) == 0.0    # not a poisoned counter
    assert bal.message_credits_daily_cap is None           # free's cap, i.e. none
    # plan_source back to NULL is what un-freezes the hourly clock renewal
    # (renew_period refuses plan_source='apple' rows) and what stops
    # active_paid_source reading the account as still paying.
    assert bal.plan_source is None
    # Apple owns the period clock and a downgrade does not rewrite history.
    assert bal.period_start == period_start
    assert bal.period_end == period_end


async def test_revoke_and_grace_expired_also_drop(grandfathered, patch_notification_decode):
    """The whole terminal set, not just the one that is easy to test."""
    for ntype, uid_ in (("GRACE_PERIOD_EXPIRED", "n-ge"), ("REVOKE", "n-rv")):
        # Re-establish the entitlement between the two.
        from app.db import async_session_maker
        from app.services.credit_service import credit_service
        async with async_session_maker() as db:
            await credit_service.activate_subscription(
                db, grandfathered.user_id, "unlimited", "apple",
                grandfathered.expires,
            )
            await db.commit()
        assert (await _balance(grandfathered.user_id)).plan_id == "unlimited"

        patch_notification_decode(_fake_txn(
            product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
            expires_dt=datetime.utcnow() - timedelta(minutes=1),
        ))
        await _deliver(_fake_notification(ntype=ntype, uuid_=uid_))
        assert (await _balance(grandfathered.user_id)).plan_id == "free", ntype


async def test_the_reconciler_drops_them_when_the_notification_never_comes(
    grandfathered, monkeypatch,
):
    """"Automatically" cannot mean "when Apple remembers to tell us". The
    channel has already dropped a message — the Sandbox Elite has read
    status='active' for 2.7 months past its expiry."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    from app.services import apple_iap_service
    from app.services.apple_iap_service import (
        APPLE_STATUS_EXPIRED, AppleSubscriptionStatus,
    )
    from app.services.apple_reconciler import reconcile_once
    from sqlalchemy import select

    # The subscription silently ends. NO notification is delivered.
    async with async_session_maker() as db:
        row = (await db.execute(select(AppleSubscription).where(
            AppleSubscription.original_transaction_id == grandfathered.txn
        ))).scalars().first()
        row.expires_date = datetime.utcnow() - timedelta(days=2)
        await db.commit()

    async def fake(txn, env):
        return AppleSubscriptionStatus(
            original_transaction_id=txn, environment=env,
            status=APPLE_STATUS_EXPIRED, product_id=LEGACY_BUILDER,
            expires_date=datetime.utcnow() - timedelta(days=2),
        )

    monkeypatch.setattr(apple_iap_service, "fetch_subscription_status", fake)
    prev = settings.apple_reconcile_apply
    settings.apple_reconcile_apply = True
    try:
        async with async_session_maker() as db:
            await reconcile_once(db)
            await db.commit()
    finally:
        settings.apple_reconcile_apply = prev

    bal = await _balance(grandfathered.user_id)
    assert bal.plan_id == "free"
    assert bal.plan_source is None
    assert (await _charge(grandfathered.user_id)).success is False


# ── 3. on EVERY channel ──────────────────────────────────────────────


async def test_after_the_drop_every_surface_refuses_identically(
    grandfathered, patch_notification_decode, test_user_id,
):
    """Every gate in the system reads the same ``credit_balances`` row, which
    is why the entitlement is materialised there and nowhere else. Prove it
    rather than assert it: one channel still saying yes is the split-brain
    this design exists to avoid.
    """
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.services.credit_service import REASON_INSUFFICIENT_MESSAGE

    # Still unlimited: every surface says yes.
    assert (await _check(test_user_id, "message")).success is True
    assert (await _check(test_user_id, "integration")).success is True

    patch_notification_decode(_fake_txn(
        product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
        expires_dt=datetime.utcnow() - timedelta(minutes=1),
    ))
    await _deliver(_fake_notification(ntype="EXPIRED", uuid_="n-chan"))

    # 1. try_charge — the llm-proxy / /agent-deduct charge path.
    charge = await _charge(test_user_id, "500")
    assert charge.success is False
    assert charge.reason == REASON_INSUFFICIENT_MESSAGE

    # 2. check_balance — the ONE pre-flight shared by llm_proxy (x2),
    #    ws_realtime (voice), connector_dispatcher and /agent-deduct.
    for bucket in ("message", "integration"):
        res = await _check(test_user_id, bucket, "5000")
        assert res.success is False, bucket

    # 3. GET /credits/preflight — authenticated on X-Agent-Key with NO User
    #    object at all, which is exactly why the entitlement had to be a plan
    #    row rather than a flag plumbed through every caller.
    key = f"agentkey-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id, hosting_mode="managed", agent_name="T",
            agent_api_key=key, llm_mode=None,
        ))
        await db.commit()

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.credits import router as credits_router

    app = FastAPI()
    app.include_router(credits_router, prefix=settings.api_prefix)
    prev = settings.credit_enforcement_enabled
    settings.credit_enforcement_enabled = True
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as ac:
            resp = await ac.get(
                f"{settings.api_prefix}/credits/preflight",
                params={"bucket": "message", "required": 500},
                headers={"X-Agent-Key": key, "X-Agent-User-Id": test_user_id},
            )
    finally:
        settings.credit_enforcement_enabled = prev
    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient"] is False
    assert body["plan_id"] == "free"


async def test_while_unlimited_the_agent_preflight_cannot_refuse(
    grandfathered, test_user_id,
):
    """The mirror image, and the reason the design is a plan ROW: this
    endpoint has no ``User`` object, so a bypass FLAG could not have been
    plumbed into it without changing its signature — while 1,000,000
    remaining makes it correct with no new code at all."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.credits import router as credits_router

    key = f"agentkey-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=test_user_id, hosting_mode="managed", agent_name="T",
            agent_api_key=key, llm_mode=None,
        ))
        await db.commit()

    app = FastAPI()
    app.include_router(credits_router, prefix=settings.api_prefix)
    prev = settings.credit_enforcement_enabled
    settings.credit_enforcement_enabled = True
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as ac:
            resp = await ac.get(
                f"{settings.api_prefix}/credits/preflight",
                params={"bucket": "message", "required": 9999},
                headers={"X-Agent-Key": key, "X-Agent-User-Id": test_user_id},
            )
    finally:
        settings.credit_enforcement_enabled = prev
    body = resp.json()
    assert body["sufficient"] is True
    assert body["plan_id"] == "unlimited"


# ── 4. coming back ───────────────────────────────────────────────────


async def test_resubscribing_restores_unlimited_immediately(
    grandfathered, patch_notification_decode, test_user_id,
):
    """No reconciliation step in between: the ordinary SUBSCRIBED path
    restores the entitlement and the very next charge goes through."""
    patch_notification_decode(_fake_txn(
        product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
        expires_dt=datetime.utcnow() - timedelta(minutes=1),
    ))
    await _deliver(_fake_notification(ntype="EXPIRED", uuid_="n-gone"))
    assert (await _balance(test_user_id)).plan_id == "free"
    assert (await _charge(test_user_id)).success is False

    new_expires = datetime.utcnow() + timedelta(days=30)
    patch_notification_decode(_fake_txn(
        product_id=UNLIMITED_PRODUCT, original_txn=grandfathered.txn,
        expires_dt=new_expires,
    ))
    await _deliver(_fake_notification(ntype="SUBSCRIBED", uuid_="n-back"))

    bal = await _balance(test_user_id)
    assert bal.plan_id == "unlimited"
    assert float(bal.message_credits_remaining) == 1_000_000.0
    assert float(bal.integration_credits_remaining) == 1_000_000.0
    assert bal.plan_source == "apple"
    # Apple's clock, not ours. (Sub-millisecond only: expiresDate round-trips
    # through epoch MILLISECONDS.)
    assert abs((bal.period_end - new_expires).total_seconds()) < 0.01

    charge = await _charge(test_user_id, "500")
    assert charge.success is True
    # Served, and nothing debited: the balance is exactly where it started.
    assert float(charge.balance_after) == 1_000_000.0
    assert float((await _balance(test_user_id)).message_credits_remaining) == 1_000_000.0


async def test_resubscribing_on_the_legacy_product_restores_the_grandfather(
    grandfathered, patch_notification_decode, test_user_id,
):
    """A grandfathered payer whose subscription lapsed and who restores it
    from iOS Settings comes back to UNLIMITED, not to Builder — the resolver
    reads ``grandfathered_at`` off the row, which survives the lapse."""
    patch_notification_decode(_fake_txn(
        product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
        expires_dt=datetime.utcnow() - timedelta(minutes=1),
    ))
    await _deliver(_fake_notification(ntype="EXPIRED", uuid_="n-gone-2"))
    assert (await _balance(test_user_id)).plan_id == "free"

    patch_notification_decode(_fake_txn(
        product_id=LEGACY_BUILDER, original_txn=grandfathered.txn,
        expires_dt=datetime.utcnow() + timedelta(days=30),
    ))
    await _deliver(_fake_notification(ntype="SUBSCRIBED", uuid_="n-back-2"))

    assert (await _balance(test_user_id)).plan_id == "unlimited"


async def test_a_repeat_subscribed_is_a_clean_no_op(
    grandfathered, patch_notification_decode, test_user_id,
):
    """A restore of a subscription that is already live must write NOTHING —
    assignment to values the row already holds, two zero deltas, no ledger
    rows. The apply_plan_change absolute branch is what makes that true."""
    from sqlalchemy import func, select
    from app.db import CreditLedger, async_session_maker

    async def _rows():
        async with async_session_maker() as db:
            return int((await db.execute(
                select(func.count(CreditLedger.id))
                .where(CreditLedger.user_id == test_user_id)
            )).scalar() or 0)

    before = await _rows()
    patch_notification_decode(_fake_txn(
        product_id=UNLIMITED_PRODUCT, original_txn=grandfathered.txn,
        expires_dt=grandfathered.expires,
    ))
    await _deliver(_fake_notification(ntype="SUBSCRIBED", uuid_="n-again"))

    bal = await _balance(test_user_id)
    assert bal.plan_id == "unlimited"
    assert float(bal.message_credits_remaining) == 1_000_000.0
    assert await _rows() == before
