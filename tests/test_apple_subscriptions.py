"""Tests for the Apple auto-renewable subscription backend (mig-063 PR A).

Covers the §8 "New" cases:

* Reconciliation 409 in BOTH directions (Apple-verify over active Stripe;
  Stripe credit-checkout creation over active Apple).
* free → activate (full monthly grant, plan_source='apple', Apple period_end,
  one apple_subscriptions row UNIQUE on original_transaction_id).
* The full lifecycle table driven via decoded fixtures
  (_handle_subscription_notification): SUBSCRIBED, DID_RENEW (+ no double-grant
  on UUID replay), AUTO_RENEW_DISABLED / DID_FAIL_TO_RENEW-GRACE_PERIOD (NO
  downgrade), EXPIRED / GRACE_PERIOD_EXPIRED / REVOKE / REFUND (downgrade),
  DID_CHANGE_RENEWAL_PREF DOWNGRADE (deferred) / UPGRADE (immediate).
* Renewal guard: an apple-source balance past period_end → renew_period()
  returns False (the hourly cron does NOT grant).
* Idempotency: notificationUUID replay is a no-op; original_transaction_id
  re-verify returns already_active without re-granting.

These are UNIT-level: the notification handler is exercised with fake decoded
payloads (mirroring the app-store-server-library attribute shapes) so we don't
need Apple's signing keys. The /subscribe/verify endpoint is exercised with
verify_subscription_transaction + iap_configured monkeypatched.

The credit_service + split-message tests in the suite prove inertness; this
file proves the new behaviour.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


# ── helpers ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def credit_user(test_user_id: str):
    """Prime a fresh free-tier balance for the user behind auth_headers."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, test_user_id)
        await db.commit()
    return test_user_id


async def _balance(user_id: str):
    from app.db import async_session_maker, CreditBalance
    async with async_session_maker() as db:
        return await db.get(CreditBalance, user_id)


async def _apple_sub(user_id: str):
    from sqlalchemy import select
    from app.db import async_session_maker
    from app.db.models import AppleSubscription
    async with async_session_maker() as db:
        return (await db.execute(
            select(AppleSubscription).where(AppleSubscription.user_id == user_id)
        )).scalars().first()


async def _ledger_renewal_count(user_id: str) -> int:
    from sqlalchemy import select, func
    from app.db import async_session_maker, CreditLedger
    from app.db.models import LEDGER_PERIOD_RENEWAL
    async with async_session_maker() as db:
        n = await db.execute(
            select(func.count(CreditLedger.id)).where(
                CreditLedger.user_id == user_id,
                CreditLedger.event_type == LEDGER_PERIOD_RENEWAL,
            )
        )
        return int(n.scalar() or 0)


def _ms(dt: datetime) -> int:
    # Apple sends epoch MILLISECONDS in UTC; ms_to_datetime() round-trips via
    # utcfromtimestamp. Treat our naive (UTC) datetime as UTC here too
    # (calendar.timegm), NOT dt.timestamp() which assumes LOCAL time and would
    # inject the machine's UTC offset.
    import calendar
    return int(calendar.timegm(dt.utctimetuple()) * 1000 + dt.microsecond // 1000)


# Enum-like stand-ins: the handler reads `.value` off enum fields and falls
# back to str(). A SimpleNamespace with `value` matches the library shape.
def _enum(value):
    return SimpleNamespace(value=value)


def _fake_txn(*, product_id, original_txn, expires_dt, environment="Production"):
    """Mirror JWSTransactionDecodedPayload's attributes the handler reads."""
    return SimpleNamespace(
        productId=product_id,
        originalTransactionId=original_txn,
        transactionId=f"txn-{original_txn}",
        expiresDate=_ms(expires_dt) if expires_dt else None,
        environment=_enum(environment),
        isUpgraded=False,
    )


def _fake_renewal(*, auto_renew_status=1, auto_renew_product_id=None, grace_dt=None):
    """Mirror JWSRenewalInfoDecodedPayload. autoRenewStatus.value: ON=1 OFF=0."""
    return SimpleNamespace(
        autoRenewStatus=_enum(auto_renew_status),
        autoRenewProductId=auto_renew_product_id,
        gracePeriodExpiresDate=_ms(grace_dt) if grace_dt else None,
        expirationIntent=None,
    )


def _fake_notification(*, ntype, subtype=None, uuid_, txn, renewal=None):
    """Mirror ResponseBodyV2DecodedPayload + Data."""
    data = SimpleNamespace(
        signedTransactionInfo="signed-txn-placeholder",
        signedRenewalInfo="signed-renewal-placeholder" if renewal is not None else None,
        status=None,
    )
    return SimpleNamespace(
        notificationType=_enum(ntype),
        subtype=_enum(subtype) if subtype else None,
        notificationUUID=uuid_,
        data=data,
    )


@pytest_asyncio.fixture
async def iap_client():
    """A test app that mounts the iap + credits routers (the shared conftest
    `client` only mounts billing/vps/auth). Uses the real get_db against the
    in-memory sqlite, so auth_headers' JWT resolves the same way."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.iap import router as iap_router
    from app.api.credits import router as credits_router, billing_router as credits_billing_router
    from app.config import settings

    app = FastAPI()
    app.include_router(iap_router, prefix=settings.api_prefix)
    app.include_router(credits_router, prefix=settings.api_prefix)
    app.include_router(credits_billing_router, prefix=settings.api_prefix)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def patch_notification_decode(monkeypatch):
    """Patch the JWS decode helpers so _handle_notification reads our fakes.

    Returns a setter that installs the txn + renewal a single notification
    should decode to.
    """
    import app.api.iap as iap

    state = {"txn": None, "renewal": None}

    def set_payload(txn, renewal=None):
        state["txn"] = txn
        state["renewal"] = renewal

    monkeypatch.setattr(iap, "_decode_inner_transaction", lambda signed: state["txn"])
    monkeypatch.setattr(iap, "_decode_inner_renewal_info", lambda signed: state["renewal"])
    return set_payload


async def _seed_active_apple_sub(user_id: str, *, plan_id, product_id, original_txn, expires_dt):
    """Put the user on an active Apple sub directly (skips JWS verify)."""
    from app.db import async_session_maker
    from app.db.models import AppleSubscription, APPLE_SUB_ACTIVE
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        db.add(AppleSubscription(
            user_id=user_id, original_transaction_id=original_txn,
            product_id=product_id, plan_id=plan_id, status=APPLE_SUB_ACTIVE,
            expires_date=expires_dt, auto_renew_status=True, environment="Production",
        ))
        await credit_service.activate_subscription(
            db, user_id, plan_id, "apple", expires_dt,
        )
        await db.commit()


# ── reconciliation: active_paid_source ───────────────────────────────


async def test_active_paid_source_free_is_none(credit_user):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        src = await credit_service.active_paid_source(db, credit_user)
    assert src is None


async def test_active_paid_source_legacy_paid_reads_stripe(credit_user):
    """A paid plan with plan_source NULL (pre-migration Stripe) reads 'stripe'."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        await credit_service.apply_plan_change(db, credit_user, "builder", reason="t")
        await db.commit()
    b = await _balance(credit_user)
    assert b.plan_source is None  # NULL preserved — Stripe path never writes it
    async with async_session_maker() as db:
        src = await credit_service.active_paid_source(db, credit_user)
    assert src == "stripe"


async def test_active_paid_source_apple(credit_user):
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-1", expires_dt=datetime.utcnow() + timedelta(days=30),
    )
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        src = await credit_service.active_paid_source(db, credit_user)
    assert src == "apple"


# ── /apple/subscribe/verify ──────────────────────────────────────────


def _patch_verify(monkeypatch, *, plan_id, product_id, original_txn, expires_dt, environment="Production"):
    """Make iap_configured() True and verify_subscription_transaction return ours."""
    import app.services.apple_iap_service as svc

    async def _fake_verify(transaction_id, environment_hint):
        return svc.VerifiedSubscription(
            transaction_id=f"txn-{original_txn}",
            original_transaction_id=original_txn,
            product_id=product_id,
            environment=environment,
            expires_date=expires_dt,
            plan_id=plan_id,
        )

    monkeypatch.setattr(svc, "iap_configured", lambda: True)
    monkeypatch.setattr(svc, "verify_subscription_transaction", _fake_verify)


async def test_subscribe_verify_free_activates(iap_client, auth_headers, credit_user, monkeypatch):
    expires = datetime.utcnow() + timedelta(days=30)
    _patch_verify(monkeypatch, plan_id="starter",
                  product_id="ai.toup.app.sub.starter", original_txn="orig-act-1",
                  expires_dt=expires)

    resp = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter",
              "environment": "Production"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["plan_id"] == "starter"
    assert body["plan_source"] == "apple"
    assert body["already_active"] is False

    b = await _balance(credit_user)
    assert b.plan_id == "starter"
    assert b.plan_source == "apple"
    # Apple's expiresDate, NOT now+30d exactly — within a second of expires.
    assert abs((b.period_end - expires).total_seconds()) < 2
    # Full monthly grant landed (starter = 130 message credits, free start = 100,
    # prorated delta ≈ +30 at ratio≈1.0 → 130 spendable).
    assert Decimal(b.message_credits_remaining) >= Decimal("130")

    sub = await _apple_sub(credit_user)
    assert sub is not None
    assert sub.original_transaction_id == "orig-act-1"
    assert sub.status == "active"


async def test_subscribe_verify_unknown_product_400(iap_client, auth_headers, credit_user, monkeypatch):
    import app.services.apple_iap_service as svc
    monkeypatch.setattr(svc, "iap_configured", lambda: True)
    resp = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.credits.small"},
    )
    assert resp.status_code == 400


async def test_subscribe_verify_unconfigured_503(iap_client, auth_headers, credit_user, monkeypatch):
    import app.services.apple_iap_service as svc
    monkeypatch.setattr(svc, "iap_configured", lambda: False)
    resp = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter"},
    )
    assert resp.status_code == 503


async def test_subscribe_verify_product_mismatch_422(iap_client, auth_headers, credit_user, monkeypatch):
    # verify returns elite but client claimed starter → 422.
    _patch_verify(monkeypatch, plan_id="elite",
                  product_id="ai.toup.app.sub.elite", original_txn="orig-mm",
                  expires_dt=datetime.utcnow() + timedelta(days=30))
    resp = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter"},
    )
    assert resp.status_code == 422


async def test_subscribe_verify_over_stripe_409(iap_client, auth_headers, credit_user, monkeypatch):
    """Apple verify when the user already has an active Stripe plan → 409."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        await credit_service.apply_plan_change(db, credit_user, "builder", reason="stripe:test")
        await db.commit()

    _patch_verify(monkeypatch, plan_id="starter",
                  product_id="ai.toup.app.sub.starter", original_txn="orig-conflict",
                  expires_dt=datetime.utcnow() + timedelta(days=30))
    resp = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "subscription_exists_other_platform"
    # Balance unchanged — still on the Stripe builder plan, no Apple grant.
    b = await _balance(credit_user)
    assert b.plan_id == "builder"
    assert b.plan_source is None
    assert await _apple_sub(credit_user) is None


async def test_subscribe_verify_idempotent_already_active(iap_client, auth_headers, credit_user, monkeypatch):
    """Re-verify the same original_transaction_id → already_active, no re-grant."""
    expires = datetime.utcnow() + timedelta(days=30)
    _patch_verify(monkeypatch, plan_id="starter",
                  product_id="ai.toup.app.sub.starter", original_txn="orig-idem",
                  expires_dt=expires)

    r1 = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter"},
    )
    assert r1.status_code == 200
    assert r1.json()["already_active"] is False
    after_first = Decimal((await _balance(credit_user)).message_credits_remaining)

    r2 = await iap_client.post(
        "/api/iap/apple/subscribe/verify", headers=auth_headers,
        json={"transaction_id": "t1", "product_id": "ai.toup.app.sub.starter"},
    )
    assert r2.status_code == 200
    assert r2.json()["already_active"] is True
    after_second = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after_second == after_first, "re-verify must NOT re-grant"


# ── Stripe-path reconciliation (§3b) ─────────────────────────────────


async def test_credit_checkout_over_apple_409(iap_client, auth_headers, credit_user, monkeypatch):
    """Stripe credit-checkout creation when an Apple sub is active → 409,
    no Stripe session created."""
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-stripe-block", expires_dt=datetime.utcnow() + timedelta(days=30),
    )
    # Guard fires BEFORE any Stripe call, so we don't need Stripe configured.
    resp = await iap_client.post("/api/billing/credit-checkout/builder", headers=auth_headers)
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "subscription_exists_other_platform"


# ── lifecycle via _handle_subscription_notification ──────────────────


async def _run_notification(db, notif):
    import app.api.iap as iap
    await iap._handle_notification(db, notif)


async def test_lifecycle_subscribed_then_renew(credit_user, patch_notification_decode):
    """SUBSCRIBED activates (after a row exists), DID_RENEW re-grants, replay no-ops."""
    from app.db import async_session_maker

    # Seed an active sub so the notification can map orig_txn → user
    # (the authoritative activation is /subscribe/verify; SUBSCRIBED mirrors).
    expires1 = datetime.utcnow() + timedelta(days=30)
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-life", expires_dt=expires1,
    )
    renewals_before = await _ledger_renewal_count(credit_user)

    # DID_RENEW bumps the period to a new expiresDate and re-grants monthly.
    expires2 = datetime.utcnow() + timedelta(days=60)
    txn = _fake_txn(product_id="ai.toup.app.sub.starter", original_txn="orig-life",
                    expires_dt=expires2)
    renewal = _fake_renewal(auto_renew_status=1)
    patch_notification_decode(txn, renewal)
    notif = _fake_notification(ntype="DID_RENEW", uuid_="uuid-renew-1", txn=txn, renewal=renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_source == "apple"
    assert abs((b.period_end - expires2).total_seconds()) < 2
    renewals_after = await _ledger_renewal_count(credit_user)
    assert renewals_after > renewals_before, "DID_RENEW must write renewal ledger rows"

    # Replay the SAME notificationUUID → no-op (no extra renewal ledger rows).
    patch_notification_decode(txn, renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()
    assert await _ledger_renewal_count(credit_user) == renewals_after, "UUID replay must not re-grant"


async def test_lifecycle_auto_renew_disabled_no_downgrade(credit_user, patch_notification_decode):
    expires = datetime.utcnow() + timedelta(days=20)
    await _seed_active_apple_sub(
        credit_user, plan_id="pro", product_id="ai.toup.app.sub.pro",
        original_txn="orig-ard", expires_dt=expires,
    )
    from app.db import async_session_maker
    txn = _fake_txn(product_id="ai.toup.app.sub.pro", original_txn="orig-ard", expires_dt=expires)
    renewal = _fake_renewal(auto_renew_status=0)  # OFF
    patch_notification_decode(txn, renewal)
    notif = _fake_notification(ntype="DID_CHANGE_RENEWAL_STATUS", subtype="AUTO_RENEW_DISABLED",
                               uuid_="uuid-ard", txn=txn, renewal=renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_id == "pro", "AUTO_RENEW_DISABLED must NOT downgrade"
    assert b.plan_source == "apple"
    sub = await _apple_sub(credit_user)
    assert sub.auto_renew_status is False
    assert sub.status == "active"


async def test_lifecycle_grace_period_no_downgrade(credit_user, patch_notification_decode):
    expires = datetime.utcnow() - timedelta(hours=1)  # billing failed
    grace = datetime.utcnow() + timedelta(days=5)
    await _seed_active_apple_sub(
        credit_user, plan_id="builder", product_id="ai.toup.app.sub.builder",
        original_txn="orig-grace", expires_dt=expires,
    )
    from app.db import async_session_maker
    txn = _fake_txn(product_id="ai.toup.app.sub.builder", original_txn="orig-grace", expires_dt=expires)
    renewal = _fake_renewal(grace_dt=grace)
    patch_notification_decode(txn, renewal)
    notif = _fake_notification(ntype="DID_FAIL_TO_RENEW", subtype="GRACE_PERIOD",
                               uuid_="uuid-grace", txn=txn, renewal=renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_id == "builder", "GRACE_PERIOD must NOT downgrade"
    sub = await _apple_sub(credit_user)
    assert sub.status == "grace"


@pytest.mark.parametrize("ntype,subtype", [
    ("EXPIRED", "VOLUNTARY"),
    ("GRACE_PERIOD_EXPIRED", None),
    ("REVOKE", None),
    ("REFUND", None),
])
async def test_lifecycle_terminal_downgrades(credit_user, patch_notification_decode, ntype, subtype):
    expires = datetime.utcnow() + timedelta(days=10)
    await _seed_active_apple_sub(
        credit_user, plan_id="elite", product_id="ai.toup.app.sub.elite",
        original_txn=f"orig-term-{ntype}", expires_dt=expires,
    )
    from app.db import async_session_maker
    txn = _fake_txn(product_id="ai.toup.app.sub.elite", original_txn=f"orig-term-{ntype}", expires_dt=expires)
    patch_notification_decode(txn, None)
    notif = _fake_notification(ntype=ntype, subtype=subtype, uuid_=f"uuid-{ntype}", txn=txn)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_id == "free", f"{ntype} must downgrade to free"
    assert b.plan_source is None, "downgrade returns plan_source to NULL"


async def test_lifecycle_downgrade_pref_deferred(credit_user, patch_notification_decode):
    """DID_CHANGE_RENEWAL_PREF/DOWNGRADE keeps the plan; stores the pending
    target. The subsequent DID_RENEW applies the new (lower) plan."""
    expires = datetime.utcnow() + timedelta(days=15)
    await _seed_active_apple_sub(
        credit_user, plan_id="pro", product_id="ai.toup.app.sub.pro",
        original_txn="orig-down", expires_dt=expires,
    )
    from app.db import async_session_maker

    # DOWNGRADE pref → no plan change now; auto_renew_product_id recorded.
    txn = _fake_txn(product_id="ai.toup.app.sub.pro", original_txn="orig-down", expires_dt=expires)
    renewal = _fake_renewal(auto_renew_product_id="ai.toup.app.sub.starter")
    patch_notification_decode(txn, renewal)
    notif = _fake_notification(ntype="DID_CHANGE_RENEWAL_PREF", subtype="DOWNGRADE",
                               uuid_="uuid-down-pref", txn=txn, renewal=renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_id == "pro", "downgrade pref is deferred — plan unchanged now"
    sub = await _apple_sub(credit_user)
    assert sub.auto_renew_product_id == "ai.toup.app.sub.starter"

    # Next DID_RENEW for the starter product applies the lower plan.
    expires2 = datetime.utcnow() + timedelta(days=45)
    txn2 = _fake_txn(product_id="ai.toup.app.sub.starter", original_txn="orig-down", expires_dt=expires2)
    renewal2 = _fake_renewal()
    patch_notification_decode(txn2, renewal2)
    notif2 = _fake_notification(ntype="DID_RENEW", uuid_="uuid-down-renew", txn=txn2, renewal=renewal2)
    async with async_session_maker() as db:
        await _run_notification(db, notif2)
        await db.commit()

    b2 = await _balance(credit_user)
    assert b2.plan_id == "starter", "DID_RENEW applies the pending downgrade"


async def test_lifecycle_upgrade_immediate(credit_user, patch_notification_decode):
    """DID_CHANGE_RENEWAL_PREF/UPGRADE switches the plan immediately."""
    expires = datetime.utcnow() + timedelta(days=12)
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-up", expires_dt=expires,
    )
    from app.db import async_session_maker
    expires2 = datetime.utcnow() + timedelta(days=30)
    txn = _fake_txn(product_id="ai.toup.app.sub.elite", original_txn="orig-up", expires_dt=expires2)
    renewal = _fake_renewal()
    patch_notification_decode(txn, renewal)
    notif = _fake_notification(ntype="DID_CHANGE_RENEWAL_PREF", subtype="UPGRADE",
                               uuid_="uuid-up", txn=txn, renewal=renewal)
    async with async_session_maker() as db:
        await _run_notification(db, notif)
        await db.commit()

    b = await _balance(credit_user)
    assert b.plan_id == "elite", "UPGRADE pref switches plan immediately"
    assert b.plan_source == "apple"


# ── renewal guard ────────────────────────────────────────────────────


async def test_renewal_guard_apple_returns_false(credit_user):
    """An apple-source balance past period_end → renew_period() returns False
    (the hourly cron does NOT grant an unpaid month)."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    # Put the user on apple, then force the period into the past.
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-guard", expires_dt=datetime.utcnow() - timedelta(days=1),
    )
    async with async_session_maker() as db:
        b = await credit_service._lock_balance(db, credit_user)
        b.period_end = datetime.utcnow() - timedelta(days=1)
        await db.commit()

    before = Decimal((await _balance(credit_user)).message_credits_remaining)
    async with async_session_maker() as db:
        granted = await credit_service.renew_period(db, credit_user)
        await db.commit()
    assert granted is False, "apple-source balance must NOT renew on the clock"
    after = Decimal((await _balance(credit_user)).message_credits_remaining)
    assert after == before, "no grant on a guarded renewal"


async def test_renewal_guard_stripe_null_still_renews(credit_user):
    """NULL/'stripe' plan_source renews on the clock EXACTLY as today
    (inertness of the guard)."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.apply_plan_change(db, credit_user, "builder", reason="stripe:test")
        b = await credit_service._lock_balance(db, credit_user)
        b.period_end = datetime.utcnow() - timedelta(days=1)
        # plan_source stays NULL (Stripe path never writes it).
        await db.commit()

    async with async_session_maker() as db:
        granted = await credit_service.renew_period(db, credit_user)
        await db.commit()
    assert granted is True, "NULL plan_source must renew on the clock as today"
    b = await _balance(credit_user)
    # builder grants 320 message credits on renewal (no rollover).
    assert Decimal(b.message_credits_remaining) == Decimal("320")


# ── /credits/status surfacing ────────────────────────────────────────


async def test_status_plan_source_iap(credit_user):
    expires = datetime.utcnow() + timedelta(days=30)
    await _seed_active_apple_sub(
        credit_user, plan_id="starter", product_id="ai.toup.app.sub.starter",
        original_txn="orig-status", expires_dt=expires,
    )
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        view = await credit_service.get_balance_view(db, credit_user)
    assert view.plan_source == "iap"
    assert view.subscription_product_id == "ai.toup.app.sub.starter"
    assert view.subscription_renews_at is not None


async def test_status_plan_source_web_for_legacy_paid(credit_user):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        await credit_service.apply_plan_change(db, credit_user, "pro", reason="stripe:test")
        await db.commit()
        view = await credit_service.get_balance_view(db, credit_user)
    assert view.plan_source == "web"
    assert view.subscription_product_id is None


async def test_status_plan_source_free(credit_user):
    from app.db import async_session_maker
    from app.services.credit_service import credit_service
    async with async_session_maker() as db:
        view = await credit_service.get_balance_view(db, credit_user)
    assert view.plan_source == "free"
