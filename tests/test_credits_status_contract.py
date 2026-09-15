"""The /api/credits/status contract — the one response that can crash a phone.

App Store build 109 is live. Its `creditView` reads `message`, `integration`
and `period_end` unconditionally, with no local boundary, and
`ChatScreen.tsx:1400` calls the same function — so an absent `message` does not
degrade a billing screen, it throws on the MAIN SCREEN of every shipped
client. `period_end` null renders a plausible WRONG date ("Resets Jan 1").

Therefore:

    message, integration, period_start and period_end are REQUIRED and
    non-Optional, forever. Fields may be ADDED. Nothing may be removed or
    relaxed.

The first test asserts that structurally, off the Pydantic model itself,
because a `= None` default is a one-character edit that no behavioural test
would notice on a populated fixture. The rest assert the values for each kind
of account this design can produce.

The Unlimited additions are all optional and all null for a free account, so
every existing client keeps rendering exactly what it rendered before. What
they buy is the answer to the question a grandfathered payer actually has:
Apple charges them $19.90 for `ai.toup.app.sub.builder` while they are
entitled to Unlimited, and a screen that showed only "Unlimited" would hide
both the subscription they can cancel and the price they are paying.
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
LEGACY_STARTER = "ai.toup.app.sub.starter"
UNLIMITED_PRODUCT = "ai.toup.app.sub.unlimited"

# Every field the shipped client dereferences without a guard.
REQUIRED_FIELDS = ("plan_id", "plan_display_name", "message", "integration",
                   "period_start", "period_end", "enforcement_enabled")


@pytest_asyncio.fixture
async def status_client():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.credits import billing_router, router as credits_router
    from app.config import settings

    app = FastAPI()
    app.include_router(credits_router, prefix=settings.api_prefix)
    app.include_router(billing_router, prefix=settings.api_prefix)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as ac:
        yield ac


async def _status(client, headers):
    from app.config import settings
    resp = await client.get(f"{settings.api_prefix}/credits/status", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _seed_apple(user_id: str, *, product: str, entitled_plan: str,
                      catalog_plan: str, grandfathered: bool = False):
    from app.db import async_session_maker
    from app.db.models import APPLE_SUB_ACTIVE, AppleSubscription
    from app.services.credit_service import credit_service

    now = datetime.utcnow()
    expires = now + timedelta(days=17)
    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, user_id)
        sub = AppleSubscription(
            user_id=user_id, original_transaction_id=f"o-{uuid.uuid4().hex[:12]}",
            product_id=product, plan_id=catalog_plan, status=APPLE_SUB_ACTIVE,
            expires_date=expires, auto_renew_status=True, environment="Production",
        )
        if grandfathered:
            sub.grandfathered_at = now
            sub.grandfathered_product_id = product
        db.add(sub)
        await credit_service.activate_subscription(
            db, user_id, entitled_plan, "apple", expires,
        )
        await db.commit()


# ── the structural guarantee ─────────────────────────────────────────


def test_the_four_load_bearing_fields_are_still_required():
    """Structural, not behavioural. Relaxing one of these to Optional is a
    one-character edit that every populated-fixture test would sail past, and
    the failure mode is a crash on the main screen of a shipped binary that
    cannot be patched from the server."""
    from app.api.credits import CreditStatusResponse

    fields = CreditStatusResponse.model_fields
    for name in REQUIRED_FIELDS:
        assert name in fields, f"{name} was REMOVED from CreditStatusResponse"
        assert fields[name].is_required(), (
            f"{name} was relaxed to Optional — App Store build 109's "
            f"creditView dereferences it with no guard, and ChatScreen calls "
            f"the same function"
        )

    # …and specifically not Optional[...] with a None default sneaking in.
    for name in ("message", "integration"):
        assert fields[name].annotation.__name__ == "BucketStatus"


def test_bucket_status_keeps_its_two_numeric_fields_required():
    """`remaining` and `monthly` feed a division in LowBalancePill and a meter
    on the phone. Null-able numbers there degrade to 0, which reads as "out of
    credits" to a user who is not."""
    from app.api.credits import BucketStatus

    fields = BucketStatus.model_fields
    assert fields["remaining"].is_required()
    assert fields["monthly"].is_required()


def test_every_new_field_is_optional_and_defaulted():
    """The other half of the contract: an ADDED field must never make an
    older client's payload invalid, and must never make the endpoint refuse
    to serialise a free account."""
    from app.api.credits import CreditStatusResponse

    fields = CreditStatusResponse.model_fields
    for name in ("unlimited", "unlimited_reason", "billed_plan_id",
                 "billed_plan_display_name",
                 "plan_source", "subscription_product_id",
                 "subscription_renews_at", "purchased_credits_remaining"):
        assert not fields[name].is_required(), name


# ── free: unchanged ──────────────────────────────────────────────────


async def test_free_account_is_semantically_unchanged(
    status_client, auth_headers, test_user_id,
):
    body = await _status(status_client, auth_headers)

    assert body["plan_id"] == "free"
    assert body["plan_source"] == "free"
    assert body["message"]["remaining"] == 100.0
    assert body["message"]["monthly"] == 100.0
    assert body["integration"]["monthly"] == 500.0
    assert body["period_end"] is not None
    # Every Unlimited addition is inert for a free account.
    assert body["unlimited"] is False
    assert body["unlimited_reason"] is None
    assert body["billed_plan_id"] is None
    assert body["billed_plan_display_name"] is None


async def test_free_account_costs_no_extra_queries(
    status_client, auth_headers, test_user_id, monkeypatch,
):
    """resolve_unlimited is two reads to answer "no", and /status is called on
    every app foreground. It must not run for an account that does not hold
    the entitlement — asserted by making it explode if it does."""
    from app.services import entitlement

    async def boom(*a, **kw):
        raise AssertionError(
            "resolve_unlimited ran for a free account — /status must not pay "
            "the controller-side query cost for the 99% case"
        )

    monkeypatch.setattr(entitlement, "resolve_unlimited", boom)
    body = await _status(status_client, auth_headers)
    assert body["unlimited"] is False


# ── legacy paid, not grandfathered ───────────────────────────────────


async def test_a_plain_apple_subscriber_sees_the_plan_they_bought(
    status_client, auth_headers, test_user_id,
):
    await _seed_apple(test_user_id, product=LEGACY_STARTER,
                      entitled_plan="starter", catalog_plan="starter")

    body = await _status(status_client, auth_headers)

    assert body["plan_id"] == "starter"
    assert body["plan_source"] == "iap"     # paywall already hidden on mobile
    assert body["unlimited"] is False
    assert body["unlimited_reason"] is None
    # Entitled plan and billed plan agree, which is the ordinary case.
    assert body["billed_plan_id"] == "starter"


# ── grandfathered: the whole point of the new fields ─────────────────


async def test_a_grandfathered_payer_sees_unlimited_AND_what_they_pay_for(
    status_client, auth_headers, test_user_id,
):
    """Apple charges $19.90 for Builder; the entitlement is Unlimited. Both
    facts have to reach the screen or the user cannot reason about the
    subscription they hold."""
    await _seed_apple(test_user_id, product=LEGACY_BUILDER,
                      entitled_plan="unlimited", catalog_plan="builder",
                      grandfathered=True)

    body = await _status(status_client, auth_headers)

    assert body["plan_id"] == "unlimited"
    assert body["plan_display_name"] == "Unlimited"
    assert body["unlimited"] is True
    assert body["unlimited_reason"] == "apple"
    # …and the subscription behind it, NAMED. Deliberately not priced — see
    # test_no_price_is_served_from_the_web_catalogue below.
    assert body["billed_plan_id"] == "builder"
    assert body["billed_plan_display_name"] == "Builder"
    assert body["subscription_product_id"] == LEGACY_BUILDER
    assert body["subscription_renews_at"] is not None
    # plan_source stays 'iap'. An unrecognised value renders a balance block
    # with NO call to action, and NULL on a paid plan reads as stripe
    # server-side and 409-blocks the account from ever buying on iOS.
    assert body["plan_source"] == "iap"
    # The load-bearing four are still real values.
    assert body["message"]["monthly"] == 1_000_000.0
    assert body["integration"]["monthly"] == 1_000_000.0
    assert body["period_end"] is not None


async def test_the_billed_plan_is_the_CATALOGUE_plan_not_the_resolver(
    status_client, auth_headers, test_user_id,
):
    """plan_for_subscription() answers 'unlimited' for a grandfathered legacy
    product — using it here would make billed_plan_id echo plan_id and destroy
    the only field that says what Apple actually charges."""
    await _seed_apple(test_user_id, product=LEGACY_BUILDER,
                      entitled_plan="unlimited", catalog_plan="builder",
                      grandfathered=True)

    body = await _status(status_client, auth_headers)
    assert body["billed_plan_id"] != body["plan_id"]
    assert body["billed_plan_id"] == "builder"


async def test_an_unlimited_product_subscriber_is_billed_for_unlimited(
    status_client, auth_headers, test_user_id,
):
    await _seed_apple(test_user_id, product=UNLIMITED_PRODUCT,
                      entitled_plan="unlimited", catalog_plan="unlimited")

    body = await _status(status_client, auth_headers)
    assert body["plan_id"] == "unlimited"
    assert body["unlimited_reason"] == "apple"
    assert body["billed_plan_id"] == "unlimited"


# ── the other two ways to be unlimited ───────────────────────────────


async def test_an_admin_is_unlimited_for_the_reason_it_has_always_been(
    status_client, auth_headers, test_user_id,
):
    """`unlimited` broadens from "role == admin" to "holds the entitlement" —
    a strict SUPERSET, so nothing that was true stops being true."""
    from app.db import User, async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        (await db.get(User, test_user_id)).role = "admin"
        await credit_service.get_or_create_balance(db, test_user_id)
        await db.commit()

    body = await _status(status_client, auth_headers)
    assert body["unlimited"] is True
    assert body["unlimited_reason"] == "admin"
    assert body["plan_id"] == "free"          # the plan row is untouched
    assert body["billed_plan_id"] is None


async def test_a_grantee_is_named_as_a_grant(status_client, auth_headers, test_user_id):
    """A comp is not a purchase. It must be distinguishable on the wire so a
    support surface can tell an entitlement from a courtesy."""
    from app.db import User, async_session_maker
    from app.db.models import APPLE_SUB_ACTIVE, AppleSubscription
    from app.services import entitlement
    from app.services.auth_service import get_password_hash
    from app.services.credit_service import credit_service

    sponsor = str(uuid.uuid4())
    txn = f"o-{uuid.uuid4().hex[:12]}"
    async with async_session_maker() as db:
        db.add(User(id=sponsor, email=f"s-{uuid.uuid4().hex[:8]}@example.com",
                    hashed_password=get_password_hash("test-password-1234"),
                    name="Sponsor"))
        await db.flush()
        await credit_service.get_or_create_balance(db, sponsor)
        await credit_service.get_or_create_balance(db, test_user_id)
        db.add(AppleSubscription(
            user_id=sponsor, original_transaction_id=txn,
            product_id=LEGACY_BUILDER, plan_id="builder", status=APPLE_SUB_ACTIVE,
            expires_date=datetime.utcnow() + timedelta(days=20),
            auto_renew_status=True, environment="Production",
        ))
        await credit_service.activate_subscription(
            db, sponsor, "builder", "apple", datetime.utcnow() + timedelta(days=20),
        )
        await entitlement.create_grant(
            db, granted_to_user_id=test_user_id, sponsor_kind="apple",
            sponsor_original_txn_id=txn, reason="second account",
            granted_by_user_id=sponsor, granted_by_email="ops@toup.ai",
        )
        await db.commit()

    body = await _status(status_client, auth_headers)
    assert body["plan_id"] == "unlimited"
    assert body["unlimited"] is True
    assert body["unlimited_reason"] == "grant"
    # The grantee has no Apple subscription of their own, so there is nothing
    # to be billed for — and inventing a price here would be a lie on a
    # billing screen.
    assert body["billed_plan_id"] is None
    # plan_source mirrors the SPONSOR's source, so the client renders
    # "Subscribed — manage in Settings" and shows no purchase affordance.
    assert body["plan_source"] == "iap"


async def test_an_unexplained_stamp_is_reported_as_such(
    status_client, auth_headers, test_user_id,
):
    """`plan` is deliberately distinguishable from `apple`/`grant`/`admin`: it
    means the balance says unlimited and nothing explains it — the
    reconciler's orphan_entitlement class. Collapsing it into 'apple' would
    make a support screen assert a subscription that does not exist."""
    from app.db import async_session_maker
    from app.services.credit_service import credit_service

    async with async_session_maker() as db:
        await credit_service.get_or_create_balance(db, test_user_id)
        await credit_service.activate_subscription(
            db, test_user_id, "unlimited", "apple",
            datetime.utcnow() + timedelta(days=30),
        )
        await db.commit()

    body = await _status(status_client, auth_headers)
    assert body["unlimited"] is True
    assert body["unlimited_reason"] == "plan"
    assert body["billed_plan_id"] is None


# ── the shape survives every kind of account ─────────────────────────


async def test_no_account_shape_can_produce_a_null_in_the_required_four(
    status_client, auth_headers, test_user_id,
):
    """One assertion over every account this design can create. The client
    crash is not conditional on the plan, so neither is the guarantee."""
    from app.db import User, async_session_maker
    from app.services.credit_service import credit_service

    async def _assert_shape(label: str):
        body = await _status(status_client, auth_headers)
        for name in REQUIRED_FIELDS:
            assert body.get(name) is not None, f"{name} is null for {label}"
        for bucket in ("message", "integration"):
            assert isinstance(body[bucket]["remaining"], (int, float)), label
            assert isinstance(body[bucket]["monthly"], (int, float)), label
        assert body["plan_source"] in ("iap", "web", "free"), label

    await _assert_shape("free")

    await _seed_apple(test_user_id, product=LEGACY_STARTER,
                      entitled_plan="starter", catalog_plan="starter")
    await _assert_shape("legacy apple paid")

    async with async_session_maker() as db:
        await credit_service.activate_subscription(
            db, test_user_id, "unlimited", "apple",
            datetime.utcnow() + timedelta(days=30),
        )
        await db.commit()
    await _assert_shape("unlimited")

    async with async_session_maker() as db:
        await credit_service.downgrade_to_free(db, test_user_id, "test")
        await db.commit()
    await _assert_shape("downgraded back to free")

    async with async_session_maker() as db:
        (await db.get(User, test_user_id)).role = "admin"
        await db.commit()
    await _assert_shape("admin")


# ── the PUBLIC catalogue ─────────────────────────────────────────────


async def test_the_public_pricing_catalogue_does_not_show_unlimited_yet(status_client):
    """`GET /api/billing/plans` is unauthenticated and filters on exactly
    `active = true`, so the plan row's `active` column IS the merchandising
    switch — for anonymous visitors, not just for subscribers.

    Shipping it active would publish, on the deploy that lands this branch and
    with no operator present, a sixth card at a price the founder has not
    chosen (UNLIMITED_PRICE_CENTS is documented as a placeholder), with an
    Upgrade button that dead-ends on "Stripe price not configured" because
    there is no Stripe price for Unlimited and deliberately never will be.
    """
    from app.config import settings

    resp = await status_client.get(f"{settings.api_prefix}/billing/plans")
    assert resp.status_code == 200, resp.text
    ids = {p["id"] for p in resp.json()["plans"]}
    assert "unlimited" not in ids, (
        "the Unlimited plan is merchandised on the public pricing page before "
        "an operator has said so"
    )
    # Anti-vacuity: the endpoint is really returning the catalogue.
    assert "free" in ids


async def test_unlimited_still_resolves_by_primary_key_while_inactive():
    """`active=false` must cost the entitlement nothing.

    Every internal path resolves the plan with `db.get(SubscriptionPlan, id)`
    and no active filter — which is precisely the property the cutover relies
    on to retire the four legacy tiers — so an inactive row must still grant,
    renew and read back normally.
    """
    from app.db import async_session_maker
    from app.db.models import SubscriptionPlan
    from app.services.credit_service import credit_service

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        plan = await db.get(SubscriptionPlan, "unlimited")
        assert plan is not None and plan.active is False
        await credit_service.get_or_create_balance(db, uid)
        await credit_service.activate_subscription(
            db, uid, "unlimited", "apple", datetime.utcnow() + timedelta(days=30),
        )
        await db.commit()

    async with async_session_maker() as db:
        bal = await credit_service.get_or_create_balance(db, uid)
    assert bal.plan_id == "unlimited"
    assert Decimal(bal.message_credits_remaining) == Decimal("1000000")


async def test_no_price_is_served_from_the_web_catalogue(
    status_client, auth_headers, test_user_id,
):
    """`billed_price_cents` shipped for one revision and was wrong in both
    amount and currency.

    It was read from `subscription_plans.price_cents` — the abandoned USD web
    ladder (1600/4000/8000/16000) that nobody has ever bought on — and it was
    only ever populated for an APPLE subscriber, because
    `view.subscription_product_id` is set only when plan_source == 'apple'.
    Apple charges 9.90/19.90/39.90/98.90, and for all three real payers in
    CAD. So the one field whose stated job was to disclose "the price they are
    paying" said roughly double it, with no currency attached, on the screen
    someone would use to decide whether to cancel.

    This schema has no source of truth for Apple storefront prices. The client
    has one — StoreKit's localised `Product.displayPrice` for
    `subscription_product_id`, which the payload still carries.
    """
    from app.api.credits import CreditStatusResponse

    assert "billed_price_cents" not in CreditStatusResponse.model_fields

    await _seed_apple(test_user_id, product=LEGACY_BUILDER,
                      entitled_plan="unlimited", catalog_plan="builder",
                      grandfathered=True)
    body = await _status(status_client, auth_headers)

    assert "billed_price_cents" not in body
    # No other key smuggles a price in either.
    assert not [k for k in body if "price" in k.lower()], sorted(body)
    # Anti-vacuity: the client still has everything it needs to ask StoreKit.
    assert body["subscription_product_id"] == LEGACY_BUILDER
    assert body["billed_plan_display_name"] == "Builder"
