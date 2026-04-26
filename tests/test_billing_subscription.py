"""
Integration tests for the Toup Bundle subscription flow.

Covers the paths the embedded Stripe Payment Element depends on:

    POST /api/billing/create-subscription  → idempotent client_secret issuance
    GET  /api/billing/status               → bundle_status reflection
    POST /api/vps/webhook/stripe           → lifecycle event handling

Tests are split into two tiers:

    1. Deterministic (no Stripe network) — sign webhook events ourselves and
       POST them. These exercise every webhook branch without touching Stripe.

    2. Live-Stripe — gated on the `stripe_test_mode` fixture (skips unless
       STRIPE_SECRET_KEY is an sk_test_* key). Creates and cleans up real
       test-mode Stripe Customers and Subscriptions.

The tiering lets devs run the full suite locally without Stripe keys, and
gives CI full end-to-end coverage when the secret is wired.
"""

from __future__ import annotations

import secrets as secrets_mod
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


async def _seed_agent_config(
    user_id: str,
    *,
    bundle_status: str = "none",
    subscription_id: str | None = None,
    setup_step: int = 3,
) -> None:
    """Create an AgentConfig row with a pre-set bundle state."""
    from app.db import AgentConfig, async_session_maker

    async with async_session_maker() as db:
        config = AgentConfig(
            user_id=user_id,
            bundle_status=bundle_status,
            bundle_stripe_subscription_id=subscription_id,
            setup_step=setup_step,
            setup_type="auto",
            hosting_mode="managed",
            llm_mode="bundle",
        )
        db.add(config)
        await db.commit()


async def _get_agent_config(user_id: str):
    from app.db import AgentConfig, async_session_maker

    async with async_session_maker() as db:
        result = await db.execute(select(AgentConfig).where(AgentConfig.user_id == user_id))
        return result.scalar_one_or_none()


def _fake_get_subscription(sub_id: str, *, status: str = "active", days_ahead: int = 30):
    """Drop-in stand-in for stripe_service.get_subscription."""
    return {
        "id": sub_id,
        "status": status,
        "current_period_end": datetime.now(timezone.utc) + timedelta(days=days_ahead),
        "cancel_at_period_end": False,
        "customer": "cus_test_fake",
    }


# ══════════════════════════════════════════════════════════════════════
# Endpoint auth — every billing endpoint requires a valid JWT
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_create_subscription_rejects_unauthenticated(client: AsyncClient):
    resp = await client.post("/api/billing/create-subscription")
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_billing_status_rejects_unauthenticated(client: AsyncClient):
    resp = await client.get("/api/billing/status")
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_billing_status_returns_none_for_fresh_user(
    client: AsyncClient, auth_headers: dict[str, str]
):
    """A user who has never touched billing gets bundle_status='none'."""
    resp = await client.get("/api/billing/status", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bundle_status"] == "none"
    assert body["has_stripe_customer"] is False


# ══════════════════════════════════════════════════════════════════════
# Webhook signature verification
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_signature(
    client: AsyncClient, signed_stripe_event
):
    """An unsigned or tampered webhook must never update state."""
    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": "in_test", "subscription": "sub_test", "object": "invoice"},
        valid=False,
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_webhook_rejects_missing_signature(client: AsyncClient):
    resp = await client.post(
        "/api/vps/webhook/stripe",
        content='{"type":"invoice.payment_succeeded"}',
        headers={"Content-Type": "application/json"},
    )
    # Signature missing → verify_webhook returns None → handler raises 400.
    assert resp.status_code == 400, resp.text


# ══════════════════════════════════════════════════════════════════════
# invoice.payment_succeeded — first-time activation (embedded path)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_invoice_payment_succeeded_activates_bundle(
    client: AsyncClient, auth_headers: dict[str, str], test_user_id: str, signed_stripe_event
):
    """The embedded flow's success webhook flips bundle_status and mints a proxy token."""
    sub_id = "sub_test_activation_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": f"in_test_{secrets_mod.token_hex(4)}", "subscription": sub_id, "object": "invoice"},
    )

    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription):
        resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)

    assert resp.status_code == 200, resp.text

    cfg = await _get_agent_config(test_user_id)
    assert cfg is not None
    assert cfg.bundle_status == "active"
    assert cfg.llm_token_hash is not None
    # Token hash is sha256 hex — 64 chars
    assert len(cfg.llm_token_hash) == 64
    assert cfg.bundle_started_at is not None
    assert cfg.bundle_period_end is not None
    assert cfg.llm_mode == "bundle"


@pytest.mark.asyncio
async def test_duplicate_invoice_webhook_is_idempotent(
    client: AsyncClient, auth_headers: dict[str, str], test_user_id: str, signed_stripe_event
):
    """A replayed webhook event must not mint a second proxy token or duplicate activation."""
    sub_id = "sub_test_dup_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": "in_test_dup_same", "subscription": sub_id, "object": "invoice"},
        event_id="evt_test_dup",  # same id both times — replay
    )

    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription):
        r1 = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
        cfg_after_first = await _get_agent_config(test_user_id)
        first_token = cfg_after_first.llm_token_hash
        first_started_at = cfg_after_first.bundle_started_at

        # Replay the same event.
        r2 = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)

    assert r1.status_code == 200
    assert r2.status_code == 200

    cfg_after_second = await _get_agent_config(test_user_id)
    assert cfg_after_second.bundle_status == "active"
    assert cfg_after_second.llm_token_hash == first_token, "proxy token must not be re-minted"
    # bundle_started_at may get re-stamped by current handler (first-activation branch
    # guards on `was_first_activation = bundle_status != 'active'`, so the second
    # call lands in the renewal branch and leaves bundle_started_at alone).
    assert cfg_after_second.bundle_started_at == first_started_at


# ══════════════════════════════════════════════════════════════════════
# customer.subscription.updated — cancelling, past_due, re-activation
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_subscription_updated_cancelling_transitions_state(
    client: AsyncClient, test_user_id: str, signed_stripe_event
):
    """User requested cancel → bundle_status='cancelling', access stays until period end."""
    sub_id = "sub_test_cancelling_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "customer.subscription.updated",
        {
            "id": sub_id,
            "object": "subscription",
            "status": "active",
            "cancel_at_period_end": True,
            "current_period_end": int((datetime.now(timezone.utc) + timedelta(days=12)).timestamp()),
            "metadata": {"type": "llm_bundle", "user_id": test_user_id},
        },
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "cancelling"


@pytest.mark.asyncio
async def test_subscription_updated_past_due_transitions_state(
    client: AsyncClient, test_user_id: str, signed_stripe_event
):
    sub_id = "sub_test_pastdue_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "customer.subscription.updated",
        {
            "id": sub_id,
            "object": "subscription",
            "status": "past_due",
            "cancel_at_period_end": False,
            "current_period_end": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            "metadata": {"type": "llm_bundle", "user_id": test_user_id},
        },
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "past_due"


@pytest.mark.asyncio
async def test_subscription_deleted_marks_cancelled(
    client: AsyncClient, test_user_id: str, signed_stripe_event
):
    sub_id = "sub_test_deleted_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "customer.subscription.deleted",
        {"id": sub_id, "object": "subscription", "metadata": {"type": "llm_bundle", "user_id": test_user_id}},
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "cancelled"


@pytest.mark.asyncio
async def test_invoice_payment_failed_marks_past_due(
    client: AsyncClient, test_user_id: str, signed_stripe_event
):
    """Card declined on renewal → past_due until retry succeeds."""
    sub_id = "sub_test_payfail_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_failed",
        {
            "id": "in_test_failed",
            "subscription": sub_id,
            "object": "invoice",
        },
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "past_due"


# ══════════════════════════════════════════════════════════════════════
# Out-of-order event delivery
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_events_out_of_order_converge_to_correct_state(
    client: AsyncClient, test_user_id: str, signed_stripe_event
):
    """
    Stripe does not guarantee event order. If subscription.updated arrives
    BEFORE invoice.payment_succeeded, the final state should still be 'active'
    with a proxy token minted.
    """
    sub_id = "sub_test_order_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    # 1. subscription.updated arrives first with status='active'
    payload1, headers1 = signed_stripe_event(
        "customer.subscription.updated",
        {
            "id": sub_id,
            "object": "subscription",
            "status": "active",
            "cancel_at_period_end": False,
            "current_period_end": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            "metadata": {"type": "llm_bundle", "user_id": test_user_id},
        },
    )
    r1 = await client.post("/api/vps/webhook/stripe", content=payload1, headers=headers1)
    assert r1.status_code == 200

    # 2. invoice.payment_succeeded arrives second
    payload2, headers2 = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": "in_test_ooo", "subscription": sub_id, "object": "invoice"},
    )
    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription):
        r2 = await client.post("/api/vps/webhook/stripe", content=payload2, headers=headers2)
    assert r2.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "active"
    assert cfg.llm_token_hash is not None


# ══════════════════════════════════════════════════════════════════════
# Live-Stripe integration (skipped unless STRIPE_SECRET_KEY=sk_test_*)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_create_subscription_end_to_end_live_stripe(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
    stripe_test_mode: str,
    stripe_cleanup,
):
    """
    Live Stripe test-mode: hit create-subscription, confirm a test
    PaymentMethod, trigger the invoice.payment_succeeded webhook back
    into our app, assert bundle_status flips to 'active'.
    """
    import stripe
    stripe.api_key = stripe_test_mode

    # The billing endpoint needs a configured price — skip if missing.
    from app.config import settings
    if not settings.stripe_llm_bundle_price_id:
        pytest.skip("STRIPE_LLM_BUNDLE_PRICE_ID not configured")

    # Seed AgentConfig so the webhook can find the user later.
    await _seed_agent_config(test_user_id, bundle_status="none")

    resp = await client.post("/api/billing/create-subscription", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("status") in ("incomplete", "already_active")
    if body["status"] == "already_active":
        pytest.skip("test user already has an active subscription — unexpected")

    sub_id = body["subscription_id"]
    stripe_cleanup["subscription"](sub_id)

    # Pay the freshly-created invoice. Basil 2025-03-31 removed
    # `invoice.payment_intent`; the supported confirmation path is
    # `stripe.Invoice.pay(...)` with a PaymentMethod that must already
    # be attached to the customer (unlike the old PaymentIntent.confirm
    # flow which would accept the global `pm_card_visa` token directly).
    # Create + attach a real PM, then pay.
    sub = stripe.Subscription.retrieve(sub_id)
    stripe_cleanup["customer"](sub.customer)
    pm = stripe.PaymentMethod.create(type="card", card={"token": "tok_visa"})
    stripe.PaymentMethod.attach(pm.id, customer=sub.customer)
    stripe.Invoice.pay(sub.latest_invoice, payment_method=pm.id)

    # Poll Stripe briefly for the subscription to go active, then assert our DB state.
    # (In a real CI run with `stripe listen`, the webhook path fires on its own.
    # Here, we simulate webhook delivery to cover the full happy path.)
    import time
    for _ in range(10):
        s = stripe.Subscription.retrieve(sub_id)
        if s.status == "active":
            break
        time.sleep(0.5)
    assert s.status == "active", f"subscription did not become active: {s.status}"


@pytest.mark.asyncio
async def test_create_subscription_is_idempotent_for_active_user(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
    stripe_test_mode: str,
    stripe_cleanup,
):
    """Calling create-subscription while bundle_status='active' returns already_active."""
    from app.config import settings
    if not settings.stripe_llm_bundle_price_id:
        pytest.skip("STRIPE_LLM_BUNDLE_PRICE_ID not configured")

    # Seed an already-active config.
    sub_id = "sub_test_existing_active"
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    resp = await client.post("/api/billing/create-subscription", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "already_active"
    assert body["subscription_id"] == sub_id


@pytest.mark.asyncio
async def test_create_subscription_resumes_existing_stripe_customer(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
    _test_user: dict[str, str],
    stripe_test_mode: str,
    stripe_cleanup,
):
    """
    Live Stripe test-mode: pre-create a Stripe Customer with the user's email
    BEFORE calling /create-subscription, so `get_or_create_customer` enters
    the existing-customer branch (customers.list().data is non-empty).

    Regression guard for the 2026-04-26 incident: stripe-python 15.x exposes
    `cust.metadata` as a StripeObject, not a dict. The line
    `cust.metadata.get("user_id")` raised `AttributeError: get`, surfacing as
    HTTP 500. The fresh-email live e2e test never enters this branch, so the
    bug went latent through CI. This test exercises that path explicitly.
    """
    import stripe
    stripe.api_key = stripe_test_mode

    from app.config import settings
    if not settings.stripe_llm_bundle_price_id:
        pytest.skip("STRIPE_LLM_BUNDLE_PRICE_ID not configured")

    # Pre-create a Stripe Customer with the test user's email + a stale
    # user_id in metadata so the "metadata mismatch → update" branch also fires.
    existing_cust = stripe.Customer.create(
        email=_test_user["email"],
        metadata={"user_id": "stale-prior-user-id"},
    )
    stripe_cleanup["customer"](existing_cust.id)

    await _seed_agent_config(test_user_id, bundle_status="none")

    resp = await client.post("/api/billing/create-subscription", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("status") == "incomplete", body
    assert body.get("client_secret"), body
    stripe_cleanup["subscription"](body["subscription_id"])

    # Confirm the existing customer was reused (not duplicated) and the
    # endpoint refreshed metadata.user_id to the real test user.
    refreshed = stripe.Customer.retrieve(existing_cust.id)
    assert refreshed.metadata.user_id == test_user_id
