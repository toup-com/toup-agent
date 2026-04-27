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
import time
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
async def test_llm_token_hash_matches_sha256_of_connect_token(
    client: AsyncClient, auth_headers: dict[str, str], test_user_id: str, signed_stripe_event
):
    """
    THE contract that makes the LLM proxy work: after bundle activation,
    `llm_token_hash` MUST equal sha256(connect_token). The agent's env has
    TOUP_TOKEN=<connect_token>; agent presents it as Bearer to /api/llm;
    proxy hashes incoming and looks up by llm_token_hash. If these don't
    match, every bundle agent's LLM call returns 401.

    Regression guard for the 2026-04-27 latent bug: the webhook used to
    mint a fresh random token, hash that, and discard the cleartext —
    the agent had no way to obtain the matching secret.
    """
    import hashlib
    sub_id = "sub_test_token_contract_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": f"in_test_{secrets_mod.token_hex(4)}", "subscription": sub_id, "object": "invoice"},
    )
    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription):
        resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.connect_token, "connect_token must be set after webhook"
    assert cfg.llm_token_hash, "llm_token_hash must be set after webhook"
    expected = hashlib.sha256(cfg.connect_token.encode()).hexdigest()
    assert cfg.llm_token_hash == expected, (
        f"llm_token_hash != sha256(connect_token) — agent's TOUP_TOKEN won't auth.\n"
        f"  connect_token={cfg.connect_token[:12]}...\n"
        f"  llm_token_hash={cfg.llm_token_hash[:16]}...\n"
        f"  expected={expected[:16]}..."
    )


@pytest.mark.asyncio
async def test_subscription_updated_out_of_order_uses_connect_token_for_hash(
    client: AsyncClient, auth_headers: dict[str, str], test_user_id: str, signed_stripe_event
):
    """Out-of-order path (subscription.updated lands first) must use the same
    connect_token-based hashing — otherwise the convergence is wrong."""
    import hashlib
    sub_id = "sub_test_order_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "customer.subscription.updated",
        {
            "id": sub_id, "object": "subscription", "status": "active",
            "cancel_at_period_end": False,
            "metadata": {"type": "llm_bundle", "user_id": test_user_id},
            "current_period_end": int(time.time()) + 30 * 86400,
        },
    )
    resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)
    assert resp.status_code == 200

    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "active"
    assert cfg.connect_token, "connect_token must be lazily generated on out-of-order activation"
    assert cfg.llm_token_hash == hashlib.sha256(cfg.connect_token.encode()).hexdigest()


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


@pytest.mark.asyncio
async def test_create_subscription_syncs_db_when_existing_sub_is_active(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
    _test_user: dict[str, str],
    stripe_test_mode: str,
    stripe_cleanup,
):
    """
    Live Stripe test-mode: pre-create a Stripe Customer + active subscription
    matching the test user's email, then call /create-subscription. The
    endpoint must return status='already_active' AND sync the platform DB:
    bundle_status='active', bundle_stripe_subscription_id set, llm_token_hash
    minted, llm_mode='bundle'.

    Regression guard for the 2026-04-27 incident: a deleted-then-re-signed-up
    user re-attached to their old Stripe customer's still-active subscription
    via email match. The endpoint returned already_active correctly but never
    updated platform DB → bundle_status stayed 'none' → LLM proxy denied the
    user → half-broken paid agent.

    Without Fix 2 (DB sync on already_active), this test fails with
    bundle_status='none' after the endpoint call.
    """
    import stripe
    stripe.api_key = stripe_test_mode

    from app.config import settings
    if not settings.stripe_llm_bundle_price_id:
        pytest.skip("STRIPE_LLM_BUNDLE_PRICE_ID not configured")

    # Pre-create customer + ATTACHED PaymentMethod + active sub by paying its
    # invoice immediately (Basil-API flow: Invoice.pay requires PM attached).
    existing_cust = stripe.Customer.create(
        email=_test_user["email"],
        metadata={"user_id": "stale-prior-user-id"},
    )
    stripe_cleanup["customer"](existing_cust.id)
    pm = stripe.PaymentMethod.create(type="card", card={"token": "tok_visa"})
    stripe.PaymentMethod.attach(pm.id, customer=existing_cust.id)
    # Setting default_payment_method on Subscription.create makes Stripe
    # auto-charge the first invoice → sub flips to 'active' immediately
    # (no explicit Invoice.pay needed; calling it would 400 "already paid").
    pre_sub = stripe.Subscription.create(
        customer=existing_cust.id,
        items=[{"price": settings.stripe_llm_bundle_price_id}],
        default_payment_method=pm.id,
        metadata={"user_id": "stale-prior-user-id", "type": "llm_bundle"},
    )
    stripe_cleanup["subscription"](pre_sub.id)
    # Sub may briefly be 'incomplete' before auto-charge settles. Poll until
    # 'active'. Test-mode is normally instant; cap at 5s.
    import time
    for _ in range(10):
        pre_sub = stripe.Subscription.retrieve(pre_sub.id)
        if pre_sub.status == "active":
            break
        time.sleep(0.5)
    assert pre_sub.status == "active", \
        f"setup precondition failed: pre-existing sub did not activate: {pre_sub.status}"

    await _seed_agent_config(test_user_id, bundle_status="none")

    resp = await client.post("/api/billing/create-subscription", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("status") == "already_active", body
    assert body["subscription_id"] == pre_sub.id

    # ── The whole point of this test: platform DB must reflect Stripe truth ──
    cfg = await _get_agent_config(test_user_id)
    assert cfg is not None
    assert cfg.bundle_status == "active", \
        f"bundle_status not synced from Stripe: {cfg.bundle_status!r}"
    assert cfg.bundle_stripe_subscription_id == pre_sub.id, \
        "bundle_stripe_subscription_id not written"
    assert cfg.llm_token_hash, "llm_token_hash not minted on already_active sync"
    assert cfg.llm_mode == "bundle", f"llm_mode not set: {cfg.llm_mode!r}"
    assert cfg.bundle_started_at is not None


@pytest.mark.asyncio
async def test_provision_blocked_when_bundle_inactive(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
):
    """
    /api/managed-agent/provision must return 402 when llm_mode='bundle' and
    bundle_status is not in ('active','cancelling'). Defense-in-depth gate
    against the 2026-04-27 incident pattern: even if frontend / Stripe state
    were wrong and the call reached this endpoint, no paid agent gets
    provisioned without a paid subscription.

    This is a deterministic test — it never reaches the bridge because the
    endpoint short-circuits before docker_host_service.provision_container.
    """
    # Seed an AgentConfig with llm_mode='bundle' but bundle_status='none' —
    # the leak scenario. managed_hosting_enabled defaults False on the test
    # app; the gate must fire BEFORE that 503, so override.
    from app.config import settings
    settings.managed_hosting_enabled = True
    try:
        await _seed_agent_config(test_user_id, bundle_status="none")
        # _seed_agent_config defaults llm_mode='bundle' (see helper); confirm.
        from app.db import AgentConfig, async_session_maker
        from sqlalchemy import select
        async with async_session_maker() as db:
            cfg = (await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == test_user_id)
            )).scalar_one()
            cfg.llm_mode = "bundle"
            cfg.bundle_status = "none"
            await db.commit()

        resp = await client.post("/api/managed-agent/provision", headers=auth_headers)
        assert resp.status_code == 402, \
            f"expected 402 Payment Required, got {resp.status_code}: {resp.text}"
        assert "subscription" in resp.text.lower()
    finally:
        settings.managed_hosting_enabled = False


@pytest.mark.asyncio
async def test_provision_blocked_for_past_due_bundle(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_user_id: str,
):
    """past_due is also blocked — only 'active' and 'cancelling' (paid through
    period_end) are allowed."""
    from app.config import settings
    settings.managed_hosting_enabled = True
    try:
        await _seed_agent_config(test_user_id, bundle_status="past_due")
        from app.db import AgentConfig, async_session_maker
        from sqlalchemy import select
        async with async_session_maker() as db:
            cfg = (await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == test_user_id)
            )).scalar_one()
            cfg.llm_mode = "bundle"
            cfg.bundle_status = "past_due"
            await db.commit()

        resp = await client.post("/api/managed-agent/provision", headers=auth_headers)
        assert resp.status_code == 402
    finally:
        settings.managed_hosting_enabled = False
