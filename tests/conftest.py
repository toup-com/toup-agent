"""
Shared pytest fixtures for the backend test suite.

Fixtures live here so any test file in backend/tests/ can use them
without duplicating boilerplate. Existing billing-critical test files
depend on:
    - `client`                — httpx.AsyncClient against the test app
    - `auth_headers`          — Bearer-authenticated headers for a fresh user
    - `test_user_id`          — the user id behind `auth_headers`
    - `stripe_test_mode`      — marks a test as requiring STRIPE_SECRET_KEY (sk_test_*)
    - `signed_stripe_event`   — factory that builds a Stripe-signed webhook payload
    - `stripe_cleanup`        — registers Stripe resources for teardown deletion

The test app mounts every router the billing/webhook flow needs —
we rebuild it inline here rather than depending on `app.main:app`
because that target doesn't include the billing or vps routers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Callable

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# ── Test app ──────────────────────────────────────────────────────────
#
# The production `platform_main` runs extra lifespan setup we don't need
# (scheduler, MCP server, etc.). Build a minimal app that includes only
# what billing/webhook tests touch: auth, agent_setup, billing, vps.


def _build_test_app() -> FastAPI:
    from app.api.auth import router as auth_router
    from app.api.agent_setup import router as agent_setup_router
    from app.api.billing import router as billing_router
    from app.api.vps import router as vps_router
    from app.api.llm_setup import router as llm_setup_router
    from app.config import settings

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(agent_setup_router, prefix=settings.api_prefix)
    app.include_router(billing_router, prefix=settings.api_prefix)
    app.include_router(vps_router, prefix=settings.api_prefix)
    app.include_router(llm_setup_router, prefix=settings.api_prefix)
    return app


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Fresh in-memory schema for every test — no cross-test pollution."""
    from app.db import init_db, drop_db

    await init_db()
    yield
    await drop_db()


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    """Unique user per test via a random email — prevents cross-test collisions."""
    email = f"test-{uuid.uuid4().hex[:12]}@toup.test"
    password = "test-password-1234"
    reg = await client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Test User"},
    )
    assert reg.status_code in (200, 201), f"register failed: {reg.status_code} {reg.text}"
    token = reg.json().get("access_token")
    if not token:
        login = await client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def test_user_id(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    me = await client.get("/api/auth/me", headers=auth_headers)
    assert me.status_code == 200, me.text
    return me.json()["id"]


# ── Stripe test-mode gate ─────────────────────────────────────────────
#
# Tests that need real Stripe API access (creating a Customer, confirming
# a PaymentMethod, etc.) are gated on this fixture. Without STRIPE_SECRET_KEY
# set to a sk_test_* key, they skip cleanly rather than fail — devs without
# keys get a clean run, CI with secrets gets full coverage.


@pytest.fixture
def stripe_test_mode() -> str:
    key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not key.startswith("sk_test_"):
        pytest.skip(
            "STRIPE_SECRET_KEY (sk_test_*) not set — skipping live-Stripe test"
        )
    return key


@pytest.fixture
def stripe_cleanup(stripe_test_mode: str):
    """Register Stripe resources for deletion in teardown."""
    import stripe

    stripe.api_key = stripe_test_mode
    created_customers: list[str] = []
    created_subscriptions: list[str] = []

    yield {
        "customer": created_customers.append,
        "subscription": created_subscriptions.append,
    }

    # Cancel subscriptions first (auto-cancels on customer delete too, but be explicit).
    for sub_id in created_subscriptions:
        try:
            stripe.Subscription.cancel(sub_id)
        except Exception:
            pass
    for cust_id in created_customers:
        try:
            stripe.Customer.delete(cust_id)
        except Exception:
            pass


# ── Webhook signing helper ────────────────────────────────────────────
#
# Stripe webhooks are signed with HMAC-SHA256 over "{timestamp}.{payload}".
# We replicate Stripe's scheme so tests can POST deterministic, valid-signed
# events without needing the `stripe` CLI or a live webhook forwarder.


def _sign_stripe_payload(payload: str, secret: str, timestamp: int | None = None) -> str:
    ts = timestamp or int(time.time())
    signed = f"{ts}.{payload}"
    sig = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


@pytest.fixture
def signed_stripe_event() -> Callable[..., tuple[str, dict[str, str]]]:
    """
    Returns a factory that builds (payload_json, headers) for a signed
    Stripe webhook event. Uses the test webhook secret from env, falling
    back to a deterministic test secret set on app.config.settings.
    """

    def _factory(
        event_type: str,
        obj: dict[str, Any],
        *,
        event_id: str | None = None,
        secret: str | None = None,
        valid: bool = True,
    ) -> tuple[str, dict[str, str]]:
        webhook_secret = (
            secret
            or os.environ.get("STRIPE_WEBHOOK_SECRET")
            or "whsec_test_deadbeef_deadbeef_deadbeef_deadbeef"
        )
        # Ensure app code sees the same secret we're signing with.
        from app.config import settings as app_settings

        app_settings.stripe_webhook_secret = webhook_secret

        event = {
            "id": event_id or f"evt_test_{uuid.uuid4().hex[:16]}",
            "object": "event",
            "api_version": "2024-06-20",
            "created": int(time.time()),
            "type": event_type,
            "livemode": False,
            "data": {"object": obj},
        }
        payload = json.dumps(event, separators=(",", ":"))
        sig_header = _sign_stripe_payload(payload, webhook_secret)
        if not valid:
            # Corrupt the signature to simulate a forged event.
            sig_header = sig_header[:-4] + "0000"
        return payload, {"stripe-signature": sig_header, "Content-Type": "application/json"}

    return _factory
