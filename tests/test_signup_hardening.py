"""Phase 3 tests — signup rate limit + Turnstile verification.

Cases per docs/onboarding/prewarm-phase0.md Phase 3:
  * Rate limit: 6th signup from same IP within an hour returns 429.
  * Turnstile (when secret configured): missing/empty token → 400.
  * Turnstile (when secret configured): invalid token → 400.
  * Turnstile skipped when no secret configured (default test env).
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def reset_signup_limiter() -> AsyncIterator[None]:
    """The limiter is module-level; clear its state between tests so
    test order doesn't matter."""
    from app.services.rate_limiter import signup_rate_limiter
    signup_rate_limiter._attempts.clear()
    yield
    signup_rate_limiter._attempts.clear()


@pytest_asyncio.fixture
async def turnstile_secret_set() -> AsyncIterator[None]:
    """Set a Turnstile secret for the duration of one test (forces the
    verify path to actually evaluate the token rather than skip)."""
    from app.config import settings
    prev = settings.turnstile_secret_key
    settings.turnstile_secret_key = "test-secret-not-really-used-because-of-mock"
    try:
        yield
    finally:
        settings.turnstile_secret_key = prev


# ─── Rate limit ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signup_rate_limit_unit_logic():
    """Limiter logic: 5 records → 6th check returns retry seconds.
    Tested at the limiter directly to sidestep the test DB's spurious
    sqlite UNIQUE constraint on `users.is_canary` (the column is a
    Postgres partial-unique index in production; sqlite drops the
    `WHERE` predicate and treats it as a full unique constraint, which
    blocks more than one user row per test schema)."""
    from app.services.rate_limiter import SignupRateLimiter

    limiter = SignupRateLimiter(max_attempts=5, window_seconds=3600)
    ip = "1.2.3.4"
    for _ in range(5):
        assert limiter.check(ip) is None
        limiter.record(ip)
    # 6th attempt: blocked, with positive retry-after seconds
    retry = limiter.check(ip)
    assert retry is not None
    assert retry > 0


@pytest.mark.asyncio
async def test_signup_endpoint_returns_429_when_limiter_blocked(client):
    """HTTP-path test: pre-record 5 hits for the test client's IP, then
    confirm a single signup attempt returns 429 + Retry-After. Avoids
    creating multiple users (sqlite is_canary unique-index quirk
    documented in the unit test above)."""
    from app.services.rate_limiter import signup_rate_limiter
    import time

    # The TestClient's reported client IP is "127.0.0.1" by default
    # (httpx's ASGITransport sets it). Pre-record 5 attempts to fill
    # the limiter's window for this IP.
    client_ip = "127.0.0.1"
    now = time.time()
    signup_rate_limiter._attempts[client_ip] = [now] * 5

    r = await client.post(
        "/api/auth/register",
        json={
            "email": f"r-{uuid.uuid4().hex[:10]}@example.com",
            "password": "password1234",
            "name": "BlockedUser",
        },
    )
    assert r.status_code == 429
    assert "Retry-After" in r.headers


@pytest.mark.asyncio
async def test_signup_rate_limit_zero_disables_at_unit_level():
    """`max_attempts=0` short-circuits to "always allowed" without
    accumulating state. Validates the dev / CI escape hatch."""
    from app.services.rate_limiter import SignupRateLimiter

    limiter = SignupRateLimiter(max_attempts=0, window_seconds=3600)
    ip = "1.2.3.4"
    for _ in range(20):
        assert limiter.check(ip) is None
        limiter.record(ip)
    # Even after 20 records, never blocked.
    assert limiter.check(ip) is None


# ─── Turnstile ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signup_with_no_turnstile_secret_skips_verification(client):
    """Default test env has no TURNSTILE_SECRET_KEY → verification
    skipped → signup succeeds without a token field."""
    r = await client.post(
        "/api/auth/register",
        json={
            "email": f"t-{uuid.uuid4().hex[:10]}@example.com",
            "password": "password1234",
            "name": "NoTurnstileUser",
        },
    )
    assert r.status_code == 201


@pytest.mark.asyncio
async def test_signup_missing_turnstile_token_returns_400(
    client, turnstile_secret_set
):
    """When the secret IS configured, missing token → 400."""
    r = await client.post(
        "/api/auth/register",
        json={
            "email": f"t-{uuid.uuid4().hex[:10]}@example.com",
            "password": "password1234",
            "name": "MissingTokenUser",
            # cf_turnstile_token: deliberately omitted
        },
    )
    assert r.status_code == 400
    assert "captcha" in r.text.lower()


@pytest.mark.asyncio
async def test_signup_invalid_turnstile_token_returns_400(
    client, turnstile_secret_set
):
    """When the secret IS configured AND the token is present BUT
    Turnstile siteverify rejects it, return 400."""
    with patch(
        "app.api.auth.verify_turnstile_token",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = False
        r = await client.post(
            "/api/auth/register",
            json={
                "email": f"t-{uuid.uuid4().hex[:10]}@example.com",
                "password": "password1234",
                "name": "BadTokenUser",
                "cf_turnstile_token": "0.fake-token-CF-rejects",
            },
        )
    assert r.status_code == 400
    assert "captcha" in r.text.lower()


@pytest.mark.asyncio
async def test_signup_valid_turnstile_token_succeeds(
    client, turnstile_secret_set
):
    """When verify_turnstile_token returns True, signup proceeds."""
    with patch(
        "app.api.auth.verify_turnstile_token",
        new_callable=AsyncMock,
    ) as mock_verify:
        mock_verify.return_value = True
        r = await client.post(
            "/api/auth/register",
            json={
                "email": f"t-{uuid.uuid4().hex[:10]}@example.com",
                "password": "password1234",
                "name": "GoodTokenUser",
                "cf_turnstile_token": "0.valid-token",
            },
        )
    assert r.status_code == 201
