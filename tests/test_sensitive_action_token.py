"""Unit tests for app.services.sensitive_action_token.

The module signs JWTs with `settings.jwt_secret` and persists redeemed
jtis to the `sensitive_action_redemptions` table for replay defense.
These tests cover the eight rejection paths verify_sensitive_action_token
must catch, plus the happy-path roundtrip.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
import uuid

import pytest
from fastapi import HTTPException
from jose import jwt

from app.config import settings
from app.db import async_session_maker
from app.db.models import SensitiveActionRedemption
from sqlalchemy import select
from app.services.sensitive_action_token import (
    DEFAULT_TTL_SECONDS,
    MAX_TTL_SECONDS,
    SensitiveActionPurpose,
    TOKEN_KIND_CLAIM,
    TOKEN_KIND_VALUE,
    issue_sensitive_action_token,
    verify_sensitive_action_token,
)


USER_A = "u-aaaa"
USER_B = "u-bbbb"


# ── Issuance ─────────────────────────────────────────────────────────


def test_issue_returns_token_and_expiry_within_ttl() -> None:
    token, exp = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT, ttl_seconds=120,
    )
    assert isinstance(token, str)
    assert len(token) > 50
    # `exp` is timezone-aware UTC so we compare against an aware now.
    delta = exp - datetime.now(timezone.utc)
    # 120s requested, allow generous slack for test machine timing.
    assert timedelta(seconds=110) < delta < timedelta(seconds=125)


def test_issue_clamps_ttl_above_max() -> None:
    _, exp = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT, ttl_seconds=99999,
    )
    delta = exp - datetime.now(timezone.utc)
    assert delta <= timedelta(seconds=MAX_TTL_SECONDS + 5)


def test_issue_clamps_ttl_below_min() -> None:
    _, exp = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT, ttl_seconds=10,
    )
    delta = exp - datetime.now(timezone.utc)
    # Floor is 60s.
    assert delta >= timedelta(seconds=55)


def test_issued_token_carries_required_claims() -> None:
    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    payload = jwt.decode(
        token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
    )
    assert payload["sub"] == USER_A
    assert payload["purpose"] == "delete_account"
    assert payload[TOKEN_KIND_CLAIM] == TOKEN_KIND_VALUE
    assert "jti" in payload
    assert "iat" in payload
    assert "exp" in payload


# ── Verification: happy path ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_roundtrip_succeeds_and_writes_redemption_row() -> None:
    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    async with async_session_maker() as db:
        await verify_sensitive_action_token(
            db, token,
            expected_user_id=USER_A,
            expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
        )
        await db.commit()
        # The jti row should now exist.
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
        )
        rows = await db.execute(
            select(SensitiveActionRedemption).where(
                SensitiveActionRedemption.jti == payload["jti"],
            )
        )
        row = rows.scalar_one()
        assert row.user_id == USER_A
        assert row.purpose == "delete_account"


# ── Verification: rejection paths ────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_rejects_replay() -> None:
    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    async with async_session_maker() as db1:
        await verify_sensitive_action_token(
            db1, token,
            expected_user_id=USER_A,
            expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
        )
        await db1.commit()
    # Second redemption attempt — same jti, fresh session.
    async with async_session_maker() as db2:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db2, token,
                expected_user_id=USER_A,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_wrong_user() -> None:
    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db, token,
                expected_user_id=USER_B,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_wrong_purpose() -> None:
    # Two purposes for the test, since the production enum has only one.
    class FakePurposes(StrEnum):
        OTHER = "other_action"

    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            # Cast: verify takes a SensitiveActionPurpose, but the wrong-
            # purpose branch only compares .value strings, so any StrEnum
            # member with a different value triggers the rejection.
            await verify_sensitive_action_token(
                db, token,
                expected_user_id=USER_A,
                expected_purpose=FakePurposes.OTHER,  # type: ignore[arg-type]
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_signature_forgery() -> None:
    token, _ = issue_sensitive_action_token(
        USER_A, SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    # Flip a character in the signature segment (last segment after '.').
    head, payload, sig = token.rsplit(".", 2)
    forged = head + "." + payload + "." + ("A" if sig[0] != "A" else "B") + sig[1:]
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db, forged,
                expected_user_id=USER_A,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_expired_token() -> None:
    # Mint an already-expired token using unix-epoch timestamps directly.
    # NOT `datetime.utcnow().timestamp()` — that silently uses local time
    # and produces a wrong unix value on any non-UTC machine.
    import time as time_module

    now_ts = int(time_module.time())
    payload = {
        "sub": USER_A,
        "purpose": "delete_account",
        TOKEN_KIND_CLAIM: TOKEN_KIND_VALUE,
        "iat": now_ts - 600,
        "exp": now_ts - 10,
        "jti": str(uuid.uuid4()),
    }
    expired = jwt.encode(
        payload, settings.jwt_secret, algorithm=settings.jwt_algorithm,
    )
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db, expired,
                expected_user_id=USER_A,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_access_token_via_kind_claim() -> None:
    """An access token (no `kind` claim) must NOT pass as a sensitive-
    action token even if its sub matches. Closes the bypass where a
    stolen access token could authorize an irreversible operation."""
    from app.services.auth_service import create_access_token

    access_token = create_access_token(USER_A)
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db, access_token,
                expected_user_id=USER_A,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_empty_token() -> None:
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as excinfo:
            await verify_sensitive_action_token(
                db, "",
                expected_user_id=USER_A,
                expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
            )
        assert excinfo.value.status_code == 401
