"""Sign in with Apple — identity-token verification + token revocation.

SIGN-IN: verifies the `identity_token` (a JWT) returned by Apple's native
AuthenticationServices flow (expo-apple-authentication on iOS) against
Apple's public JWKS. No Apple secret is required for sign-in — only the
public keys at https://appleid.apple.com/auth/keys and the expected
audience (the app's bundle id, settings.apple_client_id). Mirrors the
cached-PyJWKClient pattern used by gmail_pubsub.py.

REVOCATION (Guideline 5.1.1(v)): account deletion must revoke the user's
Apple tokens. That requires a client_secret JWT signed with a "Sign in
with Apple" key (.p8 + Key ID + Team ID) created in the Apple Developer
portal — SEPARATE from the App Store Connect API key. At sign-in we
exchange the one-time `authorization_code` for a refresh token
(`exchange_authorization_code`) and store it encrypted; at deletion we
revoke it (`revoke_token`). Both no-op (and log) when the key isn't
configured — `siwa_revocation_configured()` gates the callers — so this
module is safe to deploy before the portal key exists.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
import jwt  # PyJWT — distinct from python-jose (`jose`) used elsewhere
from jwt import PyJWKClient

from app.config import settings

logger = logging.getLogger(__name__)

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"
_JWKS_TTL_SECONDS = 3600

_jwks_client: Optional[PyJWKClient] = None


class AppleTokenError(Exception):
    """Raised when an Apple identity token fails verification."""


def _get_jwks_client() -> PyJWKClient:
    """Process-global PyJWKClient for Apple's JWKS endpoint.

    Built ONCE and reused. PyJWT's own JWK-set cache (cache_keys + lifespan)
    governs refetch — we pass lifespan=_JWKS_TTL_SECONDS so the keys are
    refreshed ~hourly instead of PyJWT's 300s default (the previous code
    wrapped a 3600s rebuild around a client whose inner cache was only 300s,
    so the wrapper TTL was dead intent and Apple was hit ~every 5 min).
    """
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(
            APPLE_JWKS_URL, cache_keys=True, lifespan=_JWKS_TTL_SECONDS,
        )
    return _jwks_client


def prewarm_apple_jwks() -> None:
    """Fetch + cache Apple's JWKS so the first Sign-in-with-Apple after a
    deploy doesn't pay the blocking network fetch. Best-effort, sync — call
    it off the event loop (asyncio.to_thread) from app startup. The cache is
    process-local, so every deploy starts cold; this warms it ahead of the
    first real sign-in.
    """
    try:
        _get_jwks_client().get_signing_keys()
    except Exception as e:  # network / Apple outage — sign-in still works cold
        logger.warning("[apple-auth] JWKS prewarm failed: %s", e)


def verify_apple_identity_token(identity_token: str, audience: str) -> dict:
    """Verify an Apple identity token and return its claims.

    Validates the RS256 signature against Apple's JWKS, the issuer, the
    audience (the app bundle id / Service ID), and expiry. Raises
    AppleTokenError on any failure.
    """
    if not identity_token:
        raise AppleTokenError("missing identity token")
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(identity_token)
        claims = jwt.decode(
            identity_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=APPLE_ISSUER,
            options={"require": ["sub", "exp", "iss", "aud"]},
        )
        return claims
    except Exception as e:  # PyJWKClientError / InvalidTokenError / network
        raise AppleTokenError(str(e)) from e


# ── Token revocation (Guideline 5.1.1(v)) ───────────────────────────


def siwa_revocation_configured() -> bool:
    """True only when the Sign in with Apple signing key is fully set.

    All three pieces come from the portal key, not the ASC API key. When
    any is missing the exchange/revoke helpers no-op so sign-in and
    account deletion keep working — the only effect is that we cannot
    revoke Apple tokens until the operator sets the env vars.
    """
    return bool(
        settings.apple_team_id
        and settings.apple_key_id
        and settings.apple_private_key
    )


def _private_key_pem() -> str:
    # Railway / .env single-line values store the PEM with literal "\n";
    # normalise back to real newlines so PyJWT can parse the EC key.
    return (settings.apple_private_key or "").replace("\\n", "\n").strip()


def _build_client_secret() -> str:
    """Build the ES256 client_secret JWT Apple's token endpoints require.

    iss=Team ID, sub=client_id (the app bundle id), aud=Apple, signed with
    the Sign in with Apple key (kid=Key ID). Short-lived (10 min); Apple
    permits up to 6 months but there's no reason to mint long-lived ones.
    """
    now = int(time.time())
    return jwt.encode(
        {
            "iss": settings.apple_team_id,
            "iat": now,
            "exp": now + 600,
            "aud": APPLE_ISSUER,
            "sub": settings.apple_client_id,
        },
        _private_key_pem(),
        algorithm="ES256",
        headers={"kid": settings.apple_key_id, "alg": "ES256"},
    )


async def exchange_authorization_code(code: str) -> Optional[dict]:
    """Exchange a one-time Apple `authorization_code` for tokens.

    Returns Apple's token response (contains `refresh_token`) or None when
    revocation isn't configured / no code. Raises AppleTokenError on an
    HTTP failure so the caller can log it (best-effort, never blocks
    sign-in).
    """
    if not code or not siwa_revocation_configured():
        return None
    data = {
        "client_id": settings.apple_client_id,
        "client_secret": _build_client_secret(),
        "grant_type": "authorization_code",
        "code": code,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(APPLE_TOKEN_URL, data=data)
    if resp.status_code != 200:
        raise AppleTokenError(
            f"authorization_code exchange failed: {resp.status_code} {resp.text[:200]}"
        )
    return resp.json()


async def revoke_token(token: str, token_type_hint: str = "refresh_token") -> bool:
    """Revoke an Apple token (refresh or access). Best-effort.

    Returns True on a 200 from Apple, False when not configured / no token
    / non-200 (logged). Apple returns 200 with an empty body on success.
    """
    if not token or not siwa_revocation_configured():
        return False
    data = {
        "client_id": settings.apple_client_id,
        "client_secret": _build_client_secret(),
        "token": token,
        "token_type_hint": token_type_hint,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(APPLE_REVOKE_URL, data=data)
    except Exception as e:
        logger.warning("[apple-auth] revoke request errored: %s", e)
        return False
    if resp.status_code != 200:
        logger.warning(
            "[apple-auth] revoke non-200: %s %s", resp.status_code, resp.text[:200]
        )
        return False
    return True
