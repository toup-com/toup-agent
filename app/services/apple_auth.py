"""Sign in with Apple — identity-token verification.

Verifies the `identity_token` (a JWT) returned by Apple's native
AuthenticationServices flow (expo-apple-authentication on iOS) against
Apple's public JWKS. No Apple secret is required for sign-in — only the
public keys at https://appleid.apple.com/auth/keys and the expected
audience (the app's bundle id, settings.apple_client_id). Mirrors the
cached-PyJWKClient pattern used by gmail_pubsub.py.

NOTE: token revocation on account deletion (Guideline 5.1.1(v)) requires
an Apple Service ID + private key (client secret) generated in the Apple
Developer portal — a separate, portal-gated follow-up NOT needed for
sign-in.
"""

from __future__ import annotations

import time
from typing import Optional

import jwt  # PyJWT — distinct from python-jose (`jose`) used elsewhere
from jwt import PyJWKClient

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
_JWKS_TTL_SECONDS = 3600

_jwks_client: Optional[PyJWKClient] = None
_jwks_loaded_at: float = 0.0


class AppleTokenError(Exception):
    """Raised when an Apple identity token fails verification."""


def _get_jwks_client() -> PyJWKClient:
    """Lazy-built, TTL-cached PyJWKClient for Apple's JWKS endpoint."""
    global _jwks_client, _jwks_loaded_at
    now = time.monotonic()
    if _jwks_client is None or (now - _jwks_loaded_at) >= _JWKS_TTL_SECONDS:
        _jwks_client = PyJWKClient(APPLE_JWKS_URL)
        _jwks_loaded_at = now
    return _jwks_client


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
