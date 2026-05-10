"""
T1d — OAuth state token signing.

The `state` parameter on the OAuth authorize URL is the trust anchor
that binds the eventual callback to a specific user_id + connector_id +
in-flight session. We sign it with HMAC-SHA256 over a JSON payload
using `oauth_state_secret` (architecture D2 — independent from
jwt_secret so rotating it does not log out users).

State lifecycle:
  1. /connect → `sign_state(user_id, connector_id)` mints a fresh
     token with a CSRF nonce and `iat` timestamp.
  2. Token + sha256(token) inserted into ConnectorOAuthSession with
     code_verifier, expires_at = now + 10 min.
  3. Browser → provider → /callback?state=<token>.
  4. /callback calls `verify_state(token)` → returns the parsed payload
     or raises StateVerificationError. The caller then looks up the
     ConnectorOAuthSession by sha256(token), atomic-deletes it
     (single-use), and exchanges the auth code for tokens.

Why HMAC and not JWT:
  - Single-use, short-lived (10 min). Replay defense lives in
    ConnectorOAuthSession's PK + atomic delete; the state itself only
    needs to prove "this hasn't been forged" and "the user_id +
    connector_id haven't been tampered with."
  - JWT brings encoding/decoding overhead, alg negotiation, claim-set
    debate, and a temptation to put fields in the token that should
    live in the session row. HMAC keeps the trust boundary tight.
  - We follow the same module shape as `sensitive_action_token.py` but
    deliberately don't share its primitives (different purpose, TTL,
    threat model — coupling them invites future "oh let's use this
    for X too" drift).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from app.config import settings


class StateVerificationError(Exception):
    """Raised when a state token is missing, malformed, expired, or
    has a bad HMAC signature. The /callback handler maps this to a 400.
    The exception message is intentionally generic — the caller logs
    a more specific reason at WARNING level."""


@dataclass(frozen=True)
class StatePayload:
    user_id: str
    connector_id: str
    csrf_nonce: str
    iat: int  # unix epoch seconds


# ─── Configuration guards ────────────────────────────────────────────


def assert_configured() -> None:
    """Call from platform lifespan when enable_connector_oauth=True.
    Fails fast if oauth_state_secret is missing in the env — matches
    the credential_crypto.py contract."""
    secret = (settings.oauth_state_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "oauth_state_secret is not set — connector OAuth cannot "
            "initialise. Set OAUTH_STATE_SECRET (e.g. "
            "`secrets.token_urlsafe(32)`)."
        )


def _secret_bytes() -> bytes:
    secret = (settings.oauth_state_secret or "").strip()
    if not secret:
        raise StateVerificationError(
            "oauth_state_secret not configured — refusing to sign or verify"
        )
    return secret.encode("utf-8")


# ─── Sign ────────────────────────────────────────────────────────────


def sign_state(user_id: str, connector_id: str) -> str:
    """Mint a fresh state token binding (user_id, connector_id) plus a
    32-byte CSRF nonce and the current unix-epoch `iat`. Returns
    `<payload_b64>.<sig_b64>` — both URL-safe base64 (no padding) so
    the token can ride a URL query string without escaping."""
    if not user_id or not connector_id:
        raise ValueError("sign_state: user_id and connector_id required")
    payload = {
        "user_id": user_id,
        "connector_id": connector_id,
        "csrf_nonce": secrets.token_urlsafe(24),
        "iat": int(time.time()),
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_b64 = _b64url_encode(body)
    sig = hmac.new(_secret_bytes(), body_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{body_b64}.{sig_b64}"


# ─── Verify ──────────────────────────────────────────────────────────


def verify_state(token: str) -> StatePayload:
    """Parse + verify a state token. Returns the payload on success;
    raises StateVerificationError otherwise.

    Failure modes (all map to 400 at the API):
      * missing / wrong-shape token
      * tampered payload (HMAC mismatch)
      * expired (`iat` more than oauth_state_ttl_seconds in the past)
      * future-dated (`iat` more than 60 seconds in the future — clock skew)
    """
    if not token or "." not in token:
        raise StateVerificationError("token missing or malformed")
    try:
        body_b64, sig_b64 = token.rsplit(".", 1)
    except ValueError:
        raise StateVerificationError("token missing signature")

    expected_sig = hmac.new(
        _secret_bytes(), body_b64.encode("ascii"), hashlib.sha256,
    ).digest()
    try:
        provided_sig = _b64url_decode(sig_b64)
    except ValueError:
        raise StateVerificationError("signature is not valid base64url")

    # constant-time compare — defends against signature timing oracles.
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise StateVerificationError("HMAC signature mismatch")

    try:
        body = _b64url_decode(body_b64)
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        raise StateVerificationError("payload is not valid JSON")

    for required in ("user_id", "connector_id", "csrf_nonce", "iat"):
        if required not in payload:
            raise StateVerificationError(f"payload missing field {required!r}")

    iat = payload["iat"]
    if not isinstance(iat, int):
        raise StateVerificationError("iat is not an integer")

    now = int(time.time())
    age = now - iat
    if age < -60:  # clock skew tolerance
        raise StateVerificationError("iat is in the future")
    if age > settings.oauth_state_ttl_seconds:
        raise StateVerificationError(
            f"state expired (age={age}s, ttl={settings.oauth_state_ttl_seconds}s)"
        )

    return StatePayload(
        user_id=payload["user_id"],
        connector_id=payload["connector_id"],
        csrf_nonce=payload["csrf_nonce"],
        iat=iat,
    )


# ─── State hash for ConnectorOAuthSession PK ─────────────────────────


def state_hash(token: str) -> str:
    """Stable hash of the state token, used as the
    ConnectorOAuthSession primary key. SHA-256 hex (64 chars) so we
    never store the redeemable token at rest — only its fingerprint.

    Caller looks up by `state_hash(token)` then atomic-deletes the row
    (single-use replay defense)."""
    return hashlib.sha256(token.encode("ascii")).hexdigest()


# ─── b64url helpers (no padding) ─────────────────────────────────────


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)
