"""Regression lock for the 2026-05-30/31 "new signup can't chat" class.

Incident: a brand-new user signs up (Gmail OAuth or /register), the pool binds
a reachable agent container, and the user's FIRST chat message ("hi") returns
"Error: Your API key is invalid. Please check your API key in Settings."

Root cause: the container was bound BEFORE the bundle LLM credentials existed.
`pool_service._build_bind_payload` forwards `connect_token` / `llm_mode` to the
bridge ONLY when truthy, so a config still on the signup defaults
(bundle_status='none', connect_token=NULL, llm_mode='manual') booted a container
with an EMPTY TOUP_TOKEN → the first proxy call 401'd → friendly-error rendered
"Your API key is invalid".

This file pins the three guarantees of the fix so a future refactor can't
silently regress them:

  1. pool_service.claim_for_user mints free-tier credentials BEFORE building the
     bind payload (the single chokepoint every container-claim path funnels
     through), and skips paid-awaiting-Stripe users.
  2. Both signup paths (auth.register + Google callback) eagerly activate before
     claiming, as explicit defense-in-depth.
  3. ws_chat._friendly_error maps the proxy's "not provisioned yet" auth
     signatures to an honest "finishing setup" message — NEVER the misleading
     "Your API key is invalid" (which a genuine BYOK bad key still gets).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "test")

BACKEND_DIR = Path(__file__).resolve().parent.parent
_POOL_SRC = (BACKEND_DIR / "app/services/pool_service.py").read_text(encoding="utf-8")
_AUTH_SRC = (BACKEND_DIR / "app/api/auth.py").read_text(encoding="utf-8")


def test_claim_for_user_mints_credentials_before_bind():
    """The credential guarantee must run INSIDE claim_for_user and BEFORE the
    bind payload is built — otherwise a fresh config (connect_token=NULL) binds
    a tokenless container and the first chat 401s."""
    idx_activate = _POOL_SRC.find("activate_free_tier")
    idx_payload = _POOL_SRC.find("payload = await _build_bind_payload")
    assert idx_activate != -1, "claim_for_user must call activate_free_tier"
    assert idx_payload != -1, "claim_for_user must build a bind payload"
    assert idx_activate < idx_payload, (
        "activate_free_tier MUST run before _build_bind_payload so the bridge "
        "bind carries a valid connect_token / TOUP_TOKEN"
    )
    # Must not clobber paid/BYOK flows: paid-awaiting-Stripe users are skipped.
    assert "is_paid_awaiting_stripe" in _POOL_SRC, (
        "the chokepoint must skip paid-plan users awaiting their Stripe webhook"
    )


def test_signup_paths_activate_free_tier_before_claim():
    """Both signup paths (register + Google callback) must eagerly mint
    credentials before claim_or_prewarm — 2 imports + 2 calls = 4 references."""
    assert _AUTH_SRC.count("activate_free_tier") >= 4, (
        "auth.register and google_auth_callback must each call activate_free_tier "
        "before claiming a container"
    )


def test_friendly_error_setup_in_progress_not_invalid_key():
    """Proxy 'not provisioned yet' 401/403 signatures must surface an honest,
    recoverable message — never the misleading 'Your API key is invalid'."""
    from app.api.ws_chat import _friendly_error

    class _E(Exception):
        def __init__(self, m, s=None):
            super().__init__(m)
            self.status_code = s

    for msg in (
        "Invalid token",
        "Missing token. Provide either 'Authorization: Bearer <TOUP_TOKEN>'",
        "Bundle subscription is not active",
    ):
        out = _friendly_error(_E(msg, 401)).lower()
        assert "finishing setup" in out, f"expected setup-in-progress for {msg!r}, got {out!r}"
        assert "api key is invalid" not in out, f"misleading invalid-key message for {msg!r}"


def test_friendly_error_real_bad_key_still_invalid():
    """A genuine bad BYOK key must STILL say the key is invalid — the new branch
    is bundle-proxy-specific and must not swallow real auth failures."""
    from app.api.ws_chat import _friendly_error

    class _E(Exception):
        def __init__(self, m, s=None):
            super().__init__(m)
            self.status_code = s

    out = _friendly_error(_E("401 authentication_error: invalid x-api-key", 401)).lower()
    assert "api key is invalid" in out
