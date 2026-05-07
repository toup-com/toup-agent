"""Cloudflare Turnstile token verification.

Phase 3 of docs/onboarding/prewarm-phase0.md. Adds a CAPTCHA gate to
/auth/register to reduce signup spam (which, post-Phase-1 prewarm-on-
Soul.save, would translate into Docker-container creation cost).

Turnstile docs: https://developers.cloudflare.com/turnstile/

Behavior:
  - When `settings.turnstile_secret_key` is empty, verification is
    skipped (dev / CI / self-hosted environments without CF in front).
    Returns True immediately so the register endpoint accepts the
    request without a token field.
  - When the secret is set, a missing or empty `cf_turnstile_token`
    field in the request returns False (caller raises 400).
  - When the token is present, POSTs to Turnstile's `siteverify`
    endpoint with `secret` + `response` + `remoteip`. Returns True iff
    Cloudflare returns `success=true`.
  - Network errors → False (fail-closed). Better to reject a real user
    than to let a bot through; the user can retry.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile_token(
    token: Optional[str],
    *,
    remote_ip: Optional[str] = None,
    timeout_s: float = 5.0,
) -> bool:
    """Return True iff the request should be accepted.

    Skipped (returns True) when no secret is configured — local dev
    and CI never have a CF Turnstile binding. Production deploys MUST
    set TURNSTILE_SECRET_KEY for the gate to actually fire.
    """
    secret = (settings.turnstile_secret_key or "").strip()
    if not secret:
        # Skip verification — no secret configured.
        return True

    if not token or not token.strip():
        return False

    payload = {"secret": secret, "response": token.strip()}
    if remote_ip:
        payload["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(TURNSTILE_VERIFY_URL, data=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("[turnstile] verification failed (network/HTTP): %s", e)
        # Fail-closed: a CF outage would otherwise wave through every signup.
        return False

    success = bool(data.get("success"))
    if not success:
        # Log the error codes (no PII) for ops visibility into legitimate
        # vs malicious failures — `timeout-or-duplicate` is the most
        # common false-positive (token reused after stale form submit).
        logger.info(
            "[turnstile] reject codes=%s",
            data.get("error-codes") or [],
        )
    return success
