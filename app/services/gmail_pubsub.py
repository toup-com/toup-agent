"""Gmail Pub/Sub trigger plumbing — platform-side.

Three responsibilities, kept in one module so the rollout fits in one
review:

  1. **JWT verification** of the Google Pub/Sub push token. Pub/Sub
     signs every push with a service-account JWT (RS256) whose `iss`
     is `accounts.google.com` and whose public keys live at
     `https://www.googleapis.com/oauth2/v1/certs`. We cache the JWKS
     for 1 h (the same TTL Google's docs recommend) and rotate on
     any verification miss. Reject pushes that fail signature, expiry,
     audience, or signer-email checks — Pub/Sub doesn't retry on 4xx
     so a forged push just dies.

  2. **`users.watch` provisioning + refresh**. Gmail watches expire
     every 7 days; we re-arm at day 6 via a system routine kind. The
     watch handle (history_id baseline + expiration timestamp) lives
     in `triggers.provider_state_json` — one watch per trigger.
     Provider-side, GCP guarantees ONE watch per (user, topic), so
     two triggers against the same Gmail account share a watch
     transparently.

  3. **History delta fetch**. Pub/Sub envelopes carry only
     `(emailAddress, historyId)` — to get the actual new message-ids
     the platform calls `users.history.list?startHistoryId=…` with
     the connector's refresh-token-derived access token. Tokens stay
     platform-side; the agent only ever sees message-ids in the
     dispatch envelope (Gate T2 fetches body via its own MCP path).

Failure modes documented per-function. Logs are structured (key=value
pairs); no PII leaks — sender / subject / body never get logged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jwt as pyjwt
from jwt import PyJWKClient

from app.config import settings
from app.services import connector_vault as _vault
from app.db.database import async_session_maker

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────


GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
# Google publishes the Pub/Sub push-token signing certs here. The
# JWKS endpoint also exists (oauth2/v3/certs) — PyJWKClient is
# happiest with the JWKS form. Both are signed by Google's root CA.
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"

# Cache the JWKS client. PyJWKClient itself caches signing keys; we
# add a wall-clock TTL on top so a compromised JWKS would stop being
# trusted after at most this many seconds.
_JWKS_TTL_SECONDS = 3600
_jwks_client: Optional[PyJWKClient] = None
_jwks_loaded_at: float = 0.0
_jwks_lock = asyncio.Lock()


# ── Exceptions ───────────────────────────────────────────────────────


class PubSubAuthError(RuntimeError):
    """Raised on JWT signature / claim verification failure. The
    webhook surface translates this to a 401. Pub/Sub will not retry
    on a 401; the forged push dies once."""


class GmailWatchError(RuntimeError):
    """Raised when the Gmail watch() / stop() / history.list() API
    returns a non-2xx response we can't recover from. The watch
    refresh job logs and continues; user-facing CRUD propagates the
    error so the operator sees the failure."""


# ── JWT verifier ─────────────────────────────────────────────────────


async def _get_jwks_client() -> PyJWKClient:
    """Lazy-build + cached PyJWKClient. Re-fetch when the cache TTL
    elapses or the client was never built."""
    global _jwks_client, _jwks_loaded_at
    now = time.time()
    if _jwks_client is not None and (now - _jwks_loaded_at) < _JWKS_TTL_SECONDS:
        return _jwks_client
    async with _jwks_lock:
        # Re-check after acquiring the lock — another coroutine might
        # have rebuilt it while we were waiting.
        now = time.time()
        if _jwks_client is None or (now - _jwks_loaded_at) >= _JWKS_TTL_SECONDS:
            _jwks_client = PyJWKClient(GOOGLE_JWKS_URL, cache_keys=True)
            _jwks_loaded_at = now
        return _jwks_client


async def verify_pubsub_jwt(authorization_header: str) -> dict[str, Any]:
    """Verify a Pub/Sub push JWT and return its claims.

    Checks:
      - Format: `Bearer <jwt>` (Google's push subscriptions use this
        exact format).
      - RS256 signature against Google's published JWKS.
      - `aud` matches `settings.pubsub_push_audience` (the webhook URL).
      - `email` matches `settings.pubsub_push_signer` when configured
        (defence against a Google-signed JWT issued for a different
        service account).
      - `iss` is `https://accounts.google.com` or
        `accounts.google.com` (Google has historically used both).
      - `exp` is in the future (PyJWT validates this automatically).

    Raises PubSubAuthError on any failure. Returns the decoded claim
    dict on success — callers use the `email` field for audit
    logging.
    """
    if not authorization_header:
        raise PubSubAuthError("missing Authorization header")
    if not authorization_header.lower().startswith("bearer "):
        raise PubSubAuthError("Authorization header is not Bearer")
    token = authorization_header.split(" ", 1)[1].strip()

    if not settings.pubsub_push_audience:
        # Fail-closed: refuse to verify when the audience hasn't been
        # configured. Otherwise any token Google ever signed for any
        # audience would pass.
        raise PubSubAuthError(
            "pubsub_push_audience not configured — refusing to verify"
        )

    try:
        jwks = await _get_jwks_client()
        signing_key = jwks.get_signing_key_from_jwt(token).key
        claims = pyjwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=settings.pubsub_push_audience,
            options={"require": ["exp", "iat", "aud"]},
        )
    except pyjwt.InvalidTokenError as e:
        raise PubSubAuthError(f"jwt invalid: {e}") from e
    except Exception as e:
        # Network / JWKS errors. Don't leak details.
        logger.exception("[pubsub_auth] jwks/verify error")
        raise PubSubAuthError(f"verification failed: {type(e).__name__}") from e

    iss = claims.get("iss", "")
    if iss not in {
        "https://accounts.google.com",
        "accounts.google.com",
    }:
        raise PubSubAuthError(f"unexpected iss: {iss!r}")

    if settings.pubsub_push_signer:
        if claims.get("email") != settings.pubsub_push_signer:
            raise PubSubAuthError(
                f"unexpected signer email: {claims.get('email')!r}"
            )

    return claims


def decode_pubsub_envelope(body: dict[str, Any]) -> tuple[str, str]:
    """Pull `emailAddress` and `historyId` out of the Pub/Sub push
    body. Pub/Sub's standard push shape is:

        {
            "message": {
                "data": "<base64-encoded JSON>",
                "messageId": "…",
                "publishTime": "…"
            },
            "subscription": "…"
        }

    The `data` field, once base64-decoded, is the Gmail notification
    payload:

        { "emailAddress": "user@gmail.com", "historyId": "12345" }

    Returns (email, history_id_str). Raises ValueError on malformed
    input — the webhook turns this into a 400 (NOT a 500: malformed
    is a permanent error, Pub/Sub shouldn't retry).
    """
    msg = body.get("message") or {}
    data_b64 = msg.get("data") or ""
    if not data_b64:
        raise ValueError("pubsub message has no data field")
    try:
        decoded = base64.b64decode(data_b64)
        data = json.loads(decoded)
    except Exception as e:
        raise ValueError(f"pubsub message.data is not valid base64+JSON: {e}") from e

    email = (data.get("emailAddress") or "").strip().lower()
    history_id = str(data.get("historyId") or "").strip()
    if not email or not history_id:
        raise ValueError(
            f"pubsub envelope missing required fields: emailAddress={email!r} "
            f"historyId={history_id!r}"
        )
    return email, history_id


# ── Token mint helper ────────────────────────────────────────────────


async def _mint_access_token(refresh_token: str) -> str:
    """Exchange a refresh token for a fresh access token via Google's
    OAuth token endpoint. Used for `users.watch` and
    `users.history.list` calls — we have to mint platform-side because
    the agent container never sees the refresh token.

    Raises GmailWatchError on a non-200 response. The caller's audit
    log captures the failure for ops to chase.
    """
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise GmailWatchError(
            "google_oauth_client_id/secret unset — cannot mint access token"
        )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if r.status_code != 200:
        raise GmailWatchError(
            f"token refresh failed status={r.status_code} "
            f"body={r.text[:200]}"
        )
    tok = r.json().get("access_token")
    if not tok:
        raise GmailWatchError("token refresh response missing access_token")
    return tok


async def _resolve_token_for(user_id: str) -> str:
    """Pull a Gmail access token for `user_id` via the connector vault.

    First checks if `DecryptedIdentity.access_token` is still valid;
    if it's expired or close to expiring (<60s remaining), mints a
    fresh one via the refresh token.

    Caller responsible for handling GmailWatchError — typically logs
    and skips this user for this push, since dropping it just means
    we miss one email's worth of history (Gmail will replay on the
    next push).
    """
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "gmail")
    if ident is None or not ident.refresh_token:
        raise GmailWatchError("no active gmail identity / refresh token")
    # If the cached access_token is still valid, prefer it (no extra
    # token-endpoint hit). connector_vault rotates this when the
    # dispatcher refreshes — we re-use the cache.
    if ident.access_token and ident.access_expires_at:
        from datetime import datetime, timezone
        remaining = (ident.access_expires_at - datetime.utcnow()).total_seconds()
        if remaining > 60:
            return ident.access_token
    return await _mint_access_token(ident.refresh_token)


# ── Watch provisioning ───────────────────────────────────────────────


@dataclass
class WatchHandle:
    """Result of a successful `users.watch` call."""

    history_id: str
    expiration_unix_ms: int

    @property
    def expiration_seconds(self) -> int:
        return self.expiration_unix_ms // 1000


async def start_watch(
    user_id: str, *, label_ids: Optional[list[str]] = None,
) -> WatchHandle:
    """Arm a Gmail watch on the user's INBOX (or specified labels).

    Returns the baseline historyId — the runner stores this in
    `triggers.provider_state_json` so the next history.list() call
    knows where to resume from. Google's docs guarantee that any
    message arriving AFTER the watch returns is captured by the next
    Pub/Sub push, no matter how short the gap.

    Watches expire ~7 days after start; `refresh_watch` re-arms them.

    Raises GmailWatchError on:
      - Token mint failure (caller likely needs to mark connector
        for reauth).
      - Non-2xx from the watch API (often = wrong topic name,
        missing Pub/Sub IAM, expired OAuth grant).
    """
    if not settings.gcp_project or not settings.pubsub_topic:
        raise GmailWatchError(
            "gcp_project or pubsub_topic unset — cannot arm watch"
        )
    topic = f"projects/{settings.gcp_project}/topics/{settings.pubsub_topic}"
    access = await _resolve_token_for(user_id)
    body: dict[str, Any] = {"topicName": topic}
    if label_ids:
        body["labelIds"] = label_ids
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{GMAIL_API_BASE}/watch",
            headers={"Authorization": f"Bearer {access}"},
            json=body,
        )
    if r.status_code >= 300:
        raise GmailWatchError(
            f"watch.start status={r.status_code} body={r.text[:300]}"
        )
    data = r.json()
    return WatchHandle(
        history_id=str(data["historyId"]),
        expiration_unix_ms=int(data["expiration"]),
    )


async def stop_watch(user_id: str) -> None:
    """Tear down the active Gmail watch for this user. Called when a
    trigger is deleted or disabled and no other trigger needs the
    watch. Idempotent — a 404 from Google means "no watch existed,"
    not an error.
    """
    try:
        access = await _resolve_token_for(user_id)
    except GmailWatchError:
        # User already disconnected Gmail — watch is implicitly dead.
        logger.info("[gmail_watch] stop skipped — no token for user=%s", user_id[:8])
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{GMAIL_API_BASE}/stop",
            headers={"Authorization": f"Bearer {access}"},
        )
    if r.status_code in (200, 204, 404):
        return
    raise GmailWatchError(
        f"watch.stop status={r.status_code} body={r.text[:300]}"
    )


async def refresh_watch(user_id: str) -> WatchHandle:
    """Re-arm a watch ahead of the 7-day expiration. Trivially the
    same call as start_watch — Gmail accepts a fresh watch over a
    live one and resets the clock. Kept as a separate function so
    the system routine that drives refresh has a semantic name.
    """
    return await start_watch(user_id)


# ── History delta fetch ──────────────────────────────────────────────


async def list_new_message_ids(
    user_id: str, *, start_history_id: str,
) -> tuple[list[str], Optional[str]]:
    """Pull every new gmail_message_id since `start_history_id`.

    Returns `(message_ids, new_history_id)`. The new history_id is
    suitable for caller to persist in `triggers.provider_state_json`
    so the next push resumes from this point. None means "no new
    history" (Google returns a 200 with no historyId when nothing
    changed).

    Pagination: history.list returns a `nextPageToken` for big
    deltas; we follow it transparently. Bounded at 50 pages
    (= ~5000 messages worst case) so a pathological history won't
    eat the webhook's 2s budget.

    Raises GmailWatchError on non-2xx. 404 from Google means the
    history_id is too old (>= 7d gap, watch expired in between);
    caller should re-arm the watch and skip this delta.
    """
    if not start_history_id:
        raise GmailWatchError("start_history_id required")
    access = await _resolve_token_for(user_id)

    msg_ids: list[str] = []
    new_history_id: Optional[str] = None
    page_token: Optional[str] = None
    pages = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            pages += 1
            if pages > 50:
                logger.warning(
                    "[gmail_history] truncating at 50 pages user=%s start=%s",
                    user_id[:8], start_history_id,
                )
                break
            params: dict[str, str] = {
                "startHistoryId": start_history_id,
                "historyTypes": "messageAdded",
            }
            if page_token:
                params["pageToken"] = page_token
            r = await client.get(
                f"{GMAIL_API_BASE}/history",
                headers={"Authorization": f"Bearer {access}"},
                params=params,
            )
            if r.status_code == 404:
                # The history_id is too old. Caller re-arms.
                raise GmailWatchError(
                    f"history.list 404 — history_id={start_history_id} too old"
                )
            if r.status_code >= 300:
                raise GmailWatchError(
                    f"history.list status={r.status_code} body={r.text[:300]}"
                )
            data = r.json()
            new_history_id = str(data.get("historyId") or new_history_id or "")
            for record in data.get("history", []) or []:
                for added in record.get("messagesAdded", []) or []:
                    mid = (added.get("message") or {}).get("id")
                    if mid:
                        msg_ids.append(str(mid))
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return msg_ids, (new_history_id or None)
