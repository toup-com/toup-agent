"""
T3 — Shared Google connector helpers.

Gmail, Calendar, and Drive all hit Google's OAuth + API endpoints with
the same error semantics. This module centralises:

  - `google_refresh()` — exchange refresh_token → fresh access_token
  - `google_revoke()`  — POST /revoke with the access_token
  - `_handle_google_error()` — map provider HTTP responses to the
    `ConnectorResult` sum type. Distinguishes 401 (reauth required),
    403 (scope missing), 429 (rate limited), 5xx (provider down),
    other 4xx (tool error).
  - `google_request()` — convenience GET/POST/PATCH wrapper that
    runs through the error handler so providers don't reimplement.

`ConnectorResult` returns happen via Python exception subclasses
(`_GoogleConnectorError`) so call sites can `try/except` and surface
the variant cleanly. Providers that need finer-grained error handling
catch the base and translate; the default `_handle_google_error` is
sensible for 95% of API endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from app.connectors.base import (
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    RefreshFailed,
    RefreshResult,
)
from app.services.provider_apps import get_provider_app_async

logger = logging.getLogger(__name__)


# Google's typical request timeout. Their P99 is ~5s on healthy days,
# 10s during incidents. 15s gives us margin without holding agent
# turns hostage to a sustained outage.
_HTTP_TIMEOUT_S = 15.0


class _GoogleConnectorError(Exception):
    """Wraps a `ConnectorResult` so we can exit a provider method via
    `raise` from anywhere in a call chain. The dispatcher's exception
    catch (T1e §6) maps unhandled exceptions to ConnectorProviderDown,
    so providers MUST catch and translate before the result bubbles."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


async def google_refresh(refresh_token: str) -> RefreshResult:
    """Exchange a refresh_token for a fresh access_token.

    Raises `RefreshFailed` on permanent failure (revoked token, bad
    grant) — the dispatcher catches it and flips the identity to
    `reauth_required`. Other failures (5xx, network) raise the
    underlying exception so the dispatcher's "unexpected refresh
    exception" branch fires.
    """
    app_cfg = await get_provider_app_async("google")
    if app_cfg is None:
        raise RefreshFailed(
            "google provider not registered — set credentials via "
            "/admin/oauth-apps or GOOGLE_OAUTH_CLIENT_ID env var"
        )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        resp = await client.post(
            app_cfg.token_url,
            data={
                "client_id": app_cfg.client_id,
                "client_secret": app_cfg.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if resp.status_code == 400 or resp.status_code == 401:
        # Google returns 400 invalid_grant when the refresh token has
        # been revoked or expired. Treat as permanent.
        raise RefreshFailed(
            f"google refresh returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        # Bubble — dispatcher's "unexpected" branch surfaces as
        # ConnectorReauthRequired (the safer fallback) and the next
        # turn retries from cold.
        resp.raise_for_status()
    if not resp.is_success:
        raise RefreshFailed(
            f"google refresh unexpected status {resp.status_code}"
        )

    body = resp.json()
    expires_in = int(body.get("expires_in", 3600))
    return RefreshResult(
        access_token=body["access_token"],
        # Google does NOT return a new refresh_token on refresh — keep
        # the existing one. None signals "reuse current".
        refresh_token=None,
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
    )


async def google_revoke(access_token: str) -> None:
    """POST to /revoke. Best-effort — provider 404s an
    already-revoked token; treat as success per the OAuth API
    docstring contract."""
    if not access_token:
        return
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        try:
            await client.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": access_token},
            )
        except Exception as e:
            logger.warning("[google_revoke] best-effort failure: %s", e)


def _handle_google_error(resp: httpx.Response, *, scope_hint: str = "") -> None:
    """Inspect the response. If it's an error, raise a
    `_GoogleConnectorError` with the appropriate ConnectorResult.
    On success, no-op (caller proceeds with the body).
    """
    if 200 <= resp.status_code < 300:
        return

    if resp.status_code == 401:
        raise _GoogleConnectorError(ConnectorReauthRequired(
            reauth_url="/agent/integrations/gmail",  # placeholder — caller passes per-connector
        ))
    if resp.status_code == 403:
        # Could be either scope missing OR Google quota exhaustion;
        # message inspection helps disambiguate.
        body = (resp.text or "").lower()
        if "insufficient" in body or "scope" in body or "permission" in body:
            raise _GoogleConnectorError(
                ConnectorScopeMissing(required_scope=scope_hint or "unknown"),
            )
        raise _GoogleConnectorError(ConnectorToolError(
            message=f"403 from google: {resp.text[:200]}",
            retryable=False,
        ))
    if resp.status_code == 429:
        # Retry-After is rare on Google APIs; default to 30s.
        retry_after = 30
        try:
            retry_after = int(resp.headers.get("Retry-After", "30"))
        except (TypeError, ValueError):
            pass
        raise _GoogleConnectorError(
            ConnectorRateLimited(retry_after_s=retry_after),
        )
    if resp.status_code >= 500:
        raise _GoogleConnectorError(ConnectorProviderDown(
            provider_status_url="https://status.cloud.google.com",
        ))
    # Other 4xx — caller error.
    raise _GoogleConnectorError(ConnectorToolError(
        message=f"{resp.status_code} from google: {resp.text[:200]}",
        retryable=False,
    ))


async def google_request(
    method: str,
    url: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    scope_hint: str = "",
) -> dict:
    """Authenticated request → JSON dict. Raises `_GoogleConnectorError`
    on any non-2xx with the right variant attached."""
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        resp = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
        )
    _handle_google_error(resp, scope_hint=scope_hint)
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return {"raw": resp.text}


def to_connector_result(result_or_error: Any) -> ConnectorResult:
    """Helper: if you caught `_GoogleConnectorError`, surface its
    inner `.result`. Otherwise wrap whatever happened as
    `ConnectorOk(content=json.dumps(...))`. Providers' `execute`
    methods use this in their top-level except blocks."""
    import json
    if isinstance(result_or_error, _GoogleConnectorError):
        return result_or_error.result
    return ConnectorOk(content=json.dumps(result_or_error, default=str))
