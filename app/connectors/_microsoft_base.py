"""Shared Microsoft Identity Platform v2 + Microsoft Graph helpers.

Same shape as `_google_base.py` — refresh / revoke / error handler /
request wrapper — but talking to:

  - https://login.microsoftonline.com/{tenant}/oauth2/v2.0/{authorize,token}
  - https://graph.microsoft.com/v1.0/*

Tenant is templated into the URLs at provider-app registration time;
this module only sees the absolute `token_url` via the provider_app
config so it stays tenant-agnostic.

Microsoft does NOT have a token revoke endpoint — once a refresh
token is issued, it's valid until the user signs the app out from
their My Account page or until the configured lifetime expires.
`microsoft_revoke()` is a best-effort no-op so connectors can call
the same shape they call for Google.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from app.connectors.base import (
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


# Microsoft Graph's P99 mirrors Google's. 15s gives a comfortable
# margin without holding agent turns hostage during a Graph incident.
_HTTP_TIMEOUT_S = 15.0


# Pooled AsyncClient — see `_get_google_client` in `_google_base.py`
# for the full rationale. Without pooling, every Graph call eats a
# fresh TLS handshake (~200-400 ms). With pooling, repeat calls reuse
# the same TCP+TLS session and drop to ~5 ms of setup overhead.
_MS_CLIENT: Optional[httpx.AsyncClient] = None
_MS_CLIENT_LOCK = asyncio.Lock()


async def _get_ms_client() -> httpx.AsyncClient:
    """Process-singleton AsyncClient — see `_google_base._get_google_client`
    for the full rationale + the defensive-fallback contract."""
    global _MS_CLIENT
    if _MS_CLIENT is not None:
        return _MS_CLIENT
    async with _MS_CLIENT_LOCK:
        if _MS_CLIENT is None:
            try:
                # HTTP/1.1 keepalive — see `_google_base`. http2=True
                # needs the `h2` extra; setting it without that extra
                # hard-crashes every Graph call on first use.
                _MS_CLIENT = httpx.AsyncClient(
                    timeout=_HTTP_TIMEOUT_S,
                    limits=httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=100,
                        keepalive_expiry=300.0,
                    ),
                )
            except Exception as e:
                logger.error(
                    "[microsoft_base] pooled AsyncClient construction "
                    "failed (%s: %s) — falling back to per-call "
                    "clients. Investigate immediately.",
                    type(e).__name__, e,
                )
                _MS_CLIENT = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    return _MS_CLIENT


async def shutdown_ms_client() -> None:
    global _MS_CLIENT
    if _MS_CLIENT is not None:
        await _MS_CLIENT.aclose()
        _MS_CLIENT = None


class _MicrosoftConnectorError(Exception):
    """Same role as `_GoogleConnectorError`. Wraps a `ConnectorResult`
    so providers can exit any call chain via `raise` and translate at
    the outer try/except."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


def _refresh_scope_param(scopes: Optional[list[str]]) -> str:
    """Build the `scope` value for Entra's refresh grant.

    This function exists because of a production incident. The original
    code sent the literal `"offline_access"`, with a comment asserting
    that Graph "echoes back the union of granted scopes". It does not.
    Entra rejects a scope set that names no RESOURCE with

        AADSTS70011: The provided request must include a 'scope' input
        parameter.

    so every Microsoft access token died at its one-hour expiry and the
    identity went `reauth_required` with no way back — silently, because
    a refresh runs on the dispatcher's timer and not in front of a user.
    Verified against the connector_events trail on 2026-08-08 (Outlook,
    two occurrences, one per connect).

    Two rules encoded here:

    - **The scopes come from the IDENTITY, never the manifest.** Entra
      will only reissue scopes the user already consented to, so a
      manifest that grows a scope after someone connects would break
      every existing identity's refresh at once — turning an additive
      change into a fleet-wide disconnect.
    - **`offline_access` is always re-appended.** Entra does NOT echo it
      in the granted `scope` response (confirmed: the stored list for a
      working Outlook identity is `profile openid email Mail.Read
      Mail.Send User.Read`), so round-tripping the stored list alone
      drops it — and without it Entra returns no rotating refresh_token,
      which converts a recoverable one-hour bug into a permanent one at
      the refresh token's own expiry.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in scopes or []:
        s = raw.strip()
        if not s or s == "offline_access" or s in seen:
            continue
        seen.add(s)
        ordered.append(s)
    if not ordered:
        # No resource scope to assert. Sending `offline_access` alone is
        # the exact shape Entra rejects, so fail loudly here instead of
        # spending a round-trip to be told so.
        raise RefreshFailed(
            "microsoft refresh has no granted resource scopes to re-assert "
            "— identity.scopes_json is empty, so the token cannot be "
            "refreshed and the user must reconnect"
        )
    ordered.append("offline_access")
    return " ".join(ordered)


async def microsoft_refresh(
    refresh_token: str,
    *,
    scopes: Optional[list[str]] = None,
) -> RefreshResult:
    """Exchange a refresh_token for a fresh access_token.

    Microsoft's v2 token endpoint accepts `grant_type=refresh_token`
    with `client_id` + `client_secret` + `refresh_token`. The response
    INCLUDES a fresh `refresh_token` (unlike Google) which we persist
    so the rotation chain stays valid past the original token's TTL.

    `scopes` is the identity's granted set — see `_refresh_scope_param`
    for why it is required and why it is not the manifest's.
    """
    app_cfg = await get_provider_app_async("microsoft")
    if app_cfg is None:
        raise RefreshFailed(
            "microsoft provider not registered — set credentials via "
            "/admin/oauth-apps or MICROSOFT_OAUTH_CLIENT_ID env var"
        )
    client = await _get_ms_client()
    resp = await client.post(
        app_cfg.token_url,
        data={
            "client_id": app_cfg.client_id,
            "client_secret": app_cfg.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "scope": _refresh_scope_param(scopes),
        },
    )
    if resp.status_code in (400, 401):
        # Microsoft returns 400 invalid_grant when the refresh token
        # is revoked / expired / consent withdrawn — permanent fail.
        raise RefreshFailed(
            f"microsoft refresh returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        resp.raise_for_status()
    if not resp.is_success:
        raise RefreshFailed(
            f"microsoft refresh unexpected status {resp.status_code}"
        )

    body = resp.json()
    expires_in = int(body.get("expires_in", 3600))
    return RefreshResult(
        access_token=body["access_token"],
        # Microsoft DOES rotate refresh tokens — capture the new one
        # if present, otherwise fall through to "reuse current".
        refresh_token=body.get("refresh_token"),
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
    )


async def microsoft_revoke(access_token: str) -> None:
    """No-op. Microsoft Identity Platform doesn't expose a token-revoke
    endpoint — tokens just age out, and refresh-token sign-out is a
    user action at https://account.live.com/consent/Manage. We expose
    the same revoke shape as Google so dispatchers don't branch."""
    return None


def _handle_microsoft_error(
    resp: httpx.Response,
    *,
    connector_id: str = "outlook",
    scope_hint: str = "",
) -> None:
    """Map Graph HTTP responses to ConnectorResult variants. Mirrors
    `_handle_google_error` so dispatcher behavior is consistent
    across providers."""
    if 200 <= resp.status_code < 300:
        return

    if resp.status_code == 401:
        raise _MicrosoftConnectorError(ConnectorReauthRequired(
            reauth_url=f"/agent/integrations/{connector_id}",
        ))
    if resp.status_code == 403:
        body = (resp.text or "").lower()
        if "insufficient" in body or "scope" in body or "permission" in body:
            raise _MicrosoftConnectorError(
                ConnectorScopeMissing(required_scope=scope_hint or "unknown"),
            )
        raise _MicrosoftConnectorError(ConnectorToolError(
            message=f"403 from microsoft graph: {resp.text[:200]}",
            retryable=False,
        ))
    if resp.status_code == 429:
        retry_after = 30
        try:
            retry_after = int(resp.headers.get("Retry-After", "30"))
        except (TypeError, ValueError):
            pass
        raise _MicrosoftConnectorError(
            ConnectorRateLimited(retry_after_s=retry_after),
        )
    if resp.status_code >= 500:
        raise _MicrosoftConnectorError(ConnectorProviderDown(
            provider_status_url="https://portal.office.com/servicestatus",
        ))
    raise _MicrosoftConnectorError(ConnectorToolError(
        message=f"{resp.status_code} from microsoft graph: {resp.text[:200]}",
        retryable=False,
    ))


async def microsoft_graph_request(
    method: str,
    url: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    connector_id: str = "outlook",
    scope_hint: str = "",
) -> Any:
    """Authenticated request → JSON dict (or raw text). Raises
    `_MicrosoftConnectorError` on any non-2xx with the right variant
    attached.

    Microsoft Graph endpoints that return no body (POST /sendMail →
    202 Accepted) come back as `{"raw": ""}` so callers can still
    return a ConnectorOk shape.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    client = await _get_ms_client()
    resp = await client.request(
        method,
        url,
        headers=headers,
        json=json_body,
        params=params,
    )
    _handle_microsoft_error(resp, connector_id=connector_id, scope_hint=scope_hint)
    if not resp.content:
        return {"raw": ""}
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return resp.json()
    return {"raw": resp.text}
