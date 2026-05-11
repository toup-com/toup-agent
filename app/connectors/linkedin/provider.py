"""LinkedIn connector provider.

Two surfaces:
  - GET /v2/userinfo (OpenID Connect /me equivalent) — returns sub,
    name, given_name, family_name, picture, email. Same shape across
    every OIDC provider including LinkedIn.
  - POST /v2/ugcPosts — share a text/link update to the user's feed.
    LinkedIn's User Generated Content API is the modern share path;
    the older /shares endpoint is deprecated.

Refresh: LinkedIn issues refresh tokens with a 365-day lifetime when
the OAuth app has been granted `Sign In with LinkedIn using OpenID
Connect` (which we require — see manifest scopes). Refresh request
shape is identical to the auth_code grant minus the `code` field.

LinkedIn does NOT have a token revoke endpoint — disconnecting on
our side just zeroes the ciphertext and lets the token age out
naturally, same shape as Microsoft.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import ClassVar, Optional

import httpx

from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    HealthResult,
    RefreshFailed,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault
from app.services.provider_apps import get_provider_app_async

logger = logging.getLogger(__name__)


LINKEDIN_API = "https://api.linkedin.com/v2"
_HTTP_TIMEOUT_S = 15.0


class _LinkedInError(Exception):
    """Same pattern as `_GoogleConnectorError` / `_MicrosoftConnectorError`."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "linkedin")
    if ident is None or not ident.access_token:
        raise _LinkedInError(
            ConnectorToolError(message="No active LinkedIn identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/linkedin")
    return result


def _handle_linkedin_error(resp: httpx.Response, *, scope_hint: str = "") -> None:
    if 200 <= resp.status_code < 300:
        return
    if resp.status_code == 401:
        raise _LinkedInError(ConnectorReauthRequired(
            reauth_url="/agent/integrations/linkedin",
        ))
    if resp.status_code == 403:
        body = (resp.text or "").lower()
        if "scope" in body or "permission" in body or "insufficient" in body:
            raise _LinkedInError(
                ConnectorScopeMissing(required_scope=scope_hint or "unknown"),
            )
        raise _LinkedInError(ConnectorToolError(
            message=f"403 from linkedin: {resp.text[:200]}",
            retryable=False,
        ))
    if resp.status_code == 429:
        retry_after = 60
        try:
            retry_after = int(resp.headers.get("Retry-After", "60"))
        except (TypeError, ValueError):
            pass
        raise _LinkedInError(ConnectorRateLimited(retry_after_s=retry_after))
    if resp.status_code >= 500:
        raise _LinkedInError(ConnectorProviderDown(
            provider_status_url="https://www.linkedin.com/help/linkedin",
        ))
    raise _LinkedInError(ConnectorToolError(
        message=f"{resp.status_code} from linkedin: {resp.text[:200]}",
        retryable=False,
    ))


async def _linkedin_request(
    method: str,
    url: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    extra_headers: Optional[dict] = None,
    scope_hint: str = "",
) -> dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        resp = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
        )
    _handle_linkedin_error(resp, scope_hint=scope_hint)
    if not resp.content:
        return {"raw": "", "x_linkedin_id": resp.headers.get("x-restli-id", "")}
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return {"raw": resp.text}


async def linkedin_refresh(refresh_token: str) -> RefreshResult:
    """Exchange a refresh_token for a fresh access_token."""
    app_cfg = await get_provider_app_async("linkedin")
    if app_cfg is None:
        raise RefreshFailed(
            "linkedin provider not registered — set credentials via "
            "/admin/oauth-apps or LINKEDIN_OAUTH_CLIENT_ID env var"
        )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        resp = await client.post(
            app_cfg.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": app_cfg.client_id,
                "client_secret": app_cfg.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code in (400, 401):
        raise RefreshFailed(
            f"linkedin refresh returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        resp.raise_for_status()
    if not resp.is_success:
        raise RefreshFailed(
            f"linkedin refresh unexpected status {resp.status_code}"
        )
    body = resp.json()
    expires_in = int(body.get("expires_in", 3600))
    return RefreshResult(
        access_token=body["access_token"],
        # LinkedIn re-issues refresh tokens periodically; capture if
        # present, otherwise reuse current.
        refresh_token=body.get("refresh_token"),
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
    )


class LinkedInProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "linkedin"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = await _resolve_token(ctx.user_id)
        except _LinkedInError as e:
            return _retarget_reauth(e.result)

        try:
            if tool_name == "linkedin__get_profile":
                # OIDC `/userinfo` endpoint — returns sub, name,
                # given_name, family_name, picture, email. The `sub`
                # is the urn:li:person id we need for share posts.
                me = await _linkedin_request(
                    "GET",
                    f"{LINKEDIN_API}/userinfo",
                    access_token=access_token,
                    scope_hint="openid profile email",
                )
                return ConnectorOk(content=json.dumps({
                    "id": me.get("sub"),
                    "name": me.get("name"),
                    "given_name": me.get("given_name"),
                    "family_name": me.get("family_name"),
                    "email": me.get("email"),
                    "picture": me.get("picture"),
                    "locale": me.get("locale"),
                }))

            if tool_name == "linkedin__share_post":
                text = (tool_input.get("text") or "").strip()
                if not text:
                    return ConnectorToolError(
                        message="text is required", retryable=False,
                    )
                link_url = (tool_input.get("link_url") or "").strip()
                visibility = (tool_input.get("visibility") or "PUBLIC").upper()
                if visibility not in ("PUBLIC", "CONNECTIONS"):
                    return ConnectorToolError(
                        message="visibility must be PUBLIC or CONNECTIONS",
                        retryable=False,
                    )

                # We need the user's urn:li:person:<id> for the
                # `author` field. The /userinfo `sub` is exactly that
                # id minus the urn prefix.
                me = await _linkedin_request(
                    "GET",
                    f"{LINKEDIN_API}/userinfo",
                    access_token=access_token,
                    scope_hint="openid profile",
                )
                sub = me.get("sub")
                if not sub:
                    return ConnectorToolError(
                        message="LinkedIn /userinfo returned no sub — token may lack `openid` scope",
                        retryable=False,
                    )
                author_urn = f"urn:li:person:{sub}"

                # Build the UGC post payload. Shape per:
                # https://learn.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/ugc-post-api
                share_content: dict = {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "NONE",
                }
                if link_url:
                    share_content["shareMediaCategory"] = "ARTICLE"
                    share_content["media"] = [{
                        "status": "READY",
                        "originalUrl": link_url,
                    }]
                payload = {
                    "author": author_urn,
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": share_content,
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": visibility,
                    },
                }
                result = await _linkedin_request(
                    "POST",
                    f"{LINKEDIN_API}/ugcPosts",
                    access_token=access_token,
                    json_body=payload,
                    extra_headers={"Content-Type": "application/json"},
                    scope_hint="w_member_social",
                )
                # The new post's urn is returned via the
                # `x-restli-id` response header (captured by
                # _linkedin_request when body is empty) or in the
                # response body under "id".
                post_id = result.get("id") or result.get("x_linkedin_id") or ""
                return ConnectorOk(content=json.dumps({
                    "post_id": post_id,
                    "url": (
                        f"https://www.linkedin.com/feed/update/{post_id}"
                        if post_id else None
                    ),
                    "visibility": visibility,
                }))

            return ConnectorToolError(
                message=f"unknown linkedin tool {tool_name!r}",
                retryable=False,
            )
        except _LinkedInError as e:
            return _retarget_reauth(e.result)

    async def revoke(self, user_id, access_token, refresh_token=None):
        # LinkedIn has no token-revoke endpoint. Same no-op shape as
        # Microsoft — disconnect zeroes our copy and the token ages
        # out at LinkedIn's side per the configured lifetime.
        return None

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await linkedin_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        try:
            access_token = await _resolve_token(ctx.user_id)
            await _linkedin_request(
                "GET",
                f"{LINKEDIN_API}/userinfo",
                access_token=access_token,
                scope_hint="openid profile",
            )
            return HealthResult(ok=True)
        except _LinkedInError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
