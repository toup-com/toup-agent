"""T3b — Gmail connector provider.

Reads + send via Gmail REST v1. The agent's user-scoped access_token
is the only auth used; we never act with a service account.

Token resolution: the dispatcher (T1e) hands us the decrypted access
token via the vault-injected ConnectorContext after refresh-on-expiring
runs. We never decrypt directly.
"""

from __future__ import annotations

import base64
import json
from typing import ClassVar

from app.connectors._google_base import (
    _GoogleConnectorError,
    google_refresh,
    google_request,
    google_revoke,
)
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.services import connector_vault as _vault
from app.db.database import async_session_maker

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


async def _resolve_access_token(user_id: str) -> str:
    """Pull the decrypted access_token for this (user, gmail). The
    dispatcher already ran refresh-on-expiring before calling us, so
    the returned token is good for the duration of THIS call. We
    re-fetch per call (not per provider instance) because providers
    are shared across requests."""
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "gmail")
    if ident is None or not ident.access_token:
        raise _GoogleConnectorError(
            ConnectorToolError(message="No active Gmail identity", retryable=False),
        )
    return ident.access_token


def _b64url_decode(s: str) -> str:
    """Gmail uses base64url for body data — translate `-_` → `+/` and
    pad to a multiple of 4. Falls back to empty string on decode
    failure rather than throwing into the LLM."""
    s = s.replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    try:
        return base64.b64decode(s + ("=" * pad)).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _b64url_encode(s: str) -> str:
    return (
        base64.urlsafe_b64encode(s.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )


def _build_rfc822(*, to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Minimal RFC 822 message — Gmail accepts the body as one base64url
    blob in the `raw` field. We construct without an MIME multipart
    because v1 of the connector only sends plain text; T3b+ may add
    HTML alongside."""
    headers = [f"To: {to}", f"Subject: {subject}"]
    if cc:
        headers.append(f"Cc: {cc}")
    if bcc:
        headers.append(f"Bcc: {bcc}")
    headers.append("Content-Type: text/plain; charset=utf-8")
    return "\r\n".join(headers) + "\r\n\r\n" + body


class GmailProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "gmail"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = await _resolve_access_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return e.result

        try:
            if tool_name == "gmail__list_messages":
                params = {"maxResults": int(tool_input.get("max_results", 10))}
                q = tool_input.get("query")
                if q:
                    params["q"] = q
                result = await google_request(
                    "GET",
                    f"{GMAIL_API_BASE}/messages",
                    access_token=access_token,
                    params=params,
                    scope_hint="gmail.readonly",
                )
                # Reduce to summary list — full bodies via gmail__get_message
                summaries = []
                for m in (result.get("messages") or [])[:int(tool_input.get("max_results", 10))]:
                    summaries.append({"id": m.get("id"), "threadId": m.get("threadId")})
                return ConnectorOk(content=json.dumps({
                    "messages": summaries,
                    "result_size": result.get("resultSizeEstimate"),
                }))

            if tool_name == "gmail__get_message":
                mid = tool_input.get("message_id")
                if not mid:
                    return ConnectorToolError(message="message_id required", retryable=False)
                result = await google_request(
                    "GET",
                    f"{GMAIL_API_BASE}/messages/{mid}",
                    access_token=access_token,
                    params={"format": "full"},
                    scope_hint="gmail.readonly",
                )
                # Pull headers + decoded body for the LLM. Skip raw
                # MIME parts beyond the first text/plain — that's
                # noise for the agent.
                headers = {
                    h["name"]: h["value"]
                    for h in (result.get("payload", {}).get("headers") or [])
                    if h["name"] in ("From", "To", "Cc", "Subject", "Date")
                }
                body_text = _extract_text_body(result.get("payload") or {})
                return ConnectorOk(content=json.dumps({
                    "id": result.get("id"),
                    "threadId": result.get("threadId"),
                    "headers": headers,
                    "snippet": result.get("snippet"),
                    "body": body_text,
                }))

            if tool_name == "gmail__send_message":
                to = tool_input.get("to")
                subject = tool_input.get("subject")
                body = tool_input.get("body")
                if not (to and subject and body):
                    return ConnectorToolError(
                        message="to/subject/body all required",
                        retryable=False,
                    )
                raw = _b64url_encode(_build_rfc822(
                    to=to, subject=subject, body=body,
                    cc=tool_input.get("cc", ""),
                    bcc=tool_input.get("bcc", ""),
                ))
                result = await google_request(
                    "POST",
                    f"{GMAIL_API_BASE}/messages/send",
                    access_token=access_token,
                    json_body={"raw": raw},
                    scope_hint="gmail.send",
                )
                return ConnectorOk(content=json.dumps({
                    "id": result.get("id"),
                    "threadId": result.get("threadId"),
                    "labelIds": result.get("labelIds"),
                }))

            if tool_name == "gmail__search_threads":
                q = tool_input.get("query")
                if not q:
                    return ConnectorToolError(message="query required", retryable=False)
                params = {
                    "q": q,
                    "maxResults": int(tool_input.get("max_results", 10)),
                }
                result = await google_request(
                    "GET",
                    f"{GMAIL_API_BASE}/threads",
                    access_token=access_token,
                    params=params,
                    scope_hint="gmail.readonly",
                )
                threads = []
                for t in (result.get("threads") or [])[:int(tool_input.get("max_results", 10))]:
                    threads.append({
                        "id": t.get("id"),
                        "snippet": t.get("snippet"),
                        "historyId": t.get("historyId"),
                    })
                return ConnectorOk(content=json.dumps({"threads": threads}))

            return ConnectorToolError(
                message=f"unknown gmail tool {tool_name!r}",
                retryable=False,
            )
        except _GoogleConnectorError as e:
            # Re-target the placeholder reauth_url that _google_base
            # produces for 401 — point at gmail's settings.
            from app.connectors.base import ConnectorReauthRequired
            if isinstance(e.result, ConnectorReauthRequired):
                return ConnectorReauthRequired(reauth_url="/agent/integrations/gmail")
            return e.result

    async def revoke(self, user_id, access_token, refresh_token=None):
        await google_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await google_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        try:
            access_token = await _resolve_access_token(ctx.user_id)
            await google_request(
                "GET",
                f"{GMAIL_API_BASE}/profile",
                access_token=access_token,
                scope_hint="gmail.readonly",
            )
            return HealthResult(ok=True)
        except _GoogleConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")


def _extract_text_body(payload: dict) -> str:
    """Walk the MIME tree, return the first text/plain content (decoded).
    Falls back to text/html if no plain part. Empty string if neither."""
    mime_type = payload.get("mimeType", "")
    body = payload.get("body") or {}
    data = body.get("data")
    if mime_type.startswith("text/plain") and data:
        return _b64url_decode(data)
    parts = payload.get("parts") or []
    for p in parts:
        if (p.get("mimeType") or "").startswith("text/plain"):
            d = (p.get("body") or {}).get("data")
            if d:
                return _b64url_decode(d)
    # Fallback: first text/html as raw
    for p in parts:
        if (p.get("mimeType") or "").startswith("text/html"):
            d = (p.get("body") or {}).get("data")
            if d:
                return _b64url_decode(d)
    return ""
