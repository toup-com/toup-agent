"""T3b — Gmail connector provider.

Reads + send via Gmail REST v1. The agent's user-scoped access_token
is the only auth used; we never act with a service account.

Token resolution: the dispatcher (T1e) hands us the decrypted access
token via the vault-injected ConnectorContext after refresh-on-expiring
runs. We never decrypt directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, ClassVar, Optional

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
    """Fallback path: pull the decrypted access_token for this
    (user, gmail) when the dispatcher didn't hand one through
    `ctx.access_token`. Production callers (the dispatcher) always
    pass the pre-decrypted token; this path exists for tests that
    build ctx by hand and for any historical code path that bypasses
    the dispatcher.

    Was the hot path pre-2026-05-12 — every Gmail tool call did a
    second vault.get() here (~100-300 ms on Railway+pgbouncer) right
    after the dispatcher had ALREADY decrypted the identity in
    pre-flight. Threading the token through ConnectorContext
    eliminates that duplicate read."""
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


def _encode_header(value: str) -> str:
    """RFC 2047 encode a header value when it is not 7-bit ASCII.

    R31-34. The D session's test mail arrived in the founder's inbox as
    `R29-D live loop test Ã¢Â€Â" Gmail push` — an em dash, mangled. The
    obvious reading is a double encode; measured, it is the opposite.
    This function used to build the Subject with a bare f-string, so
    U+2014 went onto the wire as its three raw UTF-8 bytes inside a
    header. RFC 5322 headers are 7-bit: a receiver has no way to know
    those bytes are UTF-8, reads them as Latin-1, and renders `â€"` —
    which is then mojibake'd a second time by whatever displays THAT.
    Zero encodings, not two.

    `Content-Type: charset=utf-8` below covers the BODY only and always
    did; it says nothing about the header block above it.

    The platform's other mail path (`services/email_service.py`) has
    been correct all along because `MIMEMultipart` does this for you —
    so the two writers disagreed, and the one automations can reach was
    the wrong one. Drafts use the same builder, and an automation CAN
    draft, so this is on the automations path too.

    Pure-ASCII values are returned untouched: encoding them would be
    correct but unreadable in every mail client's raw view, and this is
    the header a user is most likely to see quoted back.
    """
    if not value:
        return ""
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        from email.header import Header
        return Header(value, "utf-8").encode()


def _build_rfc822(*, to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Minimal RFC 822 message — Gmail accepts the body as one base64url
    blob in the `raw` field. We construct without an MIME multipart
    because v1 of the connector only sends plain text; T3b+ may add
    HTML alongside.

    Every header value goes through `_encode_header` (R31-34): a
    non-ASCII address display-name mangles exactly the way a non-ASCII
    subject did.
    """
    headers = [
        f"To: {_encode_header(to)}",
        f"Subject: {_encode_header(subject)}",
    ]
    if cc:
        headers.append(f"Cc: {_encode_header(cc)}")
    if bcc:
        headers.append(f"Bcc: {_encode_header(bcc)}")
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
        # Prefer the dispatcher's pre-decrypted token — skips a
        # duplicate vault.get + Fernet decrypt that used to add
        # ~100-300 ms to every Gmail call. The fallback covers tests
        # that build ctx by hand without an access_token.
        try:
            access_token = ctx.access_token or await _resolve_access_token(ctx.user_id)
        except _GoogleConnectorError as e:
            return e.result

        try:
            if tool_name == "gmail__list_messages":
                # Defaults match the manifest: max_results=25,
                # include_body=true. The dispatcher's safety net also
                # auto-injects include_body=true when the LLM omits it,
                # but mirror the default here so direct calls (tests,
                # any non-dispatcher path) get the same behaviour.
                max_results = int(tool_input.get("max_results", 25))
                params = {"maxResults": max_results}
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
                raw_ids = [
                    {"id": m.get("id"), "threadId": m.get("threadId")}
                    for m in (result.get("messages") or [])[:max_results]
                    if m.get("id")
                ]

                # FAST PATH (now the default): include_body inlines each
                # message's headers + body in this same response, fanned
                # out in parallel. Pre-2026-05-12 the LLM had to
                # explicitly opt in (default was false) and frequently
                # missed the hint, doing list → list → get_message at
                # ~10 s per call. Default is now `true` in the manifest
                # AND auto-injected by the dispatcher when omitted, so
                # this branch is the hot path. Reading: ONE call.
                include_body = tool_input.get("include_body", True)
                if include_body and raw_ids:
                    full_messages = await _fetch_messages_parallel(
                        access_token=access_token,
                        message_ids=[r["id"] for r in raw_ids],
                    )
                    return ConnectorOk(content=json.dumps({
                        "messages": full_messages,
                        "result_size": result.get("resultSizeEstimate"),
                    }))

                return ConnectorOk(content=json.dumps({
                    "messages": raw_ids,
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
                    "labelIds": result.get("labelIds") or [],
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

            if tool_name == "gmail__create_draft":
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
                    f"{GMAIL_API_BASE}/drafts",
                    access_token=access_token,
                    json_body={"message": {"raw": raw}},
                    scope_hint="gmail.compose",
                )
                msg = result.get("message") or {}
                return ConnectorOk(content=json.dumps({
                    "draft_id": result.get("id"),
                    "id": msg.get("id"),
                    "threadId": msg.get("threadId"),
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

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
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


# Cap on the include_body fan-out. Higher = more parallelism = faster
# wall time, but more pressure on the per-user Gmail quota.
# 25 is well under Google's per-user quota for read endpoints
# (~250 req/s on a fresh project) and is enough to cover the common
# "summarise my latest 5/10/20 emails" patterns.
_INCLUDE_BODY_CONCURRENCY = 25


async def _fetch_messages_parallel(
    *,
    access_token: str,
    message_ids: list[str],
) -> list[dict]:
    """Fan out a batch of GET /messages/{id}?format=full calls and
    return them in input order with parsed body + headers — the same
    shape `gmail__get_message` returns for a single id.

    Used by `gmail__list_messages` when `include_body=true`. Bounded
    by `_INCLUDE_BODY_CONCURRENCY` so a 100-email list doesn't open
    100 simultaneous httpx requests; the pooled AsyncClient in
    `_google_base` already keeps the TLS session warm so the bound
    is for politeness toward Google's per-user quota, not for our
    side."""
    sem = asyncio.Semaphore(_INCLUDE_BODY_CONCURRENCY)

    async def _one(mid: str) -> dict:
        async with sem:
            try:
                raw = await google_request(
                    "GET",
                    f"{GMAIL_API_BASE}/messages/{mid}",
                    access_token=access_token,
                    params={"format": "full"},
                    scope_hint="gmail.readonly",
                )
            except _GoogleConnectorError as e:
                # Surface the per-message error inline rather than
                # tanking the whole gather — the agent can still
                # work with the other emails it got back.
                return {
                    "id": mid,
                    "error": repr(e.result),
                }
            headers = {
                h["name"]: h["value"]
                for h in (raw.get("payload", {}).get("headers") or [])
                if h["name"] in ("From", "To", "Cc", "Subject", "Date")
            }
            return {
                "id": raw.get("id"),
                "threadId": raw.get("threadId"),
                "headers": headers,
                "snippet": raw.get("snippet"),
                "body": _extract_text_body(raw.get("payload") or {}),
            }

    return await asyncio.gather(*(_one(mid) for mid in message_ids))
