"""Outlook (Microsoft Graph mail) connector provider.

Reads + sends Outlook mail via Microsoft Graph v1.0. Auth flows
through the shared `_microsoft_base` helpers; only the API endpoint
shape differs from other Microsoft 365 surfaces (Calendar, Teams,
OneDrive) which will reuse the same base.

Read-endpoint quirk (the one that bit us): Graph's message
collection speaks two query languages and refuses to mix them —
see `_list_messages_params`.

Send-mail endpoint quirks:
  - POST /me/sendMail returns 202 Accepted with an empty body. We
    surface that as a {"sent": true} payload so the agent doesn't
    paste "no content" into the chat.
  - `toRecipients` is a list of objects, not a comma-separated
    string. We split-on-comma at the boundary and reshape for the
    LLM-friendly flat string input.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

from app.connectors._microsoft_base import (
    _MicrosoftConnectorError,
    microsoft_graph_request,
    microsoft_refresh,
    microsoft_revoke,
)
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault

GRAPH_API = "https://graph.microsoft.com/v1.0"


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "outlook")
    if ident is None or not ident.access_token:
        raise _MicrosoftConnectorError(
            ConnectorToolError(message="No active Outlook identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url="/agent/integrations/outlook")
    return result


def _split_recipient_csv(value: Optional[str]) -> list[dict]:
    """Convert a `"alice@x.com, bob@y.com"` string into the Graph
    `[{"emailAddress": {"address": ...}}]` shape. Empty / None →
    empty list so the field is just omitted from the payload."""
    if not value:
        return []
    out: list[dict] = []
    for raw in value.split(","):
        addr = raw.strip()
        if addr:
            out.append({"emailAddress": {"address": addr}})
    return out


# The manifest's own max_results ceiling. Also the ceiling on the
# over-fetch below, so a read/unread scan can never ask Graph for a
# page the tool doesn't already ship (bodies ride along in $select,
# and the docs warn that big pages of them hit the 504 gateway).
_MAX_TOP = 50

# How much extra to pull when read/unread has to be applied to a
# search page client-side (see _list_messages_params).
_READ_SCAN_HEADROOM = 4

# An open lower bound: every message in a mailbox is at or after it,
# so it changes no result. It exists only to put receivedDateTime in
# $filter — which is what makes ordering by it legal. See
# _list_messages_params. A caller-supplied `since` replaces it.
_FILTER_EPOCH = "1900-01-01T00:00:00Z"


def _graph_datetime(value: Any) -> Optional[str]:
    """`since` → the UTC literal Graph's $filter accepts, or None.

    Graph wants an ISO-8601 instant and rejects a bare date, so a
    caller that says "2026-08-30" means midnight and gets it. A value
    this cannot read is dropped rather than guessed: a malformed bound
    would 400 the whole read, and the read without it is the honest
    superset.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat() + "Z"


def _clamp_top(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        top = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(top, _MAX_TOP))


def _list_messages_params(tool_input: dict) -> tuple[dict, Optional[bool], int]:
    """Build the Graph query for outlook__list_messages.

    Graph's message collection speaks two query languages and refuses
    to mix them, so this is not a matter of appending parameters:

      - `$search` is KQL free text (from:, subject:, body:,
        hasAttachments:, received:). Graph owns the ordering of a
        search and rejects `$filter` or `$orderby` alongside it with a
        400 — so a search sends neither.
      - `$filter` is OData, and combining it with `$orderby` is legal
        only in the shape "List messages" documents: every property in
        `$orderby` must also appear in `$filter`, in the same order,
        ahead of any property that is not in `$orderby`. Otherwise
        Graph answers `InefficientFilter` ("The restriction or sort
        order is too complex for this operation"). `isRead eq false`
        alone with `$orderby=receivedDateTime desc` breaks rule one,
        which is why the filter leads with the open `_FILTER_EPOCH`
        bound on receivedDateTime.

    Dropping `$orderby` instead is not equivalent: Graph then infers
    its own sort for the filtered set, and `$top` is applied to THAT
    page — so an inbox read could return the oldest unread mail,
    which no caller of this tool wants.

    `since` is a received-from bound (R42 §5.2's "Last 24 hours"): a
    real `$filter` lower bound on its own path, and the KQL
    `received>=` restriction on the search path.

    Returns the params, the read state Graph could NOT be asked to
    apply (None whenever it is filtering server-side), and the number
    of rows the caller should return.
    """
    include_body = bool(tool_input.get("include_body", True))
    top = _clamp_top(tool_input.get("max_results"), 25)
    query = (tool_input.get("query") or "").strip()
    is_read = tool_input.get("is_read")
    if is_read is not None:
        is_read = bool(is_read)
    since = _graph_datetime(tool_input.get("since"))

    params: dict = {
        "$top": top,
        # When include_body=true we ask Graph for the full body in the
        # same list call — saves the per-message GET round-trip
        # entirely (Graph's /messages endpoint supports body inline,
        # unlike Gmail). When false, only headers + preview to keep
        # the LLM's token budget low. isRead is unconditional: it is
        # both a returned field and the key the search path filters on.
        "$select": (
            "id,subject,from,toRecipients,receivedDateTime,"
            "bodyPreview,isRead,hasAttachments"
            + (",body" if include_body else "")
        ),
    }

    if query:
        # Graph wants the whole KQL expression inside ONE pair of
        # double quotes and documents no way to escape another pair
        # inside it, so a phrase the model quoted ("subject:\"year
        # end\"") is a 400 rather than a narrower search. Dropping the
        # inner quotes keeps the terms and the request.
        if since:
            # Graph will not filter a search either, and KQL is the one
            # language left: `received>=` is a documented message
            # search restriction, date-granular.
            query = f"{query} received>={since[:10]}"
        params["$search"] = '"{}"'.format(query.replace('"', ""))
        if is_read is None:
            return params, None, top
        # Graph will not filter a search, so read/unread is applied to
        # the page here. Over-fetch first, or a page of read matches
        # answers "no unread mail" while unread ones sit one row below.
        params["$top"] = min(top * _READ_SCAN_HEADROOM, _MAX_TOP)
        return params, is_read, top

    if is_read is not None or since:
        # receivedDateTime leads whether or not it is the bound the
        # caller asked for — rule one of the filter+orderby contract
        # above, which is why the open epoch exists.
        params["$filter"] = f"receivedDateTime ge {since or _FILTER_EPOCH}"
        if is_read is not None:
            params["$filter"] += (
                f" and isRead eq {'true' if is_read else 'false'}")
    params["$orderby"] = "receivedDateTime desc"
    return params, None, top


class OutlookProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "outlook"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        # Prefer the dispatcher's pre-decrypted token — skips a
        # duplicate vault.get + Fernet decrypt in the provider.
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _MicrosoftConnectorError as e:
            return _retarget_reauth(e.result)

        try:
            if tool_name == "outlook__list_messages":
                # Default to true to match the manifest. Dispatcher
                # also auto-injects when the LLM omits the field; this
                # default covers tests + non-dispatcher call paths.
                include_body = bool(tool_input.get("include_body", True))
                params, scan_is_read, limit = _list_messages_params(tool_input)
                result = await microsoft_graph_request(
                    "GET",
                    f"{GRAPH_API}/me/messages",
                    access_token=access_token,
                    params=params,
                    connector_id="outlook",
                    scope_hint="Mail.Read",
                )
                # Trim to the fields the agent actually needs so we
                # don't blow the LLM's token budget on Graph metadata.
                msgs = []
                for m in (result.get("value") or []):
                    if (scan_is_read is not None
                            and bool(m.get("isRead")) != scan_is_read):
                        continue
                    row: dict[str, Any] = {
                        "id": m.get("id"),
                        "subject": m.get("subject", ""),
                        "from": (
                            (m.get("from") or {}).get("emailAddress", {}).get("address")
                        ),
                        "preview": m.get("bodyPreview", "")[:300],
                        "received_at": m.get("receivedDateTime"),
                        "is_read": m.get("isRead"),
                    }
                    if include_body:
                        body_obj = m.get("body") or {}
                        row["body_content_type"] = body_obj.get("contentType")
                        row["body"] = (body_obj.get("content") or "")[:50_000]
                    msgs.append(row)
                    if len(msgs) >= limit:
                        break
                return ConnectorOk(content=json.dumps({"messages": msgs}))

            if tool_name == "outlook__get_message":
                mid = tool_input.get("message_id")
                if not mid:
                    return ConnectorToolError(
                        message="message_id required", retryable=False,
                    )
                msg = await microsoft_graph_request(
                    "GET",
                    f"{GRAPH_API}/me/messages/{mid}",
                    access_token=access_token,
                    connector_id="outlook",
                    scope_hint="Mail.Read",
                )
                body_obj = msg.get("body") or {}
                return ConnectorOk(content=json.dumps({
                    "id": msg.get("id"),
                    "subject": msg.get("subject", ""),
                    "from": (
                        (msg.get("from") or {}).get("emailAddress", {}).get("address")
                    ),
                    "to": [
                        (r.get("emailAddress") or {}).get("address")
                        for r in (msg.get("toRecipients") or [])
                    ],
                    "received_at": msg.get("receivedDateTime"),
                    "body_content_type": body_obj.get("contentType"),
                    "body": (body_obj.get("content") or "")[:50_000],
                }))

            if tool_name == "outlook__send_message":
                to = tool_input.get("to")
                subject = tool_input.get("subject")
                body = tool_input.get("body")
                if not (to and subject and body):
                    return ConnectorToolError(
                        message="to, subject, and body are required",
                        retryable=False,
                    )
                payload = {
                    "message": {
                        "subject": subject,
                        "body": {
                            "contentType": "Text",
                            "content": body,
                        },
                        "toRecipients": _split_recipient_csv(to),
                    },
                    "saveToSentItems": True,
                }
                cc = _split_recipient_csv(tool_input.get("cc"))
                if cc:
                    payload["message"]["ccRecipients"] = cc
                bcc = _split_recipient_csv(tool_input.get("bcc"))
                if bcc:
                    payload["message"]["bccRecipients"] = bcc
                # /me/sendMail returns 202 Accepted with an empty body.
                # microsoft_graph_request handles that and returns
                # `{"raw": ""}`; we just surface "sent: true" so the
                # LLM doesn't paste no-content into chat.
                await microsoft_graph_request(
                    "POST",
                    f"{GRAPH_API}/me/sendMail",
                    access_token=access_token,
                    json_body=payload,
                    connector_id="outlook",
                    scope_hint="Mail.Send",
                )
                return ConnectorOk(content=json.dumps({
                    "sent": True,
                    "to": to,
                    "subject": subject,
                }))

            if tool_name == "outlook__create_draft":
                to = tool_input.get("to")
                subject = tool_input.get("subject")
                body = tool_input.get("body")
                if not (to and subject and body):
                    return ConnectorToolError(
                        message="to, subject, and body are required",
                        retryable=False,
                    )
                message = {
                    "subject": subject,
                    "body": {
                        "contentType": "Text",
                        "content": body,
                    },
                    "toRecipients": _split_recipient_csv(to),
                }
                cc = _split_recipient_csv(tool_input.get("cc"))
                if cc:
                    message["ccRecipients"] = cc
                bcc = _split_recipient_csv(tool_input.get("bcc"))
                if bcc:
                    message["bccRecipients"] = bcc
                # POST /me/messages creates the message IN DRAFTS —
                # nothing is sent (that is the whole point: send stays
                # rail-forbidden, R29 §5). Needs Mail.ReadWrite, which
                # pre-R29 connections never consented to — the Graph
                # 403 comes back through _retarget_reauth as the
                # reconnect-shaped error.
                result = await microsoft_graph_request(
                    "POST",
                    f"{GRAPH_API}/me/messages",
                    access_token=access_token,
                    json_body=message,
                    connector_id="outlook",
                    scope_hint="Mail.ReadWrite",
                )
                return ConnectorOk(content=json.dumps({
                    "draft_id": result.get("id"),
                    "id": result.get("id"),
                    "web_link": result.get("webLink"),
                    "to": to,
                    "subject": subject,
                }))

            return ConnectorToolError(
                message=f"unknown outlook tool {tool_name!r}",
                retryable=False,
            )
        except _MicrosoftConnectorError as e:
            return _retarget_reauth(e.result)

    async def revoke(self, user_id, access_token, refresh_token=None):
        # Microsoft Identity Platform has no token-revoke endpoint;
        # microsoft_revoke is a no-op so the dispatcher's
        # "best-effort revoke at provider" step doesn't branch.
        await microsoft_revoke(access_token)

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return await microsoft_refresh(refresh_token, scopes=scopes)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Cheap probe — GET /me/messageRules (which 200s with []
        when empty, doesn't fan out to the actual mailbox content,
        and is included in `Mail.Read`)."""
        try:
            access_token = await _resolve_token(ctx.user_id)
            await microsoft_graph_request(
                "GET",
                f"{GRAPH_API}/me",
                access_token=access_token,
                connector_id="outlook",
                scope_hint="User.Read",
            )
            return HealthResult(ok=True)
        except _MicrosoftConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
