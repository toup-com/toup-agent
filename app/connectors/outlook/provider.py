"""Outlook (Microsoft Graph mail) connector provider.

Reads + sends Outlook mail via Microsoft Graph v1.0. Auth flows
through the shared `_microsoft_base` helpers; only the API endpoint
shape differs from other Microsoft 365 surfaces (Calendar, Teams,
OneDrive) which will reuse the same base.

Send-mail endpoint quirks:
  - POST /me/sendMail returns 202 Accepted with an empty body. We
    surface that as a {"sent": true} payload so the agent doesn't
    paste "no content" into the chat.
  - `toRecipients` is a list of objects, not a comma-separated
    string. We split-on-comma at the boundary and reshape for the
    LLM-friendly flat string input.
"""

from __future__ import annotations

import json
from typing import ClassVar, Optional

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


class OutlookProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "outlook"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = await _resolve_token(ctx.user_id)
        except _MicrosoftConnectorError as e:
            return _retarget_reauth(e.result)

        try:
            if tool_name == "outlook__list_messages":
                params: dict = {
                    "$top": int(tool_input.get("max_results", 10)),
                    "$select": (
                        "id,subject,from,toRecipients,receivedDateTime,"
                        "bodyPreview,isRead,hasAttachments"
                    ),
                    "$orderby": "receivedDateTime desc",
                }
                if tool_input.get("query"):
                    # Graph's $search header-style: e.g. "from:alice".
                    params["$search"] = f'"{tool_input["query"]}"'
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
                msgs = [
                    {
                        "id": m.get("id"),
                        "subject": m.get("subject", ""),
                        "from": (
                            (m.get("from") or {}).get("emailAddress", {}).get("address")
                        ),
                        "preview": m.get("bodyPreview", "")[:300],
                        "received_at": m.get("receivedDateTime"),
                        "is_read": m.get("isRead"),
                    }
                    for m in (result.get("value") or [])
                ]
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

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await microsoft_refresh(refresh_token)

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
