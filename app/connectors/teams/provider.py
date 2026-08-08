"""Microsoft Teams connector provider.

Microsoft Graph v1.0, delegated (user) tokens only. Auth, refresh, revoke
and the base HTTP→ConnectorResult mapping all come from
`_microsoft_base`, which Outlook already proved out; only the endpoint
shapes below are Teams-specific.

Graph's Teams surface has more query-parameter footguns than its mail
surface, and every one of them fails quietly rather than loudly. Verified
2026-08-08 against the v1.0 reference:

  - GET /me/joinedTeams supports NO OData parameters at all, so there is
    nothing to page or trim — its cost is the user's team count.
  - GET /chats and GET /chats/{id}/messages both cap $top at 50.
  - GET /chats/{id}/messages supports DESCENDING $orderby only, and its
    $filter is silently IGNORED unless $orderby names the same property.
    It does not support $select, so message bodies always come back.
  - GET /teams/{id}/channels populates `email` at a documented
    performance cost, so we $select around it.
  - `$expand=members` on /chats returns at most 25 members per chat
    regardless of $top.

The other Teams-shaped trap is one we deliberately did not fall into:
Graph's "protected APIs" — the ones behind a request-and-approval process
and a per-message payment model — are the APPLICATION-permission export
paths. Delegated reads like these are not subject to it.
"""

from __future__ import annotations

import json
import urllib.parse
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
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault

GRAPH_API = "https://graph.microsoft.com/v1.0"

# Graph rejects anything above this on /chats and /chats/{id}/messages.
_MAX_TOP = 50

# Chat messages are short and we return up to 50 of them in one payload;
# an unbounded body would let one pasted document eat the agent's whole
# context. Outlook's 50k ceiling is per-message and single-message.
_MAX_BODY_CHARS = 4_000

# $expand=members caps at 25 server-side; keep the LLM-facing list at the
# handful that actually identify a conversation.
_MAX_MEMBERS = 20

_GRAPH_SCOPE_PREFIX = "https://graph.microsoft.com/"


# ─── Scope evidence ──────────────────────────────────────────────────
#
# Everything in this section exists to answer ONE question honestly: can
# we actually show that a scope is absent? See `_refine_403`.


def _scope_leaf(scope: str) -> str:
    """Normalise a Graph scope to a comparable leaf.

    Two spellings of the same permission reach us. The manifest declares
    `https://graph.microsoft.com/Chat.Read`; Entra's token response
    documents its `scope` field URL-ENCODED and lower-cased —
    `https%3A%2F%2Fgraph.microsoft.com%2Fmail.read` — and `oauth.py`
    stores whatever came back verbatim. A plain `in` test between those
    two is false for every Microsoft identity we hold, which would make a
    scope-absence check answer "missing" universally.
    """
    s = urllib.parse.unquote(scope).strip().casefold()
    if s.startswith(_GRAPH_SCOPE_PREFIX):
        s = s[len(_GRAPH_SCOPE_PREFIX):]
    return s


async def _granted_scope_leaves(user_id: str) -> Optional[set[str]]:
    """Scopes recorded at consent, normalised — or None when the record
    can't prove anything either way."""
    try:
        async with async_session_maker() as db:
            ident = await _vault.get(db, user_id, "teams")
    except Exception:
        # A DB hiccup is not evidence about the user's consent.
        return None
    if ident is None or not ident.scopes:
        return None
    leaves = {_scope_leaf(s) for s in ident.scopes}
    # `.default` means "every permission already consented for this app".
    # The recorded list then says nothing about any individual scope.
    if ".default" in leaves:
        return None
    return leaves


async def _refine_403(
    result: ConnectorResult,
    *,
    user_id: str,
    scope_hint: str,
) -> ConnectorResult:
    """Re-decide a `ConnectorScopeMissing` verdict against real evidence.

    `_handle_microsoft_error` reaches that verdict by substring-matching
    "insufficient" / "scope" / "permission" in the 403 body, which cannot
    separate "you never granted this scope" from the reasons Teams
    specifically 403s: the user isn't a member of that chat or team, the
    channel is private or shared, or a tenant policy blocks the app. Those
    have opposite fixes — re-consent versus nothing the user can do here —
    and telling someone to reconnect an already-correct connection is the
    confident wrong diagnosis this codebase keeps paying for.

    So we only keep the verdict when the granted-scope record proves the
    scope is absent. Otherwise the caller gets an honest unknown. Costs
    one indexed vault read, and only on the 403 path.
    """
    if not isinstance(result, ConnectorScopeMissing):
        return result

    want = _scope_leaf(scope_hint)
    granted = await _granted_scope_leaves(user_id)

    if granted is not None and want and want not in granted:
        return result  # proven absent

    if granted is None:
        detail = "we have no usable record of what was granted at consent"
    else:
        detail = f"{scope_hint} WAS granted at consent"
    return ConnectorToolError(
        message=(
            f"Microsoft Graph returned 403 for this call and {detail}, so the "
            f"cause is unconfirmed. Teams also returns 403 when the signed-in "
            f"user is not a member of the chat, team or channel, when the "
            f"channel is private or shared, or when a tenant policy blocks "
            f"the app."
        ),
        retryable=False,
    )


# ─── Request helpers ─────────────────────────────────────────────────


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "teams")
    if ident is None or not ident.access_token:
        raise _MicrosoftConnectorError(
            ConnectorToolError(message="No active Teams identity", retryable=False),
        )
    return ident.access_token


async def _graph(
    method: str,
    url: str,
    *,
    access_token: str,
    user_id: str,
    scope_hint: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Any:
    try:
        return await microsoft_graph_request(
            method,
            url,
            access_token=access_token,
            json_body=json_body,
            params=params,
            connector_id="teams",
            scope_hint=scope_hint,
        )
    except _MicrosoftConnectorError as e:
        refined = await _refine_403(
            e.result, user_id=user_id, scope_hint=scope_hint,
        )
        if refined is e.result:
            raise
        raise _MicrosoftConnectorError(refined) from e


def _path_id(raw: Any) -> str:
    """Escape a Graph id for use as one URL path segment.

    Teams ids are not opaque-looking by accident — a chat id is
    `19:…@thread.v2` and Microsoft's own examples put the `:` and `@`
    straight into the path, so those two stay in the safe set and every
    real id goes over the wire byte-identical to the reference request.
    Everything else is escaped, `/` included: these ids arrive as LLM tool
    arguments, and `quote()`'s default `safe='/'` would let one walk out of
    its segment and address a different Graph resource.
    """
    return urllib.parse.quote(str(raw), safe=":@")


def _clamp_top(raw: Any, default: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, _MAX_TOP))


def _member_rows(chat: dict) -> list[dict]:
    """Flatten `$expand=members` into name + address.

    There is deliberately no "the other participant" field. Graph flags no
    member as the signed-in user, the chat id that encodes both user ids is
    documented as opaque, and resolving it via /me would cost a User.Read
    scope no tool here otherwise needs. Naming everyone is honest; guessing
    which one is you is not.
    """
    out: list[dict] = []
    for m in (chat.get("members") or [])[:_MAX_MEMBERS]:
        out.append({
            "display_name": m.get("displayName"),
            "email": m.get("email"),
        })
    return out


def _message_row(m: dict) -> dict:
    body = m.get("body") or {}
    frm = (m.get("from") or {}).get("user") or {}
    return {
        "id": m.get("id"),
        "created_at": m.get("createdDateTime"),
        # System messages ("X joined the chat", "chat renamed") carry
        # from: null. That is data, not a parse failure.
        "sender": frm.get("displayName"),
        "message_type": m.get("messageType"),
        "body_content_type": body.get("contentType"),
        "body": (body.get("content") or "")[:_MAX_BODY_CHARS],
        "deleted_at": m.get("deletedDateTime"),
    }


class TeamsProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "teams"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        # Prefer the dispatcher's pre-decrypted token — skips a duplicate
        # vault.get + Fernet decrypt in the provider.
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _MicrosoftConnectorError as e:
            return e.result

        try:
            if tool_name == "teams__list_teams":
                # No query parameters — this endpoint documents support for
                # none, so anything sent is at best ignored.
                result = await _graph(
                    "GET",
                    f"{GRAPH_API}/me/joinedTeams",
                    access_token=access_token,
                    user_id=ctx.user_id,
                    scope_hint="Team.ReadBasic.All",
                )
                teams = [
                    {
                        "id": t.get("id"),
                        "name": t.get("displayName"),
                        "description": t.get("description"),
                        "is_archived": t.get("isArchived"),
                    }
                    # Graph returns every team property here but populates
                    # only these; the rest are null by design.
                    for t in (result.get("value") or [])
                ]
                return ConnectorOk(content=json.dumps({"teams": teams}))

            if tool_name == "teams__list_channels":
                team_id = tool_input.get("team_id")
                if not team_id:
                    return ConnectorToolError(
                        message="team_id required", retryable=False,
                    )
                result = await _graph(
                    "GET",
                    f"{GRAPH_API}/teams/{_path_id(team_id)}/channels",
                    access_token=access_token,
                    user_id=ctx.user_id,
                    scope_hint="Channel.ReadBasic.All",
                    # Populating `email` is a documented slow path on this
                    # endpoint, and nothing here needs a channel's address.
                    params={
                        "$select": (
                            "id,displayName,description,membershipType,"
                            "webUrl,isArchived"
                        ),
                    },
                )
                channels = [
                    {
                        "id": c.get("id"),
                        "name": c.get("displayName"),
                        "description": c.get("description"),
                        "membership_type": c.get("membershipType"),
                        "web_url": c.get("webUrl"),
                        "is_archived": c.get("isArchived"),
                    }
                    for c in (result.get("value") or [])
                ]
                return ConnectorOk(content=json.dumps({
                    "team_id": team_id,
                    "channels": channels,
                }))

            if tool_name == "teams__list_chats":
                top = _clamp_top(tool_input.get("max_results", 25), 25)
                result, ordered = await self._list_chats(
                    access_token=access_token, user_id=ctx.user_id, top=top,
                )
                chats = []
                for c in (result.get("value") or []):
                    members = _member_rows(c)
                    chats.append({
                        "id": c.get("id"),
                        # One-on-one chats always have topic: null.
                        "topic": c.get("topic"),
                        "chat_type": c.get("chatType"),
                        "last_updated_at": c.get("lastUpdatedDateTime"),
                        "web_url": c.get("webUrl"),
                        "members": members,
                    })
                return ConnectorOk(content=json.dumps({
                    "chats": chats,
                    # The sort is best-effort (see _list_chats); say so
                    # rather than let the agent claim "most recent".
                    "ordered_by_recent": ordered,
                }))

            if tool_name == "teams__read_chat_messages":
                chat_id = tool_input.get("chat_id")
                if not chat_id:
                    return ConnectorToolError(
                        message="chat_id required", retryable=False,
                    )
                top = _clamp_top(tool_input.get("max_results", 25), 25)
                result = await _graph(
                    "GET",
                    f"{GRAPH_API}/chats/{_path_id(chat_id)}/messages",
                    access_token=access_token,
                    user_id=ctx.user_id,
                    scope_hint="Chat.Read",
                    params={
                        "$top": top,
                        # Descending only — Graph rejects `asc` here. No
                        # $select: this endpoint doesn't support it.
                        "$orderby": "createdDateTime desc",
                    },
                )
                messages = [_message_row(m) for m in (result.get("value") or [])]
                return ConnectorOk(content=json.dumps({
                    "chat_id": chat_id,
                    "count": len(messages),
                    "messages": messages,
                }))

            if tool_name == "teams__send_chat_message":
                chat_id = tool_input.get("chat_id")
                message = tool_input.get("message")
                if not (chat_id and message):
                    return ConnectorToolError(
                        message="chat_id and message are required",
                        retryable=False,
                    )
                content_type = tool_input.get("content_type") or "text"
                if content_type not in ("text", "html"):
                    return ConnectorToolError(
                        message="content_type must be 'text' or 'html'",
                        retryable=False,
                    )
                # 201 Created with the full chatMessage back — unlike
                # Outlook's /sendMail, which 202s with no body.
                sent = await _graph(
                    "POST",
                    f"{GRAPH_API}/chats/{_path_id(chat_id)}/messages",
                    access_token=access_token,
                    user_id=ctx.user_id,
                    scope_hint="ChatMessage.Send",
                    json_body={
                        "body": {"contentType": content_type, "content": message},
                    },
                )
                return ConnectorOk(content=json.dumps({
                    "sent": True,
                    "chat_id": chat_id,
                    "message_id": sent.get("id"),
                    "created_at": sent.get("createdDateTime"),
                }))

            return ConnectorToolError(
                message=f"unknown teams tool {tool_name!r}",
                retryable=False,
            )
        except _MicrosoftConnectorError as e:
            return e.result

    async def _list_chats(
        self, *, access_token: str, user_id: str, top: int,
    ) -> tuple[dict, bool]:
        """GET /me/chats with members expanded, newest-active first.

        Graph documents `$top`, `$expand=members` and
        `$orderby=lastMessagePreview/createdDateTime desc` on this endpoint
        individually but never together, and an unsupported OData
        combination comes back 4xx. A cosmetic sort is not worth handing
        the agent an error, so a 4xx (which `_handle_microsoft_error` maps
        to ConnectorToolError — auth, throttling and outage verdicts pass
        straight through) retries once unordered and reports which it got.
        """
        url = f"{GRAPH_API}/me/chats"
        params = {"$top": top, "$expand": "members"}
        try:
            result = await _graph(
                "GET", url,
                access_token=access_token, user_id=user_id,
                scope_hint="Chat.Read",
                params={
                    **params,
                    "$orderby": "lastMessagePreview/createdDateTime desc",
                },
            )
            return result, True
        except _MicrosoftConnectorError as e:
            if not isinstance(e.result, ConnectorToolError):
                raise
        result = await _graph(
            "GET", url,
            access_token=access_token, user_id=user_id,
            scope_hint="Chat.Read", params=params,
        )
        return result, False

    async def revoke(self, user_id, access_token, refresh_token=None):
        # Documented no-op: the Microsoft identity platform exposes no
        # token-revoke endpoint. Tokens age out, and refresh-token sign-out
        # is a user action at https://myaccount.microsoft.com. Called
        # anyway so the dispatcher's revoke step doesn't branch per
        # provider, and so this stays one edit from a real endpoint if
        # Microsoft ever ships one.
        await microsoft_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await microsoft_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Cheapest authenticated round-trip this connector has: one chat,
        no expansion. A user with no chats 200s with an empty list, which
        is a healthy connection."""
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            await _graph(
                "GET",
                f"{GRAPH_API}/me/chats",
                access_token=access_token,
                user_id=ctx.user_id,
                scope_hint="Chat.Read",
                params={"$top": 1},
            )
            return HealthResult(ok=True)
        except _MicrosoftConnectorError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
