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


def _member_id(m: Any) -> str:
    """One member's stable identity, for set arithmetic.

    `userId` is the AAD object id on an `aadUserConversationMember`, and
    it is the SAME id space `chatMessage.mentions[].mentioned.user.id`
    and `chatMessage.from.user.id` use — which is what lets "am I
    mentioned" be answered without asking Graph who I am. A guest or a
    federated member can arrive without one; their address is the only
    other thing that identifies them across two chats.
    """
    if not isinstance(m, dict):
        return ""
    return str(m.get("userId") or m.get("email") or "").strip().casefold()


def _member_ids(chat: dict) -> set:
    return {i for i in (_member_id(m) for m in (chat.get("members") or [])) if i}


def _identify_self(chats: list) -> tuple[Optional[str], str]:
    """Which of these members is the signed-in user — by set arithmetic,
    not by asking.

    This connector holds `Chat.Read` and `ChatMessage.Send` and nothing
    else. Graph marks no member as the caller, the chat id that encodes
    both participants' ids is documented opaque, access tokens are
    documented opaque to clients, and `GET /me` needs `User.Read` — a
    scope every existing Teams connection would have to be re-consented
    to gain. So the answer is derived from the one thing `/me/chats`
    guarantees: the caller is a member of EVERY chat it returns.

    Two rules, both deductions rather than guesses:

      R1. A `oneOnOne` chat Graph returns with exactly ONE member is the
          user's chat with themselves — there is no other 1:1 chat a
          person can be in alone — so that member is the caller.
      R2. Otherwise, intersect the member sets. Teams allows exactly one
          `oneOnOne` chat per pair, so two of them have different
          partners and the intersection is exactly the caller. Group
          chats are folded in for reach, and the answer is accepted ONLY
          when the intersection is a single identity: two colleagues who
          share every one of the user's chats leave two, and two is not
          an answer.

    Returns `(id, reason)`; `id` is None whenever the deduction does not
    close, and `reason` says which way it failed so the caller can put
    it in front of a person instead of guessing.
    """
    typed = [c for c in chats if isinstance(c, dict)]
    for c in typed:
        if str(c.get("chatType") or "") == "oneOnOne":
            ids = _member_ids(c)
            if len(ids) == 1:
                return next(iter(ids)), "your chat with yourself names you"
    sets = [s for s in (_member_ids(c) for c in typed) if s]
    if len(sets) < 2:
        return None, (
            "we can only work out which member is you from two or more "
            "chats, and this account has fewer than that"
        )
    common = set.intersection(*sets)
    if len(common) == 1:
        return next(iter(common)), "you are the one member every chat shares"
    if not common:
        return None, (
            "no single member appears in every one of your chats, which "
            "should be impossible — Graph returned member lists this "
            "connector cannot read"
        )
    return None, (
        f"{len(common)} people appear in every one of your chats, so which "
        f"one is you cannot be told apart from here"
    )


def _member_rows(chat: dict, self_id: Optional[str]) -> list[dict]:
    """Flatten `$expand=members` into name + address, marking the caller.

    `is_self` is only ever stamped from a CLOSED deduction (see
    `_identify_self`); when self is unknown every member carries false,
    never a guess.
    """
    out: list[dict] = []
    for m in (chat.get("members") or [])[:_MAX_MEMBERS]:
        mid = _member_id(m)
        out.append({
            "display_name": m.get("displayName"),
            "email": m.get("email"),
            "is_self": bool(self_id) and mid == self_id,
        })
    return out


#: `chatMessage.from` is an identitySet: exactly one of these keys is
#: populated, and WHICH one is the author's kind. A bot posts under
#: `application`; a Teams-connected device under `device`; a system
#: message ("X joined the chat") populates none of them.
_AUTHOR_KINDS = ("user", "application", "device")


def _author(m: dict) -> tuple[str, dict]:
    """`(kind, identity)` off a chatMessage's `from`.

    Returns `("", {})` for a system message, which is neither a person
    nor a bot — "Skip bots" must not eat "Dana joined the chat".
    """
    frm = m.get("from")
    if not isinstance(frm, dict):
        return "", {}
    for kind in _AUTHOR_KINDS:
        ident = frm.get(kind)
        if isinstance(ident, dict) and ident:
            return kind, ident
    return "", {}


def _mention_rows(m: dict) -> list[dict]:
    """`chatMessage.mentions` flattened to who was named.

    `mentioned` is an identitySet like `from`, so a channel @-mention, a
    tagged bot and a named person all arrive here and are told apart by
    the same key.
    """
    out: list[dict] = []
    for mention in (m.get("mentions") or []):
        if not isinstance(mention, dict):
            continue
        target = mention.get("mentioned")
        kind, ident = ("", {})
        if isinstance(target, dict):
            for k in _AUTHOR_KINDS + ("conversation", "tag"):
                got = target.get(k)
                if isinstance(got, dict) and got:
                    kind, ident = k, got
                    break
        out.append({
            "text": mention.get("mentionText"),
            "kind": kind,
            "id": ident.get("id"),
            "display_name": ident.get("displayName"),
        })
    return out


def _message_row(m: dict, self_id: Optional[str]) -> dict:
    body = m.get("body") or {}
    author_kind, author = _author(m)
    mentions = _mention_rows(m)
    row = {
        "id": m.get("id"),
        "created_at": m.get("createdDateTime"),
        "last_modified_at": m.get("lastModifiedDateTime"),
        # System messages ("X joined the chat", "chat renamed") carry
        # from: null. That is data, not a parse failure.
        "sender": author.get("displayName"),
        # "user" | "application" | "device" | "" — the half of "Skip
        # bots" that no query language here can express, so it rides on
        # the row and the narrowing happens over the returned list.
        "author_type": author_kind,
        "message_type": m.get("messageType"),
        "body_content_type": body.get("contentType"),
        "body": (body.get("content") or "")[:_MAX_BODY_CHARS],
        "deleted_at": m.get("deletedDateTime"),
        "mentions": mentions,
    }
    if self_id:
        row["mentions_me"] = any(
            str(x.get("id") or "").strip().casefold() == self_id
            for x in mentions
        )
    return row


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
                raw = [c for c in (result.get("value") or [])
                       if isinstance(c, dict)]
                self_id, self_reason = _identify_self(raw)
                chats = []
                for c in raw:
                    members = _member_rows(c, self_id)
                    ids = _member_ids(c)
                    chats.append({
                        "id": c.get("id"),
                        # One-on-one chats always have topic: null.
                        "topic": c.get("topic"),
                        "chat_type": c.get("chatType"),
                        "last_updated_at": c.get("lastUpdatedDateTime"),
                        # The CALLER's own read mark. Graph puts it on
                        # every /chats row for a delegated token, which
                        # is the only place in this connector that
                        # answers "has this person seen it" — chat
                        # messages themselves carry no read state, so a
                        # message stamped after this is the one honest
                        # definition of unread available here.
                        "last_read_at": (
                            (c.get("viewpoint") or {})
                            .get("lastMessageReadDateTime")
                        ),
                        "web_url": c.get("webUrl"),
                        "members": members,
                        # The user's chat with THEMSELVES: the only chat
                        # this connector can post into and still be able
                        # to say nobody else reads it. A closed
                        # deduction or false — never a maybe.
                        "is_self_chat": bool(self_id) and ids == {self_id},
                    })
                return ConnectorOk(content=json.dumps({
                    "chats": chats,
                    # The sort is best-effort (see _list_chats); say so
                    # rather than let the agent claim "most recent".
                    "ordered_by_recent": ordered,
                    "self_identified": bool(self_id),
                    # Present either way: a caller that needs the self
                    # chat and finds none has to be able to tell "you
                    # have not opened one" from "we could not work out
                    # which member is you", because only the first is
                    # something the person can fix.
                    "self_note": self_reason,
                    "self_user_id": self_id,
                }))

            if tool_name == "teams__read_chat_messages":
                chat_id = tool_input.get("chat_id")
                if not chat_id:
                    return ConnectorToolError(
                        message="chat_id required", retryable=False,
                    )
                top = _clamp_top(tool_input.get("max_results", 25), 25)
                mentions_only = bool(tool_input.get("mentions_only"))
                since_read = bool(tool_input.get("since_last_read"))

                # "Mentions me" needs to know who "me" is, and this
                # connector can only deduce that from the chat list
                # (see `_identify_self`). One extra GET, and ONLY when
                # the caller asked for the narrowing.
                self_id: Optional[str] = None
                if mentions_only:
                    listed, _ = await self._list_chats(
                        access_token=access_token, user_id=ctx.user_id,
                        top=_MAX_TOP,
                    )
                    self_id, why = _identify_self(listed.get("value") or [])
                    if not self_id:
                        # Refused, not silently widened: the caller asked
                        # for messages naming THEM, and returning
                        # everything under that request is the narrowing
                        # that lies.
                        return ConnectorToolError(
                            message=(
                                f"Cannot filter to messages that mention you: "
                                f"{why}. Read the chat without mentions_only, "
                                f"or use the mentions on each message."
                            ),
                            retryable=False,
                        )

                params: dict[str, Any] = {"$top": top}
                since: Optional[str] = None
                if since_read:
                    since = await self._last_read_at(
                        chat_id, access_token=access_token,
                        user_id=ctx.user_id,
                    )
                if since:
                    # Graph IGNORES `$filter` on this endpoint unless
                    # `$orderby` names the same property (see the module
                    # docstring), so the two move together or neither
                    # does anything at all.
                    params["$orderby"] = "lastModifiedDateTime desc"
                    params["$filter"] = f"lastModifiedDateTime gt {since}"
                else:
                    # Descending only — Graph rejects `asc` here. No
                    # $select: this endpoint doesn't support it.
                    params["$orderby"] = "createdDateTime desc"

                result = await _graph(
                    "GET",
                    f"{GRAPH_API}/chats/{_path_id(chat_id)}/messages",
                    access_token=access_token,
                    user_id=ctx.user_id,
                    scope_hint="Chat.Read",
                    params=params,
                )
                messages = [_message_row(m, self_id)
                            for m in (result.get("value") or [])]
                if mentions_only:
                    # Graph exposes no mentions predicate on this
                    # endpoint, so the narrowing runs over the returned
                    # page — which is why `count` below is the count of
                    # what the caller is actually handed.
                    messages = [m for m in messages if m.get("mentions_me")]
                payload: dict[str, Any] = {
                    "chat_id": chat_id,
                    "count": len(messages),
                    "messages": messages,
                }
                if since_read:
                    # Absent viewpoint is a real state (a chat the user
                    # has never opened), and it must not read as "no new
                    # messages".
                    payload["since_last_read_at"] = since
                    payload["since_last_read_applied"] = bool(since)
                if mentions_only:
                    payload["mentions_only_applied"] = True
                return ConnectorOk(content=json.dumps(payload))

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

    async def _last_read_at(
        self, chat_id: Any, *, access_token: str, user_id: str,
    ) -> Optional[str]:
        """When this user last read this chat, or None.

        `chat.viewpoint` is the CALLER's own read mark — the only read
        state anywhere in Teams' delegated surface, since a chatMessage
        carries none. A chat the user has never opened has no viewpoint
        at all, and that is a state, not a failure: the caller is told
        the bound was not applied rather than being handed an empty
        page that reads as "nothing new".
        """
        chat = await _graph(
            "GET",
            f"{GRAPH_API}/chats/{_path_id(chat_id)}",
            access_token=access_token,
            user_id=user_id,
            scope_hint="Chat.Read",
            params={"$select": "id,viewpoint"},
        )
        mark = (chat.get("viewpoint") or {}).get("lastMessageReadDateTime")
        return str(mark) if mark else None

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

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return await microsoft_refresh(refresh_token, scopes=scopes)

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
