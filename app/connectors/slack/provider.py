"""Slack connector provider — Web API (slack.com/api).

Bearer auth like the rest, but Slack breaks three assumptions that every
other connector in this package relies on, and each one is load-bearing:

  1. **HTTP 200 is not success.** Almost every Slack Web API error comes
     back as `200 OK` with `{"ok": false, "error": "missing_scope"}`.
     `resp.raise_for_status()` and `if resp.status_code == 403` are both
     blind to it. `_check` below reads the body, always, before anything
     else looks at the response — the status code is only consulted for
     429 and 5xx, which are the two cases Slack does signal in the
     status line.

  2. **The token is a USER token (`xoxp-`), not a bot token.** See the
     long note at the top of `manifest.yaml`. Practically it means every
     call here sees exactly what the connected human sees — including
     their DMs — and `chat.postMessage` posts under their own name. It
     also means `search.messages` works at all; Slack does not offer
     search to bot tokens.

  3. **Ids are not names, anywhere.** `conversations.history` returns
     `user: "U04J1F2"` and text containing `<@U04J1F2>`; a 1:1 DM has no
     `name` field at all, only the other person's user id. Handing that
     to an LLM produces answers full of opaque ids, so this file
     resolves ids to display names through `_UserCache` and rewrites
     message text through `_render_text`. The cache is keyed by a
     fingerprint of the access token, never by user id alone — Slack
     ids are unique per workspace, not globally, so a bare id key would
     let one workspace's directory answer another workspace's lookup.

Rate limits are per-method tiers, not one global budget: `search.messages`
is Tier 2 (~20/min) while `conversations.history` is Tier 3 (~50/min).
Slack sends `Retry-After` on every 429, so `_check` trusts the header and
only falls back to `_DEFAULT_RETRY_AFTER_S` when it is absent or junk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, ClassVar, Optional

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


SLACK_API = "https://slack.com/api"

_HTTP_TIMEOUT_S = 15.0
_DEFAULT_RETRY_AFTER_S = 30
_REAUTH_URL = "/agent/integrations/slack"

# Slack's own ceilings. `limit` above 200 is accepted on paged methods but
# documented as liable to be truncated, and `search.messages` caps at 100.
_MAX_PAGE = 200
_MAX_SEARCH_COUNT = 100

# How many unresolved user ids one tool call may look up. A 30-message
# window rarely holds more than a handful of distinct speakers, and
# `users.info` is Tier 4 — but a channel full of one-line replies from
# fifty different people would otherwise turn one read into fifty API
# calls. Beyond the cap ids are left as ids, which reads badly but never
# rate-limits the user out of their own workspace.
_MAX_USER_LOOKUPS = 25
_USER_LOOKUP_CONCURRENCY = 5

# Directory entries change rarely (a display-name edit); an hour of
# staleness is invisible and saves the lookup on every subsequent read.
_USER_CACHE_TTL_S = 3600.0
_USER_CACHE_MAX = 5000

# Token states that mean "this credential is dead" rather than "this
# request was wrong". `token_expired` only appears with rotation on.
_REAUTH_ERRORS = frozenset({
    "invalid_auth",
    "not_authed",
    "token_revoked",
    "token_expired",
    "account_inactive",
    "no_permission",
})

# Slack signals its own outages in the body, not the status line.
_DOWN_ERRORS = frozenset({
    "fatal_error",
    "internal_error",
    "service_unavailable",
    "request_timeout",
})


# ─── Pooled client ───────────────────────────────────────────────────
#
# Same rationale as `_google_base` / notion: without pooling every call
# pays a fresh TLS handshake, and several tools here issue 2-3 requests
# (resolve a channel, read it, resolve the speakers).

_SLACK_CLIENT: Optional[httpx.AsyncClient] = None
_SLACK_CLIENT_LOCK = asyncio.Lock()


async def _get_slack_client() -> httpx.AsyncClient:
    global _SLACK_CLIENT
    if _SLACK_CLIENT is not None:
        return _SLACK_CLIENT
    async with _SLACK_CLIENT_LOCK:
        if _SLACK_CLIENT is None:
            try:
                _SLACK_CLIENT = httpx.AsyncClient(
                    timeout=_HTTP_TIMEOUT_S,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=300.0,
                    ),
                )
            except Exception as e:
                logger.error(
                    "[slack] pooled AsyncClient construction failed (%s: %s) — "
                    "falling back to per-call clients. Investigate immediately.",
                    type(e).__name__, e,
                )
                _SLACK_CLIENT = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    return _SLACK_CLIENT


async def shutdown_slack_client() -> None:
    global _SLACK_CLIENT
    if _SLACK_CLIENT is not None:
        await _SLACK_CLIENT.aclose()
        _SLACK_CLIENT = None


class _SlackError(Exception):
    """Wraps a `ConnectorResult` so any depth of the call chain can exit
    via `raise` and translate once, at `execute`'s outer except. Same
    pattern as `_NotionError` / `_GoogleConnectorError`."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "slack")
    if ident is None or not ident.access_token:
        raise _SlackError(
            ConnectorToolError(message="No active Slack identity", retryable=False),
        )
    return ident.access_token


def _fingerprint(access_token: str) -> str:
    """Stable, non-reversible workspace key for the directory cache.

    Never log this and never key anything user-visible on it — it is
    derived from a live credential. 16 hex chars is ~64 bits, which is
    far beyond collision range for the handful of workspaces one process
    ever sees."""
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


# ─── Error mapping ───────────────────────────────────────────────────


def _body(resp: httpx.Response) -> dict:
    try:
        parsed = resp.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _retry_after(resp: httpx.Response) -> int:
    try:
        return max(int(resp.headers.get("Retry-After", _DEFAULT_RETRY_AFTER_S)), 1)
    except (TypeError, ValueError):
        return _DEFAULT_RETRY_AFTER_S


def _check(resp: httpx.Response, *, method: str) -> dict:
    """Turn one Slack response into a parsed body, or raise `_SlackError`.

    Order matters. 429 and 5xx are read from the status line because
    Slack's proxy answers those before the API does and the body may be
    HTML. **Everything else is decided by `ok` in the body**, since a
    failed Slack call is a 200 — checking the status first and the body
    second would let `{"ok": false, "error": "invalid_auth"}` through as
    a successful result with no `channels` key, which reads downstream as
    "the workspace has no channels".
    """
    if resp.status_code == 429:
        wait = _retry_after(resp)
        logger.info("[slack] rate limited method=%s retry_after=%ss", method, wait)
        raise _SlackError(ConnectorRateLimited(retry_after_s=wait))

    if resp.status_code >= 500:
        raise _SlackError(ConnectorProviderDown(
            provider_status_url="https://slack-status.com",
        ))

    body = _body(resp)
    if body.get("ok"):
        return body

    err = str(body.get("error") or "").strip() or f"http_{resp.status_code}"

    if err in _REAUTH_ERRORS:
        raise _SlackError(ConnectorReauthRequired(reauth_url=_REAUTH_URL))

    if err == "ratelimited":
        raise _SlackError(ConnectorRateLimited(retry_after_s=_retry_after(resp)))

    if err in _DOWN_ERRORS:
        raise _SlackError(ConnectorProviderDown(
            provider_status_url="https://slack-status.com",
        ))

    if err == "missing_scope":
        # Slack names the exact scope in `needed` — the only provider
        # here that does. Pass it through verbatim so the reconnect
        # prompt can say which permission is missing instead of asking
        # the user to re-grant everything and hope.
        needed = str(body.get("needed") or "").strip()
        raise _SlackError(ConnectorScopeMissing(
            required_scope=needed or "a Slack scope Toup did not request",
        ))

    # `not_in_channel` / `channel_not_found` / `is_archived` are caller
    # errors with a real remedy, and Slack's error slugs are terse enough
    # to be unhelpful on their own.
    if err == "not_in_channel":
        raise _SlackError(ConnectorToolError(
            message=(
                "Slack says you are not a member of that channel, so its "
                "messages are not readable. Join the channel in Slack first."
            ),
            retryable=False,
        ))
    if err in ("channel_not_found", "user_not_found"):
        raise _SlackError(ConnectorToolError(
            message=(
                f"Slack could not find that {'channel' if err.startswith('channel') else 'person'} "
                f"({err}). Call slack__list_channels or slack__list_users to get a "
                f"valid id — a channel the user has never joined is invisible to "
                f"this connection even if it exists."
            ),
            retryable=False,
        ))
    if err == "is_archived":
        raise _SlackError(ConnectorToolError(
            message="That Slack channel is archived; it cannot be posted to.",
            retryable=False,
        ))

    raise _SlackError(ConnectorToolError(
        message=f"Slack {method} failed: {err}",
        retryable=err in ("service_unavailable", "request_timeout"),
    ))


async def _call(
    method: str,
    *,
    access_token: str,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
) -> dict:
    """One Slack Web API call → parsed body. Raises `_SlackError`.

    Reads use GET with query params; writes use POST with a JSON body,
    which is what Slack documents for `chat.postMessage` (a form body
    silently mangles `blocks` and any text containing `&`).
    """
    client = await _get_slack_client()
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    try:
        if json_body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            resp = await client.post(
                f"{SLACK_API}/{method}", headers=headers, json=json_body,
            )
        else:
            resp = await client.get(
                f"{SLACK_API}/{method}", headers=headers,
                params={k: v for k, v in (params or {}).items() if v not in (None, "")},
            )
    except httpx.HTTPError as e:
        raise _SlackError(ConnectorProviderDown(
            provider_status_url="https://slack-status.com",
        )) from e

    return _check(resp, method=method)


# ─── Directory cache ─────────────────────────────────────────────────


class _UserCache:
    """id → display name, per workspace, with a TTL.

    Bounded on purpose: this lives for the process lifetime and a busy
    tenant could otherwise accumulate every member of every workspace it
    has ever touched. Eviction is oldest-first on insert, which is crude
    but correct for a cache whose entries are all the same size and all
    equally cheap to refetch.
    """

    def __init__(self) -> None:
        self._d: dict[tuple[str, str], tuple[float, str]] = {}

    def get(self, fp: str, uid: str) -> Optional[str]:
        hit = self._d.get((fp, uid))
        if hit is None:
            return None
        stamped, name = hit
        if time.monotonic() - stamped > _USER_CACHE_TTL_S:
            self._d.pop((fp, uid), None)
            return None
        return name

    def put(self, fp: str, uid: str, name: str) -> None:
        if len(self._d) >= _USER_CACHE_MAX:
            for k in list(self._d.keys())[: _USER_CACHE_MAX // 10]:
                self._d.pop(k, None)
        self._d[(fp, uid)] = (time.monotonic(), name)


_USERS = _UserCache()


def _display_name(member: dict) -> str:
    """Slack stores a name in four places and populates a different one
    per account. Ordered by what a human would call the person."""
    profile = member.get("profile") or {}
    for candidate in (
        profile.get("display_name"),
        profile.get("real_name"),
        member.get("real_name"),
        member.get("name"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return str(member.get("id") or "unknown")


async def _resolve_users(
    ids: list[str], *, access_token: str,
) -> dict[str, str]:
    """Look up display names for `ids`, cache-first.

    Failures are swallowed to the raw id rather than raised: a deleted
    account or a single 429 on `users.info` must not fail the read the
    user actually asked for.
    """
    fp = _fingerprint(access_token)
    out: dict[str, str] = {}
    missing: list[str] = []
    for uid in dict.fromkeys(i for i in ids if i):
        cached = _USERS.get(fp, uid)
        if cached is not None:
            out[uid] = cached
        else:
            missing.append(uid)

    if not missing:
        return out

    if len(missing) > _MAX_USER_LOOKUPS:
        logger.info(
            "[slack] %d unresolved user ids, looking up %d",
            len(missing), _MAX_USER_LOOKUPS,
        )
        missing = missing[:_MAX_USER_LOOKUPS]

    sem = asyncio.Semaphore(_USER_LOOKUP_CONCURRENCY)

    async def one(uid: str) -> None:
        async with sem:
            try:
                data = await _call("users.info", access_token=access_token,
                                   params={"user": uid})
            except _SlackError:
                return
            member = data.get("user")
            if isinstance(member, dict):
                name = _display_name(member)
                _USERS.put(fp, uid, name)
                out[uid] = name

    await asyncio.gather(*(one(u) for u in missing))
    return out


# ─── Message text ────────────────────────────────────────────────────

_MENTION_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]*)?>")
_LINK_RE = re.compile(r"<(https?://[^|>]+)(?:\|([^>]*))?>")
_CHANNEL_RE = re.compile(r"<#([CG][A-Z0-9]+)(?:\|([^>]*))?>")
_SPECIAL_RE = re.compile(r"<!(here|channel|everyone)(?:\|[^>]*)?>")


def _render_text(text: str, names: dict[str, str]) -> str:
    """Slack's wire format is not what anyone sees on screen.

    `<@U04J1F2>`, `<#C01|general>` and `<https://x|label>` are how Slack
    encodes mentions and links; left raw, the agent quotes ids back at
    the user and cannot tell that a message mentioned them by name.
    """
    if not text:
        return ""
    out = _MENTION_RE.sub(
        lambda m: "@" + names.get(m.group(1), m.group(1)), text,
    )
    out = _CHANNEL_RE.sub(lambda m: "#" + (m.group(2) or m.group(1)), out)
    out = _LINK_RE.sub(
        lambda m: f"{m.group(2)} ({m.group(1)})" if m.group(2) else m.group(1), out,
    )
    out = _SPECIAL_RE.sub(lambda m: "@" + m.group(1), out)
    return out.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def _mention_ids(text: str) -> list[str]:
    return _MENTION_RE.findall(text or "")


def _clamp(raw: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return default


# ─── Channel / user resolution ───────────────────────────────────────

_ID_RE = re.compile(r"\A[CGD][A-Z0-9]{6,}\Z")
_USER_ID_RE = re.compile(r"\A[UW][A-Z0-9]{6,}\Z")


async def _channel_index(access_token: str) -> list[dict]:
    """Every conversation the user can see, one page set, unfiltered.

    Not cached: a channel created a minute ago is exactly the one a user
    is most likely to ask about, and `conversations.list` is Tier 2 —
    cheap enough to pay for correctness here.
    """
    channels: list[dict] = []
    cursor = ""
    for _ in range(5):  # 5 × 200 = 1000 conversations; beyond that, use the id
        data = await _call(
            "conversations.list", access_token=access_token,
            params={
                "types": "public_channel,private_channel,mpim,im",
                "exclude_archived": "true",
                "limit": _MAX_PAGE,
                "cursor": cursor,
            },
        )
        chunk = data.get("channels")
        if isinstance(chunk, list):
            channels.extend(c for c in chunk if isinstance(c, dict))
        cursor = ((data.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break
    return channels


async def _resolve_channel(raw: str, *, access_token: str, for_write: bool) -> str:
    """Accept a channel id, "#name", "@person", a bare name, or a user id.

    The agent gets its ids from `slack__list_channels`, but users speak
    in names and the agent relays them, so a tool that only accepted
    `C01ABCDEF` would fail on the most natural possible input.
    """
    value = (raw or "").strip()
    if not value:
        raise _SlackError(ConnectorToolError(
            message="`channel` is required.", retryable=False,
        ))

    if _ID_RE.match(value):
        return value

    if _USER_ID_RE.match(value) or value.startswith("@"):
        return await _open_dm(value.lstrip("@"), access_token=access_token,
                             for_write=for_write)

    name = value.lstrip("#").lower()
    for c in await _channel_index(access_token):
        if str(c.get("name") or "").lower() == name:
            return str(c.get("id"))

    raise _SlackError(ConnectorToolError(
        message=(
            f"No Slack conversation named {value!r} is visible to this account. "
            f"Call slack__list_channels to see what is — a channel nobody here "
            f"has joined does not appear, even if it exists."
        ),
        retryable=False,
    ))


async def _find_user_id(handle: str, *, access_token: str) -> Optional[str]:
    if _USER_ID_RE.match(handle):
        return handle
    wanted = handle.strip().lower()
    cursor = ""
    for _ in range(5):
        data = await _call("users.list", access_token=access_token,
                           params={"limit": _MAX_PAGE, "cursor": cursor})
        for m in (data.get("members") or []):
            if not isinstance(m, dict):
                continue
            profile = m.get("profile") or {}
            candidates = {
                str(m.get("name") or "").lower(),
                str(profile.get("display_name") or "").lower(),
                str(profile.get("real_name") or "").lower(),
            }
            if wanted in candidates and not m.get("deleted"):
                return str(m.get("id"))
        cursor = ((data.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break
    return None


async def _open_dm(handle: str, *, access_token: str, for_write: bool) -> str:
    """Channel id of the DM with `handle`, opening one if needed."""
    uid = await _find_user_id(handle, access_token=access_token)
    if uid is None:
        raise _SlackError(ConnectorToolError(
            message=(
                f"No Slack member matches {handle!r}. Call slack__list_users to "
                f"find the right handle — Slack display names and @handles often "
                f"differ."
            ),
            retryable=False,
        ))

    try:
        data = await _call("conversations.open", access_token=access_token,
                           json_body={"users": uid, "return_im": True})
        channel = data.get("channel")
        if isinstance(channel, dict) and channel.get("id"):
            return str(channel["id"])
    except _SlackError as e:
        # `im:write` is requested, but an existing install predating that
        # scope will not have it. For a write we can still fall back to
        # handing chat.postMessage the user id, which Slack accepts and
        # resolves itself; for a read there is no such fallback.
        if not (for_write and isinstance(e.result, ConnectorScopeMissing)):
            raise
        logger.info("[slack] conversations.open unavailable, posting to user id")
        return uid

    raise _SlackError(ConnectorToolError(
        message=f"Slack did not return a DM conversation for {handle!r}.",
        retryable=True,
    ))


# ─── Shaping ─────────────────────────────────────────────────────────


def _channel_row(c: dict, names: dict[str, str]) -> dict:
    kind = (
        "im" if c.get("is_im") else
        "mpim" if c.get("is_mpim") else
        "private_channel" if c.get("is_private") else
        "public_channel"
    )
    row: dict[str, Any] = {
        "id": c.get("id"),
        "type": kind,
        "is_member": bool(c.get("is_member", kind in ("im", "mpim"))),
    }
    if kind == "im":
        uid = str(c.get("user") or "")
        row["user_id"] = uid
        row["user_name"] = names.get(uid, uid)
    else:
        row["name"] = c.get("name")
        topic = ((c.get("topic") or {}).get("value") or "").strip()
        purpose = ((c.get("purpose") or {}).get("value") or "").strip()
        if topic:
            row["topic"] = topic[:300]
        if purpose and purpose != topic:
            row["purpose"] = purpose[:300]
        if c.get("num_members") is not None:
            row["num_members"] = c.get("num_members")
    return row


def _message_row(m: dict, names: dict[str, str]) -> dict:
    uid = str(m.get("user") or "")
    row: dict[str, Any] = {
        "ts": m.get("ts"),
        # A message from an app or a workflow has `bot_id` and no `user`;
        # `username` is then the only name available.
        "from": names.get(uid) or m.get("username") or uid or "(app)",
        "text": _render_text(str(m.get("text") or ""), names),
    }
    if m.get("thread_ts") and m.get("thread_ts") != m.get("ts"):
        row["in_thread_of"] = m.get("thread_ts")
    if m.get("reply_count"):
        # The channel view shows only the parent, so without this the
        # agent cannot tell that a one-line message has 40 replies under it.
        row["reply_count"] = m.get("reply_count")
        row["thread_ts"] = m.get("ts")
    files = m.get("files")
    if isinstance(files, list) and files:
        row["files"] = [
            str(f.get("name") or f.get("title") or "file")
            for f in files if isinstance(f, dict)
        ][:10]
    if m.get("subtype"):
        row["subtype"] = m.get("subtype")
    return row


def _speaker_ids(messages: list[dict]) -> list[str]:
    ids: list[str] = []
    for m in messages:
        if m.get("user"):
            ids.append(str(m["user"]))
        ids.extend(_mention_ids(str(m.get("text") or "")))
    return ids


# ─── OAuth lifecycle ─────────────────────────────────────────────────


async def slack_refresh(refresh_token: str) -> RefreshResult:
    """Slack token rotation (`grant_type=refresh_token`).

    Only reachable when the app owner has switched rotation on at
    api.slack.com — by default Slack user tokens never expire and no
    refresh_token is issued, so the vault has nothing to call this with.
    Implemented anyway so enabling rotation is a config change.

    Like the code exchange, the rotated USER token arrives nested under
    `authed_user`; the top level carries the bot token when one exists.
    """
    app_cfg = await get_provider_app_async("slack")
    if app_cfg is None:
        raise RefreshFailed("slack provider app not configured")

    client = await _get_slack_client()
    try:
        resp = await client.post(
            f"{SLACK_API}/oauth.v2.access",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": app_cfg.client_id,
                "client_secret": app_cfg.client_secret,
            },
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as e:
        raise RefreshFailed(f"slack refresh transport error: {e}") from e

    body = _body(resp)
    if not body.get("ok"):
        raise RefreshFailed(f"slack refresh rejected: {body.get('error') or resp.text[:120]}")

    nested = body.get("authed_user")
    source = nested if isinstance(nested, dict) and nested.get("access_token") else body

    access = source.get("access_token")
    if not access:
        raise RefreshFailed("slack refresh returned no access_token")

    expires_at = None
    if source.get("expires_in"):
        from datetime import datetime, timedelta
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=int(source["expires_in"]))
        except (TypeError, ValueError):
            expires_at = None

    return RefreshResult(
        access_token=str(access),
        refresh_token=source.get("refresh_token") or None,
        expires_at=expires_at,
    )


async def slack_revoke(access_token: str) -> None:
    """`auth.revoke` — best effort, like every other provider's revoke.

    A token that is already dead answers `invalid_auth`, which is the
    desired end state, so nothing here raises: the vault is about to
    zero the ciphertext regardless and a failed revoke must not leave a
    row the user believes they disconnected.
    """
    try:
        client = await _get_slack_client()
        resp = await client.post(
            f"{SLACK_API}/auth.revoke",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        body = _body(resp)
        if not body.get("ok"):
            logger.info("[slack] auth.revoke said %s", body.get("error"))
    except Exception as e:
        logger.warning("[slack] auth.revoke failed: %s: %s", type(e).__name__, e)


# ─── Provider ────────────────────────────────────────────────────────


class SlackProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "slack"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _SlackError as e:
            return e.result

        try:
            if tool_name == "slack__list_channels":
                return await self._list_channels(tool_input, access_token)
            if tool_name == "slack__read_messages":
                return await self._read_messages(tool_input, access_token)
            if tool_name == "slack__search_messages":
                return await self._search_messages(tool_input, access_token)
            if tool_name == "slack__send_message":
                return await self._send_message(tool_input, access_token)
            if tool_name == "slack__list_users":
                return await self._list_users(tool_input, access_token)
            return ConnectorToolError(
                message=f"unknown slack tool {tool_name!r}", retryable=False,
            )
        except _SlackError as e:
            return e.result

    # ── Tools ──

    async def _list_channels(self, ti: dict, token: str) -> ConnectorResult:
        types = (ti.get("types") or "public_channel,private_channel,mpim,im").strip()
        data = await _call(
            "conversations.list", access_token=token,
            params={
                "types": types,
                "exclude_archived": "true" if ti.get("exclude_archived", True) else "false",
                "limit": _clamp(ti.get("limit", 100), 100, 1, _MAX_PAGE),
                "cursor": (ti.get("cursor") or "").strip(),
            },
        )
        raw = [c for c in (data.get("channels") or []) if isinstance(c, dict)]

        # DM rows carry only the counterpart's user id, so resolve before
        # filtering — otherwise `name_contains="sara"` can never match a DM.
        names = await _resolve_users(
            [str(c.get("user")) for c in raw if c.get("is_im") and c.get("user")],
            access_token=token,
        )
        rows = [_channel_row(c, names) for c in raw]

        needle = (ti.get("name_contains") or "").strip().lower()
        if needle:
            rows = [
                r for r in rows
                if needle in str(r.get("name") or "").lower()
                or needle in str(r.get("user_name") or "").lower()
            ]

        return ConnectorOk(content=json.dumps({
            "channels": rows,
            "next_cursor": (data.get("response_metadata") or {}).get("next_cursor") or None,
        }, ensure_ascii=False))

    async def _read_messages(self, ti: dict, token: str) -> ConnectorResult:
        channel = await _resolve_channel(
            str(ti.get("channel") or ""), access_token=token, for_write=False,
        )
        thread_ts = (ti.get("thread_ts") or "").strip()
        params = {
            "channel": channel,
            "limit": _clamp(ti.get("limit", 30), 30, 1, _MAX_PAGE),
            "cursor": (ti.get("cursor") or "").strip(),
            "oldest": (ti.get("oldest") or "").strip(),
        }
        if thread_ts:
            params["ts"] = thread_ts
            data = await _call("conversations.replies", access_token=token, params=params)
        else:
            data = await _call("conversations.history", access_token=token, params=params)

        raw = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        names = await _resolve_users(_speaker_ids(raw), access_token=token)

        return ConnectorOk(content=json.dumps({
            "channel": channel,
            "thread_ts": thread_ts or None,
            "messages": [_message_row(m, names) for m in raw],
            "has_more": bool(data.get("has_more")),
            "next_cursor": (data.get("response_metadata") or {}).get("next_cursor") or None,
        }, ensure_ascii=False))

    async def _search_messages(self, ti: dict, token: str) -> ConnectorResult:
        query = (ti.get("query") or "").strip()
        if not query:
            return ConnectorToolError(
                message="`query` is required for slack__search_messages.",
                retryable=False,
            )
        sort = "timestamp" if (ti.get("sort") or "").strip() == "timestamp" else "score"
        data = await _call(
            "search.messages", access_token=token,
            params={
                "query": query,
                "sort": sort,
                "sort_dir": "desc",
                "count": _clamp(ti.get("count", 20), 20, 1, _MAX_SEARCH_COUNT),
                "page": _clamp(ti.get("page", 1), 1, 1, 100),
            },
        )
        block = data.get("messages") or {}
        raw = [m for m in (block.get("matches") or []) if isinstance(m, dict)]
        names = await _resolve_users(_speaker_ids(raw), access_token=token)

        matches = []
        for m in raw:
            uid = str(m.get("user") or "")
            ch = m.get("channel") or {}
            matches.append({
                "ts": m.get("ts"),
                "from": names.get(uid) or m.get("username") or uid or "(app)",
                "channel_id": ch.get("id"),
                "channel_name": ch.get("name") or ("(direct message)" if ch.get("is_im") else None),
                "text": _render_text(str(m.get("text") or ""), names),
                "permalink": m.get("permalink"),
            })

        paging = block.get("paging") or {}
        return ConnectorOk(content=json.dumps({
            "query": query,
            "total": block.get("total"),
            "page": paging.get("page"),
            "pages": paging.get("pages"),
            "matches": matches,
        }, ensure_ascii=False))

    async def _send_message(self, ti: dict, token: str) -> ConnectorResult:
        text = str(ti.get("text") or "").strip()
        if not text:
            return ConnectorToolError(
                message="`text` is required — Slack rejects an empty message.",
                retryable=False,
            )
        channel = await _resolve_channel(
            str(ti.get("channel") or ""), access_token=token, for_write=True,
        )
        payload: dict[str, Any] = {"channel": channel, "text": text}
        thread_ts = (ti.get("thread_ts") or "").strip()
        if thread_ts:
            payload["thread_ts"] = thread_ts
            if ti.get("reply_broadcast"):
                payload["reply_broadcast"] = True

        data = await _call("chat.postMessage", access_token=token, json_body=payload)
        return ConnectorOk(content=json.dumps({
            "sent": True,
            "channel": data.get("channel") or channel,
            "ts": data.get("ts"),
        }, ensure_ascii=False))

    async def _list_users(self, ti: dict, token: str) -> ConnectorResult:
        data = await _call(
            "users.list", access_token=token,
            params={
                "limit": _clamp(ti.get("limit", 100), 100, 1, _MAX_PAGE),
                "cursor": (ti.get("cursor") or "").strip(),
            },
        )
        include_bots = bool(ti.get("include_bots"))
        include_deactivated = bool(ti.get("include_deactivated"))
        needle = (ti.get("name_contains") or "").strip().lower()

        fp = _fingerprint(token)
        users = []
        for m in (data.get("members") or []):
            if not isinstance(m, dict):
                continue
            if m.get("deleted") and not include_deactivated:
                continue
            if (m.get("is_bot") or m.get("id") == "USLACKBOT") and not include_bots:
                continue
            name = _display_name(m)
            handle = str(m.get("name") or "")
            real = str((m.get("profile") or {}).get("real_name") or "")
            if needle and needle not in f"{name} {handle} {real}".lower():
                continue
            _USERS.put(fp, str(m.get("id")), name)
            row = {"id": m.get("id"), "name": name, "handle": handle}
            if real and real != name:
                row["real_name"] = real
            if m.get("is_bot"):
                row["is_bot"] = True
            if m.get("deleted"):
                row["deactivated"] = True
            users.append(row)

        return ConnectorOk(content=json.dumps({
            "users": users,
            "next_cursor": (data.get("response_metadata") or {}).get("next_cursor") or None,
        }, ensure_ascii=False))

    # ── Lifecycle ──

    async def revoke(self, user_id, access_token, refresh_token=None):
        await slack_revoke(access_token)

    async def refresh(self, refresh_token: str) -> RefreshResult:
        return await slack_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """One conversation. Matches `health.probe: slack__list_channels`.

        A workspace with zero visible conversations is still healthy —
        that is a brand-new account, not an outage.
        """
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            await _call(
                "conversations.list", access_token=access_token,
                params={"limit": 1, "types": "public_channel"},
            )
            return HealthResult(ok=True)
        except _SlackError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
