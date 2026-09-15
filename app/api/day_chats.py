"""
Day Chats API — list and retrieve day-level conversation containers.

These endpoints work in read-only mode regardless of USE_DAY_CHAT_CONTEXT flag:
return data if backfill has completed, return empty list if not. Never error on
the flag being off. This lets the frontend deploy independently of the flag flip.
"""

import json as _json
import logging
from datetime import date as Date, datetime, timedelta
from typing import Optional, List, Tuple
from weakref import WeakKeyDictionary

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_, or_, func, distinct
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import User, Conversation, Message
from app.db.models.conversation import HIDDEN_DAY_CHANNELS
from app.db.models.day_chat import DayChat
from app.api.auth import get_current_user
from app.api.message_cards import (
    attach_run_to_cards,
    job_card_fields,
    load_build_jobs,
    public_text,
)

logger = logging.getLogger(__name__)


def _row_channel(msg, conv_channel: Optional[str]) -> str:
    """The channel of a ROW: `Message.channel` first, the Conversation's as a
    fallback, then 'web'.

    Message-first is the only order that survives a Conversation being reused
    across surfaces — once a WhatsApp turn and a mobile turn can share a day,
    the conversation's channel is a property of where the THREAD started, not
    of who sent this row. Every row written before `Message.channel` was
    stamped has NULL there, so a released client sees exactly today's value
    for historical rows.

    Deliberately NOT `channel_util.resolve_channel`: that logs an INFO line on
    every conversation-level fallback, and this runs once per row on the
    primary history fetch — 500 rows per chat open, per tenant.
    """
    own = getattr(msg, "channel", None)
    if isinstance(own, str) and own.strip():
        return own.strip()
    if isinstance(conv_channel, str) and conv_channel.strip():
        return conv_channel.strip()
    return "web"


class _CountedRoute(APIRoute):
    """Counts 5xx answers out of this router into `health_signals`.

    The agent process is the only one whose counters are read (they ride
    `/agent/health`); the platform mounts the same router and keeps its own,
    unread copy. Diagnostic only — nothing gates on it, and the exception is
    always re-raised.
    """

    def get_route_handler(self):
        original = super().get_route_handler()

        async def _handler(request):
            try:
                return await original(request)
            except Exception as exc:
                if int(getattr(exc, "status_code", 500) or 500) >= 500:
                    try:
                        from app.services import health_signals

                        health_signals.incr("day_chats_5xx")
                    except Exception:
                        pass
                raise

        return _handler


router = APIRouter(prefix="/day-chats", tags=["Day Chats"], route_class=_CountedRoute)

# When this router is mounted in `platform_main`, the platform DB
# does NOT have the agent-only tables (`day_chats`, `conversations`,
# `messages`) — they live in each tenant's agent DB. The endpoints
# below proxy to the user's agent when one exists; when it doesn't,
# the SELECT fallback would crash with `UndefinedTableError`. The
# helper `_safe_local` swallows that specific schema-shape error and
# yields an empty result, so a brand-new platform user (no agent
# provisioned yet) gets a quiet 200 [] instead of a red 500 in their
# devtools console. Caught 2026-05-06 in the post-Install live retest.
_MISSING_TABLE_ERRORS: tuple = (ProgrammingError, OperationalError)


_PREVIEW_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _attachment_urls(message_id: str, att: dict) -> dict:
    """Compute download_url and (when applicable) preview_url for a stored
    attachment. Mirrors the live WS event payload built by ws_chat.py so
    REST-loaded history renders identically to live messages."""
    from app.config import settings as _settings
    aid = att.get("id", "")
    mime = att.get("mime_type", "")
    out = {"download_url": f"{_settings.api_prefix}/files/{message_id}/{aid}"}
    if mime in _PREVIEW_MIMES or mime.startswith("image/"):
        out["preview_url"] = f"{_settings.api_prefix}/files/{message_id}/{aid}/preview?format=html"
    # Reload path: the same inline derivative the live frame advertises, so a
    # thread opened from history is not slower than one watched live.
    if att.get("has_thumb"):
        out["thumb_url"] = f"{_settings.api_prefix}/files/{message_id}/{aid}?variant=thumb"
    return out


_META_CACHE: "WeakKeyDictionary[Message, dict]" = WeakKeyDictionary()


def _metadata(msg: Message) -> dict:
    """``Message.metadata_json``, parsed ONCE per message.

    Round 21, item 5. Four serializers below want four different keys out of
    this one column, and each of them used to `json.loads` it independently —
    so a 500-message history load parsed the same JSON two thousand times,
    and an assistant turn with a long `tool_events` array paid for all four.
    Cached on the instance, so the four calls in one row's dict literal cost
    one parse.

    A WeakKeyDictionary rather than an attribute: `Message` is a SQLAlchemy
    model and setting an unmapped attribute on one is a habit that ends in a
    column somebody meant to persist. Entries die with the row.

    **Both cache operations are guarded, not just the write.** Callers pass
    row-shaped stand-ins — `types.SimpleNamespace` in the serializer tests, a
    row proxy elsewhere — and neither a weak reference nor a hash can be taken
    of every one of them. A cache is an optimisation; a serializer that raises
    because it could not memoise is a history load that 500s.
    """
    try:
        cached = _META_CACHE.get(msg)
    except TypeError:  # not weak-referenceable — parse and do not cache
        cached = None
    if cached is not None:
        return cached
    raw = getattr(msg, "metadata_json", None)
    parsed: dict = {}
    if raw:
        try:
            loaded = _json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(loaded, dict):
                parsed = loaded
        except (TypeError, ValueError):
            parsed = {}
    try:
        _META_CACHE[msg] = parsed
    except TypeError:
        pass
    return parsed


def _serialize_attachments(msg: Message) -> Optional[List[dict]]:
    """Return Message.attachments as a client-safe list (strip storage_path,
    enrich with download_url + preview_url) or None when there are none.
    Handles both native-JSON driver returns (list) and legacy TEXT-stored
    JSON strings."""
    import json as _json
    raw = getattr(msg, "attachments", None)
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, list):
        return None
    return [
        {
            **{k: v for k, v in att.items() if k != "storage_path"},
            **_attachment_urls(msg.id, att),
        }
        for att in raw
        if isinstance(att, dict)
    ]


def _serialize_media(msg: Message) -> Optional[dict]:
    """Extract the media payload persisted in metadata_json by agent_runner.
    Shape: {"type": "youtube"|"netflix", "video_id": "...", "title": "..."}.
    Returns None when absent or malformed."""
    parsed = _metadata(msg)
    media = parsed.get("media")
    return media if isinstance(media, dict) else None


def _serialize_app_artifact(msg: Message) -> Optional[dict]:
    """The app this turn handed over. Shape: ``{"slug": "nokia-snake"}``.

    Round 18. Before this key existed the clients recovered the slug by
    regexing the `present_app` tool result for `/api/artifacts/<slug>` or
    `[[open_app:<slug>]]` — which meant an internal route and a directive
    token had to survive inside a 200-character cut of a sentence written for
    the model, and had to keep being SHOWN to the user for the card to work at
    all. The slug now rides its own field, so the prose is free to be prose.

    Emitted by all four message readers for the reason api/message_cards.py
    documents at length: a field carried by one serializer and not the others
    disappears the moment a client takes its fallback path.
    """
    parsed = _metadata(msg)
    art = parsed.get("app_artifact")
    return art if isinstance(art, dict) and art.get("slug") else None


def _serialize_admin_notice(msg: Message) -> Optional[dict]:
    """Extract the operator notice persisted in metadata_json by
    app/agent/admin_message_writer.py. Shape: {dispatch_id, mode,
    title, sender_name, sent_at}; the prose is Message.content.

    This same field is emitted by api/sessions.py and
    api/messages_recover.py — the clients fall back from this endpoint
    to those, so a notice serialized on only one path disappears on the
    fallback and the card renders as a bare assistant bubble from the
    agent, which is the one thing the feature exists to prevent.
    """
    parsed = _metadata(msg)
    notice = parsed.get("admin_notice")
    return notice if isinstance(notice, dict) else None


def _serialize_automation_card(msg: Message, key: str) -> Optional[dict]:
    """Round 26 — the two automations setup cards
    (`automation_connector_card` / `automation_grant_card`), persisted
    verbatim by app/agent/automations/cards.py. Same four-serializer
    parity contract as admin_notice above: a card serialized on only
    one path vanishes when a client takes its fallback."""
    parsed = _metadata(msg)
    card = parsed.get(key)
    return card if isinstance(card, dict) and card.get("id") else None


def _serialize_meta_card(msg: Message, key: str) -> Optional[dict]:
    """Round 29 — dict-shaped metadata payloads with no `id` field:
    `pending_action` (the confirm card, same key the chat path uses),
    `draft_card`, `memory_update`, and `fix_chip`. Same four-serializer
    parity contract as the automation cards above."""
    parsed = _metadata(msg)
    card = parsed.get(key)
    return card if isinstance(card, dict) and card else None


def _serialize_tool_events(msg: Message) -> Optional[List[dict]]:
    """Extract the ToolPillRow records persisted in metadata_json by
    agent_runner. Shape per record: {tool, started_at_ms,
    completed_at_ms, summary}. Returns None when absent or malformed
    so the frontend ToolPillRow component skips rendering entirely
    for legacy (pre-feature) messages instead of showing an empty
    pill row."""
    parsed = _metadata(msg)
    events = parsed.get("tool_events")
    if not isinstance(events, list) or not events:
        return None
    # Defensive: drop any record missing the required keys instead of
    # poisoning the whole list. The agent always writes well-formed
    # records but a hand-edited row in production should degrade
    # gracefully, not 500 every history load.
    return [
        _with_public_copy(_with_web_refs(e)) for e in events
        if isinstance(e, dict) and "tool" in e and "started_at_ms" in e
    ] or None


def _with_public_copy(rec: dict) -> dict:
    """R30 (D-01/D-03/D-17): serve every persisted record with the copy the
    dictionary allows, whatever the runner wrote at the time.

    Records persisted before this round carry no ``label`` (the client
    humanised the wire id into "List events" / "Search issues"), a raw
    vendor-JSON ``summary`` for connector tools (rendered as the
    "Site: Toup · Is last: true" detail line), and terminal-flavoured emoji
    in prose summaries ("Overall: ✅ OK" from the doctor report). The runner
    now writes clean records; THIS is the read path that makes the rows
    already in the founder's history serve clean too — the rollout-boundary
    rule: a sometimes-missing field is fixed where it is read.

    Defensive by the `_with_web_refs` precedent: these serializers are
    mounted in platform_main too, where ``app/agent`` does not exist — an
    import failure serves the record as persisted rather than 500ing the
    history load.
    """
    try:
        from app.agent.tool_display import (
            is_first_party_tool, public_step_label, strip_emoji,
        )
    except Exception:  # noqa: BLE001 — platform image has no app/agent
        return rec
    try:
        out = rec
        tool = str(rec.get("tool") or "")
        if not rec.get("label"):
            out = {**out, "label": public_step_label(tool)}
        summary = rec.get("summary")
        if isinstance(summary, str) and summary:
            cleaned = summary
            # ND-19 / R31-28. This carried its own copy of the old
            # `"__" in tool` predicate — the one `tool_display` fixed
            # at the write path — so a first-party tool's JSON was
            # blanked on every history load. Measured on real rows:
            # `routines__remind` served an empty summary, and the
            # reminder card the client builds out of that exact string
            # therefore did not render on reload.
            if (
                "__" in tool
                and not is_first_party_tool(tool)
                and cleaned.strip()[:1] in "{["
            ):
                cleaned = ""
            else:
                cleaned = strip_emoji(cleaned)
            if cleaned != summary:
                out = {**out} if out is rec else out
                out["summary"] = cleaned
        return out
    except Exception:  # noqa: BLE001 — history must load even if this can't
        return rec


def _with_web_refs(rec: dict) -> dict:
    """Stamp ``domains``/``urls`` on a web tool record that was persisted
    without them.

    The runner has written both fields on every web_search/web_fetch/browser
    record since Round 4 (``agent_runner.extract_web_refs``, fleet 2026-08-19)
    — but records persisted before that rollout carry only the summary, and
    the clients' favicon resolver reads ``domains``. Probed 2026-08-19 on the
    founder tenant: 23/23 post-rollout web_search records had the field,
    117/117 pre-rollout ones did not — those are the "Searching the web" rows
    that fell back to the generic glyph next to newer rows showing the site.
    The persisted summary is the first 2 KB of the result (header plus the
    first results, each with its URL), so the field can be derived at read
    time. Never raises; a record that already has ``domains``, is not a web
    tool, or names no URL is returned unchanged.
    """
    if rec.get("domains"):
        return rec
    try:
        from app.agent.agent_runner import WEB_DOMAIN_TOOLS, extract_web_refs
        if rec.get("tool") not in WEB_DOMAIN_TOOLS:
            return rec
        domains, urls = extract_web_refs(rec["tool"], None, rec.get("summary"))
    except Exception:  # noqa: BLE001 — history must load even if this can't
        return rec
    if not domains:
        return rec
    return {**rec, "domains": domains, "urls": urls}


# ── Agent proxy (platform mode proxies to user's VPS agent) ──────────

async def _get_agent_proxy_info(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    # WHERE THE DATA ACTUALLY IS. `serving_locally()` is true in an agent
    # container and in a monolith/dev run — the AGENT_ONLY tables are in THIS
    # process's database and there is nothing to proxy to. Without this the
    # agent can resolve its OWN `agent_configs` row and proxy to itself over
    # the public internet: harmless while a failed hop fell through to the
    # local SELECT, and a 503 over a perfectly readable local database now that
    # it does not. `tenant_proxy.agent_proxy_info` has always had this guard;
    # the three hand-rolled copies (here, sessions.py, messages_recover.py)
    # never did.
    from app.api.tenant_proxy import serving_locally
    if serving_locally():
        return None
    try:
        from app.db.models import AgentConfig
        async with db.begin_nested():
            # NOT gated on `deploy_status == "active"`.
            #
            # `deploy_status` is set to "deploying" for the whole of a redeploy
            # and to "error" by a stale-deploy sweep 15 minutes later — while
            # the container is very often still up and holding the user's
            # entire history. Requiring "active" therefore skipped the proxy
            # for exactly those users and served them the platform's own empty
            # tables instead: the 2026-08-31 defect through a second door, and
            # one a retry cannot help with because nothing failed.
            #
            # A URL and a key is the whole test. What the tenant is actually
            # doing is decided by the HTTP call — which now answers 503 rather
            # than an empty list when it does not come back.
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(AgentConfig.user_id == user_id)
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass
    return None


class AgentSaidNo(Exception):
    """The tenant answered with a 4xx. That is an answer, and it is forwarded.

    Kept apart from `AgentUnreachable` because exactly one route depends on the
    difference: `app-conversation/{app_id}` uses 404 to mean "no conversation
    for this app yet", and a client that reads a 503 as that 404 starts a
    second thread beside the one the user was already in.
    """

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


class AgentUnreachable(Exception):
    """The user's history lives in their tenant agent and we could not read it.

    Raised — never swallowed — because the alternative was the 2026-08-31
    incident: `_proxy_day_chats` returned `None` for every failure, the caller
    read that as "nothing to say" and fell through to a SELECT against the
    PLATFORM database, which does not have the `day_chats` table at all. That
    SELECT raised `UndefinedTableError`, which the handler below turns into
    `200 []` — so a ten-second timeout to one tenant reached the phone as an
    empty, successful history, and the user reported his messages deleted.

    A user with no agent (`proxy is None`) still takes the local path; that is
    what the local path is FOR. A user WITH an agent whose agent did not answer
    gets a 503 and the client keeps whatever it already had on screen.
    """

    def __init__(self, detail: str, reason: str = "agent_unreachable"):
        super().__init__(detail)
        self.detail = detail
        # `agent_unreachable` (silence, transport, 502/503/504) vs
        # `agent_error` (the tenant answered 500). The client treats both as
        # 503 — the split is for the trail and for anyone reading the header.
        self.reason = reason


# The tenant read budget, and it is the WHOLE LADDER that has to fit.
#
# The number that matters is `attempts × timeout + backoff`, not the per-attempt
# timeout: the mobile client aborts at 15 s (`api.ts DEFAULT_TIMEOUT_MS`), so a
# retry ladder that can run to 20 s re-creates the exact inversion this round
# exists to remove — the platform still politely waiting while the phone has
# already drawn an error, and the honest 503 arriving after nobody is listening.
#
# 2 × 6 s + 0.4 s = 12.4 s worst case. Generous: a healthy tenant answers this
# route in well under a second, and the 10 s ceiling was only ever reached while
# a neighbouring container was being recreated on the same host.
_PROXY_TIMEOUT_S = 6.0
# One cheap retry. The failure this exists for is a transient host stall, not a
# broken tenant: a second attempt costs a user nothing when the first attempt
# already failed, and it is the difference between "your history is gone" and
# a half-second hiccup nobody sees.
_PROXY_ATTEMPTS = 2
_PROXY_BACKOFF_S = 0.4


async def _proxy_day_chats(agent_url: str, agent_api_key: str, path: str = "", params: dict = None):
    """Proxy a day-chats request to the VPS agent.

    TKT-LAT-007: uses the shared agent_http client so calendar opens
    (which fire many day-chat hops on the chat page) don't pay a
    TLS handshake per request.

    Raises `AgentUnreachable` when the tenant does not answer, or answers with
    anything other than 200. Returning `None` for that case is what made a
    tenant hiccup indistinguishable from an empty account.
    """
    import asyncio

    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/day-chats/{path}" if path else f"{agent_url}/api/day-chats"
    last: str = "unknown"
    for attempt in range(1, _PROXY_ATTEMPTS + 1):
        try:
            client = get_agent_http_client()
            resp = await client.get(
                url,
                headers={"X-Agent-Key": agent_api_key},
                params=params or {},
                timeout=_PROXY_TIMEOUT_S,
            )
            if resp.status_code == 200:
                return resp.json()
            # A 4xx is the tenant ANSWERING — most importantly the 404 that
            # `app-conversation/{id}` returns to mean "no conversation yet".
            # Forward it verbatim; retrying it would only waste the budget and
            # translating it to 503 would tell the client to wait for an answer
            # it has already been given.
            if 400 <= resp.status_code < 500:
                raise AgentSaidNo(resp.status_code, resp.text[:300])
            # A 500 is the tenant ANSWERING, deterministically, with a
            # failure — retrying it is three times the load for the same
            # answer. That is the rule the mobile client states at
            # `api.ts RETRY_STATUSES` and the server side never adopted: on
            # 2026-09-14 one user action cost six agent 500s (3 client
            # retries x 2 proxy attempts) and up to ~25 s of platform wait.
            # 502/503/504 are a hop failing in front of the tenant and stay
            # in the retry set.
            if resp.status_code == 500:
                logger.warning(
                    "Day-chats proxy %s: tenant answered 500 — not retried "
                    "(deterministic)", url,
                )
                raise AgentUnreachable("HTTP 500", reason="agent_error")
            # A 5xx was previously silent: `_proxy_day_chats` only logged
            # inside `except`, so a tenant answering 500 produced no line at
            # all and then an empty history. Say what happened, then retry.
            last = f"HTTP {resp.status_code}"
        except (AgentSaidNo, AgentUnreachable):
            raise
        except Exception as e:
            # `str(httpx.ReadTimeout())` is the EMPTY STRING, so the old
            # "failed: %s" wrote "failed: " and the production trail for this
            # incident could not name its own cause. `repr` always says which
            # exception it was.
            last = repr(e)
        if attempt < _PROXY_ATTEMPTS:
            logger.info(
                "Day-chats proxy %s attempt %d/%d failed (%s) — retrying",
                url, attempt, _PROXY_ATTEMPTS, last,
            )
            await asyncio.sleep(_PROXY_BACKOFF_S)
    logger.warning("Day-chats proxy %s failed after %d attempts: %s",
                   url, _PROXY_ATTEMPTS, last)
    raise AgentUnreachable(last)


def _unreachable(exc: "AgentUnreachable") -> HTTPException:
    """The one shape every day-chats route returns when the tenant is silent.

    503 + `Retry-After`, never 200 and never 404: the client has to be able to
    tell "you have no history" from "we could not read your history", because
    it caches the first one and retries the second.
    """
    # The cause reaches the trail here and NOT the client: "ReadTimeout()" is
    # what an operator needs and nothing a user can act on.
    reason = getattr(exc, "reason", "agent_unreachable")
    logger.warning("[day_chats] answering 503 %s: %s",
                   reason, getattr(exc, "detail", exc))
    return HTTPException(
        status_code=503,
        detail="Your agent did not answer in time. Your history is safe — try again.",
        headers={"Retry-After": "2", "X-Toup-Reason": reason},
    )


def _agent_4xx(e: "AgentSaidNo") -> HTTPException:
    """Turn a considered tenant 4xx into the client response.

    A 401/403 FROM THE TENANT AGENT means the PLATFORM's own `X-Agent-Key` is
    stale — the agent was just rebound and the platform copy has not caught up
    (it self-heals in ~30 s). It does NOT mean the user's JWT is bad. But the
    mobile client signs the user OUT on any 401 (`api.ts` clears the token and
    `context.ts` tears the session down), so forwarding a tenant 401 verbatim
    logs the user out over a transient key blip — the D1 catastrophe of the
    2026-09-12 onboarding incident. Map 401/403 to 503 + `Retry-After` +
    `X-Toup-Reason: agent_key_stale` (the same shape `_unreachable` uses for
    5xx/transport) so the client RETRIES instead of signing out.

    Every OTHER 4xx is the agent's real answer — most importantly the 404
    `app-conversation/{id}` returns for "no conversation yet" — and is
    forwarded verbatim. A genuine PLATFORM-auth 401 never reaches here: it is
    raised by `get_current_user` before the tenant call and stays a 401.
    """
    if e.status in (401, 403):
        logger.warning(
            "[day_chats] tenant answered %s — platform agent key stale; "
            "answering 503 agent_key_stale (forwarding would sign the user out)",
            e.status,
        )
        return HTTPException(
            status_code=503,
            detail="Your agent is still starting up. Please try again in a moment.",
            headers={"Retry-After": "3", "X-Toup-Reason": "agent_key_stale"},
        )
    return HTTPException(status_code=e.status, detail=e.body or "Agent declined")


# ── Day-chat self-heal plumbing ──────────────────────────────────────
#
# The heal is a SERVICE call in a SEPARATE session (see `list_day_chats`).
# These three helpers are everything the endpoint itself needs to decide
# whether to make that call.

_HEAL_COOLDOWN_S = 60.0
#: user_id -> monotonic timestamp of the last heal ATTEMPT. Advisory only —
#: it is per-process and there are two platform replicas — but it is what
#: stops a client retrying a 503 three times from running the repair six
#: times for one user action. Real safety is that every write in the
#: rebucket is idempotent.
_HEAL_COOLDOWN: "dict[str, float]" = {}
_HEAL_COOLDOWN_MAX = 4096


def _rebucket_enabled() -> bool:
    """Flag read through `getattr` so an older Settings (or a platform
    process that has not picked up the new field) simply behaves as ON."""
    from app.config import settings as _settings

    return bool(getattr(_settings, "day_chat_rebucket_enabled", True))


def _local_today(tz_name: Optional[str]) -> Optional[Date]:
    """Today in the user's zone, or None when the zone will not load."""
    if not tz_name:
        return None
    try:
        import zoneinfo

        return datetime.now(zoneinfo.ZoneInfo(tz_name)).date()
    except Exception:
        return None


def _heal_cooldown_ok(user_id: str) -> bool:
    """True at most once per `_HEAL_COOLDOWN_S` per user, and it CLAIMS the
    window before returning True — the caller is about to await, and two
    concurrent list calls would otherwise both pass."""
    import time as _time

    now = _time.monotonic()
    last = _HEAL_COOLDOWN.get(user_id, 0.0)
    if now - last < _HEAL_COOLDOWN_S:
        return False
    if len(_HEAL_COOLDOWN) > _HEAL_COOLDOWN_MAX:
        _HEAL_COOLDOWN.clear()
    _HEAL_COOLDOWN[user_id] = now
    return True


#: Upper bound on rows the gauge will inspect. A tenant's day_chats table
#: holds one row per user per day; the SQL predicate below already reduces
#: it to today-or-later, so the cap is a backstop, not a sampling choice.
_FUTURE_GAUGE_LIMIT = 1000


async def _count_future_dated_day_chats() -> int:
    """Gauge body for `health_signals.future_dated_day_chats` — the tenant's
    CURRENT count of rows that cannot exist (zero again once healed).

    Bounded in SQL: a user's local today is within one calendar day of the
    UTC date, so any local_date that is in a user's future is >= the UTC
    date — the predicate cuts the scan to at most a couple of days' rows
    before the exact per-zone test runs in Python. Registered only where a
    tenant table lives (see the guard below): on the platform DB this table
    is a monolith leftover shared across every user and nobody reads the
    platform's copy of the counter.
    """
    from datetime import datetime as _dt, timezone as _tz
    from app.db.database import async_session_maker as _sm

    utc_today = _dt.now(_tz.utc).date()
    async with _sm() as _db:
        rows = (await _db.execute(
            select(DayChat.local_date, User.timezone)
            .join(User, DayChat.user_id == User.id)
            .where(DayChat.local_date >= utc_today)
            .limit(_FUTURE_GAUGE_LIMIT)
        )).all()
    n = 0
    for local_date, tz_name in rows:
        today = _local_today(tz_name)
        if today is not None and local_date and local_date > today:
            n += 1
    return n


def _gauge_process() -> bool:
    """The gauge measures a TENANT table; the platform process only proxies."""
    try:
        from app.config import settings as _settings
        return getattr(_settings, "run_mode", "monolith") != "platform"
    except Exception:  # pragma: no cover
        return True


try:  # pragma: no cover - registration only; the measurement is lazy
    from app.services import health_signals as _health_signals

    if _gauge_process():
        _health_signals.register_gauge(
            "future_dated_day_chats", _count_future_dated_day_chats, ttl_s=60.0,
        )
except Exception:  # pragma: no cover
    pass


@router.get("")
async def list_day_chats(
    before: Optional[str] = Query(None, description="Cursor: ISO date (YYYY-MM-DD), return days before this"),
    limit: int = Query(30, ge=1, le=90, description="Max day chats to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's day chats, newest first, cursor-paginated.

    Pagination: pass ?before=2026-04-01&limit=30 to get the 30 days before April 1.
    Default: most recent 30 days. Max limit: 90.

    Works in read-only mode regardless of feature flag state. Returns empty list
    if no day_chats exist (backfill hasn't run yet).
    """
    # Platform mode: proxy to user's VPS agent. The tenant is AUTHORITATIVE for
    # a user who has one — there is no second copy of their history down here,
    # so a failed hop must never continue into the local SELECT below.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        params = {"limit": limit}
        if before:
            params["before"] = before
        try:
            data = await _proxy_day_chats(proxy[0], proxy[1], "", params)
        except AgentSaidNo as e:
            raise _agent_4xx(e)
        except AgentUnreachable as e:
            raise _unreachable(e)
        return JSONResponse(content=data)

    query = (
        select(DayChat)
        .where(DayChat.user_id == current_user.id)
        .order_by(DayChat.local_date.desc())
        .limit(limit)
    )

    if before:
        try:
            before_date = Date.fromisoformat(before)
            query = query.where(DayChat.local_date < before_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'before' date. Use YYYY-MM-DD.")

    try:
        result = await db.execute(query)
        day_chats = result.scalars().all()
    except _MISSING_TABLE_ERRORS as exc:
        # Platform DB has no `day_chats` (AGENT_ONLY table). User
        # without an active proxy → no day chats yet. Return empty
        # so the chat shell can render its empty state instead of
        # surfacing a 500 in devtools.
        await db.rollback()
        logger.info("[day_chats] local SELECT skipped (table absent in this DB): %s", str(exc)[:120])
        return JSONResponse(content=[])

    # ── Self-heal: a future-dated day is a MIS-BUCKETED day ──
    #
    # The repair lives in `app.services.day_chat_rebucket` and runs in its
    # OWN session. The previous inline version re-pointed messages and
    # conversations and then deleted the source day with a
    # `context_budget_logs` row still referencing it; the ForeignKeyViolation
    # was caught without a rollback, so THIS request's session stayed in
    # PendingRollback and the items loop below 500'd. That is how a failed
    # heal became a dead day index for 3 hours (2026-09-14).
    #
    # `before is None` because a future day can only appear on page one, and
    # the old loop inspected whatever happened to be on the current page.
    user_tz = getattr(current_user, 'timezone', None)
    # Two signals, two meanings. The GAUGE `future_dated_day_chats` is the
    # tenant's current state: re-measured at most once a minute here and on
    # `/agent/health`, and forced after a heal that changed something so a
    # repaired tenant does not page for an hour. The COUNTER
    # `future_dated_day_chats_seen` is raised below by the number of
    # impossible rows this user had BEFORE the heal — that is the fact that
    # a tenant produced such a row, which a working heal would otherwise
    # erase from every reading.
    try:
        from app.services import health_signals as _hs

        await _hs.refresh_gauges(only="future_dated_day_chats")
    except Exception:
        pass

    if before is None and user_tz and _rebucket_enabled():
        local_today = _local_today(user_tz)
        newest = (await db.execute(
            select(func.max(DayChat.local_date)).where(DayChat.user_id == current_user.id)
        )).scalar()
        if local_today is not None and newest is not None and newest > local_today:
            try:
                from app.services import health_signals as _hs_seen
                _n_future = (await db.execute(
                    select(func.count()).select_from(DayChat).where(
                        DayChat.user_id == current_user.id,
                        DayChat.local_date > local_today,
                    )
                )).scalar() or 0
                _hs_seen.incr("future_dated_day_chats_seen", int(_n_future))
            except Exception:
                pass
        if (
            local_today is not None
            and newest is not None
            and newest > local_today
            and _heal_cooldown_ok(current_user.id)
        ):
            try:
                from app.db.database import async_session_maker as _heal_sm
                from app.services.day_chat_rebucket import rebucket_user_days
                async with _heal_sm() as _hdb:
                    res = await rebucket_user_days(
                        _hdb, current_user.id, user_tz, scope='future',
                    )
                logger.warning(
                    "[DAY-CHATS-HEAL] outcome=%s user=%s msgs=%d convs=%d cbls=%d "
                    "created=%d deleted=%d retained=%d errors=%s",
                    "rebucketed" if res.changed else "noop",
                    current_user.id[:8], res.moved_messages, res.moved_conversations,
                    res.moved_cbls, len(res.days_created), len(res.days_deleted),
                    len(res.days_retained), res.per_date_errors or "-",
                )
                if res.changed:
                    # The rows this request already read are stale now — the
                    # repair happened on a different session.
                    db.expire_all()
                    day_chats = (await db.execute(query)).scalars().all()
                    try:
                        from app.services import health_signals as _hs_after
                        await _hs_after.refresh_gauges(
                            only="future_dated_day_chats", force=True,
                        )
                    except Exception:
                        pass
            except Exception as _heal_err:
                # The request session was never handed to the rebucket, so it
                # is still usable: log the exception CLASS and serve the day
                # index unhealed rather than 500'ing it.
                logger.warning(
                    "[DAY-CHATS-HEAL] outcome=failed user=%s reason=%s: %s",
                    current_user.id[:8], type(_heal_err).__name__, _heal_err,
                )


    # TODO: collapse to single GROUP BY query — currently N+1 (one channel query
    # per day chat). For 90 days that's 91 queries. Fix when telemetry shows it matters.
    items = []
    for dc in day_chats:
        # Get distinct channels from MESSAGES (not Conversations) — because Telegram
        # sessions are long-lived and their Conversation.day_chat_id may point to a
        # different day. Message.day_chat_id is canonical for day membership.
        # `.distinct()` on the SELECT, not `distinct(<expr>)` around a function:
        # the ORM cannot locate `DISTINCT coalesce(...)` as a result column on
        # Postgres and answers NoSuchColumnError — which on this route is a
        # 500 for every tenant. Caught by the Postgres lane, invisible on SQLite.
        ch_result = await db.execute(
            select(func.coalesce(Message.channel, Conversation.channel).label("ch"))
            .select_from(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.day_chat_id == dc.id)
            .distinct()
        )
        channels = sorted({r[0] for r in ch_result.all() if r[0]})

        # Count from messages table (canonical), not from DayChat.message_count (cached, can drift)
        msg_count_result = await db.execute(
            select(func.count()).select_from(Message).where(Message.day_chat_id == dc.id)
        )
        msg_count = msg_count_result.scalar() or 0

        items.append({
            "id": dc.id,
            "local_date": dc.local_date.isoformat(),
            "message_count": msg_count,
            "channels_active": channels,
            "last_message_at": dc.last_message_at.isoformat() if dc.last_message_at else None,
            "summary_status": dc.summary_status or "up_to_date",
        })

    # TODO: add next_cursor field so frontend doesn't have to derive it from
    # the last item's local_date. Works but ugly.
    return JSONResponse(content=items)


@router.get("/{date_str}/messages")
async def get_day_chat_messages(
    date_str: str,
    limit: int = Query(500, ge=1, le=2000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages for a given local date, across all channels.

    Messages are in strict chronological order. Each message includes its
    channel and conversation_id so the frontend can show session dividers
    and channel badges.

    Content is RAW — no [channel time] annotations. Annotations are LLM-only
    and never exposed to the frontend or persisted.

    Returns 404 if no day chat exists for this date.
    Returns empty array if the day chat exists but has no messages.

    Works in read-only mode regardless of feature flag state.
    """
    # Platform mode: proxy to user's VPS agent. See `list_day_chats` — a
    # tenant that did not answer is a 503, never an empty day.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        try:
            data = await _proxy_day_chats(
                proxy[0], proxy[1], f"{date_str}/messages", {"limit": limit},
            )
        except AgentSaidNo as e:
            raise _agent_4xx(e)
        except AgentUnreachable as e:
            raise _unreachable(e)
        return JSONResponse(content=data)

    try:
        target_date = Date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Look up day chat by (user_id, local_date). When this router is
    # mounted in platform_main, the platform DB lacks the agent-only
    # `day_chats` table — return empty rather than 500'ing the client.
    try:
        dc = (await db.execute(
            select(DayChat).where(
                and_(DayChat.user_id == current_user.id, DayChat.local_date == target_date)
            )
        )).scalar_one_or_none()
    except _MISSING_TABLE_ERRORS as exc:
        await db.rollback()
        logger.info(
            "[day_chats] messages SELECT skipped (table absent in this DB): %s",
            str(exc)[:120],
        )
        return JSONResponse(content=[])

    if not dc:
        # Fall back to date-range scan (for when backfill hasn't run yet)
        # This mirrors the existing /sessions/by-date/{date}/messages logic
        from datetime import timezone as tz
        user_tz_name = current_user.timezone if hasattr(current_user, 'timezone') else None
        if user_tz_name:
            try:
                import zoneinfo
                user_tz = zoneinfo.ZoneInfo(user_tz_name)
                # Midnight in user's timezone, converted to UTC
                local_midnight = datetime(target_date.year, target_date.month, target_date.day, tzinfo=user_tz)
                day_start = local_midnight.astimezone(tz.utc).replace(tzinfo=None)
            except (KeyError, Exception):
                day_start = datetime(target_date.year, target_date.month, target_date.day)
        else:
            day_start = datetime(target_date.year, target_date.month, target_date.day)

        day_end = day_start + timedelta(days=1)

        try:
            sessions_result = await db.execute(
                select(Conversation.id, Conversation.channel)
                .where(
                    and_(
                        Conversation.user_id == current_user.id,
                        Conversation.started_at >= day_start,
                        Conversation.started_at < day_end,
                        # Autopilot tick turns were persisted raw
                        # (AUTOPILOT_* marker blocks) before 2026-07-16;
                        # ticks are headless now and terminal messages
                        # arrive as channel='routine'. Hide the
                        # historical noise without a data migration.
                        Conversation.channel.notin_(HIDDEN_DAY_CHANNELS),
                    )
                )
            )
        except _MISSING_TABLE_ERRORS as exc:
            await db.rollback()
            logger.info(
                "[day_chats] messages fallback skipped (conversations table absent): %s",
                str(exc)[:120],
            )
            return JSONResponse(content=[])
        session_rows = sessions_result.all()
        if not session_rows:
            # No conversations for this day. Return empty list rather
            # than 404 — the chat shell uses these endpoints to render
            # past-day dividers; an empty day is a valid state, not a
            # client error. (Pre-fix: 404 surfaced as a red devtools
            # entry on every day-chat poll for new users.)
            return JSONResponse(content=[])

        session_ids = [r[0] for r in session_rows]
        channel_map = {r[0]: r[1] for r in session_rows}

        msgs_result = await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(session_ids))
            # (created_at, id) — the tiebreaker `day_context_loader` already
            # has. Two rows sharing a timestamp (a presave user row and a
            # same-millisecond assistant row, or two channels' rows) came back
            # in a different order on two fetches, and in a different order
            # than the model saw.
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
        )
        messages = msgs_result.scalars().all()

        # Bulk-resolve reply targets so each replying row can render its
        # quoted card on first paint instead of flashing the "(message
        # not in current view)" stub while older days hydrate.
        from app.agent.reply_quote import resolve_reply_targets_for_serialization
        reply_targets = await resolve_reply_targets_for_serialization(db, messages)

        build_jobs = await load_build_jobs(db, messages)
        return JSONResponse(content=attach_run_to_cards([
            {
                "id": m.id,
                "role": m.role,
                "content": public_text(m.role, m.content),
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "channel": _row_channel(m, channel_map.get(m.conversation_id)),
                "conversation_id": m.conversation_id,
                "attachments": _serialize_attachments(m),
                "media": _serialize_media(m),
                "app_artifact": _serialize_app_artifact(m),
                "admin_notice": _serialize_admin_notice(m),
                "automation_connector_card": _serialize_automation_card(
                    m, "automation_connector_card"),
                "automation_grant_card": _serialize_automation_card(
                    m, "automation_grant_card"),
                "automation_notification": _serialize_automation_card(
                    m, "automation_notification"),
                "pending_action": _serialize_meta_card(m, "pending_action"),
                "draft_card": _serialize_meta_card(m, "draft_card"),
                "memory_update": _serialize_meta_card(m, "memory_update"),
                "fix_chip": _serialize_meta_card(m, "fix_chip"),
                "tool_events": _serialize_tool_events(m),
                "reply_to_message_id": getattr(m, "reply_to_message_id", None),
                "reply_to": reply_targets.get(m.id),
                **job_card_fields(m, build_jobs),
            }
            for m in messages
        ]))

    # Day chat exists — load messages via day_chat_id (fast path).
    # Wrapped: if `day_chats` lives on platform DB (legacy artifact) but
    # `messages`/`conversations` don't, this query would 500. Guard
    # mirrors the missing-table handling above.
    try:
        msgs_result = await db.execute(
            select(Message, Conversation.channel)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                or_(
                    Message.day_chat_id == dc.id,
                    # Admin-dispatch notices are the ONE row type whose
                    # Message.day_chat_id is deliberately NULL: the agent's
                    # context loader selects WHERE Message.day_chat_id = :id,
                    # and an operator's message must never enter the agent's
                    # history (see app/agent/admin_message_writer.py). The
                    # CONVERSATION still carries the day, so the day is read
                    # off it here. Without this arm the notice is written,
                    # broadcast, rendered live — and then gone on reload,
                    # because this fast path is what every client fetches.
                    # The three other readers (this route's no-DayChat
                    # fallback, /sessions/by-date, /messages/since) all scan
                    # by Conversation and already include it.
                    and_(
                        Conversation.channel == "admin",
                        Conversation.day_chat_id == dc.id,
                    ),
                ),
                # Hide historical raw autopilot tick rows (see the
                # fallback path above); mission outcomes arrive as
                # channel='routine' and still render.
                # CONTRACTS-R31 §4.1: an automation's conversation belongs
                # to that automation's thread. The R28 session path wrote
                # those turns here with a real day_chat_id, which is how
                # a thread question, its answer and its memory chip all
                # appeared in the main chat on 26 August. The one
                # sanctioned automation row in the day is the
                # notification card, and that is written on
                # channel='routine', not this one.
                Conversation.channel.notin_(HIDDEN_DAY_CHANNELS),
            )
            # (created_at, id) — the tiebreaker `day_context_loader` already
            # has. Two rows sharing a timestamp (a presave user row and a
            # same-millisecond assistant row, or two channels' rows) came back
            # in a different order on two fetches, and in a different order
            # than the model saw.
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
        )
        rows = msgs_result.all()
    except _MISSING_TABLE_ERRORS:
        await db.rollback()
        return JSONResponse(content=[])

    # Bulk-resolve reply targets so each replying row can render its
    # quoted card on first paint instead of flashing the "(message not
    # in current view)" stub while older days hydrate.
    from app.agent.reply_quote import resolve_reply_targets_for_serialization
    reply_targets = await resolve_reply_targets_for_serialization(
        db, [msg for msg, _ in rows]
    )

    # THE primary history fetch — every client asks this route first and only
    # falls back to /api/sessions when it fails. It carried no job-card
    # projection at all, so a `role='job'` row arrived with its raw marker in
    # `content` and no card fields beside it: the Round 16 P0. See
    # api/message_cards.py.
    build_jobs = await load_build_jobs(db, [msg for msg, _ in rows])
    return JSONResponse(content=attach_run_to_cards([
        {
            "id": msg.id,
            "role": msg.role,
            "content": public_text(msg.role, msg.content),
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "channel": _row_channel(msg, channel),
            "conversation_id": msg.conversation_id,
            "attachments": _serialize_attachments(msg),
            "media": _serialize_media(msg),
            "app_artifact": _serialize_app_artifact(msg),
            "admin_notice": _serialize_admin_notice(msg),
            "automation_connector_card": _serialize_automation_card(
                msg, "automation_connector_card"),
            "automation_grant_card": _serialize_automation_card(
                msg, "automation_grant_card"),
            "automation_notification": _serialize_automation_card(
                msg, "automation_notification"),
            "pending_action": _serialize_meta_card(msg, "pending_action"),
            "draft_card": _serialize_meta_card(msg, "draft_card"),
            "memory_update": _serialize_meta_card(msg, "memory_update"),
            "fix_chip": _serialize_meta_card(msg, "fix_chip"),
            "tool_events": _serialize_tool_events(msg),
            "reply_to_message_id": getattr(msg, "reply_to_message_id", None),
            "reply_to": reply_targets.get(msg.id),
            **job_card_fields(msg, build_jobs),
        }
        for msg, channel in rows
    ]))


@router.get("/app-conversation/{app_id}")
async def resolve_app_conversation(
    app_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resolve the existing Conversation for an app in today's day chat.

    Returns the conversation_id and message count if found, 404 if no
    conversation exists yet (user hasn't chatted with this app today).

    Used by the frontend to restore app conversation state on re-open.
    """
    import json as _json

    # Platform mode: proxy to user's VPS agent. Same rule as the two routes
    # above: silence upstream is a 503, not "this app has no conversation" —
    # the client uses a 404 here to decide it may start a NEW one, and doing
    # that on a timeout orphans the thread the user was already in.
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        try:
            data = await _proxy_day_chats(proxy[0], proxy[1], f"app-conversation/{app_id}")
        except AgentSaidNo as e:
            raise _agent_4xx(e)
        except AgentUnreachable as e:
            raise _unreachable(e)
        return JSONResponse(content=data)

    # Find today's day chat
    user_tz = getattr(current_user, 'timezone', None)
    if user_tz:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(user_tz)
            local_today = datetime.now(tz).date()
        except (KeyError, Exception):
            local_today = datetime.utcnow().date()
    else:
        local_today = datetime.utcnow().date()

    dc = (await db.execute(
        select(DayChat).where(
            and_(DayChat.user_id == current_user.id, DayChat.local_date == local_today)
        )
    )).scalar_one_or_none()

    if not dc:
        raise HTTPException(status_code=404, detail="No day chat for today")

    # Find app conversation in today's day chat
    candidates = (await db.execute(
        select(Conversation).where(and_(
            Conversation.user_id == current_user.id,
            Conversation.day_chat_id == dc.id,
            Conversation.channel == "app",
        ))
    )).scalars().all()

    for conv in candidates:
        try:
            meta = _json.loads(conv.metadata_json or "{}")
            if meta.get("app_id") == app_id:
                msg_count = (await db.execute(
                    select(func.count()).select_from(Message)
                    .where(Message.conversation_id == conv.id)
                )).scalar() or 0
                return JSONResponse(content={
                    "conversation_id": conv.id,
                    "message_count": msg_count,
                })
        except (ValueError, TypeError):
            continue

    raise HTTPException(status_code=404, detail="No app conversation found for today")
