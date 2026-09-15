"""Shared inbound-message handler for BaseChannel adapters.

The Toup channel framework defines an `on_message` callback on every
`BaseChannel` subclass, but until this module landed nothing wired that
callback. The result was that Discord, Slack and WhatsApp adapters
received messages, normalised them into `InboundMessage` objects, called
`BaseChannel.dispatch()` — and then `dispatch()` logged a warning and
dropped the message because `on_message` was `None`.

Telegram avoided the bug because `ToupTelegramBot` bypasses
`BaseChannel` entirely and drives the agent runtime directly.

`make_channel_handler` builds the missing bridge: a `MessageCallback`
that takes a normalised inbound, runs the user message through the
shared `AgentRunner`, and ships the final reply back via the same
channel's `send_text`. One handler per channel adapter; it's safe to
register the same handler shape for Discord, Slack and WhatsApp.

Design notes
------------
* **Per-(channel, chat_id) session cache.** Mirrors the cache pattern in
  `ToupTelegramBot._session_map` so successive turns from the same chat
  reuse the same `Conversation` row instead of creating a new session
  per message. Falls back to a DB lookup on cold cache.
* **Final replies only.** No streaming callbacks are passed to
  `AgentRunner.run`, so non-streaming surfaces (Discord, Slack,
  WhatsApp) only ever see `AgentResponse.text` — the finalised
  user-facing reply. Reasoning / tool / streaming chunks never reach
  the user.
* **Channel-agnostic outbound.** Per-channel chunking, formatting and
  retry policy live in the channel adapters' `send_text` (e.g.
  `DiscordChannel._split_message`, `WhatsAppChannel.send_text`'s
  markdown→WA + chunking + retry pipeline). This handler intentionally
  does not touch the reply text.
* **Error surfaces.** Any exception during the agent run is logged
  with full traceback but the user sees a generic apology — no Python
  internals are leaked.
* **Phone-number redaction.** WhatsApp `chat_id`s are E.164 phone
  numbers; logs only ever contain a redacted form.

Day-as-Chat invariant
---------------------
The session cache is per-channel by design, but the agent's *context*
is per-day across every channel. ``AgentRunner.run`` resolves
``Message.day_chat_id`` from the user's local date via
``app/db/message_helpers.py::resolve_day_chat_id_for_now``, which
falls back to ``User.timezone`` when no ``client_tz`` is supplied —
which is the situation here, since a Cloud-API webhook carries no
client timezone. Net effect: a user can DM the agent on WhatsApp at
3 PM and the agent will see (and reason over) their morning web chat
and noon Telegram messages because all three turns share one
``day_chat_id``. We deliberately do **not** thread ``client_tz`` from
this handler — the runner's ``_resolve_effective_tz`` (User.timezone,
then the bind payload, then the owner's phone) is the canonical path
for channels that have no per-message timezone signal, and the session
lookup below asks THAT resolver for the day rather than bucketing on
its own: a lookup that resolved the day in UTC while the runner
resolved it in the effective zone created an empty future-dated day
and minted a fresh Conversation on every message (review R2, 2026-09-14).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Optional, Tuple

from sqlalchemy import and_, select

from app.agent.agent_runner import AgentRunner
from app.agent.channels.base import (
    BaseChannel,
    ChannelType,
    InboundMessage,
    MessageCallback,
)

logger = logging.getLogger(__name__)


# Generic, user-safe failure surface. Never include exception text — it
# can leak stack traces, internal IDs, provider error messages, etc.
_GENERIC_ERROR_TEXT = (
    "⚠️ Sorry, something went wrong on my side. Please try again in a moment."
)


def _signal(name: str) -> None:
    """Best-effort process-lifetime counter; absent on older images."""
    try:
        from app.services.health_signals import incr as _incr
        _incr(name)
    except Exception:
        pass


def _redact_chat_id(channel_type: ChannelType, chat_id: str) -> str:
    """Mask phone numbers in WhatsApp logs; pass other IDs through.

    WhatsApp `chat_id`s are E.164 phone numbers and must not be logged
    in clear (privacy + GDPR). Discord/Slack/web IDs are opaque and
    safe to log.
    """
    if channel_type is not ChannelType.WHATSAPP or not chat_id:
        return chat_id
    if len(chat_id) <= 4:
        return "***"
    return f"{chat_id[:2]}***{chat_id[-2:]}"


# Resolves the effective timezone for a lookup, given the lookup's own
# DB session. Built by `make_channel_handler` over the runner's
# `_resolve_effective_tz` so the session lookup and the turn agree on the
# user's local date. `None` falls back to the tenant's `User.timezone`.
TzResolver = Callable[[object], Awaitable[Optional[str]]]

# Distinguishes "the day could not be resolved" (degraded: keep whatever
# the cache holds) from "no row exists for today yet" (a new local day:
# the cache entry is stale by definition).
_DAY_UNRESOLVED = object()


async def _tz_for_lookup(db, user_id: str, tz_resolver: Optional[TzResolver]) -> Optional[str]:
    if tz_resolver is not None:
        try:
            return await tz_resolver(db)
        except Exception:  # noqa: BLE001 — degrade to the row, never fail the lookup
            logger.debug("channel.tz_resolver_failed", exc_info=True)
    from app.db.models import User
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    return getattr(user, "timezone", None) if user else None


async def _todays_day_chat_id(db, user_id: str, tz_name: Optional[str]):
    """READ-ONLY: the id of the user's CURRENT local day, `None` when no
    such row exists yet, `_DAY_UNRESOLVED` when the question could not be
    answered. Never creates a day — `get_or_create_day_chat` is the
    runner's to call, with the zone the turn is bucketed in; a lookup that
    created the row here on a different zone left an EMPTY, future-dated
    DayChat behind on every inbound message."""
    try:
        from app.agent._day_chat_cache import get_cached_day_chat_id
        from app.agent.day_chat_resolver import resolve_local_date
        from app.db.models.day_chat import DayChat

        local_date, _ = resolve_local_date(datetime.now(timezone.utc), tz_name)
        cached = get_cached_day_chat_id(user_id, local_date)
        if cached:
            return cached
        return (await db.execute(
            select(DayChat.id).where(
                and_(DayChat.user_id == user_id, DayChat.local_date == local_date)
            )
        )).scalar_one_or_none()
    except Exception:  # noqa: BLE001
        logger.warning("channel.day_lookup_failed user=%s", (user_id or "")[:8], exc_info=True)
        return _DAY_UNRESOLVED


async def _resolve_session_id(
    user_id: str,
    channel_type: ChannelType,
    chat_id: str,
    cache: Dict[Tuple[str, str], Tuple[str, str]],
    tz_resolver: Optional[TzResolver] = None,
) -> Optional[str]:
    """Look up the active `Conversation.id` for this (channel, chat), TODAY.

    The old query filtered on `(user_id, channel, is_active)` alone. Two
    filters were missing and both of them mattered:

    * **today's day chat** — it took the newest active row for the channel
      whatever day it belonged to, so a long-lived session dragged
      yesterday's conversation into today.
    * **the chat id** — the cache key has named the chat since this was
      written, but the DB query never did, so a SECOND allowlisted
      WhatsApp contact resolved onto the FIRST contact's Conversation and
      their messages landed in a stranger's thread.

    The cache now holds `(day_chat_id, conversation_id)` and a day mismatch
    is a miss: a turn that crosses local midnight must not reuse yesterday's
    row from memory.

    Returns `None` when nothing matches; `AgentRunner.run` then creates a
    fresh `Conversation` (and today's DayChat, in the zone the turn is
    bucketed in) and the caller caches the id from the response.
    """
    key = (channel_type.value, chat_id)

    try:
        from app.db.database import async_session_maker
        from app.db.models import Conversation

        async with async_session_maker() as db:
            tz_name = await _tz_for_lookup(db, user_id, tz_resolver)
            today_dc = await _todays_day_chat_id(db, user_id, tz_name)
            cached = cache.get(key)
            if cached:
                cached_day, cached_conv = cached
                # An UNRESOLVABLE day is not a mismatch. When day
                # resolution is degraded the cached row is still the best
                # answer available, and dropping it would mint a fresh
                # Conversation on every single turn. A day that simply
                # does not EXIST yet is the opposite case: the cached
                # conversation belongs to a day that does, so it is
                # yesterday's.
                # An entry cached with an UNKNOWN day ("") is usable until
                # a real day resolves (see `_cache_session`).
                if (
                    today_dc is _DAY_UNRESOLVED
                    or (today_dc and cached_day == today_dc)
                    or (not today_dc and not cached_day)
                ):
                    return cached_conv
                cache.pop(key, None)
            if today_dc is _DAY_UNRESOLVED or not today_dc:
                return None

            row = (
                await db.execute(
                    select(Conversation)
                    .where(
                        and_(
                            Conversation.user_id == user_id,
                            Conversation.channel == channel_type.value,
                            Conversation.day_chat_id == today_dc,
                            Conversation.is_active.is_(True),
                            Conversation.metadata_json.contains(chat_id),
                        )
                    )
                    .order_by(Conversation.updated_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row:
                cache[key] = (today_dc, row.id)
                return row.id
    except Exception as exc:  # defensive — cold-cache lookup shouldn't fail the turn
        logger.warning(
            "channel.session_lookup_failed channel=%s err=%s",
            channel_type.value,
            exc,
        )
    return None


async def _cache_session(
    user_id: str,
    channel_type: ChannelType,
    chat_id: str,
    cache: Dict[Tuple[str, str], Tuple[str, str]],
    session_id: str,
    day_chat_id: Optional[str],
    tz_resolver: Optional[TzResolver] = None,
) -> None:
    """Record `(day_chat_id, conversation_id)` for the next turn.

    A session cached without the day it belongs to is the bug this
    replaces — the entry has to expire at local midnight, and only the
    day id says when that is.
    """
    if not session_id:
        return
    if not day_chat_id:
        # The runner has just persisted the turn, so today's row exists;
        # this is a read of it, never a create (see `_todays_day_chat_id`).
        try:
            from app.db.database import async_session_maker

            async with async_session_maker() as db:
                tz_name = await _tz_for_lookup(db, user_id, tz_resolver)
                found = await _todays_day_chat_id(db, user_id, tz_name)
                day_chat_id = None if found is _DAY_UNRESOLVED else found
        except Exception:
            day_chat_id = None
    # "" records an entry whose day is unknown: usable until a real day
    # resolves, at which point it reads as a mismatch and re-resolves.
    cache[(channel_type.value, chat_id)] = (day_chat_id or "", session_id)


def make_channel_handler(
    *,
    channel: BaseChannel,
    agent_runner: AgentRunner,
    user_id: str,
) -> MessageCallback:
    """Build the inbound `MessageCallback` for a channel adapter.

    The returned coroutine should be passed to
    `channel.set_message_callback()` immediately after the channel is
    started and registered. Each channel adapter gets its own handler
    instance with its own session cache.

    Args:
        channel:       The channel adapter (Discord/Slack/WhatsApp).
        agent_runner:  The shared `AgentRunner` instance.
        user_id:       The owner user id for this agent container. In
                       Toup's per-tenant container model every container
                       is single-user, so `user_id` is fixed for the
                       process lifetime.

    Returns:
        An `async def(msg: InboundMessage) -> None` coroutine suitable
        for `BaseChannel.set_message_callback`.
    """
    channel_type = channel.channel_type
    session_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
    # One turn at a time per (channel, chat). `agent_runner
    # ._get_or_create_session` is a blind INSERT for whatsapp/discord/slack
    # (they are deliberately outside INDEXED_SYSTEM_CHANNELS, so no partial
    # unique index rejects a second row), and two concurrent cold-cache
    # dispatches for one chat therefore each create a Conversation for the
    # same day. Process-local by design: one tenant is one container. It is
    # NOT durable serialization and must not be read as a substitute for a
    # DB constraint the day a tenant runs two replicas.
    session_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
    # Strong references for the fire-and-forget echo tasks below: the loop
    # holds only a weak ref to a task, so an un-referenced one can be
    # collected mid-await and the app/web sockets never see the turn.
    echo_tasks: set = set()

    def _lock_for(key: Tuple[str, str]) -> asyncio.Lock:
        lock = session_locks.get(key)
        if lock is None:
            # Bounded: a Discord/Slack tenant sees an unbounded set of chat
            # ids over a process lifetime. Idle locks are safe to drop — a
            # dropped lock is simply re-minted on the next turn.
            if len(session_locks) >= 512:
                for k in [k for k, l in session_locks.items() if not l.locked()][:256]:
                    session_locks.pop(k, None)
            lock = asyncio.Lock()
            session_locks[key] = lock
        return lock

    # The SAME zone the turn is bucketed in. `_resolve_effective_tz` is
    # User.timezone → bind payload → owner's phone (ephemeral); the phone
    # source in particular is never persisted, so a lookup reading the
    # users row alone disagreed with the runner for every phone-only
    # tenant. A stub runner without the method (tests) falls back to the
    # row.
    _tz_fn = getattr(agent_runner, "_resolve_effective_tz", None)
    tz_resolver: Optional[TzResolver] = None
    if _tz_fn is not None and inspect.iscoroutinefunction(_tz_fn):
        async def tz_resolver(db) -> Optional[str]:  # type: ignore[misc]
            return await _tz_fn(db, user_id, None, channel_type.value)

    async def handle(msg: InboundMessage) -> None:
        # 1. Skip non-content frames. Inbound debouncers / typing /
        # presence events sometimes surface here as empty messages —
        # there's nothing to run the agent on.
        if not (msg.text or msg.media_paths):
            return

        chat_id = msg.channel_chat_id
        chat_label = _redact_chat_id(channel_type, chat_id)

        # 2. Owner check. In the per-tenant container model `user_id`
        # is supposed to be set at boot from the container's .env. If
        # it is missing the agent itself is misconfigured; we should
        # neither persist nor reply.
        if not user_id:
            logger.error(
                "channel.dispatch_skipped reason=no_user_id channel=%s chat=%s",
                channel_type.value,
                chat_label,
            )
            return

        # 3-6 run under one per-chat lock: resolve, run, cache. A second
        # turn for this chat that resolved while the first was still
        # running would see the same cold cache and create a second
        # Conversation for the same day.
        async with _lock_for((channel_type.value, chat_id)):
            # 3. Resolve the session for this (channel, chat). On miss,
            # AgentRunner will create one inside its DB transaction.
            session_id = await _resolve_session_id(
                user_id, channel_type, chat_id, session_cache, tz_resolver,
            )

            # 4. Best-effort typing indicator. We never let a failing
            # indicator prevent the actual reply.
            try:
                await channel.send_typing(chat_id)
            except Exception:
                pass

            # 5. Run the agent. No streaming callbacks: non-streaming
            # surfaces should only ever receive the finalised reply text.
            try:
                response = await agent_runner.run(
                    user_message=msg.text or "",
                    user_id=user_id,
                    session_id=session_id,
                    channel=channel_type.value,
                    channel_chat_id=chat_id,
                    media_paths=list(msg.media_paths) if msg.media_paths else None,
                )
            except asyncio.CancelledError:
                logger.info(
                    "channel.run_cancelled channel=%s chat=%s",
                    channel_type.value,
                    chat_label,
                )
                raise
            except Exception:
                logger.exception(
                    "channel.run_failed channel=%s chat=%s",
                    channel_type.value,
                    chat_label,
                )
                try:
                    await channel.send_text(chat_id, _GENERIC_ERROR_TEXT)
                except Exception:
                    logger.exception(
                        "channel.error_send_failed channel=%s chat=%s",
                        channel_type.value,
                        chat_label,
                    )
                return

            # 6. Cache session for the next turn (covers both reuse and
            # the first-message-creates-session case).
            _persisted = getattr(response, "persisted", {}) or {}
            if response.session_id:
                await _cache_session(
                    user_id, channel_type, chat_id, session_cache,
                    response.session_id,
                    _persisted.get("day_chat_id") or getattr(response, "day_chat_id", None),
                    tz_resolver,
                )

        # Denominator of the channel-orphan alert (claimed − persisted).
        # `channel_events_claimed` is produced by the WhatsApp adapter
        # alone, so this side counts WhatsApp alone: a Telegram/Discord
        # turn here inflated the persisted side and hid real orphans.
        if _persisted.get("user_message_id") and channel_type is ChannelType.WHATSAPP:
            _signal("channel_events_persisted")

        # 6b. Publish the turn to this user's app / web sockets. Fire-and-
        # forget: live delivery must never sit between the agent's answer
        # and `channel.send_text` — the channel reply is the user's primary
        # surface.
        try:
            from app.agent.channel_echo import broadcast_channel_turn
            _echo = asyncio.create_task(broadcast_channel_turn(
                user_id,
                origin_channel=channel_type.value,
                session_id=response.session_id,
                persisted=_persisted,
                user_text=msg.text or "",
                assistant_text=response.text or "",
            ))
            echo_tasks.add(_echo)
            _echo.add_done_callback(echo_tasks.discard)
        except Exception:
            logger.warning(
                "channel.echo_failed channel=%s", channel_type.value, exc_info=True,
            )

        reply_text = (response.text or "").strip()
        if not reply_text:
            logger.info(
                "channel.empty_reply channel=%s chat=%s session=%s",
                channel_type.value,
                chat_label,
                (response.session_id or "")[:8],
            )
            return

        # 7. Deliver. Per-channel chunking / format conversion lives
        # inside each adapter's send_text — keep this file channel-
        # agnostic.
        try:
            await channel.send_text(chat_id, reply_text)
        except Exception:
            logger.exception(
                "channel.send_failed channel=%s chat=%s",
                channel_type.value,
                chat_label,
            )
            return

        logger.info(
            "channel.run_ok channel=%s chat=%s session=%s tokens=%d tools=%d ms=%d",
            channel_type.value,
            chat_label,
            (response.session_id or "")[:8],
            response.tokens_total,
            len(response.tool_calls),
            response.processing_time_ms,
        )

    return handle
