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
this handler — the ``User.timezone`` fallback is the canonical path
for channels that have no per-message timezone signal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Tuple

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


async def _resolve_session_id(
    user_id: str,
    channel_type: ChannelType,
    chat_id: str,
    cache: Dict[Tuple[str, str], str],
) -> Optional[str]:
    """Look up the active `Conversation.id` for this (channel, chat).

    Mirrors `ToupTelegramBot._get_session_id`. Returns `None` if no
    active session exists; in that case `AgentRunner.run` will create
    a fresh `Conversation` row and we cache the new id from the
    response.
    """
    key = (channel_type.value, chat_id)
    cached = cache.get(key)
    if cached:
        return cached

    try:
        from app.db.database import async_session_maker
        from app.db.models import Conversation

        async with async_session_maker() as db:
            row = (
                await db.execute(
                    select(Conversation)
                    .where(
                        and_(
                            Conversation.user_id == user_id,
                            Conversation.channel == channel_type.value,
                            Conversation.is_active.is_(True),
                        )
                    )
                    .order_by(Conversation.updated_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row:
                cache[key] = row.id
                return row.id
    except Exception as exc:  # defensive — cold-cache lookup shouldn't fail the turn
        logger.warning(
            "channel.session_lookup_failed channel=%s err=%s",
            channel_type.value,
            exc,
        )
    return None


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
    session_cache: Dict[Tuple[str, str], str] = {}

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

        # 3. Resolve the session for this (channel, chat). On miss,
        # AgentRunner will create one inside its DB transaction.
        session_id = await _resolve_session_id(
            user_id, channel_type, chat_id, session_cache
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
        if response.session_id:
            session_cache[(channel_type.value, chat_id)] = response.session_id

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
