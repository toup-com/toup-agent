"""Canonical resolver for `(user_id, day_chat_id, channel)` → Conversation.

Day-as-Chat invariant (Reading A, decided 2026-05-13): a `Conversation`
is one continuous UI thread on one channel inside a parent DayChat.
For **system-driven channels** (`routine`, `trigger`, `api`, `digest`),
that means **exactly one** Conversation per `(user_id, day_chat_id,
channel)` — every routine fire on the same day appends to the same
thread, never spawns a new row.

User-driven channels (`web`, `telegram`, `whatsapp`, `voice`,
`extension`, `app`) keep the existing semantics: one Conversation per
day per channel unless the user clicks "New Thread" (which sets
`force_new=True` upstream and routes around this resolver).

Before this module existed, `app.agent.routines.message_writer` and
`app.agent.triggers.message_writer` inserted a fresh Conversation per
fire. A user with one daily routine accumulated 30+ "sessions" in the
sidebar over a month — see `docs/bug-sweep-2026-05-13.md` Ticket 1.

The partial unique index installed alongside this module
(`ix_conversations_system_channel_per_day`) backs the read-then-insert
race: two concurrent fires that both miss the SELECT race to INSERT,
the loser hits an `IntegrityError`, retries the SELECT, and gets the
winner's row.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Mapping, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


# Channels with strict (user_id, day_chat_id, channel) uniqueness. Mirror
# the partial unique index predicate in `app.db.database.init_db`. Add a
# channel here only if it's system-driven (push-style, not a user UI
# thread) — see Reading A in the bug-sweep doc.
SYSTEM_CHANNELS: frozenset[str] = frozenset({"routine", "trigger", "api", "digest"})


async def resolve_or_create_day_conversation(
    db: AsyncSession,
    *,
    user_id: str,
    day_chat_id: Optional[str],
    channel: str,
    title: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> "Conversation":  # type: ignore[name-defined]
    """Return the canonical `Conversation` for this `(user, day, channel)`.

    For system channels: reuse the existing active Conversation for the
    day if one exists, otherwise insert. Concurrent inserts converge via
    the partial unique index — the loser catches `IntegrityError` and
    re-fetches.

    For non-system channels: behavior unchanged from pre-resolver code
    (always insert). Callers that need "reuse existing user session"
    semantics should resolve the session id themselves and not call this
    helper.

    `day_chat_id` may be None in degraded paths (resolver upstream
    couldn't determine a day). When None, we always create — there's no
    safe dedup key. We log loudly so the upstream omission is visible.
    """
    from app.db.models import Conversation

    is_system = channel in SYSTEM_CHANNELS
    meta_json = json.dumps(dict(metadata)) if metadata else None

    if is_system and day_chat_id is not None:
        # Look for an existing active row for today.
        existing = (
            await db.execute(
                select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.day_chat_id == day_chat_id,
                    Conversation.channel == channel,
                    Conversation.is_active.is_(True),
                ).limit(1)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

    if is_system and day_chat_id is None:
        # Degraded path: log so we can find the upstream day-chat
        # resolution gap. Don't dedupe — there's nothing to dedupe by.
        logger.warning(
            "[conversation_resolver] system channel=%s with no day_chat_id "
            "for user_id=%s — creating unconstrained row",
            channel, user_id,
        )

    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        channel=channel,
        is_active=True,
        day_chat_id=day_chat_id,
        title=title,
        metadata_json=meta_json,
    )
    db.add(conv)
    try:
        await db.flush()
        return conv
    except IntegrityError:
        # Lost the race against a concurrent fire. Roll the savepoint
        # back so the session is reusable, then read the winner.
        if not is_system or day_chat_id is None:
            raise
        await db.rollback()
        logger.info(
            "[conversation_resolver] concurrent insert collision for "
            "user=%s day=%s channel=%s — re-fetching winner",
            user_id, day_chat_id, channel,
        )
        winner = (
            await db.execute(
                select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.day_chat_id == day_chat_id,
                    Conversation.channel == channel,
                    Conversation.is_active.is_(True),
                ).limit(1)
            )
        ).scalar_one_or_none()
        if winner is None:
            # Shouldn't happen — the IntegrityError says a row exists.
            # Re-raise the original so we don't silently lose data.
            raise
        return winner
