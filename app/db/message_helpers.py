"""
Message creation helper — ensures day_chat_id is always set from the current time.

ALL code that creates Message rows should use resolve_day_chat_id_for_now()
to get the correct day_chat_id. Never read it from Conversation.day_chat_id
(which can be stale for long-lived Telegram sessions).

See day_chat_resolver.py docstring for the full semantic explanation.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def resolve_day_chat_id_for_now(
    db: AsyncSession,
    user_id: str,
    tz_override: Optional[str] = None,
    utc_now: Optional[datetime] = None,
) -> Optional[str]:
    """Resolve the DayChat ID for the current moment using the user's timezone.

    This is the canonical way to get day_chat_id for a new Message.
    Returns None if resolution fails (graceful degradation — message
    still saves, just without day_chat_id).

    Args:
        db: Active async session.
        user_id: The user's ID.
        tz_override: If provided, use this timezone instead of reading from the
            User row. This is critical for WebSocket callers that have the client's
            timezone from the message payload — avoids a race where the User.timezone
            hasn't been persisted yet or is still NULL.
        utc_now: Freeze the instant "now" resolves to. Defaults to the real
            clock, which is every production caller. Only a test that has to
            place the user on a specific side of their local midnight passes
            it; it is forwarded verbatim to `get_or_create_day_chat`.
    """
    if not user_id:
        return None
    try:
        from app.agent.day_chat_resolver import get_or_create_day_chat

        tz_name = tz_override
        if not tz_name:
            from app.db.models import User
            from sqlalchemy import select
            user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
            tz_name = getattr(user, 'timezone', None) if user else None
        dc = await get_or_create_day_chat(
            db, user_id, utc_now=utc_now, tz_name=tz_name,
        )
        return dc.id
    except Exception as exc:
        # Graceful degradation is the contract (the message still saves), but
        # a row written with day_chat_id=NULL is invisible to the day index
        # AND to load_day_context — the same "message not in canonical
        # history" shape as the 2026-09-14 incident — so it must never be
        # silent. Class only, never the message: the driver's text carries
        # DSNs.
        logger.warning(
            "[day_chat] resolve_day_chat_id_for_now failed user=%s tz=%r err=%s "
            "— message will be saved with day_chat_id=NULL",
            (user_id or "")[:8], tz_override, type(exc).__name__,
        )
        try:
            from app.services import health_signals
            health_signals.incr("day_chat_resolve_failures")
        except Exception:  # noqa: BLE001 — telemetry must never change the outcome
            pass
        return None
