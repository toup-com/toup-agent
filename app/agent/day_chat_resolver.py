"""
Day Chat Resolver — maps (user_id, timestamp, timezone) to a DayChat row.

This is the single entry point for resolving which DayChat a message belongs to.
All message save paths must call get_or_create_day_chat() to resolve the correct
DayChat for the CURRENT TIME, not read day_chat_id from the Conversation row.

CRITICAL SEMANTIC: Message.day_chat_id is CANONICAL for day membership.
Conversation.day_chat_id is a LOOSE HINT that may be stale — Telegram sessions
are long-lived and span multiple days, so their Conversation.day_chat_id points
to the day the session was created, not the current day. Never determine which
day a message belongs to by looking at its parent Conversation's day_chat_id.
Always use Message.day_chat_id directly.

Key semantics:
  - One DayChat per (user_id, local_date). Uses Postgres INSERT ... ON CONFLICT
    DO NOTHING for concurrent safety — two messages arriving at the same millisecond
    will both resolve to the same DayChat without racing.
  - A "New Thread" button in the UI creates a new Conversation (session) inside the
    current DayChat but does NOT reset agent context. The agent still sees every
    message from every session in the day.
  - Channel switches (web → telegram → vibecoding) create new sessions inside the
    SAME DayChat. Context carries across channels.
  - Day boundary uses the user's local timezone (IANA format, e.g. "America/Toronto").
    Falls back to UTC if the user has no timezone set.
"""

import logging
import uuid
import zoneinfo
from datetime import datetime, date as Date, timezone
from typing import Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def resolve_local_date(utc_now: datetime, tz_name: Optional[str]) -> Tuple[Date, str]:
    """Convert a UTC datetime to the user's local calendar date.

    Returns (local_date, resolved_timezone_name).
    """
    if utc_now.tzinfo is None:
        utc_now = utc_now.replace(tzinfo=timezone.utc)

    tz_used = "UTC"
    if tz_name:
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
            local_dt = utc_now.astimezone(tz)
            return local_dt.date(), tz_name
        except (KeyError, Exception):
            logger.warning("Invalid timezone '%s', falling back to UTC", tz_name)

    return utc_now.date(), tz_used


async def get_or_create_day_chat(
    db: AsyncSession,
    user_id: str,
    utc_now: Optional[datetime] = None,
    tz_name: Optional[str] = None,
) -> "DayChat":
    """Get or create the DayChat for a user's local calendar date.

    Concurrent-safe: uses INSERT ... ON CONFLICT DO NOTHING on the
    (user_id, local_date) unique constraint, then SELECTs the row.
    Two simultaneous callers for the same user+date will both succeed
    and return the same DayChat.

    Args:
        db: Active async session (caller manages transaction).
        user_id: The user's ID.
        utc_now: Current UTC time (defaults to now).
        tz_name: IANA timezone (e.g. "America/Toronto"). Falls back to UTC.

    Returns:
        The DayChat row for today.
    """
    from app.db.models.day_chat import DayChat

    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    local_date, resolved_tz = resolve_local_date(utc_now, tz_name)

    # Strip tzinfo for DB insertion — codebase convention is naive UTC datetimes
    utc_now_naive = utc_now.replace(tzinfo=None) if utc_now.tzinfo else utc_now

    # Fast path: check if it already exists
    existing = (await db.execute(
        select(DayChat).where(
            and_(DayChat.user_id == user_id, DayChat.local_date == local_date)
        )
    )).scalar_one_or_none()

    if existing:
        return existing

    # Slow path: create via Postgres upsert (concurrent-safe)
    new_id = str(uuid.uuid4())
    try:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(DayChat.__table__).values(
            id=new_id,
            user_id=user_id,
            local_date=local_date,
            timezone=resolved_tz,
            started_at=utc_now_naive,
            last_message_at=utc_now_naive,
            message_count=0,
            total_tokens=0,
            summary_status="up_to_date",
        ).on_conflict_do_nothing(
            index_elements=["user_id", "local_date"]
        )
        await db.execute(stmt)
        await db.flush()
    except Exception:
        # Fallback for SQLite or other DBs without ON CONFLICT
        pass

    # SELECT the row — either we created it or a concurrent writer did
    row = (await db.execute(
        select(DayChat).where(
            and_(DayChat.user_id == user_id, DayChat.local_date == local_date)
        )
    )).scalar_one_or_none()

    if row:
        return row

    # Final fallback: direct insert (SQLite path, no ON CONFLICT support)
    dc = DayChat(
        id=new_id,
        user_id=user_id,
        local_date=local_date,
        timezone=resolved_tz,
        started_at=utc_now_naive,
        last_message_at=utc_now_naive,
        message_count=0,
        total_tokens=0,
        summary_status="up_to_date",
    )
    db.add(dc)
    await db.flush()
    return dc


async def should_use_day_chat_context(
    session_maker=None,
    feature_flag_override: bool = None,
) -> bool:
    """Check if the day-chat context path should be used.

    Returns True only when BOTH conditions are met:
      1. USE_DAY_CHAT_CONTEXT feature flag is True
      2. day_chat_backfill migration is 'completed'

    If backfill hasn't finished, silently falls back to old path.
    Logs a fallback reason once per process to avoid log spam.

    Args:
        session_maker: Optional override for testing. Defaults to production session_maker.
        feature_flag_override: Optional override for testing. Defaults to settings.use_day_chat_context.
    """
    if feature_flag_override is not None:
        flag = feature_flag_override
    else:
        from app.config import settings
        flag = settings.use_day_chat_context

    if not flag:
        return False

    # Check backfill status
    try:
        if session_maker is None:
            from app.db.database import async_session_maker
            session_maker = async_session_maker

        from app.db.models.day_chat import MigrationStatus

        async with session_maker() as db:
            ms = (await db.execute(
                select(MigrationStatus).where(
                    MigrationStatus.migration_name == "day_chat_backfill"
                )
            )).scalar_one_or_none()

            if not ms or ms.status != "completed":
                if not getattr(should_use_day_chat_context, "_fallback_logged", False):
                    reason = ms.status if ms else "not_started"
                    logger.warning(
                        "day_chat_context.fallback reason=backfill_not_completed status=%s",
                        reason,
                    )
                    should_use_day_chat_context._fallback_logged = True
                return False

        return True
    except Exception as e:
        logger.warning("day_chat_context.fallback reason=check_error error=%s", e)
        return False
