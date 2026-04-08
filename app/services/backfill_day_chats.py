"""
Backfill DayChat rows from existing Conversations.

Called from agent startup as a non-blocking background task. Processes
conversations in deterministic order (ORDER BY created_at, id), creates
DayChat rows via Postgres upsert, and updates FK references.

Design constraints:
  - Idempotent: running twice produces the same result.
  - Resumable: tracks last_processed_conversation_id in MigrationStatus.
  - Per-conversation transactions: partial failures roll back cleanly.
  - UTC fallback: uses User.timezone if set, otherwise UTC.
  - Feature-flag independent: backfill running ≠ new context path active.

Env var FORCE_DAY_CHAT_BACKFILL=true resets status to not_started and
re-runs. Useful when timezone changes require re-bucketing.
"""

import json
import logging
import os
import time
import uuid
import zoneinfo
from datetime import datetime, date as Date, timezone
from typing import Optional

from sqlalchemy import select, and_, update, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("day_chat_backfill")

MIGRATION_NAME = "day_chat_backfill"


def _user_local_date(started_at: datetime, tz_name: Optional[str]) -> Date:
    """Convert a UTC started_at to the user's local calendar date."""
    if not started_at:
        return datetime.now(timezone.utc).date()
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if tz_name:
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
            return started_at.astimezone(tz).date()
        except (KeyError, Exception):
            pass
    return started_at.date()


async def get_migration_status(db: AsyncSession) -> Optional["MigrationStatus"]:
    from app.db.models.day_chat import MigrationStatus
    result = await db.execute(
        select(MigrationStatus).where(MigrationStatus.migration_name == MIGRATION_NAME)
    )
    return result.scalar_one_or_none()


async def run_backfill(session_maker) -> str:
    """Run the DayChat backfill. Returns final status string.

    Args:
        session_maker: async_sessionmaker for creating DB sessions.

    Returns:
        "completed", "already_completed", or "failed".
    """
    from app.db.models import User, Conversation, Message
    from app.db.models.day_chat import DayChat, MigrationStatus

    t0 = time.monotonic()

    # ── Check / initialize migration status ──
    async with session_maker() as db:
        status_row = await get_migration_status(db)

        # Handle FORCE_DAY_CHAT_BACKFILL
        force = os.environ.get("FORCE_DAY_CHAT_BACKFILL", "").lower() in ("true", "1", "yes")
        if force and status_row and status_row.status in ("completed", "failed"):
            logger.info("day_chat_backfill.force_reset")
            status_row.status = "not_started"
            status_row.started_at = None
            status_row.completed_at = None
            status_row.progress_json = None
            status_row.error_message = None
            await db.commit()
            status_row = await get_migration_status(db)

        if not status_row:
            status_row = MigrationStatus(
                migration_name=MIGRATION_NAME,
                status="not_started",
            )
            db.add(status_row)
            await db.commit()

        if status_row.status == "completed":
            return "already_completed"

        if status_row.status == "failed":
            logger.error(
                "day_chat_backfill.previously_failed error=%s — not auto-retrying. "
                "Set FORCE_DAY_CHAT_BACKFILL=true to re-run.",
                status_row.error_message,
            )
            return "failed"

        # Resume point
        progress = json.loads(status_row.progress_json) if status_row.progress_json else {}
        last_processed_id = progress.get("last_conversation_id")
        processed_count = progress.get("processed", 0)

    # ── Get user and total conversation count ──
    async with session_maker() as db:
        user_row = (await db.execute(select(User).limit(1))).scalar_one_or_none()
        if not user_row:
            # Empty DB — mark completed immediately
            async with session_maker() as db2:
                ms = await get_migration_status(db2)
                ms.status = "completed"
                ms.completed_at = datetime.utcnow()
                await db2.commit()
            logger.info("day_chat_backfill.completed user_id=none conversations=0 messages=0 elapsed=0s")
            return "completed"

        user_id = user_row.id
        user_tz = getattr(user_row, 'timezone', None)

        total_convos = (await db.execute(
            select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        )).scalar() or 0

        total_messages = (await db.execute(
            select(func.count()).select_from(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )).scalar() or 0

    # ── Mark in_progress ──
    async with session_maker() as db:
        ms = await get_migration_status(db)
        ms.status = "in_progress"
        ms.started_at = ms.started_at or datetime.utcnow()
        ms.error_message = None
        await db.commit()

    logger.info(
        "day_chat_backfill.started user_id=%s conversations=%d messages=%d",
        user_id, total_convos, total_messages,
    )

    is_rebucket = last_processed_id is not None or processed_count > 0 or \
        os.environ.get("FORCE_DAY_CHAT_BACKFILL", "").lower() in ("true", "1", "yes")

    try:
        # ── Load all conversations in deterministic order ──
        async with session_maker() as db:
            q = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at if hasattr(Conversation, 'created_at') else Conversation.started_at, Conversation.id)
            )
            all_convos = (await db.execute(q)).scalars().all()

        # Filter to resume point if applicable
        if last_processed_id:
            found = False
            for i, c in enumerate(all_convos):
                if c.id == last_processed_id:
                    all_convos = all_convos[i + 1:]
                    found = True
                    break
            if not found:
                # Resume ID not found — start from beginning (idempotent anyway)
                pass

        # ── Process each conversation ──
        day_chats_created = 0
        for conv in all_convos:
            async with session_maker() as db:
                try:
                    # Reload conv in this session
                    conv_row = (await db.execute(
                        select(Conversation).where(Conversation.id == conv.id)
                    )).scalar_one_or_none()
                    if not conv_row:
                        continue

                    # For re-bucket: always recompute. For initial: skip if already linked.
                    if not is_rebucket and conv_row.day_chat_id:
                        processed_count += 1
                        continue

                    # Resolve day bucket
                    local_date = _user_local_date(conv_row.started_at, user_tz)

                    # Upsert DayChat — concurrent-safe via ON CONFLICT DO NOTHING
                    dc_id = str(uuid.uuid4())
                    try:
                        stmt = pg_insert(DayChat.__table__).values(
                            id=dc_id,
                            user_id=user_id,
                            local_date=local_date,
                            timezone=user_tz or "UTC",
                            started_at=conv_row.started_at or datetime.utcnow(),
                            last_message_at=conv_row.updated_at or conv_row.started_at or datetime.utcnow(),
                            message_count=0,
                            total_tokens=0,
                            summary_status="stale",
                        ).on_conflict_do_nothing(
                            index_elements=["user_id", "local_date"]
                        )
                        await db.execute(stmt)
                        await db.flush()
                    except Exception:
                        # Fallback for SQLite (no on_conflict_do_nothing)
                        pass

                    # Fetch the actual DayChat row (may have been created by us or a concurrent write)
                    dc_row = (await db.execute(
                        select(DayChat).where(
                            and_(DayChat.user_id == user_id, DayChat.local_date == local_date)
                        )
                    )).scalar_one_or_none()

                    if not dc_row:
                        # SQLite fallback: create if not exists
                        dc_row = DayChat(
                            id=dc_id,
                            user_id=user_id,
                            local_date=local_date,
                            timezone=user_tz or "UTC",
                            started_at=conv_row.started_at or datetime.utcnow(),
                            last_message_at=conv_row.updated_at or conv_row.started_at or datetime.utcnow(),
                            message_count=0,
                            total_tokens=0,
                            summary_status="stale",
                        )
                        db.add(dc_row)
                        await db.flush()

                    actual_dc_id = dc_row.id

                    # If re-bucketing and the conversation moved to a different DayChat
                    if is_rebucket and conv_row.day_chat_id and conv_row.day_chat_id != actual_dc_id:
                        # Invalidate old DayChat's summary
                        old_dc = (await db.execute(
                            select(DayChat).where(DayChat.id == conv_row.day_chat_id)
                        )).scalar_one_or_none()
                        if old_dc:
                            old_dc.summary_status = "stale"
                            old_dc.rolling_summary = None

                        # Mark new DayChat's summary stale too
                        dc_row.summary_status = "stale"
                        dc_row.rolling_summary = None

                    # Update conversation
                    conv_row.day_chat_id = actual_dc_id

                    # Batch-update messages in chunks of 500
                    msg_result = await db.execute(
                        select(Message.id).where(
                            and_(
                                Message.conversation_id == conv.id,
                                # For re-bucket: update ALL messages; for initial: only unlinked
                                Message.day_chat_id != actual_dc_id if is_rebucket else Message.day_chat_id.is_(None),
                            )
                        )
                    )
                    msg_ids = [r[0] for r in msg_result.all()]

                    for i in range(0, len(msg_ids), 500):
                        chunk = msg_ids[i:i + 500]
                        await db.execute(
                            update(Message).where(Message.id.in_(chunk)).values(day_chat_id=actual_dc_id)
                        )

                    # Update DayChat aggregate stats
                    dc_row.message_count = (await db.execute(
                        select(func.count()).select_from(Message).where(Message.day_chat_id == actual_dc_id)
                    )).scalar() or 0
                    dc_row.last_message_at = conv_row.updated_at or conv_row.started_at or datetime.utcnow()

                    await db.commit()

                    processed_count += 1
                    if processed_count % 50 == 0:
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "day_chat_backfill.progress user_id=%s processed=%d/%d elapsed=%.1fs",
                            user_id, processed_count, total_convos, elapsed,
                        )

                except Exception as e:
                    await db.rollback()
                    logger.exception(
                        "day_chat_backfill.conversation_failed conversation_id=%s error=%s",
                        conv.id, e,
                    )
                    raise

            # Update progress after each successful conversation
            async with session_maker() as db:
                ms = await get_migration_status(db)
                ms.progress_json = json.dumps({
                    "last_conversation_id": conv.id,
                    "processed": processed_count,
                    "total": total_convos,
                })
                await db.commit()

        # ── Re-bucket cleanup: delete orphaned DayChats ──
        if is_rebucket:
            async with session_maker() as db:
                # Find DayChats with zero conversations
                orphan_dcs = (await db.execute(
                    select(DayChat.id).where(
                        and_(
                            DayChat.user_id == user_id,
                            ~DayChat.id.in_(
                                select(Conversation.day_chat_id).where(
                                    Conversation.day_chat_id.isnot(None)
                                ).distinct()
                            ),
                        )
                    )
                )).scalars().all()
                if orphan_dcs:
                    from sqlalchemy import delete as sa_delete
                    await db.execute(
                        sa_delete(DayChat).where(DayChat.id.in_(orphan_dcs))
                    )
                    logger.info("day_chat_backfill.cleanup orphaned_day_chats=%d", len(orphan_dcs))
                await db.commit()

        # ── Mark completed ──
        async with session_maker() as db:
            ms = await get_migration_status(db)
            ms.status = "completed"
            ms.completed_at = datetime.utcnow()
            ms.progress_json = json.dumps({
                "last_conversation_id": all_convos[-1].id if all_convos else None,
                "processed": processed_count,
                "total": total_convos,
            })
            await db.commit()

        elapsed = time.monotonic() - t0

        # Count day_chats created
        async with session_maker() as db:
            day_chats_created = (await db.execute(
                select(func.count()).select_from(DayChat).where(DayChat.user_id == user_id)
            )).scalar() or 0

        logger.info(
            "day_chat_backfill.completed user_id=%s day_chats_created=%d elapsed=%.1fs",
            user_id, day_chats_created, elapsed,
        )
        return "completed"

    except Exception as e:
        # Mark failed
        async with session_maker() as db:
            ms = await get_migration_status(db)
            ms.status = "failed"
            ms.error_message = str(e)[:2000]
            ms.progress_json = json.dumps({
                "last_conversation_id": None,
                "processed": processed_count,
                "total": total_convos,
            })
            await db.commit()

        logger.error(
            "day_chat_backfill.failed user_id=%s error=%s",
            user_id, e,
        )
        return "failed"
