"""
Tests for DayChat backfill logic.

Tests use an in-memory SQLite database (no Postgres required).
Since SQLite doesn't support ON CONFLICT DO NOTHING on composite unique
constraints natively, the backfill function has a SQLite fallback path.

Run: python -m pytest tests/test_backfill_day_chats.py -v
  or: python tests/test_backfill_day_chats.py  (standalone)
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.base import Base, AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES
from app.db.models.conversation import Conversation, Message
from app.db.models.day_chat import DayChat, MigrationStatus
from app.db.models.user import User


# ── Fixtures ──────────────────────────────────────────────────────────

async def _make_engine():
    """Create an in-memory SQLite engine with the tables needed for backfill tests."""
    from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, ForeignKey, Date

    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})

    # Create only the tables we actually need, avoiding Postgres-specific column types
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: c.execute(
            __import__('sqlalchemy').text("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(36) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE,
                    hashed_password VARCHAR(255),
                    password_plain VARCHAR(255),
                    name VARCHAR(255),
                    password_changed_at TIMESTAMP,
                    role VARCHAR(20) DEFAULT 'beta_user',
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    stripe_customer_id VARCHAR(255),
                    timezone VARCHAR(50)
                )
            """)
        ))
        await conn.run_sync(lambda c: c.execute(
            __import__('sqlalchemy').text("""
                CREATE TABLE IF NOT EXISTS day_chats (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    local_date DATE NOT NULL,
                    timezone VARCHAR(50) DEFAULT 'UTC',
                    started_at TIMESTAMP,
                    last_message_at TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    rolling_summary TEXT,
                    summary_up_to_message_id VARCHAR(50),
                    summary_updated_at TIMESTAMP,
                    summary_status VARCHAR(20) DEFAULT 'up_to_date',
                    UNIQUE(user_id, local_date)
                )
            """)
        ))
        await conn.run_sync(lambda c: c.execute(
            __import__('sqlalchemy').text("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    title VARCHAR(500),
                    day_chat_id VARCHAR(36) REFERENCES day_chats(id),
                    channel VARCHAR(50) DEFAULT 'api',
                    is_active BOOLEAN DEFAULT 1,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    builder_mode VARCHAR(10),
                    metadata_json TEXT
                )
            """)
        ))
        await conn.run_sync(lambda c: c.execute(
            __import__('sqlalchemy').text("""
                CREATE TABLE IF NOT EXISTS messages (
                    id VARCHAR(50) PRIMARY KEY,
                    conversation_id VARCHAR(36) REFERENCES conversations(id),
                    day_chat_id VARCHAR(36) REFERENCES day_chats(id),
                    role VARCHAR(20),
                    content TEXT,
                    created_at TIMESTAMP,
                    tokens_prompt INTEGER,
                    tokens_completion INTEGER,
                    model_used VARCHAR(50),
                    memories_retrieved_json TEXT,
                    processing_time_ms INTEGER,
                    metadata_json TEXT,
                    embedding_json TEXT,
                    embedding BLOB
                )
            """)
        ))
        await conn.run_sync(lambda c: c.execute(
            __import__('sqlalchemy').text("""
                CREATE TABLE IF NOT EXISTS migration_status (
                    migration_name VARCHAR(100) PRIMARY KEY,
                    status VARCHAR(20) DEFAULT 'not_started',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    progress_json TEXT,
                    error_message TEXT
                )
            """)
        ))
    return engine


def _make_user(user_id="user-001", tz=None):
    return User(id=user_id, email=f"{user_id}@test.local", hashed_password="x", name="Test", timezone=tz)


def _make_conv(user_id, started_at, channel="web", conv_id=None):
    return Conversation(
        id=conv_id or str(uuid.uuid4()),
        user_id=user_id,
        channel=channel,
        started_at=started_at,
        updated_at=started_at,
    )


def _make_msg(conv_id, role="user", content="hello", created_at=None):
    return Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_id,
        role=role,
        content=content,
        created_at=created_at or datetime.utcnow(),
    )


async def _seed_db(session_maker, user_tz=None, num_days=5, msgs_per_conv=10):
    """Seed a DB with conversations spanning multiple days."""
    user = _make_user(tz=user_tz)
    convos = []
    all_msgs = []

    async with session_maker() as db:
        db.add(user)
        base_date = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        for day in range(num_days):
            dt = base_date + timedelta(days=day)
            # 2 conversations per day
            for session_num in range(2):
                conv = _make_conv(user.id, dt + timedelta(hours=session_num * 3))
                convos.append(conv)
                db.add(conv)
                for m in range(msgs_per_conv):
                    msg = _make_msg(conv.id, "user" if m % 2 == 0 else "assistant",
                                    f"msg-{day}-{session_num}-{m}",
                                    dt + timedelta(hours=session_num * 3, minutes=m))
                    all_msgs.append(msg)
                    db.add(msg)
        await db.commit()

    return user, convos, all_msgs


# ── Tests ─────────────────────────────────────────────────────────────

def _asyncio_test(fn):
    """Decorator: marks as pytest.mark.asyncio if pytest is available, otherwise no-op."""
    if pytest:
        return pytest.mark.asyncio(fn)
    return fn


@_asyncio_test
async def test_backfill_idempotency():
    """Running backfill twice produces the same result."""
    from _backfill_helper import run_backfill

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    user, convos, msgs = await _seed_db(sm, num_days=3, msgs_per_conv=5)

    # First run
    result1 = await run_backfill(sm)
    assert result1 == "completed"

    async with sm() as db:
        dc_count_1 = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        linked_1 = (await db.execute(
            select(func.count()).select_from(Conversation).where(Conversation.day_chat_id.isnot(None))
        )).scalar()
        msg_linked_1 = (await db.execute(
            select(func.count()).select_from(Message).where(Message.day_chat_id.isnot(None))
        )).scalar()

    # Second run — should be no-op
    result2 = await run_backfill(sm)
    assert result2 == "already_completed"

    async with sm() as db:
        dc_count_2 = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        linked_2 = (await db.execute(
            select(func.count()).select_from(Conversation).where(Conversation.day_chat_id.isnot(None))
        )).scalar()
        msg_linked_2 = (await db.execute(
            select(func.count()).select_from(Message).where(Message.day_chat_id.isnot(None))
        )).scalar()

    # Counts unchanged
    assert dc_count_1 == dc_count_2
    assert linked_1 == linked_2
    assert msg_linked_1 == msg_linked_2

    # All conversations linked
    assert linked_1 == len(convos)
    # All messages linked
    assert msg_linked_1 == len(msgs)
    # 3 days = 3 DayChats
    assert dc_count_1 == 3

    await engine.dispose()


@_asyncio_test
async def test_backfill_resume():
    """Backfill resumes from the right conversation after a simulated crash."""
    from _backfill_helper import run_backfill, MIGRATION_NAME, MIGRATION_NAME

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    user, convos, msgs = await _seed_db(sm, num_days=3, msgs_per_conv=2)

    # Simulate a partial run: manually set some conversations as processed
    halfway = len(convos) // 2
    async with sm() as db:
        for conv in convos[:halfway]:
            # Manually create day_chats and link
            local_date = conv.started_at.date()
            existing_dc = (await db.execute(
                select(DayChat).where(and_(DayChat.user_id == user.id, DayChat.local_date == local_date))
            )).scalar_one_or_none()
            if not existing_dc:
                dc = DayChat(
                    id=str(uuid.uuid4()), user_id=user.id, local_date=local_date,
                    timezone="UTC", started_at=conv.started_at,
                    last_message_at=conv.updated_at or conv.started_at,
                )
                db.add(dc)
                await db.flush()
                dc_id = dc.id
            else:
                dc_id = existing_dc.id
            conv_row = (await db.execute(select(Conversation).where(Conversation.id == conv.id))).scalar_one()
            conv_row.day_chat_id = dc_id

        # Create migration status as if it crashed mid-run
        ms = MigrationStatus(
            migration_name=MIGRATION_NAME,
            status="in_progress",
            started_at=datetime.utcnow(),
            progress_json=json.dumps({
                "last_conversation_id": convos[halfway - 1].id,
                "processed": halfway,
                "total": len(convos),
            }),
        )
        db.add(ms)
        await db.commit()

    # Resume
    result = await run_backfill(sm)
    assert result == "completed"

    # All conversations should now be linked
    async with sm() as db:
        linked = (await db.execute(
            select(func.count()).select_from(Conversation).where(Conversation.day_chat_id.isnot(None))
        )).scalar()

    assert linked == len(convos)

    await engine.dispose()


@_asyncio_test
async def test_backfill_concurrent_writes():
    """Backfill works even when new conversations are being created concurrently."""
    from _backfill_helper import run_backfill

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    user, convos, msgs = await _seed_db(sm, num_days=2, msgs_per_conv=3)

    # Start backfill
    result = await run_backfill(sm)
    assert result == "completed"

    # Now add a new conversation to an existing day (simulating live agent)
    async with sm() as db:
        new_conv = _make_conv(user.id, datetime(2026, 4, 1, 18, 0, 0, tzinfo=timezone.utc))
        db.add(new_conv)
        new_msg = _make_msg(new_conv.id, "user", "new concurrent msg",
                            datetime(2026, 4, 1, 18, 0, 0, tzinfo=timezone.utc))
        db.add(new_msg)
        await db.commit()
        new_conv_id = new_conv.id

    # The new conversation has no day_chat_id yet (backfill already completed)
    async with sm() as db:
        conv = (await db.execute(select(Conversation).where(Conversation.id == new_conv_id))).scalar_one()
        assert conv.day_chat_id is None  # Expected: not linked yet

    # Force re-run picks it up
    os.environ["FORCE_DAY_CHAT_BACKFILL"] = "true"
    try:
        result = await run_backfill(sm)
        assert result == "completed"
    finally:
        os.environ.pop("FORCE_DAY_CHAT_BACKFILL", None)

    # Now the new conversation should be linked
    async with sm() as db:
        conv = (await db.execute(select(Conversation).where(Conversation.id == new_conv_id))).scalar_one()
        assert conv.day_chat_id is not None

    # No duplicates — still only 2 DayChats (2 days)
    async with sm() as db:
        dc_count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        assert dc_count == 2

    await engine.dispose()


@_asyncio_test
async def test_backfill_empty_db():
    """Fresh agent with zero conversations completes instantly."""
    from _backfill_helper import run_backfill

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Just a user, no conversations
    async with sm() as db:
        db.add(_make_user())
        await db.commit()

    result = await run_backfill(sm)
    assert result == "completed"

    # Migration marked completed
    async with sm() as db:
        ms = (await db.execute(
            select(MigrationStatus).where(MigrationStatus.migration_name == "day_chat_backfill")
        )).scalar_one()
        assert ms.status == "completed"

    # No day_chats created
    async with sm() as db:
        dc_count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        assert dc_count == 0

    await engine.dispose()


@_asyncio_test
async def test_rebucket_timezone_change():
    """Re-bucketing after timezone change moves messages to correct local-day buckets."""
    from _backfill_helper import run_backfill

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create a user with no timezone (defaults to UTC)
    user = _make_user(tz=None)
    async with sm() as db:
        db.add(user)

        # Message at 2026-04-01 23:30 UTC
        # In UTC: April 1. In America/Toronto (UTC-4): April 1 at 7:30pm — still April 1.
        # Message at 2026-04-02 02:30 UTC
        # In UTC: April 2. In America/Toronto: April 1 at 10:30pm — still April 1!
        conv1 = _make_conv(user.id, datetime(2026, 4, 1, 23, 30, 0), conv_id="conv-late-1")
        conv2 = _make_conv(user.id, datetime(2026, 4, 2, 2, 30, 0), conv_id="conv-late-2")
        db.add(conv1)
        db.add(conv2)
        db.add(_make_msg(conv1.id, "user", "late night msg 1", datetime(2026, 4, 1, 23, 30, 0)))
        db.add(_make_msg(conv2.id, "user", "late night msg 2", datetime(2026, 4, 2, 2, 30, 0)))
        await db.commit()

    # Initial backfill with UTC timezone
    result = await run_backfill(sm)
    assert result == "completed"

    # Under UTC: conv1 → April 1, conv2 → April 2 (two DayChats)
    async with sm() as db:
        dc_count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        assert dc_count == 2

    # Now set user timezone to America/Toronto and re-bucket
    async with sm() as db:
        u = (await db.execute(select(User).where(User.id == user.id))).scalar_one()
        u.timezone = "America/Toronto"
        await db.commit()

    os.environ["FORCE_DAY_CHAT_BACKFILL"] = "true"
    try:
        result = await run_backfill(sm)
        assert result == "completed"
    finally:
        os.environ.pop("FORCE_DAY_CHAT_BACKFILL", None)

    # Under America/Toronto: BOTH conversations are April 1 (one DayChat)
    async with sm() as db:
        dc_count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
        # conv1: 23:30 UTC = 19:30 ET (April 1)
        # conv2: 02:30 UTC = 22:30 ET April 1 (NOT April 2)
        assert dc_count == 1, f"Expected 1 DayChat after re-bucket, got {dc_count}"

        # Both conversations point to the same DayChat
        convs = (await db.execute(
            select(Conversation).where(Conversation.user_id == user.id)
        )).scalars().all()
        dc_ids = {c.day_chat_id for c in convs}
        assert len(dc_ids) == 1, f"Expected 1 unique day_chat_id, got {dc_ids}"

        # The DayChat is for April 1
        dc = (await db.execute(select(DayChat).where(DayChat.id == list(dc_ids)[0]))).scalar_one()
        assert dc.local_date.isoformat() == "2026-04-01"

        # No messages were lost
        msg_count = (await db.execute(select(func.count()).select_from(Message))).scalar()
        assert msg_count == 2

    await engine.dispose()


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_backfill_idempotency", test_backfill_idempotency),
            ("test_backfill_resume", test_backfill_resume),
            ("test_backfill_concurrent_writes", test_backfill_concurrent_writes),
            ("test_backfill_empty_db", test_backfill_empty_db),
            ("test_rebucket_timezone_change", test_rebucket_timezone_change),
        ]
        passed = 0
        failed = 0
        for name, fn in tests:
            try:
                await fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        return failed == 0

    success = asyncio.run(_run_all())
    sys.exit(0 if success else 1)
