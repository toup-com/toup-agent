"""
Tests for Day Chats REST API endpoints.

Tests the endpoint logic directly (not via HTTP) since we don't have
a running FastAPI server in tests. We test the query logic, pagination,
auth scoping, and response shapes.

Run: python tests/test_day_chats_api.py
"""

import asyncio
import re
import sys
import uuid
from datetime import datetime, date as Date, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func, text, distinct
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.user import User
from app.db.models.day_chat import DayChat
from app.db.models.conversation import Conversation, Message


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255),
                name VARCHAR(255), password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50)
            )""",
            """CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP, summary_status VARCHAR(20) DEFAULT 'up_to_date',
                archival_summary TEXT,
                archival_summary_generated_at TIMESTAMP,
                archival_summary_status VARCHAR(20) DEFAULT 'not_needed',
                UNIQUE(user_id, local_date)
            )""",
            """CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                title VARCHAR(500), day_chat_id VARCHAR(36) REFERENCES day_chats(id),
                channel VARCHAR(50) DEFAULT 'web', is_active BOOLEAN DEFAULT 1,
                started_at TIMESTAMP, ended_at TIMESTAMP, updated_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                builder_mode VARCHAR(10), metadata_json TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36) REFERENCES conversations(id),
                day_chat_id VARCHAR(36) REFERENCES day_chats(id), role VARCHAR(20),
                content TEXT, created_at TIMESTAMP, tokens_prompt INTEGER,
                tokens_completion INTEGER, model_used VARCHAR(50),
                memories_retrieved_json TEXT, processing_time_ms INTEGER,
                metadata_json TEXT, embedding_json TEXT, embedding BLOB
            , origin VARCHAR(16))""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


async def _seed_days(sm, user_id="u1", num_days=5, channels_per_day=None):
    """Seed multiple days with conversations and messages."""
    if channels_per_day is None:
        channels_per_day = ["web", "telegram"]

    async with sm() as db:
        u = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not u:
            db.add(User(id=user_id, email=f"{user_id}@test.local", hashed_password="x", name="Test"))

        day_chats = []
        for d in range(num_days):
            local_date = Date(2026, 4, 1 + d)
            dc_id = str(uuid.uuid4())
            dc = DayChat(
                id=dc_id, user_id=user_id, local_date=local_date, timezone="UTC",
                started_at=datetime(2026, 4, 1 + d, 8, 0, 0),
                last_message_at=datetime(2026, 4, 1 + d, 17, 0, 0),
                message_count=len(channels_per_day) * 3,
                summary_status="up_to_date",
            )
            db.add(dc)
            day_chats.append(dc)

            for i, ch in enumerate(channels_per_day):
                conv_id = str(uuid.uuid4())
                db.add(Conversation(
                    id=conv_id, user_id=user_id, channel=ch, day_chat_id=dc_id,
                    started_at=datetime(2026, 4, 1 + d, 8 + i * 3, 0, 0),
                    updated_at=datetime(2026, 4, 1 + d, 8 + i * 3 + 1, 0, 0),
                ))
                for m in range(3):
                    db.add(Message(
                        id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                        role="user" if m % 2 == 0 else "assistant",
                        content=f"Raw message {m} on {ch} day {d}",
                        created_at=datetime(2026, 4, 1 + d, 8 + i * 3, m * 10, 0),
                    ))

        await db.commit()
    return day_chats


# ── Helper: simulate endpoint logic ──────────────────────────────────

async def _list_day_chats(sm, user_id, before=None, limit=30):
    """Simulate GET /api/day-chats logic."""
    async with sm() as db:
        from sqlalchemy import distinct as _dist
        query = (
            select(DayChat)
            .where(DayChat.user_id == user_id)
            .order_by(DayChat.local_date.desc())
            .limit(limit)
        )
        if before:
            before_date = Date.fromisoformat(before)
            query = query.where(DayChat.local_date < before_date)

        result = await db.execute(query)
        day_chats = result.scalars().all()

        items = []
        for dc in day_chats:
            ch_result = await db.execute(
                select(_dist(Conversation.channel)).where(Conversation.day_chat_id == dc.id)
            )
            channels = sorted([r[0] for r in ch_result.all() if r[0]])
            items.append({
                "id": dc.id,
                "local_date": dc.local_date.isoformat(),
                "message_count": dc.message_count or 0,
                "channels_active": channels,
                "last_message_at": dc.last_message_at.isoformat() if dc.last_message_at else None,
                "summary_status": dc.summary_status or "up_to_date",
            })
        return items


async def _get_day_messages(sm, user_id, date_str, limit=500):
    """Simulate GET /api/day-chats/{date}/messages logic."""
    target_date = Date.fromisoformat(date_str)
    async with sm() as db:
        from sqlalchemy import and_
        dc = (await db.execute(
            select(DayChat).where(
                and_(DayChat.user_id == user_id, DayChat.local_date == target_date)
            )
        )).scalar_one_or_none()

        if not dc:
            return None  # 404

        msgs_result = await db.execute(
            select(Message, Conversation.channel)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.day_chat_id == dc.id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        rows = msgs_result.all()
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "channel": channel or "web",
                "conversation_id": msg.conversation_id,
            }
            for msg, channel in rows
        ]


# ── Tests ─────────────────────────────────────────────────────────────

async def test_list_day_chats_ordering():
    """Day chats returned newest first."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=5)

    items = await _list_day_chats(sm, "u1")
    assert len(items) == 5
    # Newest first
    dates = [i["local_date"] for i in items]
    assert dates == sorted(dates, reverse=True), f"Not newest-first: {dates}"

    await engine.dispose()


async def test_list_day_chats_pagination():
    """Cursor-based pagination with ?before works."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=5)

    # Get all
    all_items = await _list_day_chats(sm, "u1")
    assert len(all_items) == 5

    # Get before April 4 (should return April 1, 2, 3)
    page = await _list_day_chats(sm, "u1", before="2026-04-04")
    dates = [i["local_date"] for i in page]
    assert all(d < "2026-04-04" for d in dates), f"Pagination leak: {dates}"
    assert len(page) == 3

    # Limit enforcement
    limited = await _list_day_chats(sm, "u1", limit=2)
    assert len(limited) == 2

    await engine.dispose()


async def test_list_day_chats_channels_active():
    """channels_active correctly reflects which channels were used each day."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=1, channels_per_day=["web", "telegram", "vibecoding"])

    items = await _list_day_chats(sm, "u1")
    assert len(items) == 1
    assert sorted(items[0]["channels_active"]) == ["telegram", "vibecoding", "web"]

    await engine.dispose()


async def test_list_day_chats_auth_scoping():
    """User A cannot see user B's day chats."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    await _seed_days(sm, user_id="userA", num_days=3)
    await _seed_days(sm, user_id="userB", num_days=2)

    a_items = await _list_day_chats(sm, "userA")
    b_items = await _list_day_chats(sm, "userB")

    assert len(a_items) == 3
    assert len(b_items) == 2

    # No ID overlap
    a_ids = {i["id"] for i in a_items}
    b_ids = {i["id"] for i in b_items}
    assert not a_ids & b_ids

    await engine.dispose()


async def test_list_day_chats_empty():
    """Empty list when no day chats exist."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="empty-user", email="empty@test.local", hashed_password="x", name="Test"))
        await db.commit()

    items = await _list_day_chats(sm, "empty-user")
    assert items == []

    await engine.dispose()


async def test_messages_chronological_across_channels():
    """Messages returned in strict chronological order across channels."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=1, channels_per_day=["web", "telegram"])

    msgs = await _get_day_messages(sm, "u1", "2026-04-01")
    assert msgs is not None
    assert len(msgs) == 6  # 2 channels × 3 messages

    # Verify chronological order
    timestamps = [m["created_at"] for m in msgs]
    assert timestamps == sorted(timestamps), f"Not chronological: {timestamps}"

    # Verify both channels present
    channels = {m["channel"] for m in msgs}
    assert channels == {"web", "telegram"}

    await engine.dispose()


async def test_messages_raw_content_no_annotations():
    """Message content is RAW — no [channel time] annotations leak through."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=1, channels_per_day=["web"])

    msgs = await _get_day_messages(sm, "u1", "2026-04-01")
    assert msgs is not None

    annotation_pattern = re.compile(r'^\[\w+ \d+:\d+\w+\]')
    for m in msgs:
        assert not annotation_pattern.match(m["content"]), \
            f"Annotation leaked into API response: {m['content'][:50]}"
        assert m["content"].startswith("Raw message"), \
            f"Content modified: {m['content'][:50]}"

    await engine.dispose()


async def test_messages_user_scoping():
    """User can only see their own day's messages."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    await _seed_days(sm, user_id="owner", num_days=1)
    await _seed_days(sm, user_id="other", num_days=1)

    owner_msgs = await _get_day_messages(sm, "owner", "2026-04-01")
    other_msgs = await _get_day_messages(sm, "other", "2026-04-01")

    assert owner_msgs is not None
    assert other_msgs is not None

    owner_ids = {m["id"] for m in owner_msgs}
    other_ids = {m["id"] for m in other_msgs}
    assert not owner_ids & other_ids, "User scoping violated — messages overlap"

    await engine.dispose()


async def test_messages_404_for_missing_date():
    """Returns None (404) for a date with no day chat."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="no-data", email="nodata@test.local", hashed_password="x", name="Test"))
        await db.commit()

    result = await _get_day_messages(sm, "no-data", "2026-04-01")
    assert result is None, "Should return None (404) for missing date"

    await engine.dispose()


async def test_messages_empty_for_day_with_no_messages():
    """Returns empty array for a day chat that exists but has no messages."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="empty-day", email="emptyday@test.local", hashed_password="x", name="Test"))
        db.add(DayChat(
            id="dc-empty", user_id="empty-day", local_date=Date(2026, 4, 1),
            timezone="UTC", started_at=datetime(2026, 4, 1, 8, 0, 0),
            last_message_at=datetime(2026, 4, 1, 8, 0, 0),
        ))
        await db.commit()

    result = await _get_day_messages(sm, "empty-day", "2026-04-01")
    assert result == [], f"Expected empty array, got {result}"

    await engine.dispose()


async def test_response_shape_list():
    """Verify the exact shape of the list endpoint response."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=1, channels_per_day=["web"])

    items = await _list_day_chats(sm, "u1")
    assert len(items) == 1
    item = items[0]

    # Required fields
    assert "id" in item
    assert "local_date" in item
    assert "message_count" in item
    assert "channels_active" in item
    assert "last_message_at" in item
    assert "summary_status" in item

    # Types
    assert isinstance(item["id"], str)
    assert isinstance(item["local_date"], str)
    assert isinstance(item["message_count"], int)
    assert isinstance(item["channels_active"], list)
    assert isinstance(item["summary_status"], str)

    await engine.dispose()


async def test_response_shape_messages():
    """Verify the exact shape of the messages endpoint response."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _seed_days(sm, num_days=1, channels_per_day=["web"])

    msgs = await _get_day_messages(sm, "u1", "2026-04-01")
    assert len(msgs) > 0
    msg = msgs[0]

    # Required fields
    assert "id" in msg
    assert "role" in msg
    assert "content" in msg
    assert "created_at" in msg
    assert "channel" in msg
    assert "conversation_id" in msg

    # Types
    assert isinstance(msg["id"], str)
    assert isinstance(msg["role"], str)
    assert isinstance(msg["content"], str)
    assert isinstance(msg["channel"], str)
    assert isinstance(msg["conversation_id"], str)

    await engine.dispose()


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_list_day_chats_ordering", test_list_day_chats_ordering),
            ("test_list_day_chats_pagination", test_list_day_chats_pagination),
            ("test_list_day_chats_channels_active", test_list_day_chats_channels_active),
            ("test_list_day_chats_auth_scoping", test_list_day_chats_auth_scoping),
            ("test_list_day_chats_empty", test_list_day_chats_empty),
            ("test_messages_chronological_across_channels", test_messages_chronological_across_channels),
            ("test_messages_raw_content_no_annotations", test_messages_raw_content_no_annotations),
            ("test_messages_user_scoping", test_messages_user_scoping),
            ("test_messages_404_for_missing_date", test_messages_404_for_missing_date),
            ("test_messages_empty_for_day_with_no_messages", test_messages_empty_for_day_with_no_messages),
            ("test_response_shape_list", test_response_shape_list),
            ("test_response_shape_messages", test_response_shape_messages),
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
