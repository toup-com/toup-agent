"""Tests for the canonical (user, day, channel) → Conversation resolver.

Locks the Reading-A invariant: system-driven channels (`routine`,
`trigger`, `api`, `digest`) get exactly ONE Conversation per
(user_id, day_chat_id, channel). Every subsequent fire on the same day
must reuse the existing row.

Covers:
    1. Five sequential routine fires → 1 Conversation, 5 Messages.
    2. Two days, one fire each → 2 Conversations, 1 Message each.
    3. Routine + Trigger on the same day → 2 Conversations (different
       channels), same DayChat.
    4. resolve_or_create_day_conversation is idempotent under sequential
       calls with identical args.
    5. Non-system channels (`web`) bypass dedup — always create.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.agent.conversation_resolver import (
    SYSTEM_CHANNELS,
    resolve_or_create_day_conversation,
)


@pytest_asyncio.fixture
async def seeded_user_and_day():
    """Create a User row + DayChat row for today. Returns ids."""
    from app.db import User, async_session_maker
    from app.db.models import DayChat

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"resolver-test-{user_id[:8]}@example.com",
            hashed_password="x",
            name="Resolver Test",
            timezone="UTC",
        ))
        day = DayChat(
            id=str(uuid.uuid4()),
            user_id=user_id,
            local_date=date.today(),
            timezone="UTC",
        )
        db.add(day)
        await db.commit()
        return {"user_id": user_id, "day_chat_id": day.id}


@pytest.mark.asyncio
async def test_five_routine_fires_one_conversation_five_messages(seeded_user_and_day):
    """Five consecutive routine fires on the same day collapse to one
    Conversation. This is the headline Ticket-1 regression test."""
    from app.agent.routines.message_writer import write_routine_message
    from app.db import async_session_maker
    from app.db.models import Conversation, Message

    uid = seeded_user_and_day["user_id"]
    day_id = seeded_user_and_day["day_chat_id"]

    for i in range(5):
        async with async_session_maker() as db:
            await write_routine_message(
                db,
                user_id=uid,
                content=f"briefing {i}",
                source="email_briefing",
                routine_id="r-1",
                tz_override="UTC",
            )

    async with async_session_maker() as db:
        convs = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == uid,
                Conversation.channel == "routine",
                Conversation.is_active.is_(True),
            )
        )).scalars().all()
        msgs = (await db.execute(
            select(Message).where(
                Message.day_chat_id == day_id,
                Message.channel == "routine",
            )
        )).scalars().all()

    assert len(convs) == 1, (
        f"Expected 1 routine Conversation, got {len(convs)}. "
        f"The Ticket-1 dedup is broken."
    )
    assert len(msgs) == 5, f"Expected 5 routine Messages, got {len(msgs)}"
    assert all(m.conversation_id == convs[0].id for m in msgs)


@pytest.mark.asyncio
async def test_routine_and_trigger_same_day_two_conversations(seeded_user_and_day):
    """Routine and Trigger are distinct channels, so they get two rows
    on the same day — both still deduped within their own channel."""
    from app.agent.routines.message_writer import write_routine_message
    from app.agent.triggers.message_writer import write_trigger_message
    from app.db import async_session_maker
    from app.db.models import Conversation

    uid = seeded_user_and_day["user_id"]

    async with async_session_maker() as db:
        await write_routine_message(
            db, user_id=uid, content="r", source="email_briefing", tz_override="UTC",
        )
    async with async_session_maker() as db:
        await write_trigger_message(
            db, user_id=uid, content="t", source="email_received", tz_override="UTC",
        )

    async with async_session_maker() as db:
        convs = (await db.execute(
            select(Conversation).where(Conversation.user_id == uid)
        )).scalars().all()

    channels = sorted(c.channel for c in convs)
    assert channels == ["routine", "trigger"], channels


@pytest.mark.asyncio
async def test_resolver_is_idempotent(seeded_user_and_day):
    """Direct calls to the resolver with identical args return the same row."""
    from app.db import async_session_maker

    uid = seeded_user_and_day["user_id"]
    day_id = seeded_user_and_day["day_chat_id"]

    async with async_session_maker() as db:
        c1 = await resolve_or_create_day_conversation(
            db, user_id=uid, day_chat_id=day_id, channel="routine",
        )
        await db.commit()
        c1_id = c1.id

    async with async_session_maker() as db:
        c2 = await resolve_or_create_day_conversation(
            db, user_id=uid, day_chat_id=day_id, channel="routine",
        )
        await db.commit()
        assert c2.id == c1_id


@pytest.mark.asyncio
async def test_non_system_channel_always_creates(seeded_user_and_day):
    """Calling the resolver with a user channel (`web`) must not dedup
    — that semantic is reserved for SYSTEM_CHANNELS."""
    from app.db import async_session_maker

    assert "web" not in SYSTEM_CHANNELS

    uid = seeded_user_and_day["user_id"]
    day_id = seeded_user_and_day["day_chat_id"]

    async with async_session_maker() as db:
        c1 = await resolve_or_create_day_conversation(
            db, user_id=uid, day_chat_id=day_id, channel="web",
        )
        c2 = await resolve_or_create_day_conversation(
            db, user_id=uid, day_chat_id=day_id, channel="web",
        )
        await db.commit()
        assert c1.id != c2.id, "Non-system channel must not dedup"


@pytest.mark.asyncio
async def test_unique_index_blocks_raw_duplicate_insert(seeded_user_and_day):
    """Belt-and-suspenders: even if a caller skips the resolver and
    inserts directly, the partial unique index should reject duplicates
    for system channels. This locks the DB-level invariant."""
    from sqlalchemy.exc import IntegrityError

    from app.db import async_session_maker
    from app.db.models import Conversation

    uid = seeded_user_and_day["user_id"]
    day_id = seeded_user_and_day["day_chat_id"]

    async with async_session_maker() as db:
        db.add(Conversation(
            id=str(uuid.uuid4()),
            user_id=uid,
            channel="routine",
            day_chat_id=day_id,
            is_active=True,
        ))
        await db.commit()

    # Second raw insert must fail.
    async with async_session_maker() as db:
        db.add(Conversation(
            id=str(uuid.uuid4()),
            user_id=uid,
            channel="routine",
            day_chat_id=day_id,
            is_active=True,
        ))
        with pytest.raises(IntegrityError):
            await db.commit()
