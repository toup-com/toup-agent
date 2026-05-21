"""Regression test for reply-target denormalization on message-list APIs.

Live user report (2026-05-21, after PRs #53/54/58 shipped):
- On page refresh, the "(message not in current view)" stub flashes on
  each reply bubble before the full quoted content appears.
- The stub also falsely labels the reply as "Replying to you" because
  the inline card's role-attribution ternary defaulted to 'you' when
  the target hadn't been resolved yet from the in-memory map.

Root cause: every message-list serializer returned only
``reply_to_message_id`` (the raw pointer). The frontend's inline
card called `resolveReplyTarget(id)` which scans `messagesByIdRef`
— a Map that's only populated as days hydrate. On a chat with
replies pointing to messages in older days, the target wasn't in
the map yet, so the card rendered the stub until the older days
arrived.

Fix: every list serializer now bulk-resolves the targets and inlines
``reply_to: {id, role, content, created_at}`` per replying row. The
frontend reads it directly — no map scan, no flash.

This test exercises the helper used by all four serializer call
sites (``api/day_chats.py``, ``api/messages_recover.py``,
``api/sessions.py``, ``api/chat.py``) and verifies:

  1. Bulk fetch resolves all targets in a single round-trip.
  2. Cross-conversation targets are still resolved (lookup is global
     by id, mirroring the auth boundary in PR #58).
  3. Missing targets (deleted rows, stale ids) are dropped silently
     instead of crashing.
  4. Excerpt is capped at 240 chars so chatty days don't blow up the
     /api/day-chats payload.

Run:
  cd backend && env -u STRIPE_PRICE_ID_STARTER -u STRIPE_PRICE_ID_BUILDER \\
    -u STRIPE_PRICE_ID_PRO -u STRIPE_PRICE_ID_ELITE \\
    ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \\
    python tests/test_reply_target_denorm.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import date as Date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models.conversation import Conversation, Message
from app.db.models.day_chat import DayChat
from app.db.models.user import User

# Lazy import — pulls app.config indirectly, so do it inside the test
# bootstrap rather than at module-load time.
def _helper():
    import os
    os.environ.setdefault("ENVIRONMENT", "test")
    from app.agent.reply_quote import (
        resolve_reply_targets_for_serialization,
        serialize_reply_target,
        ReplyTarget,
        _INLINE_REPLY_EXCERPT_CHARS,
    )
    return (
        resolve_reply_targets_for_serialization,
        serialize_reply_target,
        ReplyTarget,
        _INLINE_REPLY_EXCERPT_CHARS,
    )


async def _make_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255),
                name VARCHAR(255), password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50),
                email_verified_at TIMESTAMP, email_verification_token VARCHAR(64),
                email_verification_sent_at TIMESTAMP, is_canary BOOLEAN DEFAULT 0
            )""",
            """CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP, summary_status VARCHAR(20) DEFAULT 'up_to_date',
                summary_failure_count INTEGER DEFAULT 0,
                summary_last_failure_at TIMESTAMP,
                summary_last_failure_reason VARCHAR(64),
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
                metadata_json TEXT, embedding_json TEXT, embedding BLOB,
                channel VARCHAR(50), source VARCHAR(50),
                reply_to_message_id VARCHAR(50), attachments TEXT
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


async def _seed(sm):
    user_id = "u-alice"
    dc_id = str(uuid.uuid4())
    conv_a_id = str(uuid.uuid4())
    conv_b_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(User(id=user_id, email="alice@test.local", hashed_password="x", name="Alice"))
        db.add(DayChat(
            id=dc_id, user_id=user_id, local_date=Date(2026, 5, 21),
            timezone="UTC",
            started_at=datetime(2026, 5, 21, 8, 0, 0),
            last_message_at=datetime(2026, 5, 21, 17, 0, 0),
            message_count=0, total_tokens=0,
            summary_status="up_to_date",
        ))
        db.add(Conversation(
            id=conv_a_id, user_id=user_id, channel="web", day_chat_id=dc_id,
            started_at=datetime(2026, 5, 21, 9, 0, 0),
            updated_at=datetime(2026, 5, 21, 10, 0, 0),
        ))
        db.add(Conversation(
            id=conv_b_id, user_id=user_id, channel="web", day_chat_id=dc_id,
            started_at=datetime(2026, 5, 21, 11, 0, 0),
            updated_at=datetime(2026, 5, 21, 12, 0, 0),
        ))
        await db.commit()
    return user_id, dc_id, conv_a_id, conv_b_id


async def _add_msg(sm, *, mid=None, conv_id, dc_id, role, content, when, reply_to=None):
    mid = mid or str(uuid.uuid4())
    async with sm() as db:
        db.add(Message(
            id=mid, conversation_id=conv_id, day_chat_id=dc_id,
            role=role, content=content, created_at=when,
            reply_to_message_id=reply_to,
        ))
        await db.commit()
    return mid


# ── Tests ─────────────────────────────────────────────────────────────


async def test_bulk_resolve_returns_target_per_replier():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id, conv_a, conv_b = await _seed(sm)
    resolve, _ser, _RT, _MAX = _helper()

    target_id = await _add_msg(
        sm, conv_id=conv_a, dc_id=dc_id, role="assistant",
        content="Original message about digital detox",
        when=datetime(2026, 5, 21, 9, 0, 0),
    )
    replier_id = await _add_msg(
        sm, conv_id=conv_b, dc_id=dc_id, role="user",
        content="can you see this?",
        when=datetime(2026, 5, 21, 11, 5, 0),
        reply_to=target_id,
    )

    async with sm() as db:
        from sqlalchemy import select
        msgs = (await db.execute(select(Message).where(Message.day_chat_id == dc_id))).scalars().all()
        out = await resolve(db, msgs)

    assert replier_id in out, f"replier message must have a resolved entry, got keys={list(out.keys())}"
    target = out[replier_id]
    assert target["id"] == target_id
    assert target["role"] == "assistant"
    assert "Original message about digital detox" in target["content"]
    assert target["created_at"] is not None
    await engine.dispose()


async def test_messages_without_reply_pointer_have_no_entry():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id, conv_a, _ = await _seed(sm)
    resolve, *_ = _helper()

    plain_id = await _add_msg(
        sm, conv_id=conv_a, dc_id=dc_id, role="user",
        content="hi", when=datetime(2026, 5, 21, 9, 5, 0),
    )

    async with sm() as db:
        from sqlalchemy import select
        msgs = (await db.execute(select(Message).where(Message.id == plain_id))).scalars().all()
        out = await resolve(db, msgs)

    assert out == {}, f"plain messages must produce no entries, got {out!r}"
    await engine.dispose()


async def test_deleted_target_skipped_silently():
    """A message with reply_to_message_id pointing at a now-deleted
    row must not crash the serializer — the entry is simply absent
    from the output. The frontend renders the neutral stub then."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id, conv_a, _ = await _seed(sm)
    resolve, *_ = _helper()

    replier_id = await _add_msg(
        sm, conv_id=conv_a, dc_id=dc_id, role="user",
        content="thanks!", when=datetime(2026, 5, 21, 9, 30, 0),
        reply_to="stale-id-no-row",
    )

    async with sm() as db:
        from sqlalchemy import select
        msgs = (await db.execute(select(Message).where(Message.id == replier_id))).scalars().all()
        out = await resolve(db, msgs)

    assert replier_id not in out, (
        f"deleted-target replies must produce no entry (frontend falls back "
        f"to neutral stub); got {out!r}"
    )
    await engine.dispose()


async def test_excerpt_is_capped_at_240_chars():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id, conv_a, conv_b = await _seed(sm)
    resolve, serialize, RT, MAX = _helper()
    assert MAX == 240

    huge = "X" * 5000
    target_id = await _add_msg(
        sm, conv_id=conv_a, dc_id=dc_id, role="assistant",
        content=huge, when=datetime(2026, 5, 21, 9, 0, 0),
    )
    replier_id = await _add_msg(
        sm, conv_id=conv_b, dc_id=dc_id, role="user",
        content="?", when=datetime(2026, 5, 21, 11, 5, 0),
        reply_to=target_id,
    )

    async with sm() as db:
        from sqlalchemy import select
        msgs = (await db.execute(select(Message).where(Message.id == replier_id))).scalars().all()
        out = await resolve(db, msgs)

    content = out[replier_id]["content"]
    assert len(content) <= MAX, (
        f"excerpt must be ≤{MAX} chars to keep day_chats payload bounded; "
        f"got {len(content)} chars"
    )
    assert content.endswith("…"), "truncated excerpt should end with ellipsis"
    await engine.dispose()


async def test_cross_conversation_target_resolves():
    """A reply that points across conversations resolves — the helper
    looks up by id alone, not by conversation_id. Matches the auth
    broadening landed in PR #58."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id, conv_a, conv_b = await _seed(sm)
    resolve, *_ = _helper()

    target_id = await _add_msg(
        sm, conv_id=conv_a, dc_id=dc_id, role="assistant",
        content="From conversation A",
        when=datetime(2026, 5, 21, 9, 0, 0),
    )
    replier_id = await _add_msg(
        sm, conv_id=conv_b, dc_id=dc_id, role="user",
        content="reply from conversation B",
        when=datetime(2026, 5, 21, 11, 0, 0),
        reply_to=target_id,
    )

    async with sm() as db:
        from sqlalchemy import select
        # Caller passes only the replier (mimics single-session endpoints
        # like /api/sessions/<id>/messages where conv A messages aren't
        # in the SELECT but the reply still needs resolving).
        msgs = (await db.execute(select(Message).where(Message.id == replier_id))).scalars().all()
        out = await resolve(db, msgs)

    assert replier_id in out, "cross-conversation reply target must resolve"
    assert "From conversation A" in out[replier_id]["content"]
    await engine.dispose()


async def test_empty_input_returns_empty_map():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    resolve, *_ = _helper()
    async with sm() as db:
        out = await resolve(db, [])
    assert out == {}
    await engine.dispose()


async def _main():
    tests = [
        ("bulk resolve returns target per replier", test_bulk_resolve_returns_target_per_replier),
        ("plain messages have no entry", test_messages_without_reply_pointer_have_no_entry),
        ("deleted target skipped silently", test_deleted_target_skipped_silently),
        ("excerpt capped at 240 chars", test_excerpt_is_capped_at_240_chars),
        ("cross-conversation target resolves", test_cross_conversation_target_resolves),
        ("empty input → empty map", test_empty_input_returns_empty_map),
    ]
    failures = []
    for name, fn in tests:
        try:
            await fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")


if __name__ == "__main__":
    asyncio.run(_main())
