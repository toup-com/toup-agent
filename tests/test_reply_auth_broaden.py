"""Regression test for the reply-target authorization broadening (Bug A
follow-up after PRs #53/#54 shipped).

Live symptom (toup.ai user report, 2026-05-21):
- User replies via the web Reply affordance on an agent message from
  a previous day (Radio video card from Apr 28). Today's day_chat.
- After send, the agent's response shows no awareness of the quoted
  digital-detox content.
- After page refresh, the violet "REPLYING TO AGENT" card above the
  user's bubble is gone — reply_to_message_id stayed NULL in DB.

Root cause: the pre-fix auth check (ws_chat.py:1370-1374) joined
through Conversation and required `Conversation.user_id == user_id`.
Some older / system-channel conversations (routine, trigger, the
Radio output path) have NULL or service-stamped user_id on the
Conversation row even though the message sits in the user's
day_chat. That tripped the silent drop at line 1387, so:
  - `reply_to_message_id` stayed None → not in INSERT → DB row had
    NULL → UI card vanishes on refresh.
  - `_reply_target_content` stayed None → no preamble prepended →
    agent saw a standalone "can you see this?" with no context.

Fix: ownership check now passes when EITHER Conversation.user_id OR
DayChat.user_id matches user_id. Day_chat is the canonical user
scope per conversation.py:21-33's Reading-A invariant.

This file tests the exact SQL fragment used by the fix to ensure:
  1. Cross-conversation reply within the user's day_chat resolves.
  2. Direct same-user Conversation ownership still resolves.
  3. Replies that DO belong to another user are correctly rejected
     (no IDOR opening).
  4. Stale ids (no row at all) return None as before.
  5. Service-stamped Conversation.user_id + user-owned day_chat
     resolves (the production case in the screenshot).

Run: cd backend && env -u STRIPE_PRICE_ID_STARTER -u STRIPE_PRICE_ID_BUILDER \\
    -u STRIPE_PRICE_ID_PRO -u STRIPE_PRICE_ID_ELITE \\
    ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \\
    python tests/test_reply_auth_broaden.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import date as Date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models.conversation import Conversation, Message
from app.db.models.day_chat import DayChat
from app.db.models.user import User


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


def _auth_query(candidate_id: str, user_id: str):
    """The exact SQL fragment the ws_chat.py auth path uses post-fix."""
    return (
        select(
            Message.id,
            Message.role,
            Message.content,
            Message.created_at,
            Conversation.user_id.label("conv_user_id"),
            DayChat.user_id.label("dc_user_id"),
        )
        .select_from(Message)
        .outerjoin(Conversation, Message.conversation_id == Conversation.id)
        .outerjoin(DayChat, Message.day_chat_id == DayChat.id)
        .where(Message.id == candidate_id)
    )


async def _is_authorized(db: AsyncSession, candidate_id: str, user_id: str) -> bool:
    """Mirrors the post-fix gate: row exists AND (conv OR day_chat owns user)."""
    row = (await db.execute(_auth_query(candidate_id, user_id))).first()
    if row is None:
        return False
    return row.conv_user_id == user_id or row.dc_user_id == user_id


# ── Fixture helpers ────────────────────────────────────────────────────


async def _seed(sm):
    """Seed two users + a day_chat for each user + relevant conversations.

    User A is the human ("u-alice"). User B is a service identity
    ("u-service") that historically owned routine/trigger/radio
    Conversations. Both have day_chats. The Apr 28 message under test
    will live in a Conversation stamped with B's user_id but a DayChat
    stamped with A's user_id — the production failure shape.
    """
    USER_A = "u-alice"
    USER_B = "u-service"
    dc_a = str(uuid.uuid4())
    dc_b = str(uuid.uuid4())

    async with sm() as db:
        for uid, name in [(USER_A, "Alice"), (USER_B, "ServiceBot")]:
            db.add(User(
                id=uid, email=f"{uid}@test.local",
                hashed_password="x", name=name,
            ))
        db.add(DayChat(
            id=dc_a, user_id=USER_A, local_date=Date(2026, 4, 28),
            timezone="UTC",
            started_at=datetime(2026, 4, 28, 8, 0, 0),
            last_message_at=datetime(2026, 4, 28, 9, 0, 0),
            message_count=0, total_tokens=0,
            summary_status="up_to_date",
        ))
        db.add(DayChat(
            id=dc_b, user_id=USER_B, local_date=Date(2026, 4, 28),
            timezone="UTC",
            started_at=datetime(2026, 4, 28, 8, 0, 0),
            last_message_at=datetime(2026, 4, 28, 9, 0, 0),
            message_count=0, total_tokens=0,
            summary_status="up_to_date",
        ))
        await db.commit()

    return USER_A, USER_B, dc_a, dc_b


async def _make_conversation(sm, *, user_id, dc_id, channel="web"):
    conv_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(Conversation(
            id=conv_id, user_id=user_id, channel=channel,
            day_chat_id=dc_id,
            started_at=datetime(2026, 4, 28, 8, 0, 0),
            updated_at=datetime(2026, 4, 28, 9, 0, 0),
        ))
        await db.commit()
    return conv_id


async def _make_message(sm, *, conv_id, dc_id, role, content, when):
    mid = str(uuid.uuid4())
    async with sm() as db:
        db.add(Message(
            id=mid, conversation_id=conv_id, day_chat_id=dc_id,
            role=role, content=content, created_at=when,
        ))
        await db.commit()
    return mid


# ── Tests ─────────────────────────────────────────────────────────────


async def test_service_conv_user_owned_day_chat_authorizes():
    """The production case: target message lives in a Conversation
    stamped with a service identity (e.g. radio output path) but its
    day_chat belongs to the human user. PRE-FIX this returned no row.
    POST-FIX day_chat ownership rescues the auth.
    """
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    USER_A, USER_B, dc_a, _dc_b = await _seed(sm)

    # Service-stamped conversation but message bucketed into Alice's day_chat.
    conv_service = await _make_conversation(sm, user_id=USER_B, dc_id=dc_a, channel="radio")
    target_id = await _make_message(
        sm, conv_id=conv_service, dc_id=dc_a, role="assistant",
        content="Digital detox video is now playing for you!",
        when=datetime(2026, 4, 28, 14, 16, 0),
    )

    async with sm() as db:
        assert await _is_authorized(db, target_id, USER_A) is True, (
            "Alice must be allowed to reply to a message in her own day_chat "
            "even if the conversation is service-stamped"
        )
    await engine.dispose()


async def test_same_user_conversation_still_authorizes():
    """Sanity: the original code path (Conversation.user_id == user_id)
    still works after the loosening."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    USER_A, _USER_B, dc_a, _dc_b = await _seed(sm)

    conv = await _make_conversation(sm, user_id=USER_A, dc_id=dc_a, channel="web")
    target_id = await _make_message(
        sm, conv_id=conv, dc_id=dc_a, role="assistant",
        content="Standard same-user reply target",
        when=datetime(2026, 4, 28, 10, 0, 0),
    )

    async with sm() as db:
        assert await _is_authorized(db, target_id, USER_A) is True
    await engine.dispose()


async def test_other_user_message_is_rejected():
    """IDOR guard: a message that genuinely belongs to a different
    user (different conversation AND different day_chat) must STILL
    be rejected. The loosening must not open a cross-account leak.
    """
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    USER_A, USER_B, _dc_a, dc_b = await _seed(sm)

    conv_b = await _make_conversation(sm, user_id=USER_B, dc_id=dc_b, channel="web")
    target_id = await _make_message(
        sm, conv_id=conv_b, dc_id=dc_b, role="user",
        content="ServiceBot's private content",
        when=datetime(2026, 4, 28, 11, 0, 0),
    )

    async with sm() as db:
        assert await _is_authorized(db, target_id, USER_A) is False, (
            "Alice must NOT be able to reply to ServiceBot's owned message"
        )
    await engine.dispose()


async def test_missing_target_returns_none():
    """Stale frontend id: no row exists. Auth must return False, not
    crash, and the calling code's WARNING log surfaces the case."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    USER_A, _USER_B, _dc_a, _dc_b = await _seed(sm)

    async with sm() as db:
        assert await _is_authorized(db, "non-existent-msg-id", USER_A) is False
    await engine.dispose()


async def test_null_conv_user_with_owned_day_chat_authorizes():
    """Edge case: Conversation.user_id is literally NULL (legacy row
    before the user_id stamp was tightened). Day_chat ownership must
    still rescue auth."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    USER_A, _USER_B, dc_a, _dc_b = await _seed(sm)

    # Raw insert because the ORM declares user_id non-null; older rows
    # in prod predate that constraint.
    conv_null = str(uuid.uuid4())
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO conversations (id, user_id, channel, day_chat_id, "
            "started_at, updated_at, is_active, message_count, total_tokens) "
            "VALUES (:id, NULL, 'routine', :dc, :ts, :ts, 1, 0, 0)"
        ), {
            "id": conv_null, "dc": dc_a,
            "ts": datetime(2026, 4, 28, 8, 0, 0),
        })
        await db.commit()

    target_id = await _make_message(
        sm, conv_id=conv_null, dc_id=dc_a, role="assistant",
        content="Legacy routine output with null conv user_id",
        when=datetime(2026, 4, 28, 12, 0, 0),
    )

    async with sm() as db:
        assert await _is_authorized(db, target_id, USER_A) is True, (
            "Legacy NULL-user_id conversation must still authorize via day_chat"
        )
    await engine.dispose()


async def _main():
    tests = [
        ("service-conv + user-owned day_chat", test_service_conv_user_owned_day_chat_authorizes),
        ("same-user conv (regression)", test_same_user_conversation_still_authorizes),
        ("other user's message rejected (IDOR guard)", test_other_user_message_is_rejected),
        ("missing target returns False", test_missing_target_returns_none),
        ("null conv user_id + user day_chat", test_null_conv_user_with_owned_day_chat_authorizes),
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
