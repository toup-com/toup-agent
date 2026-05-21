"""Regression tests for reply-context propagation into the agent prompt.

Covers two paths and four scenarios per path:

1. ``day_context_loader.load_day_context`` — active when day-chat
   context is enabled.
2. ``AgentRunner._load_history`` — fallback used when day-chat is off
   OR when ``load_day_context`` raises. Before this test landed, the
   fallback dropped every historical reply-to (Bug A).

Scenarios:
  - same-session reply  (target in same conversation_id)
  - cross-session reply (target in a different conversation_id, same user)
  - cross-channel reply (target's channel != replier's channel)
  - deleted/unreachable target → no crash, replier still returned

Run: ``cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \\
        python tests/test_reply_history.py``
"""

from __future__ import annotations

import asyncio
import importlib.util as _ilu
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

_dcl_spec = _ilu.spec_from_file_location(
    "day_context_loader",
    str(Path(__file__).resolve().parent.parent / "app" / "agent" / "day_context_loader.py"),
)
_dcl_mod = _ilu.module_from_spec(_dcl_spec)
_dcl_spec.loader.exec_module(_dcl_mod)
load_day_context = _dcl_mod.load_day_context


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


_USER_ID = "user-reply-test"
_TODAY = Date(2026, 5, 21)


async def _seed_user_and_day(sm):
    dc_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(
            User(
                id=_USER_ID,
                email=f"{_USER_ID}@test.local",
                hashed_password="x",
                name="Reply Tester",
            )
        )
        db.add(
            DayChat(
                id=dc_id,
                user_id=_USER_ID,
                local_date=_TODAY,
                timezone="UTC",
                started_at=datetime(2026, 5, 21, 8, 0, 0),
                last_message_at=datetime(2026, 5, 21, 17, 0, 0),
                message_count=0,
                total_tokens=0,
                summary_status="up_to_date",
            )
        )
        await db.commit()
    return dc_id


async def _add_conversation(sm, dc_id, channel="web"):
    conv_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(
            Conversation(
                id=conv_id,
                user_id=_USER_ID,
                channel=channel,
                day_chat_id=dc_id,
                started_at=datetime(2026, 5, 21, 9, 0, 0),
                updated_at=datetime(2026, 5, 21, 10, 0, 0),
            )
        )
        await db.commit()
    return conv_id


async def _add_message(
    sm,
    *,
    conv_id,
    dc_id,
    role,
    content,
    when,
    channel="web",
    reply_to=None,
    msg_id=None,
):
    msg_id = msg_id or str(uuid.uuid4())
    async with sm() as db:
        db.add(
            Message(
                id=msg_id,
                conversation_id=conv_id,
                day_chat_id=dc_id,
                role=role,
                content=content,
                created_at=when,
                channel=channel,
                reply_to_message_id=reply_to,
            )
        )
        await db.commit()
    return msg_id


def _load_history_fn():
    import os as _os
    _os.environ.setdefault("ENVIRONMENT", "test")
    from app.agent.agent_runner import AgentRunner
    return AgentRunner._load_history


class _DummyRunner:
    """`_load_history` doesn't touch instance state — a dummy is fine
    to bind the unbound method against."""

    async def load(self, db, session_id, client_tz=None):
        return await _load_history_fn()(self, db, session_id, client_tz=client_tz)


async def _run_same_session_reply(loader_kind: str):
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dc_id = await _seed_user_and_day(sm)
    conv = await _add_conversation(sm, dc_id, "web")
    target_id = await _add_message(
        sm, conv_id=conv, dc_id=dc_id, role="assistant",
        content="The capital of France is Paris.",
        when=datetime(2026, 5, 21, 9, 0, 0),
    )
    await _add_message(
        sm, conv_id=conv, dc_id=dc_id, role="user",
        content="and the population?",
        when=datetime(2026, 5, 21, 9, 1, 0),
        reply_to=target_id,
    )
    async with sm() as db:
        if loader_kind == "day":
            msgs = (await load_day_context(db, dc_id, model_context_tokens=200_000))["messages"]
        else:
            msgs = await _DummyRunner().load(db, conv, client_tz=None)
    joined = "\n---\n".join(m["content"] for m in msgs)
    assert "<reply_to>" in joined, f"[{loader_kind}] preamble missing"
    assert "capital of France is Paris" in joined
    assert "and the population?" in joined
    await engine.dispose()


async def _run_cross_session_reply(loader_kind: str):
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dc_id = await _seed_user_and_day(sm)
    conv_old = await _add_conversation(sm, dc_id, "web")
    conv_new = await _add_conversation(sm, dc_id, "web")
    target_id = await _add_message(
        sm, conv_id=conv_old, dc_id=dc_id, role="assistant",
        content="I'll fire off `hi jojo` once you say sent.",
        when=datetime(2026, 5, 21, 8, 30, 0),
    )
    await _add_message(
        sm, conv_id=conv_new, dc_id=dc_id, role="user",
        content="do you know what i replied?",
        when=datetime(2026, 5, 21, 11, 41, 0),
        reply_to=target_id,
    )
    async with sm() as db:
        if loader_kind == "day":
            msgs = (await load_day_context(db, dc_id, model_context_tokens=200_000))["messages"]
        else:
            msgs = await _DummyRunner().load(db, conv_new, client_tz=None)
    joined = "\n---\n".join(m["content"] for m in msgs)
    assert "<reply_to>" in joined, f"[{loader_kind}] cross-session preamble missing"
    assert "hi jojo" in joined, (
        f"[{loader_kind}] cross-session target content missing — lookup "
        f"must be global by id, not scoped to conversation_id"
    )
    await engine.dispose()


async def _run_cross_channel_reply(loader_kind: str):
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dc_id = await _seed_user_and_day(sm)
    conv_tg = await _add_conversation(sm, dc_id, "telegram")
    conv_web = await _add_conversation(sm, dc_id, "web")
    target_id = await _add_message(
        sm, conv_id=conv_tg, dc_id=dc_id, role="user",
        content="Remember the bakery I mentioned on Sunday?",
        when=datetime(2026, 5, 21, 8, 0, 0), channel="telegram",
    )
    await _add_message(
        sm, conv_id=conv_web, dc_id=dc_id, role="user",
        content="What was its name again?",
        when=datetime(2026, 5, 21, 12, 0, 0), channel="web",
        reply_to=target_id,
    )
    async with sm() as db:
        if loader_kind == "day":
            msgs = (await load_day_context(db, dc_id, model_context_tokens=200_000))["messages"]
        else:
            msgs = await _DummyRunner().load(db, conv_web, client_tz=None)
    joined = "\n---\n".join(m["content"] for m in msgs)
    assert "<reply_to>" in joined, f"[{loader_kind}] cross-channel preamble missing"
    assert "bakery" in joined
    await engine.dispose()


async def _run_deleted_target_reply(loader_kind: str):
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dc_id = await _seed_user_and_day(sm)
    conv = await _add_conversation(sm, dc_id, "web")
    await _add_message(
        sm, conv_id=conv, dc_id=dc_id, role="user",
        content="thanks for that earlier hint!",
        when=datetime(2026, 5, 21, 10, 0, 0),
        reply_to="deleted-msg-id-no-row",
    )
    async with sm() as db:
        if loader_kind == "day":
            msgs = (await load_day_context(db, dc_id, model_context_tokens=200_000))["messages"]
        else:
            msgs = await _DummyRunner().load(db, conv, client_tz=None)
    joined = "\n---\n".join(m["content"] for m in msgs)
    assert "thanks for that earlier hint" in joined, (
        f"[{loader_kind}] replier content lost when target missing"
    )
    await engine.dispose()


async def test_day_ctx_same_session_reply(): await _run_same_session_reply("day")
async def test_load_history_same_session_reply(): await _run_same_session_reply("history")
async def test_day_ctx_cross_session_reply(): await _run_cross_session_reply("day")
async def test_load_history_cross_session_reply(): await _run_cross_session_reply("history")
async def test_day_ctx_cross_channel_reply(): await _run_cross_channel_reply("day")
async def test_load_history_cross_channel_reply(): await _run_cross_channel_reply("history")
async def test_day_ctx_deleted_target_reply(): await _run_deleted_target_reply("day")
async def test_load_history_deleted_target_reply(): await _run_deleted_target_reply("history")


async def _main():
    tests = [
        ("day_ctx same-session", test_day_ctx_same_session_reply),
        ("load_history same-session", test_load_history_same_session_reply),
        ("day_ctx cross-session", test_day_ctx_cross_session_reply),
        ("load_history cross-session", test_load_history_cross_session_reply),
        ("day_ctx cross-channel", test_day_ctx_cross_channel_reply),
        ("load_history cross-channel", test_load_history_cross_channel_reply),
        ("day_ctx deleted-target", test_day_ctx_deleted_target_reply),
        ("load_history deleted-target", test_load_history_deleted_target_reply),
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
