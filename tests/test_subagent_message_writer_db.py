"""write_subagent_message — Phase 9 DB-end-to-end coverage.

Phase 4 deferred the writer's DB-end-to-end tests because the
``messages`` table has a pgvector column (``Vector()``) that
``init_db`` skips when pgvector is unavailable (SQLite). Phase 9
follows the ``test_backfill_day_chats.py`` precedent: build our
own in-memory SQLite engine and CREATE the schema manually with
pgvector-free column types. The writer code runs unchanged
against this real DB schema.

What we pin:
  - Message row written with channel='subagent', source='subagent',
    role='assistant'.
  - Conversation row created with channel='subagent'.
  - Message.metadata_json round-trips subagent_job_id, parent_job_id,
    label, outcome, credit_spent.
  - Reading-A: two writes for the same user same day collapse to
    ONE conversation row.
  - DayChat counters bump (message_count, last_message_at).
  - Token totals roll up into DayChat.total_tokens.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# Override conftest's autouse init_db — this test file owns its
# engine + schema and doesn't want conftest's app-wide init clobbering
# its setup. Mirrors the pattern in test_migration_*.py.
#
# Also clears the TKT-LAT-011 process-wide day_chat cache so a stale
# entry from a prior test (e.g., one that ran against init_db's main
# engine and populated _CACHE with a (user_id, local_date) → id
# mapping for a row that doesn't exist in our manual-schema DB)
# doesn't make get_or_create_day_chat return a phantom id.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    from app.agent import _day_chat_cache
    _day_chat_cache._CACHE.clear()
    yield
    _day_chat_cache._CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Manual schema fixture — mirrors test_backfill_day_chats.py pattern
# ──────────────────────────────────────────────────────────────────────


_USERS_DDL = """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(36) PRIMARY KEY,
        email VARCHAR(255) UNIQUE,
        hashed_password VARCHAR(255),
        name VARCHAR(255),
        password_changed_at TIMESTAMP,
        role VARCHAR(20) DEFAULT 'beta_user',
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        stripe_customer_id VARCHAR(255),
        timezone VARCHAR(50),
        notification_preferences TEXT,
        is_canary BOOLEAN NOT NULL DEFAULT 0,
        email_verified_at TIMESTAMP,
        email_verification_token VARCHAR(64),
        email_verification_sent_at TIMESTAMP
    )
"""
_DAY_CHATS_DDL = """
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
        summary_failure_count INTEGER NOT NULL DEFAULT 0,
        summary_last_failure_at TIMESTAMP,
        summary_last_failure_reason VARCHAR(60),
        archival_summary TEXT,
        archival_summary_generated_at TIMESTAMP,
        archival_summary_status VARCHAR(20) DEFAULT 'not_needed',
        UNIQUE(user_id, local_date)
    )
"""
_CONVERSATIONS_DDL = """
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
"""
_MESSAGES_DDL = """
    CREATE TABLE IF NOT EXISTS messages (
        id VARCHAR(50) PRIMARY KEY,
        conversation_id VARCHAR(36) REFERENCES conversations(id),
        day_chat_id VARCHAR(36) REFERENCES day_chats(id),
        channel VARCHAR(50),
        source VARCHAR(50),
        role VARCHAR(20),
        content TEXT,
        created_at TIMESTAMP,
        tokens_prompt INTEGER,
        tokens_completion INTEGER,
        model_used VARCHAR(50),
        memories_retrieved_json TEXT,
        processing_time_ms INTEGER,
        reply_to_message_id VARCHAR(50),
        metadata_json TEXT,
        embedding_json TEXT,
        embedding BLOB
    )
"""


@pytest_asyncio.fixture
async def db_with_manual_schema(monkeypatch):
    """Yields an AsyncSession against an in-memory SQLite DB with a
    hand-crafted schema that does NOT include the pgvector column.
    The writer code runs against this just like it would in
    production Postgres."""
    # NEW: use a shared in-memory DB so the writer (which opens its
    # own session) sees the same tables.
    engine = create_async_engine(
        "sqlite+aiosqlite:///file:phase9?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False, "uri": True},
    )

    async with engine.begin() as conn:
        for ddl in (
            _USERS_DDL, _DAY_CHATS_DDL, _CONVERSATIONS_DDL, _MESSAGES_DDL,
        ):
            await conn.execute(sa.text(ddl))

    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    # Patch the app's async_session_maker so any internal helpers
    # (resolve_day_chat_id_for_now, resolve_or_create_day_conversation)
    # that open their own sessions land on OUR engine instead of the
    # global one.
    import app.db.database as db_database
    monkeypatch.setattr(db_database, "engine", engine)
    monkeypatch.setattr(db_database, "async_session_maker", session_maker)
    import app.db as app_db
    monkeypatch.setattr(app_db, "async_session_maker", session_maker)

    async with session_maker() as s:
        yield s

    await engine.dispose()


async def _seed_user(db: AsyncSession, user_id: str) -> str:
    """Insert a user + today's day_chat. Returns the day_chat_id so
    tests can pin it.

    Pre-creating the day_chat sidesteps SQLite's lack of
    ON CONFLICT DO NOTHING (the resolver's create path uses
    pg_insert; on SQLite it falls through to a plain ORM add which
    requires every column the model declares to be present in our
    DDL). The fast-path SELECT in get_or_create_day_chat finds the
    pre-created row without needing the create path at all."""
    from datetime import date, timezone as _tz

    await db.execute(sa.text(
        "INSERT INTO users (id, email, hashed_password, name, role, "
        " created_at, updated_at, is_active, timezone) "
        "VALUES (:id, :email, 'x', 'Test', 'beta_user', :now, :now, 1, 'UTC')"
    ), {
        "id": user_id,
        "email": f"{user_id[:8]}@test.local",
        "now": datetime.utcnow(),
    })
    day_chat_id = str(uuid.uuid4())
    await db.execute(sa.text(
        "INSERT INTO day_chats "
        "(id, user_id, local_date, timezone, started_at, last_message_at, "
        " message_count, total_tokens, summary_status, summary_failure_count) "
        "VALUES (:id, :uid, :ld, 'UTC', :now, :now, 0, 0, 'up_to_date', 0)"
    ), {
        "id": day_chat_id,
        "uid": user_id,
        "ld": datetime.now(_tz.utc).date(),
        "now": datetime.utcnow(),
    })
    await db.commit()
    return day_chat_id


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_persists_message_row_with_correct_provenance(
    db_with_manual_schema,
):
    """The core happy path against a REAL Message INSERT — pinned
    channel/source/role + the metadata_json shape the frontend chat
    parser reads."""
    from app.agent.subagent_message_writer import write_subagent_message

    uid = str(uuid.uuid4())
    await _seed_user(db_with_manual_schema, uid)

    msg_id, day_chat_id = await write_subagent_message(
        db_with_manual_schema,
        user_id=uid,
        content="The answer is 42.",
        job_id="job-abc",
        parent_job_id="parent-1",
        label="research X",
        outcome="success",
        credit_spent=0.0123,
        tokens_prompt=100,
        tokens_completion=50,
    )

    assert msg_id, "writer must return a message_id"
    assert day_chat_id, "writer must return a day_chat_id"

    row = (
        await db_with_manual_schema.execute(sa.text(
            "SELECT channel, source, role, content, metadata_json, "
            "       tokens_prompt, tokens_completion "
            "FROM messages WHERE id = :mid"
        ), {"mid": msg_id})
    ).first()
    assert row is not None, "the writer must persist the row"
    assert row.channel == "subagent"
    assert row.source == "subagent"
    assert row.role == "assistant"
    assert row.content == "The answer is 42."
    assert row.tokens_prompt == 100
    assert row.tokens_completion == 50

    meta = json.loads(row.metadata_json)
    assert meta["subagent_job_id"] == "job-abc"
    assert meta["subagent_parent_job_id"] == "parent-1"
    assert meta["subagent_label"] == "research X"
    assert meta["outcome"] == "success"
    assert meta["credit_spent"] == 0.0123


@pytest.mark.asyncio
async def test_write_creates_conversation_with_subagent_channel(
    db_with_manual_schema,
):
    """A fresh write must mint a Conversation with channel='subagent'."""
    from app.agent.subagent_message_writer import write_subagent_message

    uid = str(uuid.uuid4())
    await _seed_user(db_with_manual_schema, uid)
    msg_id, _ = await write_subagent_message(
        db_with_manual_schema,
        user_id=uid, content="x", job_id="j-1", label="L",
    )

    row = (
        await db_with_manual_schema.execute(sa.text(
            "SELECT c.channel FROM conversations c "
            "JOIN messages m ON m.conversation_id = c.id "
            "WHERE m.id = :mid"
        ), {"mid": msg_id})
    ).first()
    assert row is not None
    assert row.channel == "subagent"


@pytest.mark.asyncio
async def test_reading_a_two_writes_share_one_conversation(
    db_with_manual_schema,
):
    """Reading-A invariant: one Conversation per
    (user, day_chat, channel='subagent'). Two sub-agent results on
    the same day for the same user collapse to ONE conversation
    row — matches the routine/trigger pattern."""
    from app.agent.subagent_message_writer import write_subagent_message

    uid = str(uuid.uuid4())
    await _seed_user(db_with_manual_schema, uid)

    m1, _ = await write_subagent_message(
        db_with_manual_schema, user_id=uid, content="first", job_id="j-1",
    )
    m2, _ = await write_subagent_message(
        db_with_manual_schema, user_id=uid, content="second", job_id="j-2",
    )

    rows = list((
        await db_with_manual_schema.execute(sa.text(
            "SELECT conversation_id FROM messages "
            "WHERE id IN (:m1, :m2)"
        ), {"m1": m1, "m2": m2})
    ).fetchall())
    convo_ids = {r.conversation_id for r in rows}
    assert len(convo_ids) == 1, (
        f"two same-day sub-agent writes must share one Conversation "
        f"row (Reading-A); got {convo_ids}"
    )

    # And the conversation should reflect a count of 2 messages.
    n_messages = (
        await db_with_manual_schema.execute(sa.text(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = :cid"
        ), {"cid": next(iter(convo_ids))})
    ).scalar_one()
    assert n_messages == 2


@pytest.mark.asyncio
async def test_day_chat_counters_bump(db_with_manual_schema):
    """Two writes → DayChat.message_count grows by 2;
    DayChat.total_tokens accumulates; DayChat.last_message_at
    updates."""
    from app.agent.subagent_message_writer import write_subagent_message

    uid = str(uuid.uuid4())
    await _seed_user(db_with_manual_schema, uid)

    _, dcid = await write_subagent_message(
        db_with_manual_schema, user_id=uid, content="a", job_id="j-1",
        tokens_prompt=10, tokens_completion=20,
    )
    assert dcid, "day_chat resolution must produce an id"

    _, _ = await write_subagent_message(
        db_with_manual_schema, user_id=uid, content="b", job_id="j-2",
        tokens_prompt=5, tokens_completion=5,
    )

    row = (
        await db_with_manual_schema.execute(sa.text(
            "SELECT message_count, total_tokens, last_message_at "
            "FROM day_chats WHERE id = :dcid"
        ), {"dcid": dcid})
    ).first()
    assert row is not None
    assert row.message_count == 2
    # 30 + 10 = 40
    assert row.total_tokens == 40
    assert row.last_message_at is not None
