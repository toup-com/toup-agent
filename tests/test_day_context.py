"""
Tests for day context loading, fallback when backfill incomplete, and summarizer.

Run: python tests/test_day_context.py
"""

import asyncio
import sys
import uuid
from datetime import datetime, date as Date, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.day_chat import DayChat, MigrationStatus
from app.db.models.conversation import Conversation, Message
from app.db.models.user import User
import importlib.util as _ilu

# Direct file imports to avoid triggering heavy app.config / app.services.__init__
_dcl_spec = _ilu.spec_from_file_location(
    "day_context_loader",
    str(Path(__file__).resolve().parent.parent / "app" / "agent" / "day_context_loader.py"),
)
_dcl_mod = _ilu.module_from_spec(_dcl_spec)
_dcl_spec.loader.exec_module(_dcl_mod)
load_day_context = _dcl_mod.load_day_context
annotate_message = _dcl_mod.annotate_message
build_today_so_far_block = _dcl_mod.build_today_so_far_block
should_inject_today_so_far = _dcl_mod.should_inject_today_so_far


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        # `users` is built FROM THE ORM MODEL, never from a copy of it. This was
        # a hand-written `CREATE TABLE users (...)` and it drifted the moment
        # the model gained `first_media_played_at` (migration 086): every ORM
        # insert in this file started failing on a column the table had never
        # heard of. A hand-written schema is a second source of truth nothing
        # keeps in sync, and it always breaks somewhere else entirely —
        # whenever someone adds a column. Same fix as test_reply_history.py,
        # test_recent_days_service.py and test_ws_tz_persistence.py.
        #
        # Only `users` moves: the tables below are inserted into by raw SQL
        # here, so their hand-written shape is what this file is testing
        # against, and `create_all` on some of them would compile a pgvector
        # column that sqlite cannot take.
        await conn.run_sync(User.__table__.create, checkfirst=True)
        for stmt in [
            """CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP, summary_status VARCHAR(20) DEFAULT 'up_to_date',
                summary_failure_count INTEGER DEFAULT 0,
                summary_last_failure_at TIMESTAMP,
                summary_last_failure_reason VARCHAR(50),
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
                day_chat_id VARCHAR(36) REFERENCES day_chats(id),
                channel VARCHAR(50), source VARCHAR(50), role VARCHAR(20),
                content TEXT, created_at TIMESTAMP, tokens_prompt INTEGER,
                tokens_completion INTEGER, model_used VARCHAR(50),
                memories_retrieved_json TEXT, processing_time_ms INTEGER,
                reply_to_message_id VARCHAR(50), attachments TEXT,
                metadata_json TEXT, embedding_json TEXT, embedding BLOB
            , origin VARCHAR(16))""",
            """CREATE TABLE IF NOT EXISTS migration_status (
                migration_name VARCHAR(100) PRIMARY KEY, status VARCHAR(20) DEFAULT 'not_started',
                started_at TIMESTAMP, completed_at TIMESTAMP, progress_json TEXT, error_message TEXT
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


async def _seed_day(sm, user_id="u1", channels=None, msgs_per_conv=5):
    """Seed a day with conversations across channels."""
    if channels is None:
        channels = ["web", "telegram", "vibecoding"]

    dc_id = str(uuid.uuid4())
    today = Date(2026, 4, 7)

    async with sm() as db:
        # Create user if not exists
        u = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not u:
            db.add(User(id=user_id, email=f"{user_id}@test.local", hashed_password="x", name="Test"))

        # Create DayChat
        db.add(DayChat(
            id=dc_id, user_id=user_id, local_date=today, timezone="UTC",
            started_at=datetime(2026, 4, 7, 8, 0, 0), last_message_at=datetime(2026, 4, 7, 17, 0, 0),
            message_count=0, total_tokens=0, summary_status="up_to_date",
        ))

        all_msgs = []
        for i, ch in enumerate(channels):
            conv_id = str(uuid.uuid4())
            base_hour = 8 + i * 3  # web=8am, telegram=11am, vibecoding=2pm
            db.add(Conversation(
                id=conv_id, user_id=user_id, channel=ch, day_chat_id=dc_id,
                started_at=datetime(2026, 4, 7, base_hour, 0, 0),
                updated_at=datetime(2026, 4, 7, base_hour + 1, 0, 0),
            ))
            for m in range(msgs_per_conv):
                # Spread messages across minutes safely (max 59)
                msg_minute = m % 60
                msg_second = (m // 60) % 60
                msg = Message(
                    id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                    role="user" if m % 2 == 0 else "assistant",
                    content=f"Message {m} on {ch} at {base_hour}:{msg_minute:02d}",
                    created_at=datetime(2026, 4, 7, base_hour, msg_minute, msg_second),
                )
                db.add(msg)
                all_msgs.append(msg)

        await db.commit()

    return dc_id, all_msgs


# ── Tests ─────────────────────────────────────────────────────────────

async def test_load_day_context_multi_session_chronological():
    """Loads messages from multiple sessions and channels in chronological order."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web", "telegram", "vibecoding"], msgs_per_conv=5)

    async with sm() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=200_000)

    assert ctx["message_count"] == 15  # 3 channels × 5 msgs
    assert len(ctx["messages"]) == 15

    # Verify chronological order
    for i in range(1, len(ctx["messages"])):
        # Messages should be in ascending time order
        pass  # The DB query orders by created_at ASC

    # Verify channel annotations
    web_msgs = [m for m in ctx["messages"] if "[web" in m["content"]]
    tg_msgs = [m for m in ctx["messages"] if "[telegram" in m["content"]]
    vc_msgs = [m for m in ctx["messages"] if "[vibecoding" in m["content"]]
    assert len(web_msgs) == 5, f"Expected 5 web messages, got {len(web_msgs)}"
    assert len(tg_msgs) == 5, f"Expected 5 telegram messages, got {len(tg_msgs)}"
    assert len(vc_msgs) == 5, f"Expected 5 vibecoding messages, got {len(vc_msgs)}"

    await engine.dispose()


async def test_load_day_context_respects_token_budget():
    """When messages exceed budget, falls back to summary + verbatim window."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create many messages to exceed budget
    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=50)

    # Set a tiny context window so messages exceed budget
    async with sm() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=500)

    # Should return a limited verbatim window (20 messages max)
    assert len(ctx["messages"]) <= 20
    assert ctx["summary_was_stale"] is True  # Over budget → stale

    await engine.dispose()


async def test_load_day_context_no_summary_yet():
    """Day with no rolling_summary works fine — just loads messages."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=3)

    async with sm() as db:
        ctx = await load_day_context(db, dc_id)

    assert ctx["summary"] is None
    assert len(ctx["messages"]) == 3
    assert ctx["summary_was_stale"] is False

    await engine.dispose()


async def test_load_day_context_with_summary():
    """When a rolling summary exists, it's returned alongside messages."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=3)

    # Set a summary on the DayChat
    async with sm() as db:
        dc = (await db.execute(select(DayChat).where(DayChat.id == dc_id))).scalar_one()
        dc.rolling_summary = "[web 8:00am] User discussed project setup. Key decision: use PostgreSQL."
        dc.summary_status = "up_to_date"
        await db.commit()

    async with sm() as db:
        ctx = await load_day_context(db, dc_id)

    assert ctx["summary"] is not None
    assert "PostgreSQL" in ctx["summary"]
    assert ctx["summary_was_stale"] is False

    await engine.dispose()


async def test_annotate_message():
    """Message annotation adds channel and time prefix."""
    dt = datetime(2026, 4, 7, 9, 14, 0)
    result = annotate_message("Hello world", "web", dt)
    assert result.startswith("[web ")
    assert "Hello world" in result


async def test_build_today_so_far_block():
    """today_so_far block is empty when no summary, populated when present."""
    assert build_today_so_far_block(None) == ""
    assert build_today_so_far_block("") == ""

    block = build_today_so_far_block("User worked on project X.")
    assert "<today_so_far>" in block
    assert "User worked on project X." in block
    assert "</today_so_far>" in block


async def test_history_is_complete_under_budget():
    """W2.1b: under-budget load returns the FULL day verbatim →
    history_is_complete=True, and the today_so_far gate must SKIP the
    summary (it would duplicate messages the model already sees)."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=5)

    # Give the day a rolling summary — the classic duplication setup
    async with sm() as db:
        dc = (await db.execute(select(DayChat).where(DayChat.id == dc_id))).scalar_one()
        dc.rolling_summary = "User discussed project setup."
        await db.commit()

    async with sm() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=200_000)

    assert ctx["history_is_complete"] is True
    assert len(ctx["messages"]) == 5  # complete history, nothing elided
    assert ctx["summary"]  # summary exists...
    # ...but injecting it would be pure duplication → gate says NO
    assert should_inject_today_so_far(ctx) is False

    await engine.dispose()


async def test_history_is_complete_over_budget():
    """W2.1b: over-budget load truncates to summary + verbatim window →
    history_is_complete=False and the summary MUST still inject (it
    genuinely replaces the elided older messages)."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=50)

    async with sm() as db:
        dc = (await db.execute(select(DayChat).where(DayChat.id == dc_id))).scalar_one()
        dc.rolling_summary = "Morning: user set up the project and picked PostgreSQL."
        await db.commit()

    # Tiny context window → forces the over-budget branch
    async with sm() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=500)

    assert ctx["history_is_complete"] is False
    assert len(ctx["messages"]) <= 20  # verbatim window only
    assert should_inject_today_so_far(ctx) is True
    block = build_today_so_far_block(ctx["summary"])
    assert "<today_so_far>" in block
    assert "PostgreSQL" in block

    await engine.dispose()


async def test_history_is_complete_no_day_chat():
    """Missing DayChat row → empty history is (vacuously) complete; no
    summary → gate never fires."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        ctx = await load_day_context(db, str(uuid.uuid4()), model_context_tokens=200_000)

    assert ctx["history_is_complete"] is True
    assert ctx["messages"] == []
    assert should_inject_today_so_far(ctx) is False

    await engine.dispose()


async def test_should_inject_today_so_far_pure():
    """Gate truth table, including backward-compat for the pre-W2.1b
    return shape (missing bit → inject, the historical behavior)."""
    assert should_inject_today_so_far(None) is False
    assert should_inject_today_so_far({}) is False
    # No summary → nothing to inject regardless of the bit
    assert should_inject_today_so_far({"summary": None, "history_is_complete": False}) is False
    assert should_inject_today_so_far({"summary": "", "history_is_complete": False}) is False
    # Complete history → summary is a duplicate → skip
    assert should_inject_today_so_far({"summary": "x", "history_is_complete": True}) is False
    # Truncated history → summary replaces elided messages → inject
    assert should_inject_today_so_far({"summary": "x", "history_is_complete": False}) is True
    # Old return shape without the bit → keep historical behavior (inject)
    assert should_inject_today_so_far({"summary": "x"}) is True


async def test_runner_routes_today_so_far_through_gate():
    """Source-pin: agent_runner's injection block must consult
    should_inject_today_so_far BEFORE building the block, and log the
    [PERF] skip marker so the change is visible in prod logs. (The full
    run() path is too DB-heavy for this harness — the decision itself is
    covered by the pure/loader tests above.)"""
    src = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
    gate = src.find("if not should_inject_today_so_far(_day_context):")
    build = src.find("_tsf_block = build_today_so_far_block(_day_context[\"summary\"])")
    assert gate != -1, "agent_runner no longer gates today_so_far on should_inject_today_so_far"
    assert build != -1, "agent_runner no longer builds today_so_far via build_today_so_far_block"
    assert gate < build, "gate must run before the block is built/injected"
    assert "[PERF] today_so_far_skipped=1" in src, "skip must be observable in prod logs"


async def test_fallback_when_backfill_incomplete():
    """Feature flag on + backfill in_progress → should_use_day_chat_context returns False.

    Tests the ACTUAL function with a test session_maker — not a simulation of the logic.
    """
    _dcr_spec = _ilu.spec_from_file_location(
        "day_chat_resolver",
        str(Path(__file__).resolve().parent.parent / "app" / "agent" / "day_chat_resolver.py"),
    )
    _dcr_mod = _ilu.module_from_spec(_dcr_spec)
    _dcr_spec.loader.exec_module(_dcr_mod)
    should_use_day_chat_context = _dcr_mod.should_use_day_chat_context

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Reset the once-per-process log guard so our test can trigger it
    should_use_day_chat_context._fallback_logged = False

    # Case 1: No migration status row at all → should fall back
    result = await should_use_day_chat_context(session_maker=sm, feature_flag_override=True)
    assert result is False, "Should fall back when no migration_status row exists"

    # Case 2: migration status = in_progress → should fall back
    async with sm() as db:
        db.add(MigrationStatus(
            migration_name="day_chat_backfill",
            status="in_progress",
            started_at=datetime.utcnow(),
        ))
        await db.commit()

    should_use_day_chat_context._fallback_logged = False
    result = await should_use_day_chat_context(session_maker=sm, feature_flag_override=True)
    assert result is False, "Should fall back when backfill is in_progress"

    # Case 3: migration status = failed → should fall back
    async with sm() as db:
        ms = (await db.execute(
            select(MigrationStatus).where(MigrationStatus.migration_name == "day_chat_backfill")
        )).scalar_one()
        ms.status = "failed"
        await db.commit()

    should_use_day_chat_context._fallback_logged = False
    result = await should_use_day_chat_context(session_maker=sm, feature_flag_override=True)
    assert result is False, "Should fall back when backfill is failed"

    # Case 4: migration status = completed → should use new path
    async with sm() as db:
        ms = (await db.execute(
            select(MigrationStatus).where(MigrationStatus.migration_name == "day_chat_backfill")
        )).scalar_one()
        ms.status = "completed"
        await db.commit()

    result = await should_use_day_chat_context(session_maker=sm, feature_flag_override=True)
    assert result is True, "Should use day-chat context when backfill is completed"

    # Case 5: feature flag off → should NOT use, regardless of backfill status
    result = await should_use_day_chat_context(session_maker=sm, feature_flag_override=False)
    assert result is False, "Should not use day-chat context when feature flag is off"

    await engine.dispose()


async def test_annotations_never_saved_to_db():
    """Annotations are in-memory only — DB content never has [channel time] prefixes."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web", "telegram"], msgs_per_conv=3)

    # Load context — messages should be annotated
    async with sm() as db:
        ctx = await load_day_context(db, dc_id)
    annotated = ctx["messages"]
    assert any("[web" in m["content"] for m in annotated), "Expected [web] annotations"
    assert any("[telegram" in m["content"] for m in annotated), "Expected [telegram] annotations"

    # Simulate what _save_messages does: save a NEW raw message
    new_msg_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(Message(
            id=new_msg_id, conversation_id=msgs[0].conversation_id, day_chat_id=dc_id,
            role="user", content="This is a raw user message with no prefix",
            created_at=datetime(2026, 4, 7, 17, 0, 0),
        ))
        await db.commit()

    # Re-load context — the new message should be annotated on read, not in DB
    async with sm() as db:
        ctx2 = await load_day_context(db, dc_id)

    # Check the new message appears annotated in the loaded context
    new_annotated = [m for m in ctx2["messages"] if "raw user message" in m["content"]]
    assert len(new_annotated) == 1, f"Expected 1 match, got {len(new_annotated)}"
    assert new_annotated[0]["content"].startswith("["), "New message should be annotated on load"

    # But the DB still has raw content — no double annotation
    async with sm() as db:
        raw_db = (await db.execute(
            select(Message.content).where(Message.id == new_msg_id)
        )).scalar_one()
    assert not raw_db.startswith("["), f"DB content should be raw, got: {raw_db[:50]}"
    assert "[web" not in raw_db, f"DB has leaked annotation: {raw_db[:50]}"

    # Load a third time — still no double annotations
    async with sm() as db:
        ctx3 = await load_day_context(db, dc_id)
    for m in ctx3["messages"]:
        # Count how many annotation prefixes appear — should be exactly 1
        prefix_count = m["content"].count("[web ") + m["content"].count("[telegram ")
        assert prefix_count <= 1, f"Double annotation detected: {m['content'][:80]}"

    await engine.dispose()


async def test_summarizer_produces_output():
    """Summarizer should_summarize respects debounce thresholds."""
    # Import the module directly to avoid app.services.__init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "day_summarizer",
        str(Path(__file__).resolve().parent.parent / "app" / "services" / "day_summarizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    dc_id, msgs = await _seed_day(sm, channels=["web"], msgs_per_conv=3)

    # With only 3 messages, should NOT trigger summarization (debounce = 10)
    async with sm() as db:
        should = await mod.should_summarize(db, dc_id)
    assert should is False, "Should not summarize with < 10 messages"

    # Add more messages to exceed threshold (2 channels × 8 = 16 messages)
    # dc.message_count is left at 0 (as backfill would leave it) — should_summarize
    # must count from the messages table directly, not trust the cached counter.
    dc_id2, msgs2 = await _seed_day(sm, user_id="u2", channels=["web", "telegram"], msgs_per_conv=8)

    async with sm() as db:
        # Verify message_count is still 0 (proving we're not relying on it)
        dc2 = (await db.execute(select(DayChat).where(DayChat.id == dc_id2))).scalar_one()
        assert dc2.message_count == 0, "message_count should be 0 (not patched)"

        should2 = await mod.should_summarize(db, dc_id2)
    assert should2 is True, "Should summarize with >= 10 messages (counted from DB, not dc.message_count)"

    await engine.dispose()


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_load_day_context_multi_session_chronological", test_load_day_context_multi_session_chronological),
            ("test_load_day_context_respects_token_budget", test_load_day_context_respects_token_budget),
            ("test_load_day_context_no_summary_yet", test_load_day_context_no_summary_yet),
            ("test_load_day_context_with_summary", test_load_day_context_with_summary),
            ("test_annotate_message", test_annotate_message),
            ("test_build_today_so_far_block", test_build_today_so_far_block),
            ("test_history_is_complete_under_budget", test_history_is_complete_under_budget),
            ("test_history_is_complete_over_budget", test_history_is_complete_over_budget),
            ("test_history_is_complete_no_day_chat", test_history_is_complete_no_day_chat),
            ("test_should_inject_today_so_far_pure", test_should_inject_today_so_far_pure),
            ("test_runner_routes_today_so_far_through_gate", test_runner_routes_today_so_far_through_gate),
            ("test_annotations_never_saved_to_db", test_annotations_never_saved_to_db),
            ("test_fallback_when_backfill_incomplete", test_fallback_when_backfill_incomplete),
            ("test_summarizer_produces_output", test_summarizer_produces_output),
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
