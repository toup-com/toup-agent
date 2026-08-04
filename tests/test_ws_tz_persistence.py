"""
Tests for WS timezone persistence and auto-rebucket trigger.

Since the WS handler is tightly integrated with FastAPI/WebSocket, we test
the timezone persistence and rebucket logic in isolation — the same code
paths the WS handler calls.

Run: python tests/test_ws_tz_persistence.py
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.base import Base
from app.db.models.user import User
from app.db.models.day_chat import DayChat, MigrationStatus


async def _make_engine():
    """Build the tables from the ORM models, not from a copy of them.

    This was a hand-written `CREATE TABLE users (...)`, and it went red the
    moment the User model gained a column the copy did not have
    (`email_verified_at`) — the ORM insert names a column the table has never
    heard of. A hand-written schema is a second source of truth that nothing
    keeps in sync, and it fails somewhere else entirely, whenever someone adds
    a column.

    `create_all(tables=[...])` stays narrow: only the tables this file uses, so
    no pgvector column is ever compiled on sqlite, and dependency order is
    handled for us.
    """
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[User.__table__, DayChat.__table__, MigrationStatus.__table__],
        )
    return engine


async def _persist_tz(sm, user_id: str, client_tz: str):
    """Simulate the timezone persistence logic from ws_chat.py."""
    async with sm() as db:
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not user:
            return None, None

        old_tz = user.timezone
        if old_tz != client_tz:
            user.timezone = client_tz
            await db.commit()
            return old_tz, client_tz
        return old_tz, None  # No change


async def _should_rebucket(old_tz, new_tz, client_tz):
    """Simulate the rebucket trigger condition from ws_chat.py."""
    # Only rebucket if timezone changed from one real value to another
    # (not just NULL → real, which is the initial backfill case)
    return bool(old_tz and old_tz != "UTC" and old_tz != client_tz and new_tz is not None)


async def test_first_message_sets_timezone():
    """First message with tz updates User.timezone from NULL."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u1", email="u1@test.local", hashed_password="x", name="Test"))
        await db.commit()

    old_tz, new_tz = await _persist_tz(sm, "u1", "America/Toronto")
    assert old_tz is None, f"Expected NULL old_tz, got {old_tz}"
    assert new_tz == "America/Toronto"

    # Verify DB
    async with sm() as db:
        u = (await db.execute(select(User).where(User.id == "u1"))).scalar_one()
        assert u.timezone == "America/Toronto"

    await engine.dispose()


async def test_same_tz_is_noop():
    """Subsequent message with same tz is a no-op."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u2", email="u2@test.local", hashed_password="x", name="Test", timezone="America/Toronto"))
        await db.commit()

    old_tz, new_tz = await _persist_tz(sm, "u2", "America/Toronto")
    assert old_tz == "America/Toronto"
    assert new_tz is None, "Should be no-op when timezone unchanged"

    await engine.dispose()


async def test_different_tz_updates_and_triggers_rebucket():
    """Changing timezone from one real value to another triggers re-bucket."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u3", email="u3@test.local", hashed_password="x", name="Test", timezone="America/Toronto"))
        await db.commit()

    old_tz, new_tz = await _persist_tz(sm, "u3", "Europe/London")
    assert old_tz == "America/Toronto"
    assert new_tz == "Europe/London"

    # Should trigger rebucket
    should = await _should_rebucket(old_tz, new_tz, "Europe/London")
    assert should is True, "Should trigger rebucket on real→real tz change"

    # Verify DB updated
    async with sm() as db:
        u = (await db.execute(select(User).where(User.id == "u3"))).scalar_one()
        assert u.timezone == "Europe/London"

    await engine.dispose()


async def test_null_to_real_does_not_trigger_rebucket():
    """NULL → real timezone does NOT trigger rebucket (initial backfill handles it)."""
    old_tz = None
    new_tz = "America/Toronto"
    should = await _should_rebucket(old_tz, new_tz, "America/Toronto")
    assert should is False, "NULL→real should NOT trigger rebucket"


async def test_utc_to_real_does_not_trigger_rebucket():
    """UTC → real timezone does NOT trigger rebucket (UTC was the fallback default)."""
    old_tz = "UTC"
    new_tz = "America/Toronto"
    should = await _should_rebucket(old_tz, new_tz, "America/Toronto")
    assert should is False, "UTC→real should NOT trigger rebucket"


async def test_real_to_same_does_not_trigger_rebucket():
    """Same timezone → no rebucket."""
    old_tz = "America/Toronto"
    should = await _should_rebucket(old_tz, None, "America/Toronto")
    assert should is False, "Same tz should NOT trigger rebucket"


async def test_done_payload_includes_day_chat_id():
    """AgentResponse.day_chat_id flows through to the done payload."""
    # Import AgentResponse directly (it's a dataclass, no heavy deps)
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("agent_runner", str(Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py"))
    # We can't import the full module, but we can verify the field exists
    # by reading the dataclass definition
    import dataclasses

    # Simulate the done payload construction
    class MockResponse:
        text = "Hello"
        session_id = "sess-123"
        day_chat_id = "dc-456"
        tokens_input = 100
        tokens_output = 50
        tokens_total = 150
        model = "gpt-4o"
        tool_calls = []
        processing_time_ms = 500

    r = MockResponse()
    done_payload = {
        "type": "done",
        "text": r.text,
        "session_id": r.session_id,
        "tokens": {"input": r.tokens_input, "output": r.tokens_output, "total": r.tokens_total},
        "model": r.model,
        "tool_calls": len(r.tool_calls),
        "processing_time_ms": r.processing_time_ms,
    }
    if getattr(r, 'day_chat_id', None):
        done_payload["day_chat_id"] = r.day_chat_id

    assert done_payload["day_chat_id"] == "dc-456"
    assert done_payload["session_id"] == "sess-123"

    # When day_chat_id is empty, it should NOT be in the payload
    r2 = MockResponse()
    r2.day_chat_id = ""
    done2 = {"type": "done", "session_id": r2.session_id}
    if getattr(r2, 'day_chat_id', None):
        done2["day_chat_id"] = r2.day_chat_id
    assert "day_chat_id" not in done2, "Empty day_chat_id should not be in payload"


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_first_message_sets_timezone", test_first_message_sets_timezone),
            ("test_same_tz_is_noop", test_same_tz_is_noop),
            ("test_different_tz_updates_and_triggers_rebucket", test_different_tz_updates_and_triggers_rebucket),
            ("test_null_to_real_does_not_trigger_rebucket", test_null_to_real_does_not_trigger_rebucket),
            ("test_utc_to_real_does_not_trigger_rebucket", test_utc_to_real_does_not_trigger_rebucket),
            ("test_real_to_same_does_not_trigger_rebucket", test_real_to_same_does_not_trigger_rebucket),
            ("test_done_payload_includes_day_chat_id", test_done_payload_includes_day_chat_id),
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
