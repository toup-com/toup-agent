"""
Tests for day_chat_resolver: concurrent creation, timezone handling, idempotency.

Run: python tests/test_day_chat_resolver.py
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
from app.db.models.user import User

# Late import to avoid triggering app.config at module load time
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "day_chat_resolver",
    str(Path(__file__).resolve().parent.parent / "app" / "agent" / "day_chat_resolver.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_or_create_day_chat = _mod.get_or_create_day_chat
resolve_local_date = _mod.resolve_local_date
should_use_day_chat_context = _mod.should_use_day_chat_context


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), password_plain VARCHAR(255),
                name VARCHAR(255), password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50)
            )
        """)))
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP, summary_status VARCHAR(20) DEFAULT 'up_to_date',
                UNIQUE(user_id, local_date)
            )
        """)))
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE IF NOT EXISTS migration_status (
                migration_name VARCHAR(100) PRIMARY KEY, status VARCHAR(20) DEFAULT 'not_started',
                started_at TIMESTAMP, completed_at TIMESTAMP, progress_json TEXT, error_message TEXT
            )
        """)))
    return engine


async def test_get_or_create_idempotency():
    """Calling get_or_create twice for the same user+date returns the same row."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u1", email="u1@test.local", hashed_password="x", name="Test"))
        await db.commit()

    now = datetime(2026, 4, 7, 14, 0, 0, tzinfo=timezone.utc)

    async with sm() as db:
        dc1 = await get_or_create_day_chat(db, "u1", utc_now=now)
        await db.commit()

    async with sm() as db:
        dc2 = await get_or_create_day_chat(db, "u1", utc_now=now)
        await db.commit()

    assert dc1.id == dc2.id, f"Expected same ID, got {dc1.id} vs {dc2.id}"
    assert dc1.local_date == Date(2026, 4, 7)

    # Only one row in the table
    async with sm() as db:
        count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
    assert count == 1

    await engine.dispose()


async def test_concurrent_creation():
    """5 concurrent calls for the same user+date create exactly 1 row."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u2", email="u2@test.local", hashed_password="x", name="Test"))
        await db.commit()

    now = datetime(2026, 4, 7, 14, 0, 0, tzinfo=timezone.utc)

    async def _create():
        async with sm() as db:
            dc = await get_or_create_day_chat(db, "u2", utc_now=now)
            await db.commit()
            return dc.id

    # Fire 5 concurrent creations
    results = await asyncio.gather(*[_create() for _ in range(5)])

    # All should return the same ID
    unique_ids = set(results)
    assert len(unique_ids) == 1, f"Expected 1 unique ID, got {len(unique_ids)}: {unique_ids}"

    # Exactly 1 row in DB
    async with sm() as db:
        count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
    assert count == 1

    await engine.dispose()


async def test_timezone_toronto():
    """Toronto user at 11pm UTC April 7 → still April 7 local (7pm ET)."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u3", email="u3@test.local", hashed_password="x", name="Test"))
        await db.commit()

    # 11pm UTC = 7pm ET (still April 7)
    now_11pm = datetime(2026, 4, 7, 23, 0, 0, tzinfo=timezone.utc)
    async with sm() as db:
        dc1 = await get_or_create_day_chat(db, "u3", utc_now=now_11pm, tz_name="America/Toronto")
        await db.commit()
    assert dc1.local_date == Date(2026, 4, 7)

    # 1am UTC April 8 = 9pm ET April 7 (still April 7!)
    now_1am = datetime(2026, 4, 8, 1, 0, 0, tzinfo=timezone.utc)
    async with sm() as db:
        dc2 = await get_or_create_day_chat(db, "u3", utc_now=now_1am, tz_name="America/Toronto")
        await db.commit()
    assert dc2.local_date == Date(2026, 4, 7)
    assert dc1.id == dc2.id  # Same DayChat

    # 5am UTC April 8 = 1am ET April 8 (NEW day)
    now_5am = datetime(2026, 4, 8, 5, 0, 0, tzinfo=timezone.utc)
    async with sm() as db:
        dc3 = await get_or_create_day_chat(db, "u3", utc_now=now_5am, tz_name="America/Toronto")
        await db.commit()
    assert dc3.local_date == Date(2026, 4, 8)
    assert dc3.id != dc1.id  # Different DayChat

    await engine.dispose()


async def test_timezone_missing_fallback_utc():
    """User with no timezone → UTC day boundary."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u4", email="u4@test.local", hashed_password="x", name="Test"))
        await db.commit()

    # 11:30pm UTC April 7 with no timezone → April 7 UTC
    now = datetime(2026, 4, 7, 23, 30, 0, tzinfo=timezone.utc)
    async with sm() as db:
        dc = await get_or_create_day_chat(db, "u4", utc_now=now, tz_name=None)
        await db.commit()
    assert dc.local_date == Date(2026, 4, 7)
    assert dc.timezone == "UTC"

    # 0:30am UTC April 8 → April 8 UTC (new day)
    now2 = datetime(2026, 4, 8, 0, 30, 0, tzinfo=timezone.utc)
    async with sm() as db:
        dc2 = await get_or_create_day_chat(db, "u4", utc_now=now2, tz_name=None)
        await db.commit()
    assert dc2.local_date == Date(2026, 4, 8)
    assert dc2.id != dc.id

    await engine.dispose()


async def test_resolve_local_date_function():
    """Unit test for the resolve_local_date helper."""
    # UTC fallback
    dt = datetime(2026, 4, 7, 23, 0, 0, tzinfo=timezone.utc)
    d, tz = resolve_local_date(dt, None)
    assert d == Date(2026, 4, 7)
    assert tz == "UTC"

    # Toronto
    d2, tz2 = resolve_local_date(dt, "America/Toronto")
    assert d2 == Date(2026, 4, 7)  # 11pm UTC = 7pm ET
    assert tz2 == "America/Toronto"

    # Invalid timezone → UTC fallback
    d3, tz3 = resolve_local_date(dt, "Invalid/Timezone")
    assert d3 == Date(2026, 4, 7)
    assert tz3 == "UTC"

    # Naive datetime → treated as UTC
    dt_naive = datetime(2026, 4, 7, 23, 0, 0)
    d4, tz4 = resolve_local_date(dt_naive, None)
    assert d4 == Date(2026, 4, 7)


async def test_different_days_different_day_chats():
    """Messages on different days create separate DayChats."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add(User(id="u5", email="u5@test.local", hashed_password="x", name="Test"))
        await db.commit()

    ids = []
    for day in range(5):
        now = datetime(2026, 4, 1 + day, 12, 0, 0, tzinfo=timezone.utc)
        async with sm() as db:
            dc = await get_or_create_day_chat(db, "u5", utc_now=now)
            await db.commit()
            ids.append(dc.id)

    assert len(set(ids)) == 5  # 5 unique DayChats

    async with sm() as db:
        count = (await db.execute(select(func.count()).select_from(DayChat))).scalar()
    assert count == 5

    await engine.dispose()


# ── Standalone runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    async def _run_all():
        tests = [
            ("test_get_or_create_idempotency", test_get_or_create_idempotency),
            ("test_concurrent_creation", test_concurrent_creation),
            ("test_timezone_toronto", test_timezone_toronto),
            ("test_timezone_missing_fallback_utc", test_timezone_missing_fallback_utc),
            ("test_resolve_local_date_function", test_resolve_local_date_function),
            ("test_different_days_different_day_chats", test_different_days_different_day_chats),
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
