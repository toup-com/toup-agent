"""
F8 — recent_days_service tests.

Covers:
- should_inject_recent_days gate: None / summary present / message count / fresh
- get_recent_day_summaries: archival preferred, rolling fallback, today excluded, window respected
- build_recent_days_block: empty, format, per-day char cap, relative labels

Pattern matches backend/tests/test_active_task.py and test_day_summarizer_ffb2.py:
sqlite + raw CREATE TABLE + ORM inserts, bypassing the heavy app.services init.

Direct invocation:
    cd backend && ENVIRONMENT=development python tests/test_recent_days_service.py
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "development")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.user import User
from app.db.models.day_chat import DayChat


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_recent_days_under_test",
    str(Path(__file__).resolve().parent.parent / "app" / "services" / "recent_days_service.py"),
)
_rd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rd)


# ── Test 1 — gate: None / empty / summary present / over threshold ──

def test_should_inject_recent_days_gate():
    f = _rd.should_inject_recent_days
    # None / empty
    assert f(None) is False
    assert f({}) is False
    # Has rolling summary → today's context owns continuity, skip recap
    assert f({"summary": "earlier today...", "message_count": 1}) is False
    # No summary, but lots of messages → mid-day, skip
    assert f({"summary": None, "message_count": 8}) is False
    # Fresh: no summary, low message count → INJECT
    assert f({"summary": None, "message_count": 0}) is True
    assert f({"summary": None, "message_count": 3}) is True
    # Boundary: at threshold = 4 → DO NOT inject (>=)
    assert f({"summary": None, "message_count": 4}) is False
    print("OK test_should_inject_recent_days_gate")


# ── Helper — sqlite engine ──

async def _make_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), name VARCHAR(255),
                password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50),
                is_canary BOOLEAN DEFAULT 0
            )""",
            """CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP,
                summary_status VARCHAR(20) DEFAULT 'up_to_date',
                archival_summary TEXT,
                archival_summary_generated_at TIMESTAMP,
                archival_summary_status VARCHAR(20) NOT NULL DEFAULT 'not_needed',
                summary_failure_count INTEGER NOT NULL DEFAULT 0,
                summary_last_failure_at TIMESTAMP,
                summary_last_failure_reason VARCHAR(50)
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


async def _seed_user(db, user_id):
    db.add(User(
        id=user_id, email=f"{user_id[:8]}@t.local", hashed_password="x",
        name="Test", role="beta_user", is_active=True, is_canary=False,
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    ))
    await db.flush()


async def _insert_day(db, user_id, d, archival=None, archival_status="not_needed",
                     rolling=None):
    dc_id = str(uuid.uuid4())
    await db.execute(text("""
        INSERT INTO day_chats (id, user_id, local_date, summary_status,
            archival_summary, archival_summary_status, rolling_summary)
        VALUES (:id, :uid, :d, 'up_to_date', :a, :as_, :r)
    """), {
        "id": dc_id, "uid": user_id, "d": d,
        "a": archival, "as_": archival_status, "r": rolling,
    })
    return dc_id


# ── Test 2 — fetch: archival preferred over rolling ──

async def test_get_recent_day_summaries_prefers_archival():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    today = date(2026, 5, 8)

    async with sm() as db:
        await _seed_user(db, user_id)
        # Yesterday: both archival + rolling — archival should win
        await _insert_day(db, user_id, today - timedelta(days=1),
                         archival="ARCH yesterday recap",
                         archival_status="up_to_date",
                         rolling="ROLLING yesterday")
        # 2 days ago: only rolling, archival not generated yet
        await _insert_day(db, user_id, today - timedelta(days=2),
                         rolling="ROLLING 2 days ago")
        await db.commit()

        out = await _rd.get_recent_day_summaries(db, user_id, today)

    assert len(out) == 2, f"expected 2 days, got {len(out)}"
    # Most-recent-first
    assert out[0]["local_date"] == today - timedelta(days=1)
    assert out[0]["summary"] == "ARCH yesterday recap"
    assert out[0]["is_archival"] is True
    # Fallback to rolling when archival missing
    assert out[1]["local_date"] == today - timedelta(days=2)
    assert out[1]["summary"] == "ROLLING 2 days ago"
    assert out[1]["is_archival"] is False
    print("OK test_get_recent_day_summaries_prefers_archival")


# ── Test 3 — fetch: today excluded, window respected ──

async def test_get_recent_day_summaries_excludes_today_and_old():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    today = date(2026, 5, 8)

    async with sm() as db:
        await _seed_user(db, user_id)
        # Today — must be excluded
        await _insert_day(db, user_id, today,
                         archival="TODAY", archival_status="up_to_date")
        # 1 day ago — included
        await _insert_day(db, user_id, today - timedelta(days=1),
                         archival="DAY-1", archival_status="up_to_date")
        # 5 days ago — outside default 2-day window
        await _insert_day(db, user_id, today - timedelta(days=5),
                         archival="DAY-5", archival_status="up_to_date")
        await db.commit()

        out = await _rd.get_recent_day_summaries(db, user_id, today)

    summaries = [r["summary"] for r in out]
    assert "TODAY" not in summaries, "today must be excluded"
    assert "DAY-5" not in summaries, "outside-window day must be excluded"
    assert "DAY-1" in summaries
    print("OK test_get_recent_day_summaries_excludes_today_and_old")


# ── Test 4 — fetch: pending archival NOT used (must be 'up_to_date') ──

async def test_get_recent_day_summaries_skips_pending_archival():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    today = date(2026, 5, 8)

    async with sm() as db:
        await _seed_user(db, user_id)
        # archival_status='pending' — not finalized; falls back to rolling
        await _insert_day(db, user_id, today - timedelta(days=1),
                         archival="MID-RUN", archival_status="pending",
                         rolling="ROLLING fallback")
        await db.commit()

        out = await _rd.get_recent_day_summaries(db, user_id, today)

    assert len(out) == 1
    assert out[0]["summary"] == "ROLLING fallback"
    assert out[0]["is_archival"] is False
    print("OK test_get_recent_day_summaries_skips_pending_archival")


# ── Test 5 — fetch: empty when no prior days ──

async def test_get_recent_day_summaries_empty():
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())

    async with sm() as db:
        await _seed_user(db, user_id)
        await db.commit()
        out = await _rd.get_recent_day_summaries(db, user_id, date(2026, 5, 8))

    assert out == []
    print("OK test_get_recent_day_summaries_empty")


# ── Test 6 — block format & friend-not-chatbot guard rails ──

def test_build_recent_days_block_format():
    today = date(2026, 5, 8)
    summaries = [
        {"local_date": today - timedelta(days=1), "summary": "yesterday's recap text", "is_archival": True},
        {"local_date": today - timedelta(days=2), "summary": "2 days ago recap", "is_archival": False},
    ]
    block = _rd.build_recent_days_block(summaries, today_local_date=today)
    assert block.startswith("\n<recent_days>")
    assert block.rstrip().endswith("</recent_days>")
    # Friend-not-chatbot guard MUST be present — without it the model
    # tends to open with "Yesterday you did X, Y, Z" (the bug we're avoiding)
    assert "Never" in block and "Yesterday you" in block, (
        "Block must contain explicit anti-recital guard"
    )
    # Relative labels render
    assert "Yesterday" in block
    assert "2 days ago" in block
    # Both summaries present
    assert "yesterday's recap text" in block
    assert "2 days ago recap" in block
    print("OK test_build_recent_days_block_format")


def test_build_recent_days_block_empty():
    """Empty summaries list → empty string (no block injected)."""
    assert _rd.build_recent_days_block([]) == ""
    assert _rd.build_recent_days_block([], today_local_date=date(2026, 5, 8)) == ""
    print("OK test_build_recent_days_block_empty")


def test_build_recent_days_block_truncates_long_summary():
    """Per-day char cap prevents one runaway day from ballooning the prompt."""
    today = date(2026, 5, 8)
    huge = "x" * 5000
    summaries = [
        {"local_date": today - timedelta(days=1), "summary": huge, "is_archival": True},
    ]
    block = _rd.build_recent_days_block(summaries, today_local_date=today)
    assert "…" in block, "long summary should be truncated with ellipsis"
    # Sanity: full 5000-char string is NOT in block
    assert huge not in block
    print("OK test_build_recent_days_block_truncates_long_summary")


# ── Run ──

async def _main():
    test_should_inject_recent_days_gate()
    await test_get_recent_day_summaries_prefers_archival()
    await test_get_recent_day_summaries_excludes_today_and_old()
    await test_get_recent_day_summaries_skips_pending_archival()
    await test_get_recent_day_summaries_empty()
    test_build_recent_days_block_format()
    test_build_recent_days_block_empty()
    test_build_recent_days_block_truncates_long_summary()
    print("\nALL F8 RECENT_DAYS TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(_main())
