"""A day chat dated in the user's FUTURE is the whole incident (E1/E2).

Replayed from the log, exactly:

    01:40:21Z  WhatsApp turn, tz unknown  -> DayChat(local_date=2026-09-14, tz='UTC')
    01:40:53Z  mobile turn, tz=America/Toronto -> DayChat(local_date=2026-09-13)

Two rows for one conversation, 32 seconds apart, and the WhatsApp half was
invisible in the app for the rest of the day. Nothing in this repo asserted
that a user may not hold a day chat whose local_date is in their own future,
so the stranding was legal.

This file states that invariant and pins the tz-learn gate that repairs it —
including the case the SHIPPED guard returns False for (NULL -> real), which
is precisely how the incident survived its own self-heal.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_unknown_tz_never_strands_a_day.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import uuid
from datetime import date as Date, datetime, timezone

import pytest
import pytest_asyncio

TORONTO = "America/Toronto"
# The two timestamps from the incident log.
T1 = datetime(2026, 9, 14, 1, 40, 21, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 14, 1, 40, 53, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def day_tables():
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages", "context_budget_logs"]
    async with engine.begin() as conn:
        for n in names:
            table = Base.metadata.tables.get(n)
            if table is None:
                continue
            try:
                await conn.run_sync(table.create, checkfirst=True)
            except Exception:
                pass
    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


@pytest.fixture(autouse=True)
def _clear_day_cache():
    """The day-chat id cache is keyed (user, local_date) and process-wide;
    a leaked entry answers the second resolve before it reaches the DB."""
    from app.agent._day_chat_cache import _CACHE

    _CACHE.clear()
    yield
    _CACHE.clear()


async def _mk_user(tz=None):
    from app.db.database import async_session_maker
    from app.db.models import User

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"strand-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password="x", name="Strand", timezone=tz,
        ))
        await db.commit()
    return uid


async def _resolve_day(user_id, utc_now, tz_name):
    from app.db.database import async_session_maker
    from app.agent.day_chat_resolver import get_or_create_day_chat

    async with async_session_maker() as db:
        dc = await get_or_create_day_chat(db, user_id, utc_now=utc_now, tz_name=tz_name)
        await db.commit()
        return dc.id, dc.local_date, dc.timezone


async def _all_days(user_id):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models.day_chat import DayChat

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(DayChat).where(DayChat.user_id == user_id)
        )).scalars().all()
        return [(r.id, r.local_date, r.timezone) for r in rows]


# ── the stranding, reproduced ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tz_less_turn_at_0140z_strands_a_day_in_the_future(day_tables):
    """Not a bug in the resolver — it is doing exactly what it was told.
    This is the STATE the rest of the file has to be able to repair."""
    uid = await _mk_user(tz=None)

    _, d1, tz1 = await _resolve_day(uid, T1, None)
    assert (d1, tz1) == (Date(2026, 9, 14), "UTC")

    _, d2, tz2 = await _resolve_day(uid, T2, TORONTO)
    assert (d2, tz2) == (Date(2026, 9, 13), TORONTO)

    days = sorted(d for _, d, _ in await _all_days(uid))
    assert days == [Date(2026, 9, 13), Date(2026, 9, 14)], (
        "two day chats 32 s apart — the WhatsApp half is now unreachable "
        "from the app, which only ever asks for the local date"
    )


# ── the tz-learn gate (D5; ws_chat.should_rebucket_on_tz_change) ──────


def _gate():
    from app.api import ws_chat

    if not hasattr(ws_chat, "should_rebucket_on_tz_change"):
        pytest.skip("should_rebucket_on_tz_change not landed yet — lane B2 (D5)")
    return ws_chat.should_rebucket_on_tz_change


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (None, TORONTO, True),        # THE incident case
        ("", TORONTO, True),
        ("UTC", TORONTO, True),       # the default masquerading as a real value
        (TORONTO, TORONTO, False),
        (TORONTO, "Europe/Berlin", True),
        (TORONTO, None, False),       # losing a tz teaches nothing
        (TORONTO, "", False),
    ],
)
def test_should_rebucket_on_tz_change(old, new, expected):
    """NULL -> real and 'UTC' -> real are the two the shipped guard answered
    False for, and they are the only two that can strand a day: a user whose
    tz was never known is the only user whose days were bucketed by UTC."""
    assert _gate()(old, new) is expected


# ── the invariant ─────────────────────────────────────────────────────


def _rebucket():
    svc = pytest.importorskip(
        "app.services.day_chat_rebucket",
        reason="day_chat_rebucket service not landed yet — lane B3 (D5)",
    )
    return svc


@pytest.mark.asyncio
async def test_no_user_holds_a_day_chat_dated_in_their_own_future(day_tables):
    """THE invariant, after the tz becomes known. `scope='future'` is the
    D5 default: only days the user cannot have lived yet are candidates."""
    svc = _rebucket()
    from app.db.database import async_session_maker

    uid = await _mk_user(tz=None)
    await _resolve_day(uid, T1, None)
    await _resolve_day(uid, T2, TORONTO)

    async with async_session_maker() as db:
        result = await svc.rebucket_user_days(db, uid, TORONTO, now_utc=T2)
        await db.commit()

    assert result.changed is True

    local_today = T2.astimezone(__import__("zoneinfo").ZoneInfo(TORONTO)).date()
    future = [d for _, d, _ in await _all_days(uid) if d > local_today]
    assert not future, (
        f"stranded day chat(s) {future} dated after the user's local today "
        f"({local_today}) — their history is unreachable from every client"
    )


@pytest.mark.asyncio
async def test_a_known_tz_throughout_needs_no_rebucket_at_all(day_tables):
    """ANTI-VACUITY. With the tz known from the first turn there is nothing
    to repair, and the service must say so rather than churn rows — a
    rebucket that always 'succeeds' would satisfy the test above by moving
    everything to today, which is the shipped heal's actual bug."""
    svc = _rebucket()
    from app.db.database import async_session_maker

    uid = await _mk_user(tz=TORONTO)
    await _resolve_day(uid, T1, TORONTO)
    await _resolve_day(uid, T2, TORONTO)

    before = sorted(await _all_days(uid))
    async with async_session_maker() as db:
        result = await svc.rebucket_user_days(db, uid, TORONTO, now_utc=T2)
        await db.commit()

    assert result.changed is False
    assert result.moved_messages == 0 and result.moved_conversations == 0
    assert sorted(await _all_days(uid)) == before


@pytest.mark.asyncio
async def test_a_spring_forward_local_0130_is_never_a_future_date(day_tables):
    """DST edge. 2026-03-08 is spring-forward in America/Toronto; 01:30 local
    is a real instant on the 8th and 06:30Z. A resolver that converted by a
    fixed offset would land on the 9th and strand it."""
    uid = await _mk_user(tz=TORONTO)
    utc = datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc)
    _, d, tz = await _resolve_day(uid, utc, TORONTO)
    assert (d, tz) == (Date(2026, 3, 8), TORONTO)
