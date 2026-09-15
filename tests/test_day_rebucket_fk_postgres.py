"""The rebucket, against an engine that actually enforces foreign keys.

`day_chats.id` has THREE dependents — `conversations.day_chat_id`,
`messages.day_chat_id` and `context_budget_logs.day_chat_id` — and the shipped
self-heal moved two of them before deleting the parent. On 2026-09-14 that
raised ForeignKeyViolation inside a `try/except` that logged and did NOT roll
back, so the request session stayed in PendingRollback and the enclosing loop
kept dereferencing expired ORM attributes: one FK error became 65 consecutive
500s on `GET /api/day-chats` over three hours, and the whole day index was
gone for every client.

This is the Postgres half of the group. The sqlite half
(tests/test_day_rebucket_service.py) enforces FKs on a private engine with an
explicit PRAGMA; here they are simply real. The adversarial control below runs
the OLD sequence and asserts it RAISES, so this file cannot pass on permissive
semantics — the exact way an FK test proves nothing.

CI: named in the "Run pool / reclaim / proxy / pre-save suites on Postgres"
step. Being named there also removes it from the sqlite sweep, which has no
asyncpg and crashed on a file like this once before (CI run 34146791609).

Local run:
    TEST_PG_URL='postgresql+asyncpg://postgres:test@127.0.0.1:5432/toup_presave' \
    TEST_PG_REQUIRED=1 PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_day_rebucket_fk_postgres.py -v --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import os
import uuid
from datetime import date as Date, datetime, timezone

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

PG_URL = os.environ.get("TEST_PG_URL", "postgresql+asyncpg://localhost/toup_lane_agent")
PG_REQUIRED = bool(os.environ.get("TEST_PG_REQUIRED"))

TORONTO = "America/Toronto"
NOW = datetime(2026, 9, 14, 1, 40, 53, tzinfo=timezone.utc)   # local 2026-09-13 21:40
LOCAL_TODAY = Date(2026, 9, 13)
FUTURE_DAY = Date(2026, 9, 14)

_TABLES = ("users", "day_chats", "conversations", "messages", "context_budget_logs")


def _no_postgres(reason: str):
    """Under TEST_PG_REQUIRED a skip would mean the one lane this file is
    meant to run in had quietly proven nothing."""
    if PG_REQUIRED:
        pytest.fail(f"TEST_PG_REQUIRED is set but {reason}", pytrace=False)
    pytest.skip(reason)


def _svc():
    return pytest.importorskip(
        "app.services.day_chat_rebucket",
        reason="day_chat_rebucket service not landed yet — lane B3 (D5)",
    )


async def _engine():
    from sqlalchemy import text

    try:
        # Inside the try on purpose: create_async_engine imports the driver.
        from sqlalchemy.ext.asyncio import create_async_engine
        eng = create_async_engine(PG_URL)
    except Exception as e:  # noqa: BLE001
        _no_postgres(f"no asyncpg driver for {PG_URL}: {e}")
    try:
        async with eng.begin() as c:
            await c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as e:  # noqa: BLE001
        await eng.dispose()
        _no_postgres(f"no scratch Postgres at {PG_URL}: {e}")
    return eng


async def _fresh_tenant():
    """A tenant carrying the incident's five tables and one owner."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.db.models import User
    from app.db.models.base import Base

    eng = await _engine()
    tables = [Base.metadata.tables[n] for n in _TABLES]
    async with eng.begin() as c:
        await c.run_sync(lambda s: Base.metadata.drop_all(s, tables=tables[::-1], checkfirst=True))
        await c.run_sync(lambda s: Base.metadata.create_all(s, tables=tables, checkfirst=True))

    Session = async_sessionmaker(eng, expire_on_commit=False)
    uid = str(uuid.uuid4())
    async with Session() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@toup.ai", name="Rebucket",
                    hashed_password="x", timezone=TORONTO))
        await db.commit()
    return eng, Session, uid


async def _seed_incident(Session, uid, *, with_cbl=True, days=1):
    """`days` future day chats, each with a conversation, two messages and
    (optionally) one context_budget_logs child."""
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import ContextBudgetLog, DayChat

    ids = []
    async with Session() as db:
        db.add(DayChat(id="D_TODAY", user_id=uid, local_date=LOCAL_TODAY, timezone=TORONTO))
        await db.flush()
        for n in range(days):
            dc = f"D_FUT_{n}"
            ids.append(dc)
            db.add(DayChat(id=dc, user_id=uid, local_date=Date(2026, 9, 14 + n), timezone="UTC"))
            await db.flush()
            db.add(Conversation(id=f"C_{n}", user_id=uid, channel="whatsapp", day_chat_id=dc,
                                started_at=datetime(2026, 9, 14, 1, 40, 14)))
            for k, sec in enumerate((21, 29)):
                db.add(Message(id=f"M_{n}_{k}", conversation_id=f"C_{n}", day_chat_id=dc,
                               channel="whatsapp", role="user", content=f"m{n}{k}",
                               created_at=datetime(2026, 9, 14, 1, 40, sec)))
            if with_cbl:
                db.add(ContextBudgetLog(id=f"CBL_{n}", user_id=uid, day_chat_id=dc,
                                        created_at=datetime(2026, 9, 14, 1, 40, 30)))
        await db.commit()
    return ids


# ── the adversarial control ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_old_sequence_really_does_raise_here():
    """ANTI-VACUITY. Move messages + conversations, then delete the day —
    the shipped heal, byte for byte in shape — and prove Postgres refuses it.

    Without this, every assertion below would read identically on an engine
    that ignores foreign keys, and the file would be proving nothing.
    """
    from sqlalchemy import delete as sa_delete, update as sa_update
    from sqlalchemy.exc import IntegrityError
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat

    eng, Session, uid = await _fresh_tenant()
    try:
        await _seed_incident(Session, uid)
        async with Session() as db:
            await db.execute(sa_update(Message).where(Message.day_chat_id == "D_FUT_0")
                             .values(day_chat_id="D_TODAY"))
            await db.execute(sa_update(Conversation).where(Conversation.day_chat_id == "D_FUT_0")
                             .values(day_chat_id="D_TODAY"))
            with pytest.raises(IntegrityError) as ei:
                await db.execute(sa_delete(DayChat).where(DayChat.id == "D_FUT_0"))
                await db.commit()
        assert "ForeignKeyViolation" in str(ei.value) or "foreign key" in str(ei.value).lower()
    finally:
        await eng.dispose()


# ── the fix ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stranded_day_with_a_budget_log_child_rebuckets_cleanly():
    m = _svc()
    from sqlalchemy import func, select
    from app.db.models import Message
    from app.db.models.day_chat import ContextBudgetLog

    eng, Session, uid = await _fresh_tenant()
    try:
        await _seed_incident(Session, uid)
        async with Session() as db:
            res = await m.rebucket_user_days(db, uid, TORONTO, now_utc=NOW)
            await db.commit()

        assert res.changed is True
        async with Session() as db:
            days = set((await db.execute(
                select(Message.day_chat_id).where(Message.id.in_(["M_0_0", "M_0_1"]))
            )).scalars().all())
            cbls = (await db.execute(select(func.count()).select_from(ContextBudgetLog))).scalar()
        assert days == {"D_TODAY"}, f"messages did not land on the local day: {days}"
        assert cbls == 1, "the budget log row was destroyed rather than moved or retained"
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_a_failed_rebucket_leaves_the_session_usable():
    """The mechanism behind the 3-hour outage, isolated. After an exception
    inside the rebucket, a SELECT in the SAME session must still work — a
    PendingRollbackError here is what turned one FK error into 65 x 500."""
    m = _svc()
    from sqlalchemy import select
    from app.db.models.day_chat import DayChat

    eng, Session, uid = await _fresh_tenant()
    try:
        await _seed_incident(Session, uid)

        import app.services.day_chat_rebucket as mod
        real_apply = mod.apply_rebucket

        async def boom(db, user_id, plan, **kw):  # noqa: ANN001
            # Issue a statement that Postgres will reject, so the failure is a
            # real aborted transaction rather than a Python-level raise the
            # session never saw.
            from sqlalchemy import text
            try:
                await db.execute(text("SELECT 1 FROM a_table_that_does_not_exist"))
            except Exception:
                raise RuntimeError("injected rebucket failure")

        mod.apply_rebucket = boom
        try:
            async with Session() as db:
                with pytest.raises(Exception):
                    await m.rebucket_user_days(db, uid, TORONTO, now_utc=NOW)
                await db.rollback()
                rows = (await db.execute(
                    select(DayChat).where(DayChat.user_id == uid)
                )).scalars().all()
                assert len(rows) == 2
        finally:
            mod.apply_rebucket = real_apply
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_list_day_chats_returns_200_when_the_rebucket_raises():
    """The operational invariant that matters most: a heal that cannot run is
    a MISSING REPAIR, never a broken day index. D5 puts the heal in its own
    session precisely so this is a property of the shape."""
    _svc()
    import app.api.day_chats as day_chats_api

    eng, Session, uid = await _fresh_tenant()
    try:
        await _seed_incident(Session, uid)

        import app.services.day_chat_rebucket as mod
        real = mod.rebucket_user_days

        async def boom(*a, **k):
            raise RuntimeError("injected")

        mod.rebucket_user_days = boom
        if hasattr(day_chats_api, "rebucket_user_days"):
            day_chats_api.rebucket_user_days = boom
        # `managed_containers` is PLATFORM_ONLY and this scratch tenant holds
        # five tables; the proxy probe is not what this test is about.
        real_proxy = day_chats_api._get_agent_proxy_info

        async def _no_proxy(*a, **k):
            return None

        day_chats_api._get_agent_proxy_info = _no_proxy
        try:
            from app.db.models import User
            from sqlalchemy import select
            async with Session() as db:
                user = (await db.execute(select(User).where(User.id == uid))).scalar_one()
                out = await day_chats_api.list_day_chats(
                    limit=30, before=None, current_user=user, db=db,
                )
        finally:
            mod.rebucket_user_days = real
            day_chats_api._get_agent_proxy_info = real_proxy
            if hasattr(day_chats_api, "rebucket_user_days"):
                day_chats_api.rebucket_user_days = real

        # The route answers a JSONResponse (the platform-proxy branch returns
        # the tenant's JSON verbatim, so the local branch does too).
        if hasattr(out, "body"):
            import json as _json
            out = _json.loads(bytes(out.body).decode())
        rows = out if isinstance(out, list) else (out.get("day_chats") or out.get("items") or [])
        assert rows, "a failed heal returned an EMPTY day list — the released "\
                     "client's history-honesty guard reads that as data loss"
        assert any(str(r.get("local_date") if isinstance(r, dict) else r) for r in rows)
    finally:
        await eng.dispose()


@pytest.mark.asyncio
async def test_many_stranded_days_rebucket_without_a_per_row_commit_storm():
    """A tenant can hold dozens of stranded days (one per tz-less day since
    provisioning). The heal runs on a READ path, so a per-row commit is a
    latency bomb on exactly the request a user is waiting on."""
    m = _svc()
    eng, Session, uid = await _fresh_tenant()
    try:
        await _seed_incident(Session, uid, days=8)
        async with Session() as db:
            res = await m.rebucket_user_days(db, uid, TORONTO, now_utc=NOW, limit_days=90)
            await db.commit()
        assert res.changed is True
        assert res.moved_messages == 16
    finally:
        await eng.dispose()
