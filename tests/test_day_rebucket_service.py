"""The rebucket service: per-MESSAGE dates, dependents before parent, FK-safe.

The heal this replaces (day_chats.py:580-641) got four things wrong and each
one is a case below:

  1. it moved a WHOLE BUCKET to "today", so two messages that straddle local
     midnight both landed on the wrong day;
  2. it updated `messages` and `conversations` and then DELETED the day chat
     while `context_budget_logs` still pointed at it — the ForeignKeyViolation
     that turned into 65 consecutive 500s;
  3. it deleted a source day unconditionally, including one whose local_date
     had since become a legitimate calendar day;
  4. its UPDATEs carried no `user_id` predicate, so isolation rested on uuid
     collision odds rather than on a WHERE clause.

The apply half runs on its OWN engine with `PRAGMA foreign_keys=ON`. The shared
test engine has no such pragma anywhere in this repo (grep: zero hits), so an
FK assertion written against it passes vacuously — which is exactly how a
delete-before-move survived every existing test. The Postgres half of this
group is tests/test_day_rebucket_fk_postgres.py.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_day_rebucket_service.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import uuid
from datetime import date as Date, datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

TORONTO = "America/Toronto"
NOW = datetime(2026, 9, 14, 1, 40, 53, tzinfo=timezone.utc)   # local 2026-09-13 21:40
LOCAL_TODAY = Date(2026, 9, 13)


def svc():
    return pytest.importorskip(
        "app.services.day_chat_rebucket",
        reason="day_chat_rebucket service not landed yet — lane B3 (D5)",
    )


def _rows():
    m = svc()
    missing = [n for n in ("MsgRow", "ConvRow", "CblRow", "DayRow", "plan_rebucket")
               if not hasattr(m, n)]
    if missing:
        pytest.skip(f"planner surface incomplete ({missing}) — lane B3 (D5)")
    return m


# ══════════════════════════════════════════════════════════════════════
# The pure planner — no DB
# ══════════════════════════════════════════════════════════════════════

def _plan(*, days, messages, conversations=(), cbls=(), tz=TORONTO, now=NOW, scope=None):
    m = _rows()
    if scope is None:
        scope = {d.id for d in days if d.local_date > LOCAL_TODAY}
    return m.plan_rebucket(
        days=list(days), messages=list(messages), conversations=list(conversations),
        cbls=list(cbls), tz_name=tz, now_utc=now, scope_day_ids=set(scope),
    )


def test_targets_are_per_message_not_per_bucket():
    """The incident's two WhatsApp messages both belong to 2026-09-13 under
    Toronto — and the point is that the planner asks EACH message, so a pair
    that straddles local midnight cannot be answered with one date."""
    m = _rows()
    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 14))]
    msgs = [
        m.MsgRow(id="M1", created_at=datetime(2026, 9, 14, 1, 40, 21, tzinfo=timezone.utc),
                 day_chat_id="D_FUT", conversation_id="C_WA"),
        m.MsgRow(id="M2", created_at=datetime(2026, 9, 14, 1, 40, 29, tzinfo=timezone.utc),
                 day_chat_id="D_FUT", conversation_id="C_WA"),
    ]
    plan = _plan(days=days, messages=msgs)
    assert plan.message_moves == {"M1": Date(2026, 9, 13), "M2": Date(2026, 9, 13)}
    assert plan.noop is False


def test_a_conversation_straddling_local_midnight_does_not_move():
    """Two targets for one conversation means there is no conversation-level
    answer — `Message.day_chat_id` is canonical for day membership, so the
    messages split and the conversation stays where it is."""
    m = _rows()
    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 15))]
    msgs = [
        m.MsgRow(id="M1", created_at=datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc),
                 day_chat_id="D_FUT", conversation_id="C"),   # local 09-13 16:00
        m.MsgRow(id="M2", created_at=datetime(2026, 9, 14, 5, 0, tzinfo=timezone.utc),
                 day_chat_id="D_FUT", conversation_id="C"),   # local 09-14 01:00
    ]
    convs = [m.ConvRow(id="C", day_chat_id="D_FUT", channel="whatsapp", is_active=True)]
    plan = _plan(days=days, messages=msgs, conversations=convs,
        now=datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc),
    )

    assert set(plan.message_moves.values()) == {Date(2026, 9, 13), Date(2026, 9, 14)}
    assert plan.conversation_moves == {}, (
        "a split conversation was given a single target date — one of its "
        "messages would then disagree with its own conversation's day"
    )


def test_a_future_created_at_is_clamped_to_local_today():
    """Clock skew on a container has been measured at +14 minutes. A message
    whose timestamp maps past local today must NOT produce a new future day —
    that is the very state being repaired."""
    m = _rows()
    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 20))]
    msgs = [m.MsgRow(id="M1", created_at=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
                     day_chat_id="D_FUT", conversation_id="C")]
    plan = _plan(days=days, messages=msgs)
    assert plan.message_moves == {"M1": LOCAL_TODAY}
    assert all(d <= LOCAL_TODAY for d in plan.days_needed)


@pytest.mark.parametrize("tz", [None, "", "Mars/Olympus"])
def test_an_unknown_timezone_is_a_noop_and_never_guesses_utc(tz):
    """Guessing UTC is what created the stranded day. The planner declines."""
    m = _rows()
    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 14))]
    msgs = [m.MsgRow(id="M1", created_at=NOW, day_chat_id="D_FUT", conversation_id="C")]
    plan = _plan(days=days, messages=msgs, tz=tz, scope={"D_FUT"})
    assert plan.noop is True
    assert not plan.message_moves and not plan.days_deletable


def test_an_emptied_day_that_is_no_longer_in_the_future_is_kept():
    """Rule (g). The day the incident created — 2026-09-14 — becomes a real
    calendar day the moment the user lives it. Deleting it then strands the
    `_day_chat_cache` entry that already points at its id."""
    m = _rows()
    days = [m.DayRow(id="D", local_date=LOCAL_TODAY)]
    msgs = [m.MsgRow(id="M1", created_at=datetime(2026, 9, 12, 20, 0, tzinfo=timezone.utc),
                     day_chat_id="D", conversation_id="C")]
    plan = _plan(days=days, messages=msgs, scope={"D"})
    assert "D" in plan.days_emptied
    assert "D" not in plan.days_deletable


def test_a_day_still_in_the_future_and_provably_empty_is_deletable():
    """ANTI-VACUITY for the case above: the deletable set must be reachable,
    or 'never delete' would satisfy both tests."""
    m = _rows()
    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 14))]
    msgs = [m.MsgRow(id="M1", created_at=datetime(2026, 9, 14, 1, 40, 21, tzinfo=timezone.utc),
                     day_chat_id="D_FUT", conversation_id="C")]
    plan = _plan(days=days, messages=msgs)
    assert plan.days_deletable == {"D_FUT"}


def test_a_partial_index_collision_becomes_a_deactivation_not_a_move():
    """`conversations` carries a partial unique index over the system
    channels; moving a second active one onto a day that already holds one
    raises. Deactivating removes it from the index predicate instead — and
    INDEXED_SYSTEM_CHANNELS is IMPORTED, so this expectation follows the
    product rather than a copy of it."""
    m = _rows()
    try:
        from app.agent.conversation_resolver import INDEXED_SYSTEM_CHANNELS
    except Exception:  # noqa: BLE001
        pytest.skip("INDEXED_SYSTEM_CHANNELS not importable on this build")
    ch = next(iter(INDEXED_SYSTEM_CHANNELS))

    days = [m.DayRow(id="D_FUT", local_date=Date(2026, 9, 14)),
            m.DayRow(id="D_TODAY", local_date=LOCAL_TODAY)]
    msgs = [m.MsgRow(id="M1", created_at=datetime(2026, 9, 14, 1, 40, tzinfo=timezone.utc),
                     day_chat_id="D_FUT", conversation_id="C_MOVING")]
    convs = [
        m.ConvRow(id="C_MOVING", day_chat_id="D_FUT", channel=ch, is_active=True),
        m.ConvRow(id="C_SITTING", day_chat_id="D_TODAY", channel=ch, is_active=True),
    ]
    plan = _plan(days=days, messages=msgs, conversations=convs)
    assert "C_MOVING" not in plan.conversation_moves
    assert plan.conversation_deactivate.get("C_MOVING") == LOCAL_TODAY


def test_a_plan_with_nothing_to_do_is_a_noop():
    """The idempotence property, stated on the planner where it is cheap."""
    m = _rows()
    days = [m.DayRow(id="D", local_date=LOCAL_TODAY)]
    plan = _plan(days=days, messages=[], scope=set())
    assert plan.noop is True


# ══════════════════════════════════════════════════════════════════════
# The applier — own engine, FKs ON
# ══════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def fk_db(tmp_path):
    """A private engine with foreign keys ENFORCED.

    Without the pragma SQLite ignores every FK and the delete-ordering
    assertions below prove nothing — the failure mode this file exists for.
    """
    from app.db.models.base import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/rebucket.db")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    names = ("users", "day_chats", "conversations", "messages", "context_budget_logs")
    tables = [Base.metadata.tables[n] for n in names]
    async with engine.begin() as conn:
        for t in tables:
            try:
                await conn.run_sync(lambda sc, tt=t: tt.create(sc, checkfirst=True))
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"cannot create {t.name} on this backend: {e}")
        rows = await conn.exec_driver_sql("PRAGMA foreign_keys")
        assert rows.scalar() == 1, "PRAGMA foreign_keys did not take — the FK cases are vacuous"

    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed_incident(maker, user_id="u1", with_cbl=True):
    """The incident's exact shape: a future day holding a WhatsApp
    conversation with two messages and one context_budget_logs child."""
    from app.db.models import Conversation, Message, User
    from app.db.models.day_chat import DayChat, ContextBudgetLog

    async with maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@t.local", hashed_password="x",
                    name="U", timezone=TORONTO))
        db.add(DayChat(id="D_FUT", user_id=user_id, local_date=Date(2026, 9, 14),
                       timezone="UTC"))
        db.add(DayChat(id="D_TODAY", user_id=user_id, local_date=LOCAL_TODAY,
                       timezone=TORONTO))
        await db.flush()
        db.add(Conversation(id="C_WA", user_id=user_id, channel="whatsapp",
                            day_chat_id="D_FUT",
                            started_at=datetime(2026, 9, 14, 1, 40, 14)))
        for mid, sec in (("M1", 21), ("M2", 29)):
            db.add(Message(id=mid, conversation_id="C_WA", day_chat_id="D_FUT",
                           channel="whatsapp", role="user", content=mid,
                           created_at=datetime(2026, 9, 14, 1, 40, sec)))
        if with_cbl:
            db.add(ContextBudgetLog(id="CBL1", user_id=user_id, day_chat_id="D_FUT",
                                    created_at=datetime(2026, 9, 14, 1, 40, 30)))
        await db.commit()


async def _day_of(maker, msg_id):
    from app.db.models import Message

    async with maker() as db:
        return (await db.execute(
            select(Message.day_chat_id).where(Message.id == msg_id)
        )).scalar_one()


async def _day_ids(maker, user_id="u1"):
    from app.db.models.day_chat import DayChat

    async with maker() as db:
        return sorted((await db.execute(
            select(DayChat.id).where(DayChat.user_id == user_id)
        )).scalars().all())


@pytest.mark.asyncio
async def test_the_incident_rebuckets_with_foreign_keys_enforced(fk_db):
    """THE regression: both messages land on 2026-09-13, and the
    context_budget_logs child never causes a ForeignKeyViolation."""
    m = svc()
    await _seed_incident(fk_db)

    async with fk_db() as db:
        res = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    assert res.changed is True
    assert res.moved_messages == 2
    assert await _day_of(fk_db, "M1") == "D_TODAY"
    assert await _day_of(fk_db, "M2") == "D_TODAY"


@pytest.mark.asyncio
async def test_a_source_day_still_holding_a_budget_log_is_retained_not_deleted(fk_db):
    """The dependent that the shipped heal did not know about. If the CBL row
    cannot be placed, the day it points at must SURVIVE — retention is the
    successful outcome here, not an error."""
    m = svc()
    await _seed_incident(fk_db)

    async with fk_db() as db:
        res = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    ids = await _day_ids(fk_db)
    if "D_FUT" in ids:
        assert "D_FUT" in res.days_retained, (
            "D_FUT survived but the result does not say why — an operator "
            "reading [rebucket] cannot tell a retention from a silent failure"
        )
    else:
        assert res.moved_cbls >= 1, (
            "D_FUT was deleted while a context_budget_logs row still named it"
        )
    from app.db.models.day_chat import ContextBudgetLog
    async with fk_db() as db:
        assert (await db.execute(
            select(func.count()).select_from(ContextBudgetLog)
        )).scalar() == 1, "the budget log row was destroyed rather than moved"


@pytest.mark.asyncio
async def test_rebucket_is_idempotent(fk_db):
    """A second pass must plan nothing — the heal runs on a read path and a
    retrying client calls it repeatedly."""
    m = svc()
    await _seed_incident(fk_db)

    async with fk_db() as db:
        await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()
    snapshot = (await _day_ids(fk_db), await _day_of(fk_db, "M1"), await _day_of(fk_db, "M2"))

    async with fk_db() as db:
        second = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    assert second.changed is False
    assert second.moved_messages == 0
    assert (await _day_ids(fk_db), await _day_of(fk_db, "M1"), await _day_of(fk_db, "M2")) == snapshot


@pytest.mark.asyncio
async def test_a_second_users_identically_dated_day_is_untouched(fk_db):
    """User isolation, asserted POSITIVELY by seeding a decoy and re-reading
    it. The shipped heal's UPDATEs filtered on `day_chat_id` alone."""
    m = svc()
    await _seed_incident(fk_db, user_id="u1")

    from app.db.models import Conversation, Message, User
    from app.db.models.day_chat import DayChat

    async with fk_db() as db:
        db.add(User(id="u2", email="u2@t.local", hashed_password="x", name="U2",
                    timezone=TORONTO))
        db.add(DayChat(id="D_FUT_U2", user_id="u2", local_date=Date(2026, 9, 14),
                       timezone="UTC"))
        await db.flush()
        db.add(Conversation(id="C_U2", user_id="u2", channel="whatsapp",
                            day_chat_id="D_FUT_U2",
                            started_at=datetime(2026, 9, 14, 1, 40, 14)))
        db.add(Message(id="M_U2", conversation_id="C_U2", day_chat_id="D_FUT_U2",
                       channel="whatsapp", role="user", content="decoy",
                       created_at=datetime(2026, 9, 14, 1, 40, 25)))
        await db.commit()

    async with fk_db() as db:
        await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    assert await _day_of(fk_db, "M_U2") == "D_FUT_U2"
    assert "D_FUT_U2" in await _day_ids(fk_db, "u2")


@pytest.mark.asyncio
async def test_target_counters_and_summaries_are_refreshed(fk_db):
    """Messages arrive BEHIND the target day's summary watermark, so both the
    rolling summary and the counters on that day are now lies."""
    m = svc()
    await _seed_incident(fk_db)

    from app.db.models.day_chat import DayChat

    async with fk_db() as db:
        target = (await db.execute(
            select(DayChat).where(DayChat.id == "D_TODAY")
        )).scalar_one()
        target.summary_status = "up_to_date"
        target.rolling_summary = "a summary written before those rows existed"
        await db.commit()

    async with fk_db() as db:
        await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    async with fk_db() as db:
        target = (await db.execute(
            select(DayChat).where(DayChat.id == "D_TODAY")
        )).scalar_one()
    assert target.summary_status == "stale"
    assert target.rolling_summary is None
    assert (target.message_count or 0) == 2


@pytest.mark.asyncio
async def test_the_day_chat_cache_is_invalidated_for_the_whole_user(monkeypatch, fk_db):
    """Per-date invalidation is not enough: the plan touches several dates and
    a deleted id must go too. A stale entry keeps the process resolving the
    pre-heal day for up to the cache TTL."""
    m = svc()
    await _seed_incident(fk_db)

    seen: list[str] = []
    monkeypatch.setattr(
        "app.agent._day_chat_cache.invalidate_user", lambda uid: seen.append(uid),
    )

    async with fk_db() as db:
        await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    assert seen == ["u1"], f"expected exactly one whole-user invalidation, got {seen}"


@pytest.mark.asyncio
async def test_two_concurrent_rebuckets_leave_one_surviving_day(fk_db):
    """The heal fires from a read path; two tabs is the ordinary case."""
    import asyncio

    m = svc()
    await _seed_incident(fk_db)

    async def run():
        async with fk_db() as db:
            try:
                r = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
                await db.commit()
                return r
            except Exception as e:  # noqa: BLE001
                await db.rollback()
                return e

    await asyncio.gather(run(), run())

    from app.db.models.day_chat import DayChat
    async with fk_db() as db:
        dates = (await db.execute(
            select(DayChat.local_date).where(DayChat.user_id == "u1")
        )).scalars().all()
    assert len(dates) == len(set(dates)), f"duplicate day rows after a race: {dates}"
    assert await _day_of(fk_db, "M1") == await _day_of(fk_db, "M2")


@pytest.mark.asyncio
async def test_the_harness_itself_enforces_foreign_keys(fk_db):
    """ANTI-VACUITY for this whole file, and the sqlite twin of the Postgres
    adversarial control: run the OLD sequence — move messages and
    conversations, then DELETE the day — and assert it RAISES.

    Without this, every FK assertion above would pass just as happily on an
    engine that ignores foreign keys, which is the default everywhere else in
    this suite (no PRAGMA foreign_keys anywhere in conftest or database.py).
    """
    from sqlalchemy import delete as sa_delete, update as sa_update
    from sqlalchemy.exc import IntegrityError
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat

    await _seed_incident(fk_db)

    with pytest.raises(IntegrityError):
        async with fk_db() as db:
            await db.execute(sa_update(Message).where(Message.day_chat_id == "D_FUT")
                             .values(day_chat_id="D_TODAY"))
            await db.execute(sa_update(Conversation).where(Conversation.day_chat_id == "D_FUT")
                             .values(day_chat_id="D_TODAY"))
            await db.execute(sa_delete(DayChat).where(DayChat.id == "D_FUT"))
            await db.commit()


@pytest.mark.asyncio
async def test_a_failing_delete_keeps_every_move(fk_db, monkeypatch):
    """Review R3 P2: the delete phase ran OUTSIDE any savepoint on the same
    uncommitted transaction as the moves, so one failed DELETE rolled back
    every date's repair with `per_date_errors` empty. Each delete now sits in
    its own savepoint: the day is retained with the error, the moves commit."""
    m = svc()
    await _seed_incident(fk_db, with_cbl=False)  # D_FUT becomes provably empty → deletable

    real_count = m._count_dependents

    async def _boom(db, day_id):
        if day_id == "D_FUT":
            raise RuntimeError("simulated failure inside the delete phase")
        return await real_count(db, day_id)

    monkeypatch.setattr(m, "_count_dependents", _boom)

    async with fk_db() as db:
        res = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW)
        await db.commit()

    assert res.changed is True and res.moved_messages == 2
    assert await _day_of(fk_db, "M1") == "D_TODAY", "the moves were rolled back with the failed delete"
    assert "D_FUT" in res.days_retained and "delete failed" in res.days_retained["D_FUT"]
    assert "D_FUT" not in res.days_deleted
    assert "D_FUT" in await _day_ids(fk_db), "a day whose delete failed must still exist"


@pytest.mark.asyncio
async def test_the_kill_switch_stops_the_service_before_it_reads(fk_db, monkeypatch):
    """`rebucket_user_days` is the only door for every automatic caller (the
    endpoint heal and the ws_chat tz-learn trigger), so the flag is honoured
    HERE — one env flip stops them all. `force=True` is the operator's
    explicit override for the ops script."""
    m = svc()
    from app.config import settings
    await _seed_incident(fk_db)

    monkeypatch.setattr(settings, "day_chat_rebucket_enabled", False, raising=False)

    class _Untouchable:
        async def execute(self, *a, **k):
            raise AssertionError("the service touched the database while disabled")

    res = await m.rebucket_user_days(_Untouchable(), "u1", TORONTO, now_utc=NOW)
    assert res.changed is False and res.moved_messages == 0
    assert await _day_of(fk_db, "M1") == "D_FUT", "a disabled service moved a row"

    async with fk_db() as db:
        forced = await m.rebucket_user_days(db, "u1", TORONTO, now_utc=NOW, force=True)
        await db.commit()
    assert forced.changed is True and await _day_of(fk_db, "M1") == "D_TODAY"


def test_the_ws_chat_trigger_and_the_script_read_the_same_switch():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ws = (root / "app" / "api" / "ws_chat.py").read_text()
    gate = ws.index('getattr(settings, "day_chat_rebucket_enabled", True)')
    task = ws.index("asyncio.create_task(_trigger_scoped_rebucket())")
    assert gate < task < gate + 400, "the tz-learn rebucket task is created outside the flag gate"
    script = (root / "scripts" / "rebucket_day_chats.py").read_text()
    assert "rebucket_enabled()" in script and '"--force"' in script
    addon = (root.parent / "bridge" / "pool_addon.py").read_text()
    tuple_body = addon.split("_FEATURE_FLAG_ENVS = (")[1].split("\n)")[0]
    assert '"DAY_CHAT_REBUCKET_ENABLED"' in tuple_body, (
        "the kill switch is not forwarded to pool containers"
    )


@pytest.mark.asyncio
async def test_the_ops_script_refuses_to_apply_while_disabled_unless_forced(monkeypatch, capsys):
    """EXECUTED, not grepped: `main()` with --apply while the switch is off
    returns 2 before it reads a single row; --force proceeds (to an empty
    candidate list here, which is a 0)."""
    from types import SimpleNamespace
    import scripts.rebucket_day_chats as script
    from app.config import settings

    monkeypatch.setattr(settings, "day_chat_rebucket_enabled", False, raising=False)

    async def _no_candidates(*a, **k):
        return []

    monkeypatch.setattr(script, "_candidate_user_ids", _no_candidates)
    args = SimpleNamespace(apply=True, force=False, user_id=None, scope="future", limit_days=90, json=None)
    assert await script.main(args) == 2
    assert "refusing --apply" in capsys.readouterr().out

    args.force = True
    assert await script.main(args) == 0

    # ANTI-VACUITY: with the switch ON the plain --apply path runs.
    monkeypatch.setattr(settings, "day_chat_rebucket_enabled", True, raising=False)
    args.force = False
    assert await script.main(args) == 0
