"""The save's SHAPE, measured by the instrument that will measure it in prod.

WHY THIS FILE EXISTS

`[PERF] phase3_save` is one timer around the save, and one 2026-09-20 sample
turn spent 5.023 s inside it (PRODUCTION OBSERVATION, from the supervisor's
platform-log extract). Round 1's map counted six statements there; this file
measures five cursor executions on sqlite. Round 1 produced four mechanisms
that all fit that number — a fresh dial (NullPool is the SOURCE DEFAULT, not a
verified runtime value), a PgBouncer server-connection wait, a row lock, a
starved loop — and nothing in the deployed trail separates them. `db_span`
narrows them; it refutes none of them, and it is a CLIENT-SIDE instrument that
can never separate server-side SQL duration from network from pooler wait.

An instrument that reports the wrong shape is worse than none, so this file
drives the REAL `AgentRunner._save_messages` inside the real
`async with session_maker()` envelope against a file-backed sqlite engine
pinned to NullPool, and asserts what the production line says: one connection,
one commit, and FIVE statements in the order the round-1 map predicted — five
and not the six that map counted, because the ORM batches the two `messages`
INSERTs; the test name says five so that a reader grepping for the save's shape
finds the number the instrument actually prints.

Two properties are being pinned, and they fail differently:

  * the COUNTS and the ORDER — if the save grows a seventh statement or
    re-orders its locks (`conversations` then `day_chats`, both held across
    both INSERTs), this file says so;
  * the ATTRIBUTION — a global engine listener on a process that runs ~20
    background loops is only safe because the span rides a ContextVar and
    greenlet propagates `gr_context`. `test_two_concurrent_spans_do_not_mix`
    is the check on that, and it is the check a2's skeptic asked for when they
    objected that a global listener "becomes a fabrication generator at
    exactly the moment the box is busy".

Local sqlite proves STRUCTURE, never latency. Every ms figure here is
meaningless; only the counts and the order are claims.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

#: Synthetic, and a FULL uuid on purpose — see the canary-validation tests
#: at the bottom of this file: matching is exact and a short entry is
#: rejected, so a test keyed on an 8-character prefix would test the mistake.
CANARY = "00000000-aaaa-4bbb-8ccc-000000000001"
OTHER_USER = "00000000-aaaa-4bbb-8ccc-000000000002"

AGENT_RUNNER_SRC = (
    Path(__file__).resolve().parent.parent / "app/agent/agent_runner.py"
).read_text()
WS_CHAT_SRC = (
    Path(__file__).resolve().parent.parent / "app/api/ws_chat.py"
).read_text()
DB_SRC = (
    Path(__file__).resolve().parent.parent / "app/db/database.py"
).read_text()


class _StubTools:
    """`_save_messages` touches `self` only through `self.tools`."""

    def __init__(self) -> None:
        self._last_media = None
        self._last_pending_action = None
        self.pending_attachments = []


class _StubRunner:
    def __init__(self) -> None:
        self.tools = _StubTools()


def _fields(line: str) -> dict:
    out = {}
    for token in line[len("[PERF] db_span "):].split(" "):
        k, _, v = token.partition("=")
        out[k] = v
    return out


@pytest.fixture()
async def save_engine(tmp_path):
    """A file-backed sqlite engine pinned to NullPool — the deployed agent's
    pool shape (`config.agent_db_pool_size = 0`), which is what makes every
    session a fresh `connect` and therefore makes `conns`/`dials` mean
    something."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
    from sqlalchemy.pool import NullPool

    from app.db.models import Base

    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/save.db",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
        hide_parameters=True,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(
        eng, class_=AsyncSession, expire_on_commit=False,
        autocommit=False, autoflush=False,
    )
    try:
        yield eng, maker
    finally:
        await eng.dispose()


async def _seed(maker, user_id: str, session_id: str) -> None:
    from app.db.models import Conversation, User
    from app.db.models.day_chat import DayChat
    from app.agent.day_chat_resolver import resolve_local_date

    local_date, tz = resolve_local_date(datetime.now(timezone.utc), "UTC")
    async with maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@t.test",
                    hashed_password="x", timezone="UTC"))
        db.add(Conversation(id=session_id, user_id=user_id, title="t",
                            channel="web"))
        db.add(DayChat(id=str(uuid.uuid4()), user_id=user_id,
                       local_date=local_date, timezone=tz,
                       started_at=datetime.utcnow(),
                       last_message_at=datetime.utcnow(),
                       message_count=0, total_tokens=0,
                       summary_status="up_to_date"))
        await db.commit()


@pytest.mark.asyncio
async def test_the_save_is_one_connection_one_commit_and_five_statements(
    caplog, save_engine, monkeypatch,
):
    from app.agent.agent_runner import AgentRunner
    import app.db.db_span as ds

    eng, maker = save_engine
    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.install(eng)
    ds.begin_turn(user_id="u-save", client_msg_id="cmid-save", channel="web",
                  replace=True)

    user_id, session_id = "u-save", str(uuid.uuid4())
    await _seed(maker, user_id, session_id)

    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        async with ds.db_span("save"), maker() as db:
            await AgentRunner._save_messages(
                _StubRunner(),
                db=db,
                session_id=session_id,
                user_id=user_id,
                user_message="hello",
                assistant_response="OK",
                tokens_input=10,
                tokens_output=1,
                model="gpt-5.6-terra",
                processing_time_ms=1,
                save_user_message=True,
                client_tz="UTC",
                channel="web",
                client_msg_id="cmid-save",
            )
            await db.commit()
    ds._reset_for_tests()

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[PERF] db_span phase=save")]
    assert len(lines) == 1, lines
    f = _fields(lines[0])

    # ONE connection: the save opens a single session and nothing inside it
    # opens a second. On NullPool `conns` and `dials` are equal by
    # construction — that equality IS the pool-shape reading, and it is what
    # would change the day `AGENT_DB_POOL_SIZE` is set on a slot.
    assert f["conns"] == "1", lines[0]
    assert f["dials"] == "1", lines[0]

    # FIVE cursor executions, not the six the round-1 map counted: SQLAlchemy's
    # unit of work folds this turn's two `messages` INSERTs into ONE
    # `executemany`, which `em=1` says out loud. `autoflush=False` is why that
    # insert lands AFTER both UPDATEs — the locks are taken conversations →
    # day_chats and held across the insert and the commit.
    #
    # THE CAVEAT THAT MATTERS: this is sqlite. The batching decision is the
    # ORM's, not the driver's, so the SHAPE should hold on asyncpg — but an
    # asyncpg executemany of two rows is one Parse plus two Bind/Execute, so
    # "5 statements" is NOT "5 round trips", and the round-trip arithmetic a1
    # and a2 both built on "6 statements" has to be re-derived against this
    # field on a real Postgres. Local sqlite proves the shape, never the wire.
    assert f["stmts"] == "5", lines[0]
    assert f["em"] == "1", lines[0]
    assert f["seq"] == (
        "select:conversations,select:day_chats,"
        "update:conversations,update:day_chats,"
        "insert:messages"
    ), lines[0]

    # The phase's own envelope accounts for itself. The buckets are disjoint
    # ONLY because this span holds ONE connection — they are sequential
    # intervals on it — so the precondition is asserted, not assumed: on a
    # span holding two connections concurrently (phase 1 does) the same sum
    # adds overlapping intervals and can exceed `total_ms` without anything
    # being wrong. That is why the acceptance gate in NOTES is restricted to
    # `conns<=1` lines rather than applied to every span.
    assert f["conns"] == "1", lines[0]
    total = int(f["total_ms"])
    parts = (int(f["dial_ms"]) + int(f["begin_ms"]) + int(f["sql_ms"])
             + int(f["commit_ms"]))
    assert parts <= total + 2, (parts, total, lines[0])
    # `stmt1_ms` is a SUBSET of `sql_ms` and must never be added into that sum
    # — it is the same wall time, bucketed a second way so the pooler's
    # signature is a field rather than an inference (R48 review, B-1).
    assert int(f["stmt1_ms"]) <= int(f["sql_ms"]), lines[0]
    # The sample ring covered the whole span, so the lag figure is a maximum
    # and not a floor.
    assert f["lag_cov"] == "1", lines[0]
    assert f["cmid_h"] != "00000000" and f["ch"] == "web"


@pytest.mark.asyncio
async def test_two_concurrent_spans_do_not_mix(caplog, save_engine, monkeypatch):
    """The listeners are GLOBAL on the engine, and this process runs ~20
    background loops on it. Attribution is a ContextVar, read inside the
    greenlet SQLAlchemy spawns from the awaiting task — so two turns in flight
    at once each keep their own statements. If that ever stopped holding, the
    instrument would fabricate rather than fail, which is why this is a test
    and not a comment."""
    from sqlalchemy import text

    import app.db.db_span as ds

    eng, _maker = save_engine
    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.install(eng)
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)

    async def leg(n: int, phase: str):
        async with ds.db_span(phase), eng.connect() as conn:
            for _ in range(n):
                await conn.execute(text("SELECT id FROM conversations"))
                await asyncio.sleep(0)

    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        await asyncio.gather(leg(2, "left"), leg(5, "right"))
    ds._reset_for_tests()

    by_phase = {}
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith("[PERF] db_span"):
            f = _fields(m)
            by_phase[f["phase"]] = f
    assert by_phase["left"]["stmts"] == "2", by_phase
    assert by_phase["right"]["stmts"] == "5", by_phase


@pytest.mark.asyncio
async def test_voice_slow_query_probe_and_db_span_count_the_same_query(
    caplog, save_engine, monkeypatch,
):
    """The Voice probe and the owner span both listen to cursor events.

    Production installs the Voice listener first. Both observers must finish
    the same statement without changing its result or leaving it in flight.
    """
    from sqlalchemy import text

    import app.db.db_span as ds
    from app.db import slow_query_probe

    eng, _maker = save_engine
    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr(
        "app.config.settings.turn_db_span_canary_user_ids", CANARY, raising=False,
    )
    ds._reset_for_tests()
    slow_query_probe.install(eng, warn_s=0)
    ds.install(eng)
    ds.begin_turn(user_id=CANARY, client_msg_id="voice-probe", channel="mobile", replace=True)
    try:
        with caplog.at_level(logging.INFO):
            async with ds.db_span("voice_probe_interop"), eng.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar_one() == 1
    finally:
        ds._reset_for_tests()

    spans = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[PERF] db_span phase=voice_probe_interop ")]
    slow = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[SLOW_SQL] ")]
    assert len(spans) == 1 and _fields(spans[0])["stmts"] == "1"
    assert len(slow) == 1
    assert slow_query_probe.active_snapshot() == "-"


@pytest.mark.asyncio
async def test_a_dial_is_timed_and_a_reused_connection_is_not(
    caplog, tmp_path, monkeypatch,
):
    """`dial_ms` is the fresh-connection cost — the number that decides
    whether "enable the pool" is the fix at all. It must count only real
    dials, or a warm pool would read as if it were still re-handshaking."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import QueuePool

    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/warm.db",
        poolclass=QueuePool, pool_size=2, max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ds.install(eng)
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    try:
        with caplog.at_level(logging.INFO, logger="app.db.db_span"):
            async with ds.db_span("cold"), eng.connect() as conn:
                await conn.execute(text("SELECT 1"))
            async with ds.db_span("warm"), eng.connect() as conn:
                await conn.execute(text("SELECT 1"))
    finally:
        await eng.dispose()
        ds._reset_for_tests()

    got = {}
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith("[PERF] db_span"):
            f = _fields(m)
            got[f["phase"]] = f
    assert got["cold"]["conns"] == "1" and got["cold"]["dials"] == "1"
    # Same checkout count, no dial: the pool handed back the warm connection.
    assert got["warm"]["conns"] == "1" and got["warm"]["dials"] == "0", got


# ── where the pooler's wait actually lands (R48 review, B-1) ─────────

def test_begin_ms_cannot_hold_a_round_trip_on_the_drivers_we_ship():
    """`begin_ms` is ORM-compile time, not a BEGIN round trip, and the module
    must not claim otherwise.

    The first draft of this instrument existed largely FOR that claim: a
    PgBouncer transaction-pooling wait happens at the transaction's first
    query, and `before_cursor_execute` was said to be blind to it. The wire
    half is true; the HOOK half is not. SQLAlchemy dispatches the `begin` event
    immediately before `dialect.do_begin()`, and on BOTH drivers this repo runs
    `do_begin` is inherited from `DefaultDialect`, whose body is `pass` —
    asyncpg issues the real BEGIN lazily inside `cursor.execute()`, i.e.
    between this module's before/after cursor hooks.

    So this test pins the semantics we ACTUALLY get, in the direction that
    matters: it goes RED the day a SQLAlchemy or asyncpg upgrade gives the
    dialect a real `do_begin` — which is the day `begin_ms` starts meaning
    something else and the docstring would otherwise still be right by
    accident."""
    from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
    from sqlalchemy.dialects.sqlite.aiosqlite import SQLiteDialect_aiosqlite
    from sqlalchemy.engine.default import DefaultDialect

    assert PGDialect_asyncpg.do_begin is DefaultDialect.do_begin
    assert SQLiteDialect_aiosqlite.do_begin is DefaultDialect.do_begin

    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "app/db/db_span.py").read_text()
    head = src[:src.index("PRIVACY.")]
    # The refuted claim, in the words it was written in, must be gone…
    assert "`begin_ms`\ndominant" not in head
    assert "begin_ms` dominant with fast statements" not in head
    # …and the correction must be present and reachable by a reader of a line.
    assert "ORM" in head and "stmt1_ms" in head


@pytest.mark.asyncio
async def test_stmt1_ms_is_the_first_statement_on_a_fresh_dial(
    caplog, tmp_path, monkeypatch,
):
    """The bucket that makes the pooler's signature MACHINE-READABLE.

    On asyncpg the server assignment, the per-connection type introspection
    and the BEGIN all happen inside the first `cursor.execute()` of a fresh
    connection. Without this field the line would read
    `begin_ms=1 sql_ms=3950 slow=select:conversations@3900` and the module's
    own decoder would call it a row lock on `conversations` — the confident
    fiction the round-1 critic flagged.

    Both directions are asserted, because a probe that only checks the
    present-case tests the direction failures do not run in: a delay on the
    FIRST statement lands in `stmt1_ms`, and the identical delay on the SECOND
    does not."""
    import time as _t

    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)

    async def run_with_delay_on(nth: int) -> dict:
        ds._reset_for_tests()
        eng = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/s1-{nth}.db",
            poolclass=NullPool,
            connect_args={"check_same_thread": False},
        )
        ds.install(eng)
        seen = {"n": 0}

        # Registered AFTER db_span's listeners, so it runs between this
        # module's `before_cursor_execute` (which starts the clock) and the
        # execution itself — exactly where a pooler wait sits on the wire.
        @event.listens_for(eng.sync_engine, "before_cursor_execute")
        def _slow(conn, cursor, statement, parameters, context, executemany):
            seen["n"] += 1
            if seen["n"] == nth:
                _t.sleep(0.12)

        ds.begin_turn(user_id="u1", client_msg_id="c", channel="web",
                      replace=True)
        try:
            with caplog.at_level(logging.INFO, logger="app.db.db_span"):
                caplog.clear()
                async with ds.db_span("p"), eng.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    await conn.execute(text("SELECT 2"))
            line = [r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("[PERF] db_span")][0]
        finally:
            await eng.dispose()
            ds._reset_for_tests()
        return _fields(line)

    first = await run_with_delay_on(1)
    assert first["dials"] == "1", first
    assert int(first["stmt1_ms"]) >= 100, first
    assert int(first["stmt1_ms"]) <= int(first["sql_ms"]), first

    second = await run_with_delay_on(2)
    assert second["dials"] == "1", second
    assert int(second["sql_ms"]) >= 100, second
    # The ABSENT direction: the same 120 ms, one statement later, must not
    # appear in stmt1_ms. Without this the field would be indistinguishable
    # from `sql_ms` and would settle nothing.
    assert int(second["stmt1_ms"]) < 100, second


@pytest.mark.asyncio
async def test_a_span_that_did_not_dial_reports_no_stmt1_ms(
    caplog, tmp_path, monkeypatch,
):
    """`dials=0 ⇒ stmt1_ms=0`, the ABSENT direction of the field's own
    docstring, on a pool that RETAINS connections.

    The mark a dial leaves on a connection was a bare `True`, and only the
    first statement executed on it consumed it. A span that dialled and ran
    NOTHING (a session opened and closed without a query) left it behind, and
    the next span to check that same warm connection out consumed it and
    charged its own first statement to `stmt1_ms` — while printing `dials=0`
    on the same line, the one combination the docstring says cannot occur.
    Reproduced LOCAL on QueuePool at the R48 final review
    (`conns=1 dials=0 sql_ms=125 stmt1_ms=125`).

    Reachability is low — it needs a connection-retaining pool, i.e. exactly
    the per-slot `AGENT_DB_POOL_SIZE` override this patch flags as possible
    and unverified — but the failure is a confident wrong reading of the one
    field the fresh-connection hypothesis rests on, which is worse than a
    missing one."""
    import time as _t

    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import QueuePool

    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/stale.db",
        poolclass=QueuePool, pool_size=2, max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ds.install(eng)

    # 120 ms on span B's one statement. Without it the misattributed value
    # would be a sub-millisecond `stmt1_ms=0` and the assertion below would
    # hold for the WRONG reason — the fast path cannot tell a correct zero
    # from a wrongly-charged one.
    @event.listens_for(eng.sync_engine, "before_cursor_execute")
    def _slow(conn, cursor, statement, parameters, context, executemany):
        _t.sleep(0.12)

    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    try:
        with caplog.at_level(logging.INFO, logger="app.db.db_span"):
            # Span A: dials, executes nothing, returns the connection warm.
            async with ds.db_span("dialled_only"), eng.connect():
                pass
            # Span B: reuses it. Its first statement is NOT a fresh-connection
            # first statement, and must not be reported as one.
            async with ds.db_span("reused"), eng.connect() as conn:
                await conn.execute(text("SELECT 1"))
    finally:
        await eng.dispose()
        ds._reset_for_tests()

    got = {}
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith("[PERF] db_span"):
            f = _fields(m)
            got[f["phase"]] = f
    assert got["dialled_only"]["dials"] == "1", got
    assert got["dialled_only"]["stmts"] == "0", got
    assert got["reused"]["dials"] == "0", got
    assert got["reused"]["stmts"] == "1", got
    assert int(got["reused"]["sql_ms"]) >= 100, got   # the delay landed
    assert got["reused"]["stmt1_ms"] == "0", got      # …and not here


def test_the_pending_entry_holds_its_connection_so_an_id_cannot_be_recycled():
    """`pending` is keyed on `id(Connection)`, and a `commit` entry is closed
    only by the NEXT wire event on that connection or by span exit. A span
    that opens two sessions (phase 1 opens at least two) could therefore have
    session B's Connection allocated at the address session A's freed one had,
    and charge the Python between A's commit and B's first statement to
    `commit_ms`. Holding the object makes the address unreusable for as long
    as the entry lives."""
    import app.db.db_span as ds

    s = ds._Span("x")
    token = ds._SPAN.set(s)
    try:
        class _Conn:
            pass

        c = _Conn()
        ds._on_commit(c)
        held, kind, _t0 = s.pending[id(c)]
        assert kind == "commit"
        assert held is c, "the entry does not pin its connection"
    finally:
        ds._SPAN.reset(token)


@pytest.mark.asyncio
async def test_lag_cov_says_so_when_the_ring_rolled_over_inside_the_span(
    monkeypatch,
):
    """`loop_lag_max_ms` from a ring that has rolled over is a FLOOR, not a
    maximum. The ring holds ~6.8 minutes at 50 ms, which an agentic turn can
    outlast — so the under-reading is labelled rather than silent, the same
    rule `src=` follows on the host line."""
    from collections import deque

    import app.db.db_span as ds

    monkeypatch.setattr(ds, "_LAG", deque(maxlen=2))
    now = 1000.0
    assert ds._lag_covers(now) is True          # empty: nothing was dropped
    ds._LAG.append((now + 1.0, 1.0))
    ds._LAG.append((now + 2.0, 1.0))
    assert ds._lag_covers(now) is False         # oldest sample is after t0
    ds._LAG.appendleft((now - 1.0, 1.0))
    assert ds._lag_covers(now) is True


# ── flag off: nothing installed, nothing logged, nothing changed ─────

def test_the_listeners_are_not_installed_when_the_flag_is_off(monkeypatch):
    """"Off" means ABSENT, not "present and returning early". `_build_engine`
    asks `armed()` and skips `install` entirely, so an ordinary container
    carries no db_span listeners. The Voice slow-query probe is separate."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids", "",
                        raising=False)
    assert ds.armed() is False

    from app.db.database import _build_engine

    eng = _build_engine("sqlite+aiosqlite:///:memory:")
    assert ds.is_installed(eng) is False

    # …and the canary list ALONE arms it, or a per-tenant rollout would be
    # impossible: the fleet-wide flag is the thing we do not want to flip.
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY, raising=False)
    assert ds.armed() is True
    eng2 = _build_engine("sqlite+aiosqlite:///:memory:")
    assert ds.is_installed(eng2) is True


@pytest.mark.asyncio
async def test_a_user_off_the_canary_list_gets_no_span_and_no_line(
    caplog, save_engine, monkeypatch,
):
    from sqlalchemy import text

    import app.db.db_span as ds

    eng, _maker = save_engine
    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY, raising=False)
    ds._reset_for_tests()
    ds.install(eng)  # installed, because the canary list arms the process
    ds.begin_turn(user_id=OTHER_USER, client_msg_id="c", channel="web",
                  replace=True)
    assert ds.turn_enabled() is False

    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        async with ds.db_span("save"), eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    assert not [r for r in caplog.records
                if r.getMessage().startswith("[PERF] db_span")]

    # …and the canary user on the same process DOES get one.
    ds.begin_turn(user_id=CANARY, client_msg_id="c", channel="web",
                  replace=True)
    assert ds.turn_enabled() is True
    with caplog.at_level(logging.INFO, logger="app.db.db_span"):
        async with ds.db_span("save"), eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    ds._reset_for_tests()
    assert [r for r in caplog.records
            if r.getMessage().startswith("[PERF] db_span")]


@pytest.mark.asyncio
async def test_an_unarmed_span_allocates_nothing(monkeypatch):
    """The no-op path is reused, not rebuilt: `db_span(...)` off-canary must
    not construct a span object per call site per turn on a 1-CPU cgroup."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids", "",
                        raising=False)
    ds._reset_for_tests()
    assert ds.db_span("a") is ds.db_span("b") is ds._NULL
    async with ds.db_span("a"):
        pass
    assert ds._SPAN.get() is None


@pytest.mark.asyncio
async def test_the_lag_sampler_runs_only_while_a_span_is_open(monkeypatch):
    """A 50 ms wake-up loop that outlived its span would be a permanent tax on
    a container that is already CPU-capped."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    assert ds._lag_task is None
    async with ds.db_span("a"):
        assert ds._lag_task is not None and not ds._lag_task.done()
        await asyncio.sleep(0.12)
    assert ds._lag_task is None
    ds._reset_for_tests()


@pytest.mark.asyncio
async def test_the_span_reports_a_loop_block_that_happened_inside_it(monkeypatch):
    """The whole point: a phase that was slow because the PROCESS could not
    run must not read as a slow database. `loop_health` cannot answer this —
    it samples at 250 ms and `snapshot()` covers a fixed 30 s window shared
    with every background loop and with the adjacent turn."""
    import time as _t

    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)
    async with ds.db_span("blocked"):
        await asyncio.sleep(0.06)
        _t.sleep(0.35)          # a GIL-holding block, exactly what starves it
        await asyncio.sleep(0.12)
    assert ds._lag_max_since(0.0) >= 250, ds._lag_max_since(0.0)
    ds._reset_for_tests()


# ── source-order probes: guards whose precondition can be destroyed ──

def test_the_save_span_wraps_the_async_with_not_the_phase3_timer():
    """`t_phase3` starts ~40 lines of pure-Python bookkeeping above the
    session. A span anchored there could never have its buckets add up, and
    the acceptance gate written against it would be unmeasurable — which is
    exactly what a1's skeptic objected to in the first sketch of this patch."""
    src = AGENT_RUNNER_SRC
    i_t3 = src.index("t_phase3 = time.perf_counter()")
    i_span = src.index('async with _db_span("save"), async_session_maker() as db:')
    i_save = src.index("_persisted = await self._save_messages(")
    assert i_t3 < i_span < i_save
    between = src[i_t3:i_span]
    assert "_db_span(" not in between, "the span moved up onto the timer"


def test_every_span_wraps_a_session_and_none_of_them_re_indents_its_body():
    """The combined `async with A, B as c:` form is what keeps this a one-line
    change at each call site. A span that does NOT carry its session on the
    same statement is wrapping something else — most likely a block that ends
    before the session's checkin, which is where a commit's cost lands."""
    for src, name in ((AGENT_RUNNER_SRC, "agent_runner"), (WS_CHAT_SRC, "ws_chat")):
        for m in re.finditer(r"async with _?_?db_span\([^)]*\)([^\n]*)", src):
            tail = m.group(1)
            assert "session_maker" in tail or "_sm()" in tail or "_maker()" in tail, (
                f"{name}: a db_span does not wrap a session on its own line: "
                f"{m.group(0)!r}"
            )


def test_the_turn_is_armed_above_the_first_db_work_of_the_turn():
    """The pre-turn lookups are the FIRST database work of a turn and the
    phase the sample showed at ~1.0 s. Arming below them would measure
    everything except the thing being investigated."""
    src = WS_CHAT_SRC
    i_arm = src.index("_db_span_mod.begin_turn(")
    i_lookups = src.index('_pt.start("lookups")')
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    assert i_arm < i_lookups < i_task, (i_arm, i_lookups, i_task)
    # …and `replace=True`, because the handler's context is shared by every
    # later iteration of the receive loop: the second message on a socket must
    # mint its own turn rather than inherit the first one's identity.
    assert "replace=True," in src[i_arm:i_arm + 400]


def test_ws_preturn_identity_uses_the_same_validated_id_as_the_runner():
    """A rejected client id must not create a pre-turn join key that the
    runner cannot carry after the WS handler removes that id from `msg`."""
    src = WS_CHAT_SRC
    i_local = src.index('_trace_client_msg_id = msg.get("client_msg_id")')
    i_arm = src.index("_db_span_mod.begin_turn(")
    i_late = src.index('_client_msg_id_top = msg.get("client_msg_id")')
    assert i_local < i_arm < i_late
    guard = src[i_local:i_arm]
    assert "not isinstance(_trace_client_msg_id, str)" in guard
    assert "len(_trace_client_msg_id) > _MAX_CLIENT_MSG_ID_LEN" in guard
    assert "_trace_client_msg_id = None" in guard
    assert "client_msg_id=_trace_client_msg_id," in src[i_arm:i_late]


def test_the_runner_never_overwrites_the_handlers_arming():
    """`create_task` copies the context, so a WS turn reaches `_run_inner`
    already armed — with the handler's `cmid_h` and, more importantly, the
    host-counter baseline taken BEFORE the pre-turn. A second `begin_turn`
    that replaced it would silently re-base the turn_host deltas."""
    src = AGENT_RUNNER_SRC
    i = src.index("_db_span_mod.begin_turn(")
    call = src[i:src.index(")", i)]
    assert "replace" not in call, call


def test_the_waterfall_gains_its_identity_only_when_the_instrument_is_armed():
    """`meta` is splatted into the JSON whole, so a `cmid_h: None` would still
    change every flag-off `[TURN_WATERFALL]` line. The key must be ADDED under
    the gate, never set inside the `meta.update({...})` literal."""
    src = AGENT_RUNNER_SRC
    i_update = src.index("_wf.meta.update({")
    literal = src[i_update:src.index("})", i_update)]
    assert "cmid_h" not in literal, (
        "cmid_h is inside the unconditional meta literal — flag-off lines move"
    )
    i_set = src.index('_wf.meta["cmid_h"]')
    guard = src[i_set - 900:i_set]
    assert "if _db_span_mod.turn_enabled():" in guard, guard
    # …and it is the one flag-on telemetry write at the very END of a turn,
    # after the answer has streamed, so it coerces its raw-client-input
    # argument the way `begin_turn` does and cannot raise (R48 review N3).
    assert 'str(client_msg_id or "")[:100]' in src[i_set:i_set + 200], (
        "the waterfall's cmid_h does not coerce client_msg_id — a non-str id "
        "would raise after the answer streamed"
    )
    assert "try:" in guard.rsplit("if _db_span_mod.turn_enabled():", 1)[-1], (
        "the waterfall's cmid_h write is not wrapped"
    )


@pytest.mark.asyncio
async def test_a_broken_instrument_never_costs_the_turn(monkeypatch, caplog):
    """Telemetry on the hot path fails silent. A field whose rendering blows
    up may not take the user's turn with it — the same rule `_PreTurn.emit`
    already follows."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", True, raising=False)
    ds._reset_for_tests()
    ds.begin_turn(user_id="u1", client_msg_id="c", channel="web", replace=True)

    def _boom(_t0):
        raise RuntimeError("no")

    monkeypatch.setattr(ds, "_lag_max_since", _boom)
    reached = False
    async with ds.db_span("broken"):
        reached = True
    assert reached
    # …and the span still released its ContextVar and its sampler.
    assert ds._SPAN.get() is None
    assert ds._lag_task is None
    ds._reset_for_tests()


def test_the_engine_asks_armed_before_installing():
    """`_build_engine` is the one place that decides; `install` itself is
    unconditional so a test can install on its own engine."""
    i = DB_SRC.index("from app.db import db_span as _db_span")
    tail = DB_SRC[i:i + 400]
    assert "_db_span.armed()" in tail
    assert tail.index("_db_span.armed()") < tail.index("_db_span.install(")


def test_the_flag_reads_its_env_var(monkeypatch):
    """A default-off flag nobody can turn on is worse than no flag: the code
    it guards is dead and the measurement never happens."""
    from app.config import Settings

    assert Settings.model_fields["turn_db_span"].default is False
    assert Settings.model_fields["turn_db_span_canary_user_ids"].default == ""
    monkeypatch.setenv("TURN_DB_SPAN", "true")
    monkeypatch.setenv("TURN_DB_SPAN_CANARY_USER_IDS", "abc,def")
    s = Settings()
    assert s.turn_db_span is True
    assert s.turn_db_span_canary_user_ids == "abc,def"


@pytest.mark.asyncio
async def test_the_unconditional_bookkeeping_is_silent_with_the_flag_off(
    caplog, monkeypatch,
):
    """Seven things in this patch run whether or not the flag is on: the
    imports, `enter_run`, `begin_turn`, `emit_turn_host`, `end_turn`,
    `exit_run`, and the `db_span(...)` call at each of the nine sites.
    "Default-off" is only true if each of them is inert, so this drives the
    whole unconditional path with the flag off and the canary list empty and
    asserts it produces NOTHING at any log level — no span object, no host
    line, no sampler task, no waterfall key."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids", "",
                        raising=False)
    ds._reset_for_tests()

    with caplog.at_level(logging.DEBUG, logger="app.db.db_span"):
        _tok = ds.enter_run()
        ds.begin_turn(user_id=CANARY, client_msg_id="c", channel="web",
                      replace=True)
        assert ds.turn_enabled() is False
        assert ds.db_span("save") is ds._NULL
        async with ds.db_span("save"):
            pass
        assert ds.emit_turn_host() is None
        ds.end_turn()
        ds.exit_run(_tok)

    assert caplog.records == [], [r.getMessage() for r in caplog.records]
    assert ds._SPAN.get() is None
    assert ds._lag_task is None
    # …and the depth is restored, so the next run() in this context is not
    # mistaken for a child of this one.
    assert ds._RUN_NEST.get() == 0
    ds._reset_for_tests()


# ── a truncated canary id must be LOUD, never a silent no-op ─────────
#
# This patch's own round-2 rollout instructions told an operator to set
# TURN_DB_SPAN_CANARY_USER_IDS to the 8-character prefix that names the
# CONTAINER. Matching is exact, so that would have armed the process, opened
# no span, printed no line — and every acceptance gate would have read like a
# healthy, quiet control. These tests pin the three halves of the fix: the
# prefix still does NOT enable anything, the mistake is reported at ERROR, and
# a list of nothing but bad entries does not arm the engine at all.

def _errors(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.ERROR]


def test_a_truncated_canary_id_does_not_enable_the_span_and_it_says_so(
    caplog, monkeypatch,
):
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY[:8], raising=False)
    ds._reset_for_tests()

    with caplog.at_level(logging.ERROR, logger="app.db.db_span"):
        enabled = ds.span_enabled(CANARY)
    ds._reset_for_tests()

    # The half that matters most: a prefix is reported, never honoured.
    assert enabled is False
    msgs = _errors(caplog)
    assert msgs, "a truncated id was a silent no-op"
    joined = " ".join(msgs)
    assert "TURN_DB_SPAN_CANARY_USER_IDS" in joined, joined
    assert "EXACT" in joined, joined
    assert CANARY[:8] in joined, joined
    # BOTH detections must fire, asserted separately. The R48 final review
    # found this test green under two different single-mechanism mutations
    # (the prefix report removed; `_MIN_PLAUSIBLE_ID_LEN = 1`) because either
    # message on its own satisfied the joined assertions above — a test that
    # cannot tell which half of the fix is alive.
    assert [m for m in msgs if "PREFIX" in m], msgs
    assert [m for m in msgs if "characters" in m and "uuid" in m], msgs


def test_the_truncation_error_fires_once_per_process_per_entry(
    caplog, monkeypatch,
):
    """It sits on the per-turn path, so an un-deduplicated ERROR would be one
    line per turn forever on a mistyped canary — the log-flood failure the
    round-1 finding about a blocking log pipe is about."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY[:8], raising=False)
    ds._reset_for_tests()
    with caplog.at_level(logging.ERROR, logger="app.db.db_span"):
        for _ in range(25):
            assert ds.span_enabled(CANARY) is False
    ds._reset_for_tests()

    prefix = [m for m in _errors(caplog) if "PREFIX" in m]
    assert len(prefix) == 1, prefix


def test_an_implausibly_short_entry_does_not_arm_the_engine(caplog, monkeypatch):
    """`armed()` installs eight engine listeners that fire on every statement
    of every background loop. A list whose every entry `span_enabled` will
    refuse can never open a span, so arming for it is pure cost for a
    measurement that cannot happen — and it is the state that reads as a
    healthy control."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY[:8], raising=False)
    ds._reset_for_tests()
    with caplog.at_level(logging.ERROR, logger="app.db.db_span"):
        armed = ds.armed()
    ds._reset_for_tests()

    assert armed is False
    joined = " ".join(_errors(caplog))
    assert "TURN_DB_SPAN_CANARY_USER_IDS" in joined, joined
    assert "NOT armed" in joined, joined


def test_a_mixed_list_still_arms_on_its_one_usable_entry(caplog, monkeypatch):
    """The absent direction of the two tests above: the rejection drops the bad
    entry, it does not disable the feature. A rollout that named the right user
    AND a stale prefix must still measure the right user."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        f"{CANARY[:8]},{CANARY}", raising=False)
    ds._reset_for_tests()
    with caplog.at_level(logging.ERROR, logger="app.db.db_span"):
        armed = ds.armed()
        good = ds.span_enabled(CANARY)
        other = ds.span_enabled(OTHER_USER)
    ds._reset_for_tests()

    assert armed is True
    assert good is True
    assert other is False
    assert [m for m in _errors(caplog) if "characters" in m], "bad entry unreported"


def test_a_correct_full_uuid_is_silent_and_is_never_written_to_the_log(
    caplog, monkeypatch,
):
    """The other absent direction, and a privacy one: a correctly configured
    canary must produce NO diagnostic at all, and the id it is configured with
    must not be echoed into the log by the code that exists to complain about
    bad ones."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        CANARY, raising=False)
    ds._reset_for_tests()
    with caplog.at_level(logging.DEBUG, logger="app.db.db_span"):
        assert ds.armed() is True
        assert ds.span_enabled(CANARY) is True
        assert ds.span_enabled(OTHER_USER) is False
    ds._reset_for_tests()

    assert _errors(caplog) == []
    assert not [r for r in caplog.records if CANARY in r.getMessage()]


@pytest.mark.parametrize("keep", [9, 20, 30, 31, 32, 35])
def test_a_nearly_complete_entry_is_reported_without_printing_the_id(
    caplog, monkeypatch, keep,
):
    """An entry long enough to be most of a real uuid is still a prefix, so it
    is still reported — but the ERROR may not carry it whole, or the guard
    against pasting ids into logs would be defeated by the guard against
    pasting prefixes into config.

    Parametrised ACROSS the old truncation boundary. The shipped `_show`
    truncated only at `_MIN_PLAUSIBLE_ID_LEN` (32), so a 31-character entry —
    86 % of a hyphenated uuid — printed WHOLE, from the one function whose
    docstring promised it could not. Measured LOCAL at R48 final review
    (31 → whole, 32 → truncated), and the round-2 test probed 35 characters
    only: the safe side. 9 is the first length past `_SHOW_CHARS`."""
    import app.db.db_span as ds

    nearly = CANARY[:keep]
    assert len(nearly) == keep
    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                        nearly, raising=False)
    ds._reset_for_tests()
    with caplog.at_level(logging.ERROR, logger="app.db.db_span"):
        assert ds.span_enabled(CANARY) is False
    ds._reset_for_tests()

    joined = " ".join(_errors(caplog))
    assert "PREFIX" in joined, joined
    assert nearly not in joined, joined
    # …and what IS printed is bounded, not merely "not the whole entry".
    assert nearly[:ds._SHOW_CHARS] in joined, joined
    assert nearly[:ds._SHOW_CHARS + 1] not in joined, joined


def test_no_entry_of_any_length_prints_more_than_eight_characters():
    """The direct property, independent of who calls `_show`. An id in a log
    file is the failure this whole rejection path exists to avoid, so the
    bound is asserted on the renderer rather than inferred from one message."""
    import app.db.db_span as ds

    assert ds._SHOW_CHARS == 8
    alphabet = (CANARY.replace("-", "x") * 3)
    for n in range(0, 60):
        entry = alphabet[:n]
        assert len(entry) == n
        shown = ds._show(entry)
        assert entry[:ds._SHOW_CHARS] in shown or not entry, (n, shown)
        if n > ds._SHOW_CHARS:
            assert entry not in shown, (n, shown)
            assert entry[:ds._SHOW_CHARS + 1] not in shown, (n, shown)
            assert str(n) in shown, (n, shown)


def test_the_rejection_never_raises_on_a_hostile_setting(monkeypatch):
    """It runs on the per-turn path above the handler's own validation, so it
    is handed whatever the environment holds."""
    import app.db.db_span as ds

    monkeypatch.setattr("app.config.settings.turn_db_span", False, raising=False)
    for raw in (",,,", "   ", "a" * 5000, "x,\n,y", CANARY + ",", 12345, None):
        monkeypatch.setattr("app.config.settings.turn_db_span_canary_user_ids",
                            raw, raising=False)
        ds._reset_for_tests()
        assert ds.span_enabled(CANARY) in (True, False)
        assert ds.armed() in (True, False)
    ds._reset_for_tests()
