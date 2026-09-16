"""The agent gets a bounded warm pool, not a connection per query.

Measured on the VPS, 2026-09-15. With `NullPool`, every agent DB session was a
fresh TCP + SCRAM handshake through the shared pgbouncer: the pooler's own log
shows **14 connect/close pairs in one second** for a single tenant at 14:49:50,
and there are ~111 tenant containers. Postgres is at `max_connections = 300`,
and `server login failed: FATAL remaining connection slots are reserved...`
appears during ORDINARY load — 42 s before that day's signup, hours before the
outage. Every pooler restart then produced a thundering herd against a ceiling
that had no headroom to begin with.

The platform process was moved off NullPool for exactly this reason (see the
comment in `_build_engine_inner`); the agent never got the fix, because a
connectionless pool made the generic->tenant rebind trivially clean. That is
still the one hazard, and it is answered rather than avoided:
`rebind_database()` disposes the old engine, and `dispose_engine()` exists for
the identity-only `/admin/bind` path, which changes the tenant WITHOUT changing
DATABASE_URL.

`agent_db_pool_size = 0` restores NullPool — the rollback switch, asserted here
so it cannot rot.

This file builds NO real Postgres engine: it captures the kwargs
`create_async_engine` would receive, so it runs in the plain sweep (the sweep
venv has no asyncpg) and asserts the DECISION rather than a driver.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. \
        pytest tests/test_agent_engine_pooling.py -q
"""
from __future__ import annotations

import pytest

PG_URL = "postgresql+asyncpg://u:p@host.docker.internal:6432/toup_agent_feed0082"

pytestmark = pytest.mark.asyncio


def _captured(monkeypatch, *, run_mode: str, pool_size=None, url: str = PG_URL) -> dict:
    """The kwargs `_build_engine_inner` hands to SQLAlchemy for this config."""
    import app.db.database as db
    from app.config import settings

    seen: dict = {}

    def _fake(u, **kw):
        seen.update(kw)
        seen["url"] = u
        return object()

    monkeypatch.setattr(db, "create_async_engine", _fake)
    monkeypatch.setattr(settings, "run_mode", run_mode, raising=False)
    if pool_size is not None:
        monkeypatch.setattr(settings, "agent_db_pool_size", pool_size, raising=False)
    db._build_engine_inner(url)
    return seen


def _is_nullpool(kw: dict) -> bool:
    from sqlalchemy.pool import NullPool
    return kw.get("poolclass") is NullPool


# ── anti-vacuity ─────────────────────────────────────────────────


async def test_the_capture_sees_the_real_builder(monkeypatch):
    kw = _captured(monkeypatch, run_mode="platform")
    assert kw["url"].startswith("postgresql+asyncpg://")
    assert "prepared_statement_cache_size=0" in kw["url"], (
        "the builder was not actually exercised — every assertion below would "
        "be reading an empty dict"
    )


# ── the change ───────────────────────────────────────────────────


async def test_the_agent_gets_a_bounded_warm_pool(monkeypatch):
    kw = _captured(monkeypatch, run_mode="agent", pool_size=3)
    assert not _is_nullpool(kw), (
        "the agent is back on NullPool: every session is a fresh TCP+SCRAM "
        "handshake through the shared pooler, ~111 containers against "
        "max_connections=300"
    )
    assert kw["pool_size"] == 3
    assert kw["max_overflow"] >= 6, (
        "no headroom: this container runs several independent background "
        "loops (routines, reminders, radio, health, memory) alongside a turn "
        "that demonstrably holds a session for tens of seconds "
        "(`phase3_save: 34505ms`, 2026-09-15)"
    )
    assert kw["pool_recycle"] == 300
    assert kw["pool_pre_ping"] is True, (
        "pre_ping on a WARM connection is one cheap round-trip that catches a "
        "connection the pooler already dropped — that is the case this change "
        "creates"
    )


async def test_zero_restores_nullpool_as_the_rollback_switch(monkeypatch):
    kw = _captured(monkeypatch, run_mode="agent", pool_size=0)
    assert _is_nullpool(kw)
    assert "pool_size" not in kw


async def test_the_platform_pool_is_untouched(monkeypatch):
    kw = _captured(monkeypatch, run_mode="platform")
    assert kw["pool_size"] == 10 and kw["max_overflow"] == 10
    assert kw["pool_recycle"] == 300


async def test_sqlite_is_untouched(monkeypatch):
    kw = _captured(monkeypatch, run_mode="agent", pool_size=3,
                   url="sqlite+aiosqlite:///./toup.db")
    assert _is_nullpool(kw), (
        "file-backed sqlite must keep NullPool — a shared connection lets a "
        "background rollback eat a request handler's flushed writes"
    )


# ── the hazard the bounded pool creates, and its answer ──────────


async def test_dispose_engine_drops_warm_connections_and_says_so(tmp_path):
    """`/admin/bind` swaps the tenant identity without swapping DATABASE_URL,
    so nothing else would drop a connection opened under the previous one.

    A file-backed sqlite on a real queue pool stands in for the Postgres
    pool: what matters is that the engine HOLDS connections it can drop."""
    import app.db.database as db
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import AsyncAdaptedQueuePool

    throwaway = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/warm.db",
        poolclass=AsyncAdaptedQueuePool, pool_size=2, max_overflow=0,
    )
    async with throwaway.connect() as c:
        await c.execute(text("select 1"))
    assert throwaway.pool.checkedin() == 1, "the pool should be holding a warm connection"
    prev = db._holder["engine"]
    db._holder["engine"] = throwaway
    try:
        assert await db.dispose_engine(reason="admin_bind") is True
        assert throwaway.pool.checkedin() == 0, "dispose left the warm connection in the pool"
    finally:
        db._holder["engine"] = prev
    await throwaway.dispose()


async def test_dispose_engine_leaves_an_in_memory_database_alone():
    """On a StaticPool the one connection IS the database (`_build_engine_inner`'s
    in-memory shape). Disposing it dropped every table under a live bind —
    `/admin/bind` answered 200 and the next query raised `no such table`."""
    import app.db.database as db
    from sqlalchemy import text

    throwaway = db._build_engine("sqlite+aiosqlite:///:memory:")
    async with throwaway.begin() as c:
        await c.execute(text("create table probe (id integer primary key)"))
        await c.execute(text("insert into probe (id) values (7)"))
    prev = db._holder["engine"]
    db._holder["engine"] = throwaway
    try:
        assert await db.dispose_engine(reason="admin_bind") is False
        async with throwaway.connect() as c:
            rows = (await c.execute(text("select id from probe"))).scalars().all()
        assert rows == [7], "the in-memory database did not survive dispose_engine"
    finally:
        db._holder["engine"] = prev
        await throwaway.dispose()


async def test_dispose_engine_is_a_no_op_on_nullpool(tmp_path):
    """Safe to call unconditionally: the caller must not have to know which
    pool this process built."""
    import app.db.database as db

    throwaway = db._build_engine(f"sqlite+aiosqlite:///{tmp_path}/probe.db")
    prev = db._holder["engine"]
    db._holder["engine"] = throwaway
    try:
        assert await db.dispose_engine(reason="admin_bind") is False
    finally:
        db._holder["engine"] = prev
        await throwaway.dispose()


async def test_dispose_engine_never_raises_into_a_bind(monkeypatch):
    """A bind must not fail because a connection refused to close."""
    import app.db.database as db

    class _Boom:
        class pool:  # noqa: D106 - only its class NAME is read
            pass

        async def dispose(self):
            raise OSError("broken pipe")

    prev = db._holder["engine"]
    db._holder["engine"] = _Boom()
    try:
        assert await db.dispose_engine(reason="admin_bind") is False
    finally:
        db._holder["engine"] = prev


# ── fix lane A: saturation must fail fast and namably ─────────────


async def test_the_agent_pool_bounds_how_long_a_checkout_waits(monkeypatch):
    """SQLAlchemy's default `pool_timeout` is 30 s of SILENT waiting, and the
    `sqlalchemy.exc.TimeoutError` it finally raises is in `_SA_INFRA` — so a
    pool-exhaustion bug would render as a 503 `backend_unavailable` with
    nothing naming the cause. Bounded, and overridable per container."""
    kw = _captured(monkeypatch, run_mode="agent", pool_size=3)
    assert "pool_timeout" in kw, "a saturated agent pool stalls for 30 s"
    assert 0 < kw["pool_timeout"] <= 15


async def test_the_pool_timeout_is_a_setting_not_a_constant(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "agent_db_pool_timeout_s", 4.0, raising=False)
    kw = _captured(monkeypatch, run_mode="agent", pool_size=3)
    assert kw["pool_timeout"] == 4.0


async def test_bound_parameters_never_reach_an_exception_string(monkeypatch):
    """`StatementError.__str__` renders `[parameters: (...)]`, and a failed
    message INSERT logged with `logger.exception` would put the user's own
    text in the trail."""
    for mode in ("agent", "platform"):
        kw = _captured(monkeypatch, run_mode=mode, pool_size=3)
        assert kw.get("hide_parameters") is True, mode


async def test_admin_bind_actually_calls_dispose_engine():
    """The cases above prove the HELPER works when called. Nothing asserted
    that `/admin/bind` — the one identity swap that does not change
    DATABASE_URL, and therefore the whole reason the helper exists — calls it,
    or that it calls it BEFORE the first post-bind session is opened.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "api"
           / "admin_pool.py").read_text()
    body = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == "admin_bind":
            body = ast.unparse(node)
            break
    assert body, "admin_bind not found"

    dispose = body.find("dispose_engine(")
    assert dispose != -1, (
        "/admin/bind no longer disposes the engine — warm connections opened "
        "under the PREVIOUS tenant identity now serve the new one"
    )
    write = body.find("write_runtime(")
    assert write != -1
    assert write < dispose, (
        "the dispose runs before the new identity is written; the pool would "
        "be refilled under the old one"
    )
    # The other bound, and the one presence alone would miss: the dispose must
    # also come BEFORE the first session opened under the new identity (step 2b
    # writes the owner user row). Moved after it, the warm connections it drops
    # are ones the NEW tenant just opened, and the stale ones already served.
    first_session = body.find("real_user_name")
    assert first_session != -1, (
        "admin_bind no longer builds the owner user row — re-anchor this probe "
        "on whatever now opens the first post-bind session"
    )
    assert dispose < first_session, (
        "the dispose runs after the first post-bind session; a connection "
        "opened under the PREVIOUS tenant identity served the new one"
    )


# ── fix lane A: the rollout posture and the saturation signal ──────


async def test_the_shipped_default_is_the_rollback_value():
    """Warm connections multiply by ~111 containers, so this round ships the
    code with the switch OFF (decision D26): the bounded pool is enabled per
    slot with AGENT_DB_POOL_SIZE, and only after the host's
    max_client_conn/max_db_connections step in bridge/INSTALL.md is verified.
    A default of 3 would enable it fleet-wide on the next agent image."""
    from app.config import Settings

    assert Settings.model_fields["agent_db_pool_size"].default == 0


async def test_background_loops_cannot_starve_a_turn(monkeypatch):
    """The container runs several independent background loops (routines,
    reminders, radio, health, memory) alongside a turn that demonstrably holds
    a session for tens of seconds. Overflow is the headroom that keeps a slow
    success a success."""
    for size in (1, 3, 5):
        kw = _captured(monkeypatch, run_mode="agent", pool_size=size)
        assert kw["max_overflow"] >= max(6, size * 3), (
            f"pool_size={size} left only {kw['max_overflow']} of overflow"
        )


async def test_a_bounded_pool_publishes_its_occupancy(monkeypatch):
    """Saturation otherwise surfaces only as a `pool_timeout` wait and then a
    503 `backend_unavailable`, with nothing naming the cause."""
    import app.db.database as db
    from app.services import health_signals as hs

    class _Pool:
        def checkedout(self):
            return 4

        def overflow(self):
            return 2

    class _Eng:
        pool = _Pool()

    hs._GAUGES.pop("db_pool_checked_out", None)
    db._register_pool_gauge(_Eng())
    assert "db_pool_checked_out" in hs._GAUGES
    await hs.refresh_gauges(only="db_pool_checked_out", force=True)
    assert hs.snapshot()["db_pool_checked_out"] == 6

    # NullPool has no occupancy to report, and a registration there would
    # publish a permanent 0 that reads as "measured and healthy".
    from sqlalchemy.pool import NullPool

    class _NullEng:
        pool = NullPool(creator=lambda: None)

    hs._GAUGES.pop("db_pool_checked_out", None)
    db._register_pool_gauge(_NullEng())
    assert "db_pool_checked_out" not in hs._GAUGES


async def test_the_gauge_key_is_part_of_the_stable_shape():
    """A reader must be able to tell "this image does not report it" from
    "it reported zero"."""
    from app.services import health_signals as hs

    assert "db_pool_checked_out" in hs.KNOWN_SIGNALS
