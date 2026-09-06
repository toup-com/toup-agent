"""Cross-replica dedupe — the two prewarms 19 ms apart.

At 18:17:39.262 and 18:17:39.281 on 2026-09-06 BOTH platform replicas started
a prewarm for the same freshly-registered user, and both drove a full cold
`provision_container` (POST /v1/tenants — the NAMED path) for a user the pool
was already binding. Nothing serialised them:

  * `prewarm_service.schedule_prewarm`'s status check is a read on one
    replica's snapshot,
  * `docker_host_service._ENV_PUSH_LOCKS` is an in-PROCESS asyncio.Lock,
  * `pool_service.claim_for_user`'s advisory lock covers the CLAIM path only —
    `provision_container` takes no lock at all.

Two halves, both real:
  1. the primitive — `pg_try_advisory_xact_lock` really does exclude a second
     session, on a real Postgres, and really does release on commit;
  2. the wiring — `_run_prewarm` consults it and a loser does not provision.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest

PG_DSN = os.environ.get("LANE_C_PG_DSN", "postgresql://localhost/postgres")
_port = itertools.count(9500)


# ── 1. The primitive, on real Postgres ───────────────────────────────────

@pytest.mark.asyncio
async def test_advisory_xact_lock_excludes_a_second_replica():
    """MUTATION: swap `pg_try_advisory_xact_lock` for
    `pg_advisory_xact_lock` (the blocking form claim_for_user used) and the
    second acquire never returns — the test hangs instead of proving
    exclusion, which is precisely the difference between "observe" and
    "queue behind a 30 s bridge call"."""
    asyncpg = pytest.importorskip("asyncpg")
    try:
        admin = await asyncpg.connect(PG_DSN)
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"no local Postgres: {e}")
    db = f"toup_lane_c_{uuid.uuid4().hex[:8]}"
    await admin.execute(f'CREATE DATABASE "{db}"')
    await admin.close()
    dsn = PG_DSN.rsplit("/", 1)[0] + "/" + db
    key_sql = "SELECT pg_try_advisory_xact_lock(hashtext($1)::bigint)"
    key = f"provision_drive:{uuid.uuid4()}"
    try:
        a = await asyncpg.connect(dsn)
        b = await asyncpg.connect(dsn)
        try:
            ta = a.transaction(); await ta.start()
            tb = b.transaction(); await tb.start()

            assert await a.fetchval(key_sql, key) is True, "the first driver wins"
            assert await b.fetchval(key_sql, key) is False, (
                "a second replica must NOT also get the drive — this is the "
                "18:17:39.262 / .281 duplicate"
            )
            # A different user is a different lock: dedupe must not serialise
            # unrelated signups behind each other.
            assert await b.fetchval(key_sql, f"provision_drive:{uuid.uuid4()}") is True

            # Releasing is structural: the lock dies with the transaction, so a
            # replica killed mid-bridge-call (a Railway redeploy) frees it with
            # no TTL bookkeeping and no stuck lease.
            await ta.commit()
            assert await b.fetchval(key_sql, key) is True, (
                "the lock must release on commit"
            )
            await tb.rollback()
        finally:
            await a.close()
            await b.close()
    finally:
        admin = await asyncpg.connect(PG_DSN)
        await admin.execute(
            f'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
            f"WHERE datname = '{db}'"
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db}"')
        await admin.close()


# ── 2. The wiring ────────────────────────────────────────────────────────

async def _seed(db):
    from app.db.models import User, AgentConfig
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                name="T", is_active=True))
    await db.flush()
    db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                       bundle_status="active", llm_mode="bundle"))
    await db.commit()
    return uid


@pytest.mark.asyncio
async def test_only_one_concurrent_prewarm_provisions(monkeypatch):
    """Two `_run_prewarm` tasks for the same user, exactly as the two replicas
    did. One must drive; the other must observe.

    The gate below stands in for the Postgres lock (the test DB is sqlite,
    where `try_take_provision_drive` correctly answers True — there is only
    one process). `gate.consulted` is the falsifier: the pre-fix `_run_prewarm`
    never asks anyone whether it should drive, so it reads 0 and BOTH
    provisions run.
    """
    from app.db import async_session_maker
    from app.services import prewarm_service as pw
    from app.services import pool_service as ps
    from app.config import settings

    monkeypatch.setattr(settings, "provision_discovery_enabled", False, raising=False)

    async with async_session_maker() as db:
        uid = await _seed(db)

    gate = {"granted": 0, "consulted": 0}

    async def _one_shot(_db, _uid):
        gate["consulted"] += 1
        if gate["granted"] == 0:
            gate["granted"] += 1
            return True
        return False
    monkeypatch.setattr(ps, "try_take_provision_drive", _one_shot)

    provisions = {"n": 0}

    async def _fake_provision(db, user_id, agent_config=None, **kw):
        provisions["n"] += 1
        await asyncio.sleep(0.2)          # the bridge call the winner is inside
        from app.db.models import ManagedContainer
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=user_id,
            container_name="toup-agent-pool-73", host_port=next(_port),
            db_name="d", status="running",
        ))
        agent_config.agent_url = f"https://agent-{user_id[:8]}.agents.toup.ai"
        agent_config.agent_api_key = "k"
        await db.commit()
        return None
    from app.services import docker_host_service as dhs
    monkeypatch.setattr(dhs, "provision_container", _fake_provision)

    async def _no_boot_poll(*a, **k):
        return None
    monkeypatch.setattr(pw, "_await_boot_ready", _no_boot_poll)

    await asyncio.gather(pw._run_prewarm(uid), pw._run_prewarm(uid))

    assert gate["consulted"] == 2, (
        f"both prewarms must ASK whether they own the drive; asked "
        f"{gate['consulted']} time(s) — 0 is the pre-fix tree, where nothing "
        f"serialises the two replicas"
    )
    assert provisions["n"] == 1, (
        f"exactly one driver may provision; {provisions['n']} did — this is "
        f"the 2026-09-06 duplicate"
    )

    from sqlalchemy import select
    from app.db.models import ManagedContainer
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == uid)
        )).scalars().all()
    assert len(rows) == 1, f"one assignment per user; got {len(rows)}"


@pytest.mark.asyncio
async def test_a_losing_prewarm_observes_instead_of_going_quiet(monkeypatch):
    """The loser must not simply return: nobody would then notice if the
    winner's bridge call lost its response. It starts discovery."""
    from app.db import async_session_maker
    from app.services import prewarm_service as pw
    from app.services import pool_service as ps
    from app.config import settings

    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db)

    async def _denied(_db, _uid):
        return False
    monkeypatch.setattr(ps, "try_take_provision_drive", _denied)

    started: list = []
    monkeypatch.setattr(
        ps, "ensure_discovery",
        lambda u, **k: started.append((u, k.get("reason"))),
    )
    monkeypatch.setattr(pw, "_await_boot_ready", lambda *a, **k: asyncio.sleep(0))

    from app.services import docker_host_service as dhs

    async def _must_not_run(*a, **k):
        raise AssertionError("a losing driver must not provision")
    monkeypatch.setattr(dhs, "provision_container", _must_not_run)

    await pw._run_prewarm(uid)
    assert started and started[0][0] == uid, (
        "the observer must start discovery, not go quiet"
    )


@pytest.mark.asyncio
async def test_a_failed_provision_starts_discovery(monkeypatch):
    """The winner's own timeout path. `provision_container` raising is NOT
    evidence that nothing was provisioned — at 18:18:09 it had been."""
    from app.db import async_session_maker
    from app.services import prewarm_service as pw
    from app.services import pool_service as ps
    from app.config import settings
    import httpx

    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)
    async with async_session_maker() as db:
        uid = await _seed(db)

    async def _granted(_db, _uid):
        return True
    monkeypatch.setattr(ps, "try_take_provision_drive", _granted)

    started: list = []
    monkeypatch.setattr(
        ps, "ensure_discovery",
        lambda u, **k: started.append((u, k.get("reason"))),
    )

    from app.services import docker_host_service as dhs

    async def _timeout(*a, **k):
        raise RuntimeError(f"bridge unreachable: {httpx.ReadTimeout('')!r}")
    monkeypatch.setattr(dhs, "provision_container", _timeout)

    await pw._run_prewarm(uid)
    assert started and started[0][0] == uid, (
        "a provisioning call that lost its response must be followed up"
    )


@pytest.mark.asyncio
async def test_drive_lock_fails_open(monkeypatch):
    """A lock we cannot evaluate is not a reason to skip provisioning: a
    duplicate drive is idempotent, a skipped one is a dead agent."""
    from app.services import pool_service as ps

    class _Boom:
        async def execute(self, *a, **k):
            raise RuntimeError("connection reset")

    from app.db import database as _dbmod

    class _PgDialect:
        name = "postgresql"

    class _Eng:
        dialect = _PgDialect()

    monkeypatch.setattr(_dbmod, "get_engine", lambda: _Eng())
    assert await ps.try_take_provision_drive(_Boom(), "u") is True


@pytest.mark.asyncio
async def test_lock_timeout_really_bounds_an_advisory_lock():
    """`claim_for_user`'s bounded wait is `SET LOCAL lock_timeout` + the
    BLOCKING `pg_advisory_xact_lock` — because the blocking form is what gives
    the loser the winner's committed row when it finally gets in, and only the
    timeout makes the wait finite.

    That only works if `lock_timeout` applies to ADVISORY lock acquisition.
    It is not obvious that it does (the docs describe it in terms of table and
    row locks), and if it did not, the "observe instead of queueing" branch
    would be unreachable code that every unit test still passes — the exact
    shape of defect this repo's guards exist to catch. Measured here against a
    real Postgres. VERIFIED PG 14.19: raises LockNotAvailableError at ~2.08 s.
    """
    import time
    asyncpg = pytest.importorskip("asyncpg")
    try:
        admin = await asyncpg.connect(PG_DSN)
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"no local Postgres: {e}")
    db = f"toup_lane_c_{uuid.uuid4().hex[:8]}"
    await admin.execute(f'CREATE DATABASE "{db}"')
    await admin.close()
    dsn = PG_DSN.rsplit("/", 1)[0] + "/" + db
    key = f"pool_claim:{uuid.uuid4()}"
    lock_sql = "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)"
    try:
        a = await asyncpg.connect(dsn)
        b = await asyncpg.connect(dsn)
        try:
            ta = a.transaction(); await ta.start()
            await a.execute(lock_sql, key)          # the winner holds it

            tb = b.transaction(); await tb.start()
            await b.execute("SET LOCAL lock_timeout = '2s'")
            t0 = time.monotonic()
            with pytest.raises(asyncpg.exceptions.LockNotAvailableError):
                await b.execute(lock_sql, key)
            waited = time.monotonic() - t0
            assert 1.0 < waited < 5.0, (
                f"the loser waited {waited:.2f}s — the bound must be the "
                f"configured one, not zero (a try-lock) and not forever"
            )
            await tb.rollback()

            # And the timeout is per-transaction, so it cannot leak onto the
            # bridge-call statements that follow in the winner's transaction.
            tb2 = b.transaction(); await tb2.start()
            await b.execute("SET LOCAL lock_timeout = 0")
            assert await b.fetchval("SHOW lock_timeout") == "0"
            await tb2.rollback()
            await ta.rollback()
        finally:
            await a.close()
            await b.close()
    finally:
        admin = await asyncpg.connect(PG_DSN)
        await admin.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db}'"
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db}"')
        await admin.close()


def test_claim_lock_wait_is_actually_wired():
    """A guard whose precondition something above it destroys is invisible to
    every other check in this repo. The bounded wait only exists if BOTH the
    SET LOCAL and the blocking acquire are in the source, in that order."""
    import inspect
    from app.services import pool_service as ps
    src = inspect.getsource(ps.claim_for_user)
    i_set = src.find("SET LOCAL lock_timeout")
    i_lock = src.find("pg_advisory_xact_lock")
    i_raise = src.find("ProvisionDriveTaken")
    assert i_set != -1, "the bounded wait lost its lock_timeout"
    assert i_lock != -1
    assert i_set < i_lock, (
        "lock_timeout must be set BEFORE the acquire it is meant to bound"
    )
    assert i_raise > i_lock, (
        "a loser must raise ProvisionDriveTaken so claim_or_prewarm knows not "
        "to start a second cold provision"
    )


def test_the_drive_lock_is_taken_before_provisioning_and_nothing_commits_between():
    """ORDER, not just presence. The lock is transaction-scoped, so any commit
    between taking it and the bridge call silently downgrades the dedupe to a
    no-op — a guard whose precondition something above it destroys, which is
    invisible to every other check in this repo.
    """
    import inspect
    from app.services import prewarm_service as pw
    src = inspect.getsource(pw._run_prewarm)
    i_lock = src.find("try_take_provision_drive")
    i_prov = src.find("provision_container(")
    assert i_lock != -1, "the cross-replica dedupe is gone"
    assert i_prov != -1
    assert i_lock < i_prov, "the drive must be claimed BEFORE provisioning"
    between = src[i_lock:i_prov]
    assert "commit()" not in between, (
        "a commit between the advisory lock and the bridge call releases the "
        "lock — the dedupe would compile, log, and do nothing"
    )
    assert "ensure_discovery" in src, (
        "a loser that goes quiet is worse than a duplicate: nobody would then "
        "notice if the winner's bridge call lost its response"
    )


def test_only_a_real_lock_timeout_counts_as_contention():
    """MUTATION: widen `_is_lock_timeout` to `return True` and a connection
    reset during the lock becomes "somebody else is driving" — claim_or_prewarm
    returns True, nobody provisions, and the user ends up with no agent at all
    while every log line says the system is fine."""
    from app.services.pool_service import _is_lock_timeout

    class _PgErr(Exception):
        sqlstate = "55P03"

    class _Wrapped(Exception):
        def __init__(self, orig):
            self.orig = orig

    class _Reset(Exception):
        sqlstate = "08006"          # connection_failure

    assert _is_lock_timeout(_PgErr())
    assert _is_lock_timeout(_Wrapped(_PgErr()))
    assert not _is_lock_timeout(_Reset())
    assert not _is_lock_timeout(_Wrapped(_Reset()))
    assert not _is_lock_timeout(RuntimeError("boom"))
    assert not _is_lock_timeout(ValueError())

    class _AsyncpgStyle(Exception):
        pass
    _AsyncpgStyle.__name__ = "LockNotAvailableError"
    assert _is_lock_timeout(_AsyncpgStyle())


@pytest.mark.asyncio
async def test_a_real_sqlalchemy_lock_timeout_is_classified_as_contention():
    """The end of the chain, measured rather than assumed: what SQLAlchemy +
    asyncpg actually raise when `lock_timeout` fires, and whether
    `_is_lock_timeout` recognises it.

    MEASURED PG 14.19 / SQLAlchemy asyncpg: `sqlalchemy.exc.DBAPIError` whose
    `.orig` is a bare `asyncpg.exceptions.Error` — class name "Error", NOT
    "LockNotAvailableError". So the class-name branch alone would have missed
    it and `claim_for_user` would have re-raised a contention as a real
    failure. The SQLSTATE (55P03) is what carries the meaning.
    """
    asyncpg = pytest.importorskip("asyncpg")
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    from app.services.pool_service import _is_lock_timeout

    try:
        admin = await asyncpg.connect(PG_DSN)
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"no local Postgres: {e}")
    db = f"toup_lane_c_{uuid.uuid4().hex[:8]}"
    await admin.execute(f'CREATE DATABASE "{db}"')
    await admin.close()
    key = f"pool_claim:{uuid.uuid4()}"
    holder = await asyncpg.connect(PG_DSN.rsplit("/", 1)[0] + "/" + db)
    t = holder.transaction()
    await t.start()
    await holder.execute(
        "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", key,
    )
    eng = create_async_engine(f"postgresql+asyncpg://localhost/{db}")
    try:
        with pytest.raises(Exception) as ei:
            async with eng.begin() as conn:
                await conn.execute(text("SET LOCAL lock_timeout = '1s'"))
                await conn.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:k)::bigint)"),
                    {"k": key},
                )
        assert getattr(ei.value.orig, "sqlstate", None) == "55P03"
        assert _is_lock_timeout(ei.value), (
            "a genuine lock timeout must be classified as contention, or the "
            'observe branch is unreachable code that every unit test passes'
        )
    finally:
        await eng.dispose()
        await t.rollback()
        await holder.close()
        admin = await asyncpg.connect(PG_DSN)
        await admin.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db}'"
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db}"')
        await admin.close()
