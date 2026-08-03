"""The leak instrument must name the abandoning function — and stay quiet otherwise.

A diagnostic that does not fire is worse than none: it converts "we don't know"
into "there's nothing there". So the load-bearing test here deliberately
abandons a session and asserts the reported stack contains the name of the
function that did it.

    cd backend && ENVIRONMENT=development python -m pytest tests/test_pool_leak_debug.py
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.db import pool_debug  # noqa: E402


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _fresh_engine(tmpdir):
    # NullPool mirrors the agent's real configuration: a fresh connection per
    # session, closed on checkin — so an abandoned one is exactly this leak.
    path = os.path.join(tmpdir, "probe.db")
    return create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)


def _run(coro_factory):
    """Run, then force collection so finalizers fire deterministically."""
    async def _main():
        await coro_factory()
    asyncio.run(_main())
    for _ in range(3):
        gc.collect()


def test_it_names_the_function_that_abandoned_the_connection():
    handler = _Catch()
    pool_debug.logger.addHandler(handler)
    pool_debug.logger.setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as tmp:
        engine = _fresh_engine(tmp)
        assert pool_debug.install(engine)
        maker = async_sessionmaker(engine, expire_on_commit=False)

        async def a_function_that_forgets_to_close():
            session = maker()
            await session.execute(text("SELECT 1"))
            # deliberately no close() and no context manager

        try:
            _run(a_function_that_forgets_to_close)
        finally:
            pool_debug.logger.removeHandler(handler)

    leaks = [m for m in handler.records if "[pool-leak]" in m]
    assert leaks, (
        "the instrument stayed SILENT on a deliberately abandoned session — "
        "a diagnostic that does not fire is worse than none"
    )
    assert "a_function_that_forgets_to_close" in leaks[0], (
        "the report did not name the abandoning function; stack was:\n" + leaks[0]
    )


def test_a_properly_closed_session_is_not_reported():
    """No false positives, or the signal is unusable."""
    handler = _Catch()
    pool_debug.logger.addHandler(handler)
    pool_debug.logger.setLevel(logging.ERROR)
    before = pool_debug.stats()["abandoned"]
    with tempfile.TemporaryDirectory() as tmp:
        engine = _fresh_engine(tmp)
        assert pool_debug.install(engine)
        maker = async_sessionmaker(engine, expire_on_commit=False)

        async def a_well_behaved_function():
            async with maker() as session:
                await session.execute(text("SELECT 1"))

        try:
            _run(a_well_behaved_function)
        finally:
            pool_debug.logger.removeHandler(handler)

    assert pool_debug.stats()["abandoned"] == before, (
        "a correctly closed session was reported as abandoned: "
        + "\n".join(m for m in handler.records if "[pool-leak]" in m)
    )


def test_counters_move():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _fresh_engine(tmp)
        pool_debug.install(engine)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        before = pool_debug.stats()

        async def _work():
            async with maker() as session:
                await session.execute(text("SELECT 1"))

        _run(_work)
        after = pool_debug.stats()

    assert after["checkouts"] > before["checkouts"]
    assert after["checkins"] > before["checkins"]


def test_install_is_survivable_on_a_bad_engine():
    """It must never be able to break boot."""
    class _NoPool:
        pass

    assert pool_debug.install(_NoPool()) is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


def test_arming_by_workspace_sentinel():
    """Tenant env is bridge-built and append-only across a blue-green upgrade,
    so arming ONE tenant by env is a deploy rather than a switch. The sentinel
    makes it a `docker exec touch` + restart — the same pattern the blue-green
    promote marker already uses."""
    import tempfile

    # Off by default, both routes negative.
    assert pool_debug.should_enable(False) is False
    # The settings flag alone is enough.
    assert pool_debug.should_enable(True) is True

    with tempfile.TemporaryDirectory() as tmp:
        sentinel = os.path.join(tmp, ".pool_leak_debug")
        original = pool_debug.SENTINEL
        pool_debug.SENTINEL = sentinel
        try:
            assert pool_debug.should_enable(False) is False, "armed with no sentinel"
            open(sentinel, "w").close()
            assert pool_debug.should_enable(False) is True, (
                "sentinel present but the instrument stayed disarmed — it "
                "could never be switched on for a single tenant"
            )
        finally:
            pool_debug.SENTINEL = original
