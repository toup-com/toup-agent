"""Harness for the agent memory verification suite.

Why this exists separately from backend/tests/conftest.py
---------------------------------------------------------
The repo's CI conftest defaults to `sqlite+aiosqlite:///:memory:` with
`RUN_MODE=platform`. Under those settings `init_db()` does not create a single
AGENT_ONLY table, and the pgvector `embedding` column cannot compile at all — so
the entire agent-side memory system (vector search, tsvector search, entity
graph) has never executed in CI. Every assertion in this package would be
vacuous on that stack.

This package therefore requires a REAL Postgres with pgvector and
`RUN_MODE=agent`. It refuses to run otherwise rather than passing vacuously.

Run it with `python backend/scripts/memory_verify.py` (or `make memory-verify`),
which provisions the database, exports the env, and writes a machine-readable
result file.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio

# ── Hard environment contract ────────────────────────────────────────────
# These MUST already be set by the runner, before backend/tests/conftest.py
# imported (it uses os.environ.setdefault, so an exported value wins).

REQUIRED_DB_PREFIX = "postgresql"

_DB_URL = os.environ.get("DATABASE_URL", "")
_RUN_MODE = os.environ.get("RUN_MODE", "")

_ENV_OK = _DB_URL.startswith(REQUIRED_DB_PREFIX) and _RUN_MODE == "agent"

collect_ignore_glob: List[str] = []
if not _ENV_OK:  # pragma: no cover - guard path
    # Fail loudly at collection rather than silently skipping: a vacuous green
    # run on sqlite is exactly the failure mode this package exists to prevent.
    raise RuntimeError(
        "memverify requires RUN_MODE=agent and a postgresql DATABASE_URL "
        f"(got RUN_MODE={_RUN_MODE!r}, DATABASE_URL={_DB_URL[:40]!r}). "
        "Run via: python backend/scripts/memory_verify.py"
    )


# ── Tables this suite owns and truncates between tests ───────────────────
#
# v3: `memory_files` and `memory_file_changes` are THE product. `memories`
# and the entity tables stay on the list because they still exist (the
# legacy table is the rollback, and the graph still has consumers) — a
# truncate of a table nothing writes is free, and leaving one off is how a
# scenario inherits another's rows.
_MEMORY_TABLES = [
    "memory_file_changes",
    "memory_files",
    "memory_capture_outbox",
    "retrieval_events",
    "memory_events",
    "memory_relationships",
    "entity_links",
    "entity_relationships",
    "entities",
    "memories",
    "brain_stats",
    "users",
]


async def _truncate_all() -> None:
    from sqlalchemy import text
    from app.db.database import engine

    async with engine.begin() as conn:
        existing = set(
            (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='public'"
                    )
                )
            ).scalars()
        )
        targets = [t for t in _MEMORY_TABLES if t in existing]
        if targets:
            await conn.execute(
                text(f"TRUNCATE TABLE {', '.join(targets)} RESTART IDENTITY CASCADE")
            )


def _ensure_schema_once() -> None:
    """Create the full agent-side schema if this database is empty."""

    async def _go() -> None:
        from sqlalchemy import text
        from app.db import init_db
        from app.db.database import engine

        async with engine.begin() as conn:
            has = (
                await conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name='memory_files'"
                    )
                )
            ).scalar()
        if not has:
            await init_db()
        await engine.dispose()

    asyncio.run(_go())


@pytest.fixture(scope="session", autouse=True)
def _schema() -> None:
    _ensure_schema_once()


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Override the repo-wide autouse fixture.

    The parent fixture does a full init_db()/drop_db() per test, which on a real
    Postgres with 45 tables costs seconds per test. Truncating the memory tables
    gives the same isolation for this package at a fraction of the cost.

    The per-test `engine.dispose()` is NOT optional: asyncpg connections are
    bound to the event loop that created them, and pytest-asyncio builds a fresh
    loop per test.
    """
    from app.db.database import engine

    await _truncate_all()
    try:
        yield
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db():
    """A session against the tenant database."""
    from app.db.database import async_session_maker

    async with async_session_maker() as session:
        yield session


# ── Synthetic users ──────────────────────────────────────────────────────

async def make_user(session, *, name: str = "Test User", email: Optional[str] = None) -> str:
    """Create a synthetic user and return its id.

    Every test user is synthetic and lives only in the throwaway verification
    database. Nothing in this package may ever point at a production DSN — the
    env guard above enforces the database, and `memory_verify.py` enforces the
    database NAME.
    """
    from app.db.models.user import User

    uid = str(uuid.uuid4())
    session.add(
        User(
            id=uid,
            email=email or f"{uid[:8]}@memverify.local",
            hashed_password="!synthetic",
            name=name,
        )
    )
    await session.commit()
    return uid


@pytest_asyncio.fixture
async def user_a(db) -> str:
    return await make_user(db, name="Alice Verifier", email=f"a-{uuid.uuid4().hex[:8]}@memverify.local")


@pytest_asyncio.fixture
async def user_b(db) -> str:
    return await make_user(db, name="Bob Verifier", email=f"b-{uuid.uuid4().hex[:8]}@memverify.local")


# ── Labeled corpus: executed once per run, concurrently ──────────────────
#
# Every labeled scenario gets its OWN synthetic user and its OWN DB session,
# so they can run concurrently against the real writer without interfering.
# Results are plain data, so the per-test truncation that follows cannot
# invalidate them.
#
# The user's display name is a real-looking one on purpose: the identity
# resolver reports `known=False` for a placeholder ("Agent Owner", "Test
# User" — see user_identity.PLACEHOLDER_ALIASES), and the user-not-in-people
# rule fails SOFT in that state. Naming these users "Test User" would
# silently disable the rule for the whole corpus and make the routing
# numbers look better than production's. The placeholder shape is covered
# separately, in test_g_isolation.py.

from .corpus import LABELED_USER_NAME  # noqa: E402


async def _execute_labeled(scenarios):
    from app.db.database import async_session_maker, engine
    from .pipeline import ScenarioResult, run_scenario

    limit = int(os.environ.get("MEMVERIFY_CONCURRENCY", "8"))
    sem = asyncio.Semaphore(limit)
    out: Dict[str, Any] = {}

    async def _one(sc, dirty=False):
        key = sc.id + ("[dirty]" if dirty else "")
        async with sem:
            async with async_session_maker() as session:
                uid = await make_user(
                    session,
                    name=LABELED_USER_NAME,
                    email=f"{key.lower().replace('[', '-').replace(']', '')}"
                          f"-{uuid.uuid4().hex[:6]}@memverify.local",
                )
                try:
                    res = await run_scenario(session, uid, sc, dirty=dirty)
                    err = None
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    res = ScenarioResult(key, missed=[m.id for m in sc.must_capture])
                    err = f"{type(exc).__name__}: {exc}"
                out[key] = {"result": res, "user_id": uid, "error": err}

    jobs = [_one(sc) for sc in scenarios]
    # The BELT run: a scenario carrying an `injected` suffix is additionally
    # driven with the string ws_chat actually builds. See pipeline.py.
    jobs += [
        _one(sc, dirty=True)
        for sc in scenarios
        if any(t.injected for t in sc.turns)
    ]

    started = time.time()
    await asyncio.gather(*jobs)
    await engine.dispose()
    record_metric("labeled_wall_clock_s", round(time.time() - started, 1))
    return out


@pytest.fixture(scope="session")
def labeled_run():
    """scenario_id -> {result: ScenarioResult, user_id, error}"""
    from .corpus import ALL_LABELED

    return asyncio.run(_execute_labeled(ALL_LABELED))


# ── Machine-readable results ─────────────────────────────────────────────

_RESULTS: Dict[str, Dict[str, Any]] = {}
_METRICS: Dict[str, Any] = {}


def record_metric(key: str, value: Any) -> None:
    """Attach a metric to the run's JSON artifact."""
    _METRICS[key] = value


@pytest.fixture
def metrics():
    return record_metric


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" and not (report.when == "setup" and report.outcome != "passed"):
        return
    entry = _RESULTS.setdefault(
        item.nodeid,
        {
            "scenario_id": item.nodeid.split("::")[-1],
            "file": item.nodeid.split("::")[0],
            "category": _category_of(item.nodeid),
            "outcome": "passed",
            "duration_s": 0.0,
            "message": "",
        },
    )
    entry["duration_s"] = round(entry["duration_s"] + report.duration, 4)
    if report.outcome != "passed":
        entry["outcome"] = report.outcome
        entry["message"] = str(report.longrepr)[-4000:] if report.longrepr else ""


def _category_of(nodeid: str) -> str:
    base = Path(nodeid.split("::")[0]).name
    if base.startswith("test_") and len(base) > 6 and base[5].isalpha() and base[6] == "_":
        return base[5].upper()
    return "?"


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("MEMVERIFY_RESULTS")
    if not out:
        return
    results = list(_RESULTS.values())
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"passed": 0, "failed": 0, "skipped": 0})
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    payload = {
        "run_id": os.environ.get("MEMVERIFY_RUN_ID", "adhoc"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exit_status": int(exitstatus),
        "totals": {
            "tests": len(results),
            "passed": sum(1 for r in results if r["outcome"] == "passed"),
            "failed": sum(1 for r in results if r["outcome"] == "failed"),
            "skipped": sum(1 for r in results if r["outcome"] == "skipped"),
        },
        "by_category": by_cat,
        "metrics": _METRICS,
        "scenarios": sorted(results, key=lambda r: r["scenario_id"]),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, default=str))
