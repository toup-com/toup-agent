"""A cold database must not be able to delete toup.ai.

`platform_main` serves the React SPA from the same process as the API (the
static mount at the bottom of that file). `await init_db()` sat unwrapped in
the lifespan, so a database that was merely slow to accept connections aborted
the lifespan, exited the process, and replaced the product with Railway's own
error page. Every other DB-touching boot step in that file was already wrapped
with an explicit "don't take the whole API down for one subsystem"; this was
the exception, and it is the step most likely to meet a cold database.

The contract these tests pin:

  * a transient failure is retried, not fatal;
  * an unrecoverable one boots DEGRADED instead of dying, so static assets keep
    serving and the restart budget (10, per railway.json) stays intact;
  * `/health` answers 503 in that state, so a deploy that genuinely cannot
    reach its database is still never promoted;
  * a runtime query blip does NOT do that — it stays 200/"degraded", or one
    slow moment would cycle otherwise-healthy replicas;
  * a degraded replica heals itself, or "don't crash" would only mean
    "permanently broken but up".
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from unittest.mock import AsyncMock, patch

import pytest

import platform_main

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _no_db():
    """These tests never touch Postgres — override conftest's autouse fixture."""
    yield


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Real backoff is 1+2+4+8s; the behaviour under test is the retrying.

    It still awaits a REAL zero-length sleep rather than an AsyncMock. An
    AsyncMock completes without ever yielding to the event loop, so the healer's
    `while` loop becomes uninterruptible and a `wait_for` timeout around it can
    never fire — a healer that failed to exit would hang the suite instead of
    failing it, which is how this fixture's first version let a mutation pass.
    """
    real_sleep = asyncio.sleep

    async def _instant(_delay):
        await real_sleep(0)

    monkeypatch.setattr(platform_main.asyncio, "sleep", _instant)
    monkeypatch.setattr(platform_main, "_DB_HEAL_INTERVAL", 0)


@pytest.fixture(autouse=True)
def _restore_ready_flag():
    prior = getattr(platform_main.app.state, "db_ready", None)
    yield
    if prior is None:
        if hasattr(platform_main.app.state, "db_ready"):
            del platform_main.app.state.db_ready
    else:
        platform_main.app.state.db_ready = prior


# ── Boot ─────────────────────────────────────────────────────────────────


async def test_a_transient_database_is_retried_and_boots_healthy():
    """Two refusals then success is the common case — a Postgres restart, or
    both replicas opening connections at the same instant during a deploy."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("connection refused")

    with patch.object(platform_main, "init_db", flaky):
        assert await platform_main._init_db_with_retry() is True
    assert calls["n"] == 3


async def test_an_unreachable_database_boots_degraded_instead_of_exiting():
    """The whole point: returning False keeps the process — and the SPA — alive.

    If this ever raises again, uvicorn aborts the lifespan and toup.ai serves
    an infrastructure error page instead of the product.
    """
    with patch.object(platform_main, "init_db", AsyncMock(side_effect=OSError("down"))) as db:
        assert await platform_main._init_db_with_retry() is False
    assert db.await_count == platform_main._DB_INIT_ATTEMPTS


async def test_the_lifespan_never_calls_init_db_directly():
    """The retry wrapper is only protection if boot actually goes through it.

    Asserted structurally because a direct `await init_db()` reintroduces the
    exact original bug while every other test here still passes.
    """
    tree = ast.parse(inspect.getsource(platform_main.lifespan).strip())
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "_init_db_with_retry" in called, "boot must use the retrying wrapper"
    assert "init_db" not in called, (
        "a bare init_db() in the lifespan takes the whole site down on a cold database"
    )


# ── Healing ──────────────────────────────────────────────────────────────


async def test_a_degraded_replica_heals_itself_and_the_healer_stops():
    """Without this, booting degraded only converts a crash into a silent
    outage that lasts until a human restarts the service."""
    calls = {"n": 0}

    async def down_then_up():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("still down")

    platform_main.app.state.db_ready = False
    with patch.object(platform_main, "init_db", down_then_up):
        await asyncio.wait_for(platform_main._heal_db_schema(platform_main.app), timeout=5)

    assert platform_main.app.state.db_ready is True
    assert calls["n"] == 3, "the healer must stop retrying once the schema is up"


async def test_a_degraded_boot_actually_starts_the_healer():
    """Self-healing only exists if boot schedules it.

    Structural for the same reason as the init_db check: dropping the
    `create_task` leaves every other test here green while a degraded replica
    stays degraded forever.
    """
    tree = ast.parse(inspect.getsource(platform_main.lifespan).strip())
    called = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "_heal_db_schema" in called, (
        "a boot that degrades must schedule the healer, or it never recovers"
    )


# ── /health ──────────────────────────────────────────────────────────────


class _OkSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, *a, **k):
        return None


async def test_health_answers_503_while_the_schema_is_down():
    """Boot no longer crashes to signal an unusable deploy, so the probe has to.

    Railway gates promotion on `healthcheckPath: /health`. Without this, a
    replica that never got its schema would answer 200 and be promoted.
    """
    platform_main.app.state.db_ready = False
    with patch("app.db.database.async_session_maker", _OkSession):
        resp = await platform_main.health()

    assert getattr(resp, "status_code", 200) == 503


async def test_a_runtime_query_blip_stays_200(monkeypatch):
    """Deliberately narrower than the schema check.

    A failed `SELECT 1` is usually a moment, not a broken deploy. 503-ing on it
    would let one slow instant cycle replicas that are otherwise fine — so this
    keeps the pre-existing 200 + "degraded" body.
    """
    class _Boom:
        async def __aenter__(self):
            raise OSError("pool exhausted")

        async def __aexit__(self, *a):
            return False

    platform_main.app.state.db_ready = True
    with patch("app.db.database.async_session_maker", _Boom):
        resp = await platform_main.health()

    assert getattr(resp, "status_code", 200) == 200
    assert resp["status"] == "degraded"
    assert "error" in resp["database"]
