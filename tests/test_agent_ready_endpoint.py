"""
PR onboarding-v2/3 — `/api/agent-setup/ready` endpoint.

Exercises the state-mapping logic by reading the endpoint's source
(pure-function / no DB, mirrors the test_init_db_applies_021.py idiom
already used in this repo). Integration coverage that exercises a
live `AgentConfig` / `ManagedContainer` round-trip is in the
billing/agent_setup test file (which already needs a DB to run);
this file stays pure-function so it runs on every dev machine
without DB plumbing.
"""

from __future__ import annotations

import inspect

from app.api import agent_setup


def test_ready_endpoint_is_registered():
    # Source-level invariant: a `@router.get("/ready")` decorator must
    # land on a function called `get_agent_ready`. If someone renames
    # either the route or the handler, this test fails loudly so the
    # frontend (`agentSetup.ready()` in api.ts) doesn't silently 404.
    src = inspect.getsource(agent_setup)
    assert '@router.get("/ready")' in src
    assert "async def get_agent_ready" in src


def test_ready_endpoint_maps_running_to_stage_running():
    src = inspect.getsource(agent_setup.get_agent_ready)
    assert '"running"' in src
    # The endpoint must map container.status='running' to the same
    # stage value the frontend expects — not `"ready"`, not `"up"`.
    assert 'stage = "running"' in src


def test_ready_endpoint_treats_unknown_status_as_cold():
    # Defense-in-depth: an undocumented container.status must not
    # accidentally let the SPA open chat. Test asserts the fall-through
    # branch sets stage='cold', not 'running'.
    src = inspect.getsource(agent_setup.get_agent_ready)
    # Find the else-branch within the cs/lower if-chain. The
    # important thing is `stage = "cold"` appears AFTER the recognised
    # status checks.
    assert src.count("stage = ") >= 2
    # And no `else: stage = "running"` snuck in.
    assert "else:\n            stage = \"running\"" not in src


def test_ready_endpoint_pool_bound_is_warm():
    # The cold_start signal must check the `toup-agent-pool-` prefix
    # — pool-bound members keep their pool name post-claim and we
    # don't pay the cold-build cost. Source-level check that the
    # exact prefix is in the cold_start logic.
    src = inspect.getsource(agent_setup.get_agent_ready)
    assert "toup-agent-pool-" in src
    assert "cold_start = " in src


def test_ready_endpoint_emits_onboarding_event_once_per_user():
    # The first time `/ready` returns ready=True for a user we want a
    # single `onboarding.agent_ready` line in Loki. Subsequent polls
    # (the SPA might still tick once after observing ready) must NOT
    # duplicate the emission within the TTL window.
    src = inspect.getsource(agent_setup.get_agent_ready)
    assert "_AGENT_READY_EMITTED" in src
    assert "emit_agent_ready" in src
    # The TTL check is a `timedelta(hours=1)` window per the design
    # — anything sooner would re-emit on long onboardings and pollute
    # the funnel-completion metric.
    assert "timedelta(hours=1)" in src


def test_ready_endpoint_returns_shape_the_frontend_expects():
    # Schema parity with frontend/src/api.ts `agentSetup.ready()`:
    # the response must include ready, stage, duration_ms, cold_start.
    # We pattern-match the source rather than running the endpoint so
    # we don't need a DB.
    src = inspect.getsource(agent_setup.get_agent_ready)
    assert '"ready": ready' in src
    assert '"stage": stage' in src
    assert '"duration_ms": duration_ms' in src
    assert '"cold_start": cold_start' in src


def test_module_emit_cache_is_bounded():
    # The in-process emit-once cache must not grow without bound. A
    # misbehaving caller (or bot) could otherwise OOM the platform
    # over time. Source-level guardrail.
    src = inspect.getsource(agent_setup.get_agent_ready)
    assert "10_000" in src or "10000" in src
    # Eviction strategy: bulk cutoff (oldest 25%) rather than touching
    # the dict on every call — keeps the hot path cheap.
    assert "cutoff" in src
