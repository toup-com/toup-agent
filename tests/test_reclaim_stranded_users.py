"""Regression for the 2026-07-03/04 "Waking your agent… forever" incident.

Production failure classes that left users with no reachable agent and no
self-heal path:
  * A managed user with NO ManagedContainer row — their signup's
    fire-and-forget finalize died after priming the config (Railway
    redeploys kill background tasks) and nothing retried. 11 real users,
    including Apple's App Store review accounts.
  * A pool-bound row flipped to status='orphan' by the rollout's
    whois-404 quarantine (pool containers are named `toup-agent-pool-NN`,
    so the bridge's by-name whois can never find them).
  * A FRESH signup whose finalize died so early the AgentConfig was never
    created or never primed off the 'self-hosted' schema default (Apple
    signup 8dwm74…@privaterelay) — invisible to any config-based query.

Fixes under test:
  * pool_service._stranded_user_ids — enumerates exactly those classes,
    while never matching real self-hosters (ssh_host set), 'local' users,
    deactivated accounts, or old abandoned signups in the died-early class.
  * rollout_service._running_tenants — excludes pool-bound containers so
    rollouts can never orphan-quarantine them again.
"""
import os
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("ENVIRONMENT", "test")

import itertools

import pytest

_port = itertools.count(9900)  # managed_containers.host_port is UNIQUE


async def _seed(db, *, hosting="managed", ssh_host=None, active=True,
                config=True, created_at=None, container=None,
                mc_updated_at=None):
    """Create user (+ optional agent_config, container). Returns user_id."""
    from app.db.models import User, AgentConfig, ManagedContainer
    uid = str(uuid.uuid4())
    u = User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
             name="T", is_active=active)
    if created_at is not None:
        u.created_at = created_at
    db.add(u)
    # PG enforces FKs at flush; SQLAlchemy only orders inserts via
    # relationship(), which these models don't declare — flush the user
    # row before adding its dependents.
    await db.flush()
    if config:
        db.add(AgentConfig(user_id=uid, hosting_mode=hosting, ssh_host=ssh_host))
    if container is not None:
        name, status = container
        mc = ManagedContainer(id=str(uuid.uuid4()), user_id=uid,
                              container_name=name, host_port=next(_port),
                              db_name=f"db_{uid[:8]}", status=status)
        if mc_updated_at is not None:
            mc.updated_at = mc_updated_at
        db.add(mc)
    await db.commit()
    return uid


@pytest.mark.asyncio
async def test_stranded_predicate_matches_only_broken_users():
    from app.db import async_session_maker
    from app.services.pool_service import _stranded_user_ids

    old = datetime.utcnow() - timedelta(days=90)
    stale = datetime.utcnow() - timedelta(minutes=60)
    async with async_session_maker() as db:
        # Stranded — must be reclaimed:
        managed_no_row = await _seed(db)
        pool_orphan = await _seed(db, container=("toup-agent-pool-07", "orphan"))
        no_config_fresh = await _seed(db, config=False)
        unprimed_fresh = await _seed(db, hosting="self-hosted")  # schema default, never primed
        named_error_fresh = await _seed(db, container=("toup-agent-feedf00d", "error"))
        stuck_prov_fresh = await _seed(db, container=("toup-agent-beefcafe", "provisioning"),
                                       mc_updated_at=stale)

        # Healthy or out of scope — must NOT be touched:
        pool_running = await _seed(db, container=("toup-agent-pool-08", "running"))
        named_orphan_old = await _seed(db, created_at=old,
                                       container=("toup-agent-deadbeef", "orphan"))
        inflight_prov = await _seed(db, container=("toup-agent-cafef00d", "provisioning"))
        real_self_hoster = await _seed(db, hosting="self-hosted", ssh_host="1.2.3.4")
        local_mode = await _seed(db, hosting="local")
        local_mode_err = await _seed(db, hosting="local",
                                     container=("toup-agent-0ddba11", "error"))
        inactive = await _seed(db, active=False)
        no_config_old = await _seed(db, config=False, created_at=old)

        ids = await _stranded_user_ids(db, limit=50)

    assert managed_no_row in ids, "managed user with no container row must be reclaimed"
    assert pool_orphan in ids, "orphaned pool row must be reclaimed"
    assert no_config_fresh in ids, "fresh signup with no config (finalize died early) must be reclaimed"
    assert unprimed_fresh in ids, "fresh signup stuck on the self-hosted schema default must be reclaimed"
    assert named_error_fresh in ids, "fresh signup with a dead named row must be reclaimed"
    assert stuck_prov_fresh in ids, "fresh signup wedged in stale 'provisioning' must be reclaimed"

    assert pool_running not in ids, "healthy pool user must not be touched"
    assert named_orphan_old not in ids, "OLD named-container orphans stay manual"
    assert inflight_prov not in ids, "an actively-provisioning row must not be raced"
    assert real_self_hoster not in ids, "real self-hosters (ssh_host set) are never adopted"
    assert local_mode not in ids, "hosting_mode='local' users are never adopted"
    assert local_mode_err not in ids, "'local' users are never adopted even with dead rows"
    assert inactive not in ids
    assert no_config_old not in ids, "old abandoned signups must not be mass-provisioned"


@pytest.mark.asyncio
async def test_running_tenants_excludes_pool_bound():
    from app.db import async_session_maker
    from app.services.rollout_service import _running_tenants

    async with async_session_maker() as db:
        named = await _seed(db, container=("toup-agent-cafebabe", "running"))
        pooled = await _seed(db, container=("toup-agent-pool-12", "running"))

        tenants = await _running_tenants(db)
        got = {t.user_id for t in tenants}

    assert named in got, "named tenants still roll out"
    assert pooled not in got, (
        "pool-bound tenants must never enter the per-tenant rollout — "
        "the bridge's by-name whois 404s them and the orphan quarantine "
        "breaks healthy users"
    )


@pytest.mark.asyncio
async def test_claim_force_bypasses_running_early_return(monkeypatch):
    """force=True must reach the bridge even when the row says 'running' —
    that's how a keyless-but-honest row gets its secrets re-pushed. Without
    force, the early-return wins and no bridge call happens."""
    from app.db import async_session_maker
    from app.services import pool_service as ps

    uid = await _run_seed(container=("toup-agent-pool-42", "running"))

    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    calls = {"n": 0}

    class _BoomClient:
        async def __aenter__(self):
            calls["n"] += 1
            raise RuntimeError("stop-at-bridge")  # proves we got past the guard

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(
        "app.services.docker_host_service._bridge_client", lambda *a, **k: _BoomClient()
    )

    async with async_session_maker() as db:
        # Without force: early-return, bridge never touched.
        c = await ps.claim_for_user(db, uid)
        assert c is not None and calls["n"] == 0

    async with async_session_maker() as db:
        # With force: proceeds to the bridge (our stub raises there).
        try:
            await ps.claim_for_user(db, uid, force=True)
        except RuntimeError:
            pass
    assert calls["n"] == 1, "force=True must reach the bridge call"


async def _run_seed(**kw):
    from app.db import async_session_maker
    async with async_session_maker() as db:
        return await _seed(db, **kw)


@pytest.mark.asyncio
async def test_keyless_sweep_runs_with_zero_stranded_candidates(monkeypatch):
    """Steady state (no stranded users) must NOT short-circuit the keyless
    sweep — the original early-return did, leaving restart-keyless agents
    broken forever while reclaim reported all-quiet (2026-07-04)."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    # One healthy-looking pool user whose agent will reject its key.
    uid = await _run_seed(container=("toup-agent-pool-77", "running"))
    from app.db import async_session_maker
    from sqlalchemy import update
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        await db.execute(update(AgentConfig).where(AgentConfig.user_id == uid)
                         .values(agent_url="https://agent-x.test", agent_api_key="k"))
        await db.commit()

    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)

    class _Resp:
        status_code = 401

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    fake_container = type("C", (), {"container_name": "toup-agent-pool-77"})()
    claim = AsyncMock(return_value=fake_container)
    monkeypatch.setattr(ps, "claim_for_user", claim)

    summary = await ps.reclaim_stranded_users()

    assert summary.get("candidates") == 0, "steady state: no stranded users"
    assert summary.get("keyless") == 1, "keyless sweep must still run and detect"
    assert summary.get("rebound") == 1
    assert claim.await_args.kwargs.get("force") is True, "keyless heal must force past the running-row early-return"


@pytest.mark.asyncio
async def test_sweep_probes_named_containers_and_restarts_on_401(monkeypatch):
    """The authenticated sweep must cover NAMED containers too (the oldest
    users were invisible to the original pool-only sweep). A named tenant
    that 401s its own key gets a bridge restart, NOT a pool force-claim."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _run_seed(container=(f"toup-agent-{'a1b2c3d4'}", "running"))
    from app.db import async_session_maker
    from sqlalchemy import update
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        await db.execute(update(AgentConfig).where(AgentConfig.user_id == uid)
                         .values(agent_url="https://agent-n.test", agent_api_key="k"))
        await db.commit()

    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)

    class _Resp:
        status_code = 401

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    claim = AsyncMock()
    monkeypatch.setattr(ps, "claim_for_user", claim)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)

    summary = await ps.reclaim_stranded_users()

    assert summary.get("keyless") == 1
    claim.assert_not_awaited()
    restart.assert_awaited_once()
    assert summary.get("restarted") == 1


@pytest.mark.asyncio
async def test_sweep_restarts_sick_agent_after_two_consecutive_5xx(monkeypatch):
    """5xx/timeout probes accumulate strikes; the SECOND consecutive sick
    tick restarts the container (the poisoned-DB class that previously sat
    broken for days). One sick tick alone must NOT restart."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _run_seed(container=("toup-agent-pool-88", "running"))
    from app.db import async_session_maker
    from sqlalchemy import update
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        await db.execute(update(AgentConfig).where(AgentConfig.user_id == uid)
                         .values(agent_url="https://agent-s.test", agent_api_key="k"))
        await db.commit()

    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    ps._PROBE_STRIKES.clear()

    class _Resp:
        status_code = 500

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)

    s1 = await ps.reclaim_stranded_users()
    assert s1.get("sick", 0) == 0, "first sick tick only records a strike"
    restart.assert_not_awaited()

    s2 = await ps.reclaim_stranded_users()
    assert s2.get("sick") == 1, "second consecutive sick tick acts"
    restart.assert_awaited_once()
    assert uid not in ps._PROBE_STRIKES, "strikes cleared after restart"


@pytest.mark.asyncio
async def test_sweep_mass_transport_failure_records_no_strikes(monkeypatch):
    """When the majority of probes fail at the TRANSPORT level in one tick,
    that's a platform-egress problem, not N sick agents — the sweep must
    not accumulate strikes (a fleet-wide restart storm would turn a network
    blip into an outage)."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _run_seed(container=("toup-agent-pool-89", "running"))
    from app.db import async_session_maker
    from sqlalchemy import update
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        await db.execute(update(AgentConfig).where(AgentConfig.user_id == uid)
                         .values(agent_url="https://agent-m.test", agent_api_key="k"))
        await db.commit()

    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    ps._PROBE_STRIKES.clear()

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): raise ConnectionError("egress down")

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)

    await ps.reclaim_stranded_users()
    await ps.reclaim_stranded_users()

    restart.assert_not_awaited()
    assert not ps._PROBE_STRIKES, "transport mass-failure must not strike"
