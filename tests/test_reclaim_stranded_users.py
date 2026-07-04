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
