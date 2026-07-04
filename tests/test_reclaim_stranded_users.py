"""Regression for the 2026-07-03/04 "Waking your agent… forever" incident.

Two production failure classes left users with no reachable agent and no
self-heal path:
  * A managed user with NO ManagedContainer row at all — their signup's
    fire-and-forget finalize died (Railway redeploys kill background
    tasks) and nothing retried (Apple signup 8dwm74…@privaterelay).
  * A pool-bound row flipped to status='orphan' by the rollout's
    whois-404 quarantine (pool containers are named `toup-agent-pool-NN`,
    so the bridge's by-name whois can never find them).

Fixes under test:
  * pool_service._stranded_user_ids — enumerates exactly those two classes.
  * rollout_service._running_tenants — excludes pool-bound containers so
    rollouts can never orphan-quarantine them again.
"""
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import itertools

import pytest

_port = itertools.count(9900)  # managed_containers.host_port is UNIQUE


async def _seed(db, *, hosting="managed", active=True, container=None):
    """Create user + agent_config (+ optional container). Returns user_id."""
    from app.db.models import User, AgentConfig, ManagedContainer
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                name="T", is_active=active))
    db.add(AgentConfig(user_id=uid, hosting_mode=hosting))
    if container is not None:
        name, status = container
        db.add(ManagedContainer(id=str(uuid.uuid4()), user_id=uid,
                                container_name=name, host_port=next(_port),
                                db_name=f"db_{uid[:8]}", status=status))
    await db.commit()
    return uid


@pytest.mark.asyncio
async def test_stranded_predicate_matches_only_broken_users():
    from app.db import async_session_maker
    from app.services.pool_service import _stranded_user_ids

    async with async_session_maker() as db:
        no_row = await _seed(db)                                            # stranded: no container at all
        pool_orphan = await _seed(db, container=("toup-agent-pool-07", "orphan"))   # stranded: quarantined pool row
        pool_running = await _seed(db, container=("toup-agent-pool-08", "running")) # healthy — leave alone
        named_orphan = await _seed(db, container=("toup-agent-deadbeef", "orphan")) # named orphan — operator-only
        self_hosted = await _seed(db, hosting="self-hosted")                # not ours to provision
        inactive = await _seed(db, active=False)                            # deactivated account

        ids = await _stranded_user_ids(db, limit=50)

    assert no_row in ids, "user with no container row must be reclaimed"
    assert pool_orphan in ids, "orphaned pool row must be reclaimed"
    assert pool_running not in ids, "healthy pool user must not be touched"
    assert named_orphan not in ids, "named-container orphans stay manual"
    assert self_hosted not in ids
    assert inactive not in ids


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
