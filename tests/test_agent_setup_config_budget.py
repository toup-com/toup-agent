"""PUT /api/agent-setup/config must return inside the INSTALLED client's
budget, whatever the bridge is doing.

All 42 field devices run build 109 (2026-08-29). Its `api.ts` sets
`DEFAULT_TIMEOUT_MS = 15000` and `updateAgentConfig` does not override it, so
every REST call is aborted at 15 s. The handler's VPS/bridge push ran on the
bridge's flat `bridge_request_timeout_s = 30` (and for a POOL member, up to
THREE such attempts inside `update_container_env`). On 2026-09-06 that produced
"[AGENT-SETUP] Managed container provision/sync failed: " at 18:18:09, twice,
for a row that had been committed ~30 s earlier — the user saw
"Could not save — the server took too long" for a save that had been saved.

The platform row is committed BEFORE this block; only the push is bounded.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from httpx import AsyncClient
from sqlalchemy import select


async def _seed(user_id, *, container=("toup-agent-pool-73", "running")):
    from app.db import async_session_maker
    from app.db.models import AgentConfig, ManagedContainer
    async with async_session_maker() as db:
        db.add(AgentConfig(
            user_id=user_id, hosting_mode="managed", llm_mode="bundle",
            bundle_status="active", setup_step=3, setup_completed=True,
            agent_model="gpt-4o-mini",
        ))
        if container:
            name, status = container
            db.add(ManagedContainer(
                id=str(uuid.uuid4()), user_id=user_id, container_name=name,
                host_port=18073, db_name="toup_agent_feed0073", status=status,
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_config_save_returns_fast_while_the_bridge_hangs(
    client: AsyncClient, auth_headers, test_user_id, monkeypatch,
):
    """MUTATION: delete the budget (await `update_container_env` inline) and
    this takes as long as the bridge does — 30 s on the incident's numbers,
    twice the client's whole patience."""
    from app.config import settings
    from app.services import docker_host_service as dhs

    await _seed(test_user_id)
    monkeypatch.setattr(settings, "agent_setup_sync_timeout_s", 0.4, raising=False)

    sync = {"started": 0, "finished": 0}

    async def _hanging_bridge(db, user_id, cfg):
        sync["started"] += 1
        await asyncio.sleep(1.5)          # the bridge, stalled by docker work
        sync["finished"] += 1
        return object()
    monkeypatch.setattr(dhs, "update_container_env", _hanging_bridge)

    t0 = time.monotonic()
    resp = await client.put(
        "/api/agent-setup/config", headers=auth_headers,
        # agent_model is in _BRIDGE_FIELDS, so this is a real env change and
        # the handler cannot take its no-op short-circuit.
        json={"agent_model": "gpt-4o"},
    )
    dt = time.monotonic() - t0

    assert resp.status_code == 200, resp.text
    assert dt < 1.0, (
        f"the save took {dt:.2f}s; build 109 aborts at 15 s and the whole "
        f"point is to answer well inside that"
    )
    body = resp.json()
    assert body["config_synced"] is False
    assert body["sync_deferred"] is True, (
        "the response must SAY the push is still running — otherwise the "
        "client cannot tell 'saved, syncing' from 'failed'"
    )
    assert sync["started"] == 1

    # The platform row is durable regardless of the bridge — that is the half
    # the user actually asked for.
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()
    assert cfg.agent_model == "gpt-4o", "the save must have been SAVED"

    # And the deferred push is not cancelled — it completes on its own.
    assert sync["finished"] == 0, "fixture invariant: it was still running"
    await asyncio.sleep(1.6)
    assert sync["finished"] == 1, (
        "the deferred sync must run to completion in the background, not be "
        "cancelled: cancelling between the bridge applying the config and the "
        "row recording it manufactures the lost-response state"
    )


@pytest.mark.asyncio
async def test_a_sync_that_fits_the_budget_still_reports_synced(
    client: AsyncClient, auth_headers, test_user_id, monkeypatch,
):
    """The budget must not turn every save into a deferred one."""
    from app.config import settings
    from app.services import docker_host_service as dhs

    await _seed(test_user_id)
    monkeypatch.setattr(settings, "agent_setup_sync_timeout_s", 5.0, raising=False)

    async def _fast(db, user_id, cfg):
        return object()
    monkeypatch.setattr(dhs, "update_container_env", _fast)

    resp = await client.put(
        "/api/agent-setup/config", headers=auth_headers,
        json={"agent_model": "gpt-4o"},
    )
    assert resp.status_code == 200
    assert resp.json()["config_synced"] is True
    assert resp.json()["sync_deferred"] is False


@pytest.mark.asyncio
async def test_repeated_saves_are_idempotent(
    client: AsyncClient, auth_headers, test_user_id, monkeypatch,
):
    """The retry a user makes after "the server took too long" must not queue
    a second bridge recreate. Re-PUTting the same values changes no bridge
    field, so the handler takes its no-op short-circuit."""
    from app.config import settings
    from app.services import docker_host_service as dhs

    await _seed(test_user_id)
    monkeypatch.setattr(settings, "agent_setup_sync_timeout_s", 5.0, raising=False)
    calls = {"n": 0}

    async def _count(db, user_id, cfg):
        calls["n"] += 1
        return object()
    monkeypatch.setattr(dhs, "update_container_env", _count)

    body = {"agent_model": "gpt-4o", "setup_step": 4}
    for _ in range(3):
        r = await client.put(
            "/api/agent-setup/config", headers=auth_headers, json=body,
        )
        assert r.status_code == 200
        assert r.json()["sync_deferred"] is False
    assert calls["n"] == 1, (
        f"three identical saves must cost ONE bridge push; cost {calls['n']}"
    )


@pytest.mark.asyncio
async def test_a_cold_provision_is_deferred_not_awaited(
    client: AsyncClient, auth_headers, test_user_id, monkeypatch,
):
    """The branch that actually fired at 18:18:09: no container row, a
    completed setup, so the handler fell through to `provision_container` —
    30-90 s of bridge work inside a request the client abandons at 15 s."""
    from app.config import settings
    from app.services import docker_host_service as dhs
    from app.services import pool_service as ps

    await _seed(test_user_id, container=None)
    monkeypatch.setattr(settings, "agent_setup_sync_timeout_s", 0.3, raising=False)
    monkeypatch.setattr(settings, "provision_discovery_enabled", True, raising=False)

    discovered: list = []
    monkeypatch.setattr(
        ps, "ensure_discovery", lambda u, **k: discovered.append(u),
    )

    async def _slow_provision(db, user_id, cfg=None, **kw):
        await asyncio.sleep(1.2)
        return None
    monkeypatch.setattr(dhs, "provision_container", _slow_provision)

    t0 = time.monotonic()
    resp = await client.put(
        "/api/agent-setup/config", headers=auth_headers,
        json={"agent_model": "gpt-4o"},
    )
    dt = time.monotonic() - t0
    assert resp.status_code == 200, resp.text
    assert dt < 1.0, f"{dt:.2f}s — a cold provision must not hold the request"
    assert resp.json()["sync_deferred"] is True
    assert discovered == [test_user_id], (
        "a deferred provision for a pool tenant must also ask the bridge "
        "whether the bind already exists"
    )
    await asyncio.sleep(1.3)


@pytest.mark.asyncio
async def test_the_identity_write_through_still_fails_closed_but_in_budget(
    client: AsyncClient, auth_headers, test_user_id, monkeypatch,
):
    """agent_name / agent_color must still reach the tenant or the save fails
    (the L-1 fail-closed rule). What changes is that it fails at ~1 s instead
    of at up to 32 s — one 15 s `_sync_soul_to_vps` attempt, a 2 s sleep, and
    a second 15 s attempt, against a client that gave up at 15 s."""
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    from app.api import soul as soul_mod

    await _seed(test_user_id)
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()
        cfg.agent_url = "https://agent-x.agents.toup.ai"
        cfg.agent_api_key = "k"
        await db.commit()

    monkeypatch.setattr(
        settings, "agent_setup_identity_sync_budget_s", 0.5, raising=False,
    )

    async def _never_answers(*a, **k):
        await asyncio.sleep(30)
        return True
    monkeypatch.setattr(soul_mod, "_sync_soul_to_vps", _never_answers)

    t0 = time.monotonic()
    resp = await client.put(
        "/api/agent-setup/config", headers=auth_headers,
        json={"agent_name": "Aria"},
    )
    dt = time.monotonic() - t0
    assert resp.status_code == 502, (
        "the identity write-through is fail-closed; that must not change"
    )
    assert dt < 5.0, (
        f"it must fail INSIDE the client's budget; took {dt:.2f}s"
    )


@pytest.mark.asyncio
async def test_every_bridge_push_in_agent_setup_is_bounded():
    """`update_container_env` retries a POOL member three times at
    `bridge_request_timeout_s` = 30 s, so a single un-bounded `await` here is a
    90 s request. Every call site in this module must go through a budget.

    MUTATION: restore a bare `await update_container_env(db, user_id, config)`
    at either WhatsApp seed site → red.
    """
    import inspect
    import re
    from app.api import agent_setup

    src = inspect.getsource(agent_setup)
    # The invariant is the SESSION, which is also what makes the bound safe:
    # a push that may outlive the response must never run on the request's
    # session (FastAPI closes it at response time — the "non-checked-in
    # connection" class in services/background_tasks.py). Every remaining
    # bridge push therefore takes a private session, and every one of them is
    # awaited through a budget.
    on_request_session = re.findall(
        r"await (?:update_container_env|provision_container)\(\s*db\b", src,
    )
    assert not on_request_session, (
        f"{len(on_request_session)} bridge push(es) still run on the REQUEST "
        f"session — un-bounded they hold it for up to 90 s, and bounded they "
        f"keep writing through a session FastAPI has closed"
    )
    assert "_bridge_push_within_budget" in src
    assert src.count("asyncio.wait({") >= 2, (
        "every bridge push must be awaited through a budget, not inline"
    )


@pytest.mark.asyncio
async def test_a_slow_whatsapp_seed_does_not_hold_the_request(monkeypatch):
    """The WhatsApp allowlist seeds are explicitly best-effort ("Never block
    the pairing flow on the seed") — which was not true of a 90 s await."""
    from app.config import settings
    from app.api import agent_setup

    monkeypatch.setattr(settings, "agent_setup_sync_timeout_s", 0.3, raising=False)
    state = {"done": False}

    async def _slow():
        await asyncio.sleep(1.0)
        state["done"] = True
        return object()

    t0 = time.monotonic()
    ok = await agent_setup._bridge_push_within_budget(
        _slow(), user_id="deadbeef-0000", what="test",
    )
    dt = time.monotonic() - t0
    assert ok is False and dt < 0.9, f"{dt:.2f}s"
    assert state["done"] is False
    await asyncio.sleep(1.0)
    assert state["done"] is True, "the deferred push must not be cancelled"
