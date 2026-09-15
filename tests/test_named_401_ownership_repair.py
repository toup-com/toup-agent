"""Repair a named-401 from bridge truth, or refuse it with evidence.

The 2026-09-12 incident's persistent phase, in one sentence: the platform row
named `toup-agent-f261b564` while the public hostname routed to pool slot 81,
so the authenticated sweep's probe 401'd every tick for 2 h 28 m — and the
branch it landed in had no repair at all. Pre-PR-738 it restarted the named
container every tick forever (a restart cannot rewrite a platform row);
post-738 it counted and logged every tick forever. Neither converged.

What converges is asking the BRIDGE which container owns the user, and acting
only on what it says:

  * bridge says pool slot P, no named container, route agrees, platform row
    says something else  → ADOPT P into the platform row. One tick.
  * bridge says BOTH a pool bind and a named container → two databases, two
    candidate owners; do NOT act, page with both names so an operator can
    decide which one holds the user's life. (Four accounts sat in this state
    with nothing paging, and one went dark on the next Caddy restart.)
  * bridge says no bind at all → escalate. NEVER force-claim: that binds an
    established account to a fresh empty slot (R40).
  * no evidence → defer and alert, exactly as before. A parse failure must
    never be read as "there is no named container".

Direction is load-bearing (L4 skeptic C4): the repair adopts the ROUTED
container's identity INTO the platform row. Pushing the platform's stored key
at whatever the route happens to hit would have bound the named key onto a
live pool member — manufacturing the ambiguous-ownership state this branch
exists to detect.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q \
        tests/test_named_401_ownership_repair.py
"""

from __future__ import annotations

import itertools
import os
import uuid
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

_port = itertools.count(9700)


async def _seed_named_401_user(container_name: str = "toup-agent-a1b2c3d4"):
    """A user whose platform row names a NAMED container and whose probe 401s."""
    from app.db import async_session_maker
    from app.db.models import User, AgentConfig, ManagedContainer

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@t.local", hashed_password="",
                    name="T", is_active=True))
        await db.flush()
        db.add(AgentConfig(user_id=uid, hosting_mode="managed",
                           agent_url=f"https://agent-{uid[:8]}.test",
                           agent_api_key="platform-key"))
        db.add(ManagedContainer(
            id=str(uuid.uuid4()), user_id=uid, container_name=container_name,
            host_port=next(_port), db_name=f"toup_agent_{uid[:8]}",
            status="running",
        ))
        await db.commit()
    return uid


def _install_401_probe(monkeypatch):
    class _Resp:
        status_code = 401

    class _FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


async def _probe_row(uid: str):
    from app.db import async_session_maker
    from app.db.models import AgentProbeState
    from sqlalchemy import select
    async with async_session_maker() as db:
        return (await db.execute(
            select(AgentProbeState).where(AgentProbeState.user_id == uid)
        )).scalar_one_or_none()


# ── The classifier, pure ──────────────────────────────────────────────

def test_no_bridge_truth_is_unknown_not_an_adopt():
    """The safety property the whole branch rests on: absence of evidence is
    never evidence of absence. A body we could not read must not authorise a
    rewrite of a live user's row."""
    from app.services import pool_service as ps

    outcome, ev = ps.classify_named_401_ownership(
        platform_container="toup-agent-a1b2c3d4", slot=None, truth=None,
    )
    assert outcome == ps.OWNERSHIP_UNKNOWN
    assert ev["reason"] == "no_bridge_truth"


def test_pool_and_named_both_present_is_ambiguous():
    from app.services import pool_service as ps

    outcome, ev = ps.classify_named_401_ownership(
        platform_container="toup-agent-51d4ed2f",
        slot=None,
        truth={"pool_container": "toup-agent-pool-17", "pool_db": "pool_17",
               "named_container": "toup-agent-51d4ed2f",
               "named_db": "toup_agent_51d4ed2f",
               "route_container": "toup-agent-51d4ed2f"},
    )
    assert outcome == ps.OWNERSHIP_AMBIGUOUS
    # Both candidate containers AND both databases must reach the operator:
    # deciding this needs to know which side holds the user's messages.
    assert ev["pool_container"] == "toup-agent-pool-17"
    assert ev["named_container"] == "toup-agent-51d4ed2f"
    assert ev["pool_db"] == "pool_17"
    assert ev["named_db"] == "toup_agent_51d4ed2f"


def test_no_bind_anywhere_is_an_escalation():
    from app.services import pool_service as ps

    outcome, _ = ps.classify_named_401_ownership(
        platform_container="toup-agent-a1b2c3d4", slot=None,
        truth={"pool_container": None, "named_container": None},
    )
    assert outcome == ps.OWNERSHIP_NO_BIND


def test_the_12_sep_shape_is_an_adopt():
    """Platform row names the named container; the bridge says the user is on
    pool-81, there is no named container, and Caddy dials pool-81."""
    from app.services import pool_service as ps

    outcome, ev = ps.classify_named_401_ownership(
        platform_container="toup-agent-f261b564",
        slot={"container_name": "toup-agent-pool-81", "db_name": "pool_81"},
        truth={"pool_container": "toup-agent-pool-81", "pool_db": "pool_81",
               "named_container": None,
               "route_container": "toup-agent-pool-81"},
    )
    assert outcome == ps.OWNERSHIP_ADOPT
    assert ev["pool_container"] == "toup-agent-pool-81"


def test_a_route_that_disagrees_with_the_bind_is_ambiguous_not_an_adopt():
    """Even with only one container reported, bind != route IS the split."""
    from app.services import pool_service as ps

    outcome, ev = ps.classify_named_401_ownership(
        platform_container="toup-agent-f261b564",
        slot=None,
        truth={"pool_container": "toup-agent-pool-81",
               "named_container": None,
               "route_container": "toup-agent-pool-26"},
    )
    assert outcome == ps.OWNERSHIP_AMBIGUOUS
    assert ev["reason"] == "route_disagrees_with_bind"


def test_a_named_only_answer_is_not_an_adopt():
    """`_adopt_discovered_bind` goes through `claim_for_user`, which binds a
    POOL slot. A named container's key drift is an operator's problem, not
    something to answer with a pool claim."""
    from app.services import pool_service as ps

    outcome, ev = ps.classify_named_401_ownership(
        platform_container="toup-agent-a1b2c3d4", slot=None,
        truth={"pool_container": None,
               "named_container": "toup-agent-a1b2c3d4"},
    )
    assert outcome == ps.OWNERSHIP_UNKNOWN
    assert ev["reason"] == "named_only"


# ── The adapter ───────────────────────────────────────────────────────

def test_tenant_truth_view_reads_the_nested_shape():
    from app.services import pool_service as ps

    v = ps._tenant_truth_view({
        "prefix": "f261b564",
        "pool": {"container_name": "toup-agent-pool-81", "db_name": "pool_81"},
        "named": {"container_name": "toup-agent-f261b564",
                  "db_name": "toup_agent_f261b564"},
        "route": {"container_name": "toup-agent-pool-81",
                  "upstream": "127.0.0.1:9573"},
    })
    assert v["pool_container"] == "toup-agent-pool-81"
    assert v["named_container"] == "toup-agent-f261b564"
    assert v["route_upstream"] == "127.0.0.1:9573"


def test_tenant_truth_view_reads_the_flat_shape():
    """The bridge is another lane's code and is not in this repository, so the
    adapter tolerates spelling — within the limit below."""
    from app.services import pool_service as ps

    v = ps._tenant_truth_view({
        "prefix": "f261b564",
        "pool_container": "toup-agent-pool-81",
        "named_container": None,
        "route_target": "127.0.0.1:9573",
    })
    assert v["pool_container"] == "toup-agent-pool-81"
    assert v["named_container"] is None


def test_an_unreadable_body_answers_none_never_an_empty_view():
    """This is the load-bearing half of the tolerance. An empty view would say
    'no named container', which is exactly the evidence ADOPT requires — so a
    parse failure would authorise a rewrite. None routes to UNKNOWN."""
    from app.services import pool_service as ps

    assert ps._tenant_truth_view({"totally": "unexpected"}) is None
    assert ps._tenant_truth_view({}) is None
    assert ps._tenant_truth_view(None) is None
    # …but an EXPLICIT "I know this prefix and it has nothing" is an answer.
    assert ps._tenant_truth_view({"prefix": "x", "found": False}) is not None


@pytest.mark.asyncio
async def test_tenant_truth_degrades_once_when_the_route_is_not_deployed(monkeypatch):
    """79 rows x every 195 s is a lot of 404s to keep asking for."""
    from app.services import pool_service as ps

    calls = {"n": 0}

    class _Resp:
        status_code = 404

    class _Client:
        async def get(self, *a, **k):
            calls["n"] += 1
            return _Resp()

    async def _get_client():
        return _Client()

    monkeypatch.setattr(
        "app.services.docker_host_service.get_bridge_client", _get_client,
    )
    monkeypatch.setattr(ps, "_TENANT_TRUTH_UNAVAILABLE", False, raising=False)

    assert await ps.bridge_tenant_truth("abcd1234") is None
    assert await ps.bridge_tenant_truth("abcd1234") is None
    assert calls["n"] == 1, "degrade once, not once per subject per tick"
    monkeypatch.setattr(ps, "_TENANT_TRUTH_UNAVAILABLE", False, raising=False)


# ── The sweep, end to end, with a fake bridge ─────────────────────────

@pytest.mark.asyncio
async def test_sweep_adopts_bridge_truth_and_converges_in_one_tick(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _seed_named_401_user("toup-agent-f261b564")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_401_probe(monkeypatch)

    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value={
        "container_name": "toup-agent-pool-81", "db_name": "pool_81",
        "state": "ASSIGNED", "bound": True,
    }))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value={
        "pool_container": "toup-agent-pool-81", "pool_db": "pool_81",
        "named_container": None, "route_container": "toup-agent-pool-81",
    }))
    adopt = AsyncMock(return_value="https://agent-x.test")
    monkeypatch.setattr(ps, "_adopt_discovered_bind", adopt)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)

    summary = await ps.reclaim_stranded_users()

    assert summary.get("repaired") == 1
    assert summary.get("keyless_named_deferred", 0) == 0
    restart.assert_not_awaited()
    adopt.assert_awaited_once()
    # DIRECTION: the routed container's identity goes INTO the platform row.
    _args, kwargs = adopt.await_args
    assert _args[1]["container_name"] == "toup-agent-pool-81"
    assert kwargs.get("bridge_confirmed_sole_owner") is True
    # Converged → the streak is gone, so the next tick starts clean.
    assert await _probe_row(uid) is None


@pytest.mark.asyncio
async def test_sweep_refuses_to_act_on_ambiguous_ownership_and_pages(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    await _seed_named_401_user("toup-agent-51d4ed2f")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_401_probe(monkeypatch)

    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value={
        "pool_container": "toup-agent-pool-17", "pool_db": "pool_17",
        "named_container": "toup-agent-51d4ed2f",
        "named_db": "toup_agent_51d4ed2f",
        "route_container": "toup-agent-51d4ed2f",
    }))
    adopt = AsyncMock(return_value="https://x")
    monkeypatch.setattr(ps, "_adopt_discovered_bind", adopt)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)
    alerts: list = []

    async def _alert(category, level, message, **kw):
        alerts.append((category, level, message, kw.get("subject")))
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", _alert)

    summary = await ps.reclaim_stranded_users()

    adopt.assert_not_awaited()
    restart.assert_not_awaited()
    assert summary.get(f"named_401_{ps.OWNERSHIP_AMBIGUOUS}") == 1
    amb = [a for a in alerts if a[0] == "pool-ownership-ambiguous"]
    assert len(amb) == 1, alerts
    assert amb[0][1] == "critical"
    body = amb[0][2]
    for token in ("toup-agent-pool-17", "toup-agent-51d4ed2f",
                  "pool_17", "toup_agent_51d4ed2f"):
        assert token in body, f"{token} missing from the page: {body!r}"


@pytest.mark.asyncio
async def test_sweep_escalates_but_never_force_claims_when_there_is_no_bind(monkeypatch):
    """R40: a force-claim for an established user with no bridge bind binds an
    EMPTY database. The correct action is a page."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    await _seed_named_401_user("toup-agent-deadbeef")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_401_probe(monkeypatch)

    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value={
        "pool_container": None, "named_container": None,
    }))
    claim = AsyncMock()
    monkeypatch.setattr(ps, "claim_for_user", claim)
    adopt = AsyncMock(return_value="https://x")
    monkeypatch.setattr(ps, "_adopt_discovered_bind", adopt)
    alerts: list = []

    async def _alert(category, level, message, **kw):
        alerts.append((category, level, message))
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", _alert)

    summary = await ps.reclaim_stranded_users()

    claim.assert_not_awaited()
    adopt.assert_not_awaited()
    assert summary.get(f"named_401_{ps.OWNERSHIP_NO_BIND}") == 1
    assert any("NO bind" in a[2] and a[1] == "critical" for a in alerts), alerts


@pytest.mark.asyncio
async def test_no_bridge_evidence_defers_exactly_as_before(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    await _seed_named_401_user("toup-agent-b0b0b0b0")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_401_probe(monkeypatch)
    monkeypatch.setattr(
        ps, "bridge_lookup_user_slot",
        AsyncMock(side_effect=RuntimeError("bridge down")),
    )
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value=None))
    adopt = AsyncMock(return_value="https://x")
    monkeypatch.setattr(ps, "_adopt_discovered_bind", adopt)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)

    summary = await ps.reclaim_stranded_users()

    adopt.assert_not_awaited()
    restart.assert_not_awaited()
    assert summary.get("keyless_named_deferred") == 1


@pytest.mark.asyncio
async def test_the_tick_counter_survives_and_carries_the_streak(monkeypatch):
    """"46th consecutive tick, since 19:12" is the sentence an operator acts
    on. It has to survive a redeploy, which a module dict never did."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _seed_named_401_user("toup-agent-c0ffee11")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    _install_401_probe(monkeypatch)
    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "_restart_sick_container", AsyncMock(return_value=True))

    await ps.reclaim_stranded_users()
    await ps.reclaim_stranded_users()
    await ps.reclaim_stranded_users()

    row = await _probe_row(uid)
    assert row is not None
    assert row.consecutive_failures == 3
    assert row.last_class == "401"
    assert row.first_seen_at is not None


@pytest.mark.asyncio
async def test_a_200_clears_the_streak(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _seed_named_401_user("toup-agent-1234abcd")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value=None))
    _install_401_probe(monkeypatch)
    await ps.reclaim_stranded_users()
    assert (await _probe_row(uid)).consecutive_failures == 1

    class _Ok:
        status_code = 200

    class _OkClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Ok()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _OkClient)
    await ps.reclaim_stranded_users()
    assert await _probe_row(uid) is None


@pytest.mark.asyncio
async def test_a_different_failure_class_restarts_the_streak(monkeypatch):
    """"46 consecutive 401s" must never be 20 timeouts plus 26 401s — the two
    have different causes and different fixes."""
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _seed_named_401_user("toup-agent-77665544")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)
    monkeypatch.setattr(ps, "bridge_lookup_user_slot", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "bridge_tenant_truth", AsyncMock(return_value=None))
    monkeypatch.setattr(ps, "_restart_sick_container", AsyncMock(return_value=True))
    _install_401_probe(monkeypatch)
    await ps.reclaim_stranded_users()
    await ps.reclaim_stranded_users()
    assert (await _probe_row(uid)).consecutive_failures == 2

    class _Boom:
        status_code = 500

    class _BoomClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Boom()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)
    await ps.reclaim_stranded_users()
    row = await _probe_row(uid)
    assert row.last_class == "5xx"
    assert row.consecutive_failures == 1, "the 401 streak does not carry over"


# ── The per-user restart cap ──────────────────────────────────────────

def test_restart_cap_is_per_user_per_hour_and_shared():
    from app.services import pool_service as ps

    st = ps._probe_state_blank("u")
    for _ in range(ps.MAX_RESTARTS_PER_USER_PER_HOUR):
        assert ps._restart_allowed(st)
        ps._record_restart(st)
    assert ps._restart_allowed(st) is False, (
        "the platform must not override the bridge's own crash-loop cap"
    )
    # …and the window rolls.
    st["restart_window_started_at"] = (
        datetime.utcnow() - timedelta(seconds=ps.RESTART_WINDOW_S + 1)
    )
    assert ps._restart_allowed(st) is True


@pytest.mark.asyncio
async def test_sick_restarts_stop_at_the_cap_and_keep_probing(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services import pool_service as ps

    uid = await _seed_named_401_user("toup-agent-pool-91")
    monkeypatch.setattr(ps.settings, "use_container_pool", True, raising=False)

    class _Boom:
        status_code = 500

    class _BoomClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Boom()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)
    restart = AsyncMock(return_value=True)
    monkeypatch.setattr(ps, "_restart_sick_container", restart)
    alerts: list = []

    async def _alert(category, level, message, **kw):
        alerts.append((level, message))
        return True

    monkeypatch.setattr("app.services.alerting.send_infra_alert", _alert)

    # Each restart takes 2 ticks (strike, then act), so 8 ticks is 4 attempts
    # against a cap of 3.
    for _ in range(8):
        await ps.reclaim_stranded_users()

    assert restart.await_count == ps.MAX_RESTARTS_PER_USER_PER_HOUR
    row = await _probe_row(uid)
    assert row is not None and row.escalated_at is not None
    assert any("restarts in an" in m for _l, m in alerts), alerts


# ── Direction, as a structural invariant ──────────────────────────────

def test_the_adopt_escape_hatch_has_exactly_one_producer():
    """`bridge_confirmed_sole_owner` is the one way past the refusal that keeps
    a named tenant off a pool slot. It must only ever be set by the path that
    obtained the evidence — a caller that sets it without the tenant-truth read
    re-opens the R40 data-loss door."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app" / "services" / "pool_service.py").read_text()
    setters = src.count("bridge_confirmed_sole_owner=True")
    assert setters == 1, f"{setters} callers set the escape hatch"
    # …and it is set inside `_adopt_bridge_truth`, whose evidence comes from
    # classify_named_401_ownership.
    body = src[src.index("async def _adopt_bridge_truth"):]
    assert "bridge_confirmed_sole_owner=True" in body[:body.index("\n\n\n")
                                                      if "\n\n\n" in body
                                                      else len(body)]
