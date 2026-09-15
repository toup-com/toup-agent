"""`POST /v1/tenants` must not mint a second tenant over a live pool slot.

On 2026-09-12 the platform's named fallback fired for prefix `f261b564` while
pool slot 81 was ASSIGNED to that user, bound, healthy and routed. `create_tenant`
made no reference whatsoever to the pool registry, so it:

  * minted a fresh 43-char `AGENT_API_KEY`,
  * ran `create_tenant_db` → a SECOND database and role for the same user,
  * started a second container,
  * and repointed `agent-f261b564.agents.toup.ai` off the bound pool slot.

Twice — once per Railway replica, 3 s apart, the second force-removing the
container the first had just created. Nothing released slot 81 and nothing
reverses any of it: four accounts still carry two containers, two databases and
one user each, and all four are excluded from Caddy route restoration for it.

The guard here is the bridge's own backstop for that class. It is deliberately
NOT "the platform should stop calling this": the platform is two replicas with no
leader election and the destructive call is its fallback path, so the refusal has
to live at the thing that owns the resource.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_create_tenant_pool_guard.py
"""
from __future__ import annotations

import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _bridge_main_loader import load_bridge_main  # noqa: E402

PREFIX = "f261b564"


def _member(slot="81", prefix=PREFIX, state="ASSIGNED", port=9573):
    return {
        "slot": slot,
        "state": state,
        "port": port,
        "container_name": f"toup-agent-pool-{slot}",
        "assigned_prefix": prefix,
        "assigned_user_id": f"{prefix}-0000-0000-0000-000000000000",
        "db_name": f"toup_agent_feed00{slot}",
    }


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """A loaded `main.py` whose /data/agents is a temp dir we control."""
    mod = load_bridge_main()
    agents = tmp_path / "agents"
    (agents / "_pool").mkdir(parents=True)
    monkeypatch.setattr(mod, "AGENTS_DATA_DIR", agents)
    # No test here may reach Caddy; every case states the dial explicitly.
    monkeypatch.setattr(mod, "_caddy_tenant_route_dial", lambda p: None)
    mod._test_agents = agents
    return mod


def _write_registry(mod, members):
    (mod._test_agents / "_pool" / "members.json").write_text(json.dumps(members))


def _named_env(mod, prefix, port=9055):
    d = mod._test_agents / prefix
    d.mkdir(parents=True, exist_ok=True)
    (d / ".env").write_text(
        f"AGENT_PORT={port}\nAGENT_API_KEY=existing-key\n"
        f"DATABASE_URL=postgresql+asyncpg://u:pw@127.0.0.1:6432/toup_agent_{prefix}\n"
    )


# ── the pure decision ──────────────────────────────────────────────


def test_assigned_pool_member_owns_the_prefix(bridge):
    conflict = bridge._pool_ownership_for_prefix(PREFIX, [_member()], None)
    assert conflict is not None
    assert conflict["reason"] == "pool_slot_assigned"
    assert conflict["slot"] == "81"
    assert conflict["container_name"] == "toup-agent-pool-81"


def test_a_route_pointing_at_a_pool_member_owns_the_prefix(bridge):
    """The claim that is still ASSIGNING has no ASSIGNED row yet — but it is
    already routed, and routing is what the named path would steal."""
    m = _member(state="ASSIGNING")
    assert bridge._pool_ownership_for_prefix(PREFIX, [m], "127.0.0.1:9573") is not None
    assert (
        bridge._pool_ownership_for_prefix(PREFIX, [m], "127.0.0.1:9573")["reason"]
        == "tenant_route_points_at_pool_member"
    )


def test_a_route_pointing_at_a_named_port_is_not_a_conflict(bridge):
    assert bridge._pool_ownership_for_prefix(PREFIX, [], "127.0.0.1:9055") is None


def test_another_users_assigned_slot_is_not_a_conflict(bridge):
    members = [_member(slot="26", prefix="3703be10", port=9558)]
    assert bridge._pool_ownership_for_prefix(PREFIX, members, None) is None


def test_an_unreadable_registry_never_raises(bridge):
    (bridge._test_agents / "_pool" / "members.json").write_text("{not json")
    assert bridge._pool_members_registry() == []
    (bridge._test_agents / "_pool" / "members.json").write_text('{"members": []}')
    assert bridge._pool_members_registry() == []


# ── the endpoint ───────────────────────────────────────────────────


def test_create_tenant_refuses_a_pool_owned_prefix(bridge):
    from fastapi import HTTPException

    _write_registry(bridge, [_member()])
    req = bridge.CreateTenantReq(prefix=PREFIX, user_id="u-1")
    with pytest.raises(HTTPException) as e:
        bridge.create_tenant(req)
    assert e.value.status_code == 409
    detail = e.value.detail
    assert detail["error"] == "pool_slot_owns_prefix"
    # The body has to name the slot, or the operator cannot act on it.
    assert detail["slot"] == "81"
    assert detail["container_name"] == "toup-agent-pool-81"
    assert "force_named" in detail["override"]


def test_the_refusal_happens_before_any_mutation(bridge, monkeypatch):
    """A 409 that has already created the network, the DB or the key is not a
    refusal — it is the incident with a different status code."""
    from fastapi import HTTPException

    touched = []
    monkeypatch.setattr(
        bridge.docker_client.networks, "create",
        lambda *a, **k: touched.append("network"),
    )
    monkeypatch.setattr(bridge, "_create_tenant_db", lambda p: touched.append("db") or "x")
    monkeypatch.setattr(bridge, "_next_port", lambda **k: touched.append("port") or 9999)
    monkeypatch.setattr(bridge, "_write_tenant_env", lambda *a: touched.append("env"))

    _write_registry(bridge, [_member()])
    with pytest.raises(HTTPException):
        bridge.create_tenant(bridge.CreateTenantReq(prefix=PREFIX, user_id="u-1"))
    assert touched == []


def test_force_named_overrides_the_refusal(bridge, monkeypatch):
    """The operator escape hatch still has to work — reconciling the four
    scarred accounts may well need it."""
    _write_registry(bridge, [_member()])
    reached = []
    monkeypatch.setattr(
        bridge.docker_client.networks, "create",
        lambda *a, **k: reached.append("network") or _Stub(),
    )
    monkeypatch.setattr(bridge, "_next_port", lambda **k: 9999)
    monkeypatch.setattr(bridge, "_create_tenant_db", lambda p: "pw")
    monkeypatch.setattr(bridge, "_write_tenant_env", lambda *a: None)
    monkeypatch.setattr(bridge.docker_client.containers, "run",
                        lambda *a, **k: _Stub(id="c" * 20))
    monkeypatch.setattr(bridge, "_caddy_add_tenant_route", lambda *a: None)

    resp = bridge.create_tenant(
        bridge.CreateTenantReq(prefix=PREFIX, user_id="u-1", force_named=True)
    )
    assert resp.host_port == 9999
    assert reached == ["network"]


def test_an_existing_named_tenant_stays_idempotent(bridge, monkeypatch):
    """`create_tenant` is how an env refresh and an image upgrade land on a
    tenant that IS named. That path reuses the key, the DB and the port, so it
    is not the destructive one and must not be refused — even while a stale
    pool row still claims the prefix (which is exactly the four-account state)."""
    _write_registry(bridge, [_member()])
    _named_env(bridge, PREFIX, port=9055)

    monkeypatch.setattr(bridge.docker_client.networks, "create", lambda *a, **k: _Stub())
    made_db = []
    monkeypatch.setattr(bridge, "_create_tenant_db", lambda p: made_db.append(p) or "pw")
    monkeypatch.setattr(bridge, "_write_tenant_env", lambda *a: None)
    monkeypatch.setattr(bridge.docker_client.containers, "run",
                        lambda *a, **k: _Stub(id="c" * 20))
    monkeypatch.setattr(bridge, "_caddy_add_tenant_route", lambda *a: None)

    resp = bridge.create_tenant(bridge.CreateTenantReq(prefix=PREFIX, user_id="u-1"))
    assert resp.host_port == 9055            # reused, not reallocated
    assert resp.agent_api_key == "existing-key"  # reused, not rotated
    assert made_db == []                     # no second database


def test_a_prefix_nobody_owns_still_provisions(bridge, monkeypatch):
    _write_registry(bridge, [_member(slot="26", prefix="3703be10", port=9558)])
    monkeypatch.setattr(bridge.docker_client.networks, "create", lambda *a, **k: _Stub())
    monkeypatch.setattr(bridge, "_next_port", lambda **k: 9101)
    monkeypatch.setattr(bridge, "_create_tenant_db", lambda p: "pw")
    monkeypatch.setattr(bridge, "_write_tenant_env", lambda *a: None)
    monkeypatch.setattr(bridge.docker_client.containers, "run",
                        lambda *a, **k: _Stub(id="c" * 20))
    monkeypatch.setattr(bridge, "_caddy_add_tenant_route", lambda *a: None)

    resp = bridge.create_tenant(bridge.CreateTenantReq(prefix="51d4ed2f", user_id="u-2"))
    assert resp.host_port == 9101


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.__dict__.setdefault("id", "stub")
        self.__dict__.setdefault("name", "stub")
