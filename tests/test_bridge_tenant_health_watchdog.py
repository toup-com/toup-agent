"""The named-container watchdog: a boot grace, a host-wide quorum, one budget.

Three separate defects, all live on 2026-09-12:

1. **No start-up grace.** Three failed probes ~90 s apart can land inside a
   cold first boot. (The premise that a named boot *structurally* cannot make
   that deadline is REFUTED — an uninterrupted one measured 56.7 s — so what
   is added here is a margin, not a new budget.)

2. **No quorum guard.** `pool_addon` refuses to restart ASSIGNED members when a
   majority fail in the same tick, because a majority failing at once is a
   docker daemon / disk / network fault and restarting them all turns a blip
   into an outage. `tenant_health.py` had no such logic under any name. At
   02:00:50 on 13 Sep the pool was correctly spared with 71/76 sick, and a
   named container was restarted 34 s later.

3. **`POST /v1/tenants/{p}/restart` bypassed the budget and the route.** The
   platform posted it 92 times in 2 h 28 m from both replicas, at a container
   nobody was routed to; each restart produced an agent that answered
   `/agent/health` long enough to reset `failure_count`, so the CRASH-LOOP
   give-up re-fired forever and never stopped anything.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_tenant_health_watchdog.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import time
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _bridge_main_loader import load_bridge_main  # noqa: E402

HEALTH_PY = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "tenant_health.py"


def _load_health() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"tenant_health_uut_{time.time_ns()}", HEALTH_PY
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def th():
    mod = _load_health()
    mod._state.clear()
    mod._restart_history.clear()
    return mod


def _tenant(prefix, port, started_at=None):
    return {
        "prefix": prefix,
        "container_name": f"toup-agent-{prefix}",
        "host_port": port,
        "started_at": started_at,
    }


def _drive(th, tenants, probe_ok, restarts=None):
    """Run one tick with the docker list and the probe stubbed out."""
    restarts = restarts if restarts is not None else []

    async def _fake_list():
        return tenants

    async def _fake_probe(port):
        return probe_ok(port)

    async def _fake_restart(name):
        restarts.append(name)
        return True

    th._list_assigned_tenants = lambda: tenants
    th._probe_health = _fake_probe
    th._docker_restart = _fake_restart
    return asyncio.run(th.healthcheck_tick()), restarts


# ── StartedAt parsing ──────────────────────────────────────────────


def test_docker_nanosecond_timestamps_parse(th):
    """The two ends of the uninterrupted 12 Sep named boot, 56.7 s apart."""
    import datetime as dt

    start = th._parse_docker_time("2026-09-12T19:16:14.632812345Z")
    ready = th._parse_docker_time("2026-09-12T19:17:11.337000000+00:00")
    assert start == dt.datetime(
        2026, 9, 12, 19, 16, 14, 632812, tzinfo=dt.timezone.utc
    ).timestamp()
    assert abs((ready - start) - 56.7) < 0.05


@pytest.mark.parametrize("value", ["", None, "not a time", "2026-13-45T99:99:99Z", 17])
def test_unparseable_timestamps_are_none_not_exceptions(th, value):
    assert th._parse_docker_time(value) is None


def test_a_never_started_container_has_no_grace(th):
    """Docker's zero time. It must parse to "long ago", not to "just now"."""
    ts = th._parse_docker_time("0001-01-01T00:00:00Z")
    assert ts is None or time.time() - ts > th.START_GRACE_S


# ── 1. start-up grace ──────────────────────────────────────────────


def test_a_freshly_started_container_is_not_probed(th):
    now = time.time()
    summary, restarts = _drive(
        th, [_tenant("aaaaaaaa", 9001, started_at=now - 5)], lambda p: False
    )
    assert summary["starting"] == 1
    assert summary["probed"] == 0
    assert summary["failing"] == 0
    assert restarts == []


def test_grace_expires_and_the_watchdog_resumes(th):
    now = time.time()
    old = now - th.START_GRACE_S - 1
    summary, _ = _drive(th, [_tenant("aaaaaaaa", 9001, started_at=old)], lambda p: False)
    assert summary["probed"] == 1
    assert summary["failing"] == 1


def test_an_unknown_started_at_does_not_grant_grace(th):
    """A container whose attrs we could not read must stay watched — the
    grace is a margin, not a way to fall silent."""
    summary, _ = _drive(th, [_tenant("aaaaaaaa", 9001, started_at=None)], lambda p: False)
    assert summary["probed"] == 1


def test_our_own_restart_grants_grace_to_the_next_tick(th):
    old = time.time() - 3600
    tenants = [_tenant("aaaaaaaa", 9001, started_at=old)]
    restarts = []
    for _ in range(th.FAILS_BEFORE_RESTART):
        _drive(th, tenants, lambda p: False, restarts)
    assert restarts == ["toup-agent-aaaaaaaa"]
    # `started_at` in our stub is unchanged (docker would have moved it), so
    # without note_started() the very next tick would strike again.
    summary, restarts2 = _drive(th, tenants, lambda p: False, [])
    assert summary["starting"] == 1
    assert restarts2 == []


def test_an_external_restart_also_grants_grace(th):
    old = time.time() - 3600
    th.note_started("aaaaaaaa")
    summary, restarts = _drive(
        th, [_tenant("aaaaaaaa", 9001, started_at=old)], lambda p: False
    )
    assert summary["starting"] == 1
    assert restarts == []


# ── 2. host-wide quorum ────────────────────────────────────────────


def _fleet(n, started_at):
    return [_tenant(f"{i:08x}", 9000 + i, started_at) for i in range(n)]


def test_a_majority_failing_at_once_restarts_nothing(th):
    old = time.time() - 3600
    tenants = _fleet(9, old)
    restarts = []
    for _ in range(th.FAILS_BEFORE_RESTART):
        summary, _ = _drive(th, tenants, lambda p: False, restarts)
    assert summary["health_quorum_skip"] == 9
    assert restarts == []


def test_one_sick_agent_among_healthy_ones_is_still_restarted(th):
    old = time.time() - 3600
    tenants = _fleet(9, old)
    sick_port = tenants[3]["host_port"]
    restarts = []
    for _ in range(th.FAILS_BEFORE_RESTART):
        summary, _ = _drive(th, tenants, lambda p: p != sick_port, restarts)
    assert "health_quorum_skip" not in summary
    assert restarts == [tenants[3]["container_name"]]


def test_quorum_needs_a_quorum_to_mean_anything(th):
    """Two of two sick is two sick agents, not a sick host."""
    old = time.time() - 3600
    tenants = _fleet(2, old)
    restarts = []
    for _ in range(th.FAILS_BEFORE_RESTART):
        summary, _ = _drive(th, tenants, lambda p: False, restarts)
    assert "health_quorum_skip" not in summary
    assert len(restarts) == 2


def test_containers_in_grace_do_not_count_toward_the_quorum(th):
    """Otherwise a deploy that restarts most of the fleet would disarm the
    watchdog for the one container that is genuinely wedged."""
    now = time.time()
    tenants = _fleet(5, now - 5)          # all booting
    tenants.append(_tenant("ffffffff", 9500, started_at=now - 3600))
    restarts = []
    for _ in range(th.FAILS_BEFORE_RESTART):
        summary, _ = _drive(th, tenants, lambda p: False, restarts)
    assert summary["probed"] == 1
    assert "health_quorum_skip" not in summary
    assert restarts == ["toup-agent-ffffffff"]


# ── 3. the shared restart budget ───────────────────────────────────


def test_the_budget_is_per_prefix_and_bounded(th):
    for i in range(th.MAX_RESTARTS_PER_WINDOW):
        allowed, used = th.restart_allowed("aaaaaaaa")
        assert allowed and used == i
        th.record_restart("aaaaaaaa")
    allowed, used = th.restart_allowed("aaaaaaaa")
    assert not allowed and used == th.MAX_RESTARTS_PER_WINDOW
    assert th.restart_allowed("bbbbbbbb")[0] is True


def test_the_window_rolls(th):
    th._restart_history["aaaaaaaa"] = th.deque(
        [time.time() - th.RESTART_WINDOW_S - 1] * 5
    )
    assert th.restart_allowed("aaaaaaaa") == (True, 0)


def test_a_failed_docker_restart_does_not_consume_the_budget(th):
    old = time.time() - 3600
    tenants = [_tenant("aaaaaaaa", 9001, started_at=old)]

    async def _fail(name):
        return False

    th._list_assigned_tenants = lambda: tenants
    th._probe_health = lambda port: _false()
    th._docker_restart = _fail

    async def _false():
        return False

    for _ in range(th.FAILS_BEFORE_RESTART):
        asyncio.run(th.healthcheck_tick())
    assert th.restarts_in_window("aaaaaaaa") == 0


# ── 3b. the endpoint honours both ──────────────────────────────────


@pytest.fixture
def bridge(tmp_path, monkeypatch, th):
    mod = load_bridge_main()
    agents = tmp_path / "agents"
    (agents / "_pool").mkdir(parents=True)
    monkeypatch.setattr(mod, "AGENTS_DATA_DIR", agents)
    monkeypatch.setattr(mod, "_caddy_tenant_route_dial", lambda p: None)
    monkeypatch.setattr(mod, "_pool_members_registry", lambda: [])
    # The endpoint imports tenant_health lazily; give it this test's instance.
    sys.modules["tenant_health"] = th
    yield mod
    sys.modules.pop("tenant_health", None)


class _Container:
    def __init__(self):
        self.restarts = 0

    def restart(self, timeout=10):
        self.restarts += 1


def test_restart_endpoint_refuses_a_pool_routed_prefix(bridge, monkeypatch):
    from fastapi import HTTPException

    member = {
        "slot": "81", "state": "ASSIGNED", "port": 9573,
        "container_name": "toup-agent-pool-81", "assigned_prefix": "f261b564",
    }
    monkeypatch.setattr(bridge, "_pool_members_registry", lambda: [member])
    monkeypatch.setattr(bridge, "_caddy_tenant_route_dial", lambda p: "127.0.0.1:9573")
    got = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: got)

    with pytest.raises(HTTPException) as e:
        bridge.restart_tenant("f261b564")
    assert e.value.status_code == 409
    assert e.value.detail["slot"] == "81"
    assert "restart-member" in e.value.detail["use_instead"]
    assert got.restarts == 0


def test_restart_endpoint_allows_a_named_routed_prefix(bridge, monkeypatch, th):
    member = {
        "slot": "17", "state": "ASSIGNED", "port": 9517,
        "container_name": "toup-agent-pool-17", "assigned_prefix": "51d4ed2f",
    }
    monkeypatch.setattr(bridge, "_pool_members_registry", lambda: [member])
    monkeypatch.setattr(bridge, "_caddy_tenant_route_dial", lambda p: "127.0.0.1:9083")
    got = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: got)

    assert bridge.restart_tenant("51d4ed2f")["status"] == "restarting"
    assert got.restarts == 1


def test_restart_endpoint_spends_the_watchdog_budget(bridge, monkeypatch, th):
    from fastapi import HTTPException

    got = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: got)

    for i in range(th.MAX_RESTARTS_PER_WINDOW):
        assert bridge.restart_tenant("aaaaaaaa")["restarts_in_window"] == i + 1
    with pytest.raises(HTTPException) as e:
        bridge.restart_tenant("aaaaaaaa")
    assert e.value.status_code == 429
    assert got.restarts == th.MAX_RESTARTS_PER_WINDOW
    # And the watchdog sees those restarts as its own.
    assert th.restart_allowed("aaaaaaaa") == (False, th.MAX_RESTARTS_PER_WINDOW)


def test_an_external_restart_does_not_reset_the_failure_count_into_a_new_strike(
    bridge, monkeypatch, th
):
    """The 92-restart loop worked because each platform restart made the
    container answer for a while, zeroing `failure_count` and re-arming the
    whole cycle. The restart now books grace instead."""
    got = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: got)
    bridge.restart_tenant("aaaaaaaa")
    assert th._state["aaaaaaaa"]["last_status"] == "starting"
    assert th._state["aaaaaaaa"]["grace_until"] > time.time()


def test_restart_endpoint_still_404s_for_an_unknown_container(bridge, monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        bridge.restart_tenant("deadbeef")
    assert e.value.status_code == 404


def test_restart_endpoint_rejects_a_bogus_prefix(bridge):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        bridge.restart_tenant("../../etc")
    assert e.value.status_code == 400
