"""`named_and_pool_ownership_ambiguous` — resolve it, see it, or end it.

Four accounts have been in this state continuously since 12 Sep 22:44. Both
inventories claim the prefix, so `_restore_caddy_routes_from_disk_v2` blocked it
from both, restored nothing, and recorded the refusal in a dict nobody logged.
The platform's self-heal sweep defers a 4xx to "the bridge route reconciler owns
that" and the bridge route reconciler had decided not to act — two healers each
correctly deferring to the other is zero healers. On the next Caddy restart those
hostnames get no route at all.

Three changes, tested here:

1. **The live route is the tie-break.** Whichever side the hostname dials right
   now IS what serves that user; adopt it and touch nothing else. Deliberately
   NOT the platform's `agent_configs` row: the bridge makes no platform call
   today, a boot-path dependency on the platform trades four dark accounts for a
   dark fleet, and at 19:14:23 on 12 Sep that row pointed at the empty named
   container — following it would have made the outage worse.

2. **It is logged, once per prefix per pass**, and the reconciler's error list
   finally reaches the journal.

3. **An operator can end it**, from either side, without deleting anything.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_ownership_conflicts.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _bridge_main_loader import load_bridge_main  # noqa: E402

POOL_SIDE = "f261b564"   # pool-81 routed, named side is the orphan
NAMED_SIDE = "51d4ed2f"  # named :9083 routed, pool-17 holds a 4-message fragment


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    mod = load_bridge_main()
    agents = tmp_path / "agents"
    (agents / "_pool").mkdir(parents=True)
    monkeypatch.setattr(mod, "AGENTS_DATA_DIR", agents)
    monkeypatch.setattr(mod, "RETIRED_AGENTS_DIR", tmp_path / "retired")
    mod._test_agents = agents
    mod._test_retired = tmp_path / "retired"
    return mod


def _conflict(mod, prefix, pool_port, named_port, dial, slot="81"):
    """Both inventories claim `prefix`; the hostname dials `dial`."""
    member = {
        "slot": slot, "state": "ASSIGNED", "port": pool_port,
        "container_name": f"toup-agent-pool-{slot}",
        "assigned_prefix": prefix,
        "assigned_user_id": f"{prefix}-1111-2222-3333-444455556666",
        "db_name": f"toup_agent_feed00{slot}",
    }
    (mod._test_agents / "_pool" / "members.json").write_text(json.dumps([member]))
    d = mod._test_agents / prefix
    d.mkdir(parents=True, exist_ok=True)
    (d / ".env").write_text(
        f"AGENT_PORT={named_port}\nAGENT_API_KEY=k\n"
        f"DATABASE_URL=postgresql+asyncpg://u:secret@127.0.0.1:6432/toup_agent_{prefix}\n"
    )
    routes = []
    if dial is not None:
        routes.append({
            "@id": f"tenant_{prefix}",
            "match": [{"host": [f"agent-{prefix}.agents.toup.ai"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": dial}]}],
        })
    return member, routes


# ── 1. the tie-break ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "dial,expected",
    [
        ("127.0.0.1:9573", ("pool", 9573)),
        ("127.0.0.1:9083", ("named", 9083)),
        (None, (None, None)),                  # Caddy lost the route
        ("127.0.0.1:9999", (None, None)),      # dials a third thing
        ("", (None, None)),
    ],
)
def test_the_live_route_decides_and_nothing_else(bridge, dial, expected):
    assert bridge._ambiguity_live_route_side(dial, 9573, 9083) == expected


def test_two_sides_on_one_port_cannot_be_told_apart(bridge):
    """A dial that matches both is not evidence for either."""
    assert bridge._ambiguity_live_route_side("127.0.0.1:9000", 9000, 9000) == (None, None)


def test_restore_adopts_the_routed_side_and_writes_nothing(bridge, monkeypatch):
    _member, routes = _conflict(bridge, POOL_SIDE, 9573, 9055, "127.0.0.1:9573")
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({POOL_SIDE: 9573}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({POOL_SIDE: 9055}, set(), 0, []),
    )
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp(routes))
    wrote = []
    monkeypatch.setattr(bridge, "_restore_one_caddy_route",
                        lambda p, port: wrote.append((p, port)) or "restored")

    summary = bridge._restore_caddy_routes_from_disk_v2()
    assert summary["conflicts_preserved"] == 0
    assert summary["verified"] == 1
    assert wrote == []            # the route is already right; touch nothing
    assert any("resolved:live_route:pool" in d["error"] for d in summary["details"])


def test_restore_refuses_when_no_route_names_a_side(bridge, monkeypatch):
    """The armed P0: after a Caddy restart there is no route to read, so the
    ambiguity stands. It must still refuse — and now it says so out loud."""
    _conflict(bridge, POOL_SIDE, 9573, 9055, None)
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({POOL_SIDE: 9573}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({POOL_SIDE: 9055}, set(), 0, []),
    )
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp([]))
    wrote = []
    monkeypatch.setattr(bridge, "_restore_one_caddy_route",
                        lambda p, port: wrote.append(p) or "restored")

    summary = bridge._restore_caddy_routes_from_disk_v2()
    assert summary["conflicts_preserved"] == 1
    assert wrote == []            # never guess a side
    assert any(d["error"] == "named_and_pool_ownership_ambiguous"
               for d in summary["details"])
    assert any(r[1] == "caddy_ownership_ambiguous_unresolved"
               for r in bridge.logger.records)


def test_an_unambiguous_prefix_is_unaffected(bridge, monkeypatch):
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({"aaaaaaaa": 9501}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk", lambda: ({}, set(), 0, [])
    )
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp([]))
    restored = []
    monkeypatch.setattr(bridge, "_restore_one_caddy_route",
                        lambda p, port: restored.append((p, port)) or "restored")
    summary = bridge._restore_caddy_routes_from_disk_v2()
    assert restored == [("aaaaaaaa", 9501)]
    assert summary["conflicts_preserved"] == 0


def test_a_retired_data_dir_is_not_a_named_candidate_nor_an_error(bridge):
    """The operator's rename used to come back as `named_prefix_invalid` on
    every pass, forever."""
    d = bridge._test_agents / f"{POOL_SIDE}.retired-1789300000"
    d.mkdir(parents=True)
    (d / ".env").write_text("AGENT_PORT=9055\n")
    named, blocked, skipped, errors = bridge._named_route_candidates_from_disk()
    assert named == {} and blocked == set() and errors == []
    assert skipped == 1


# ── 2. the report an operator acts from ────────────────────────────


def test_the_report_names_both_sides(bridge, monkeypatch):
    _conflict(bridge, NAMED_SIDE, 9517, 9083, "127.0.0.1:9083", slot="17")
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({NAMED_SIDE: 9517}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({NAMED_SIDE: 9083}, set(), 0, []),
    )
    routes = [{
        "@id": f"tenant_{NAMED_SIDE}",
        "match": [{"host": [f"agent-{NAMED_SIDE}.agents.toup.ai"]}],
        "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9083"}]}],
    }]
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp(routes))

    report = bridge._ownership_conflicts_report()
    assert report["count"] == 1
    c = report["conflicts"][0]
    assert c["routed_side"] == "named"
    assert c["pool"]["slot"] == "17"
    assert c["pool"]["container_name"] == "toup-agent-pool-17"
    assert c["pool"]["db_name"] == "toup_agent_feed0017"
    assert c["named"]["container_name"] == f"toup-agent-{NAMED_SIDE}"
    assert c["named"]["db_name"] == f"toup_agent_{NAMED_SIDE}"
    # Each action is offered only for the side that is NOT routed.
    assert c["release_pool_slot_available"] is True
    assert c["retire_named_available"] is False


def test_the_report_never_carries_a_credential(bridge, monkeypatch):
    _conflict(bridge, NAMED_SIDE, 9517, 9083, "127.0.0.1:9083", slot="17")
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({NAMED_SIDE: 9517}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({NAMED_SIDE: 9083}, set(), 0, []),
    )
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp([]))
    assert "secret" not in json.dumps(bridge._ownership_conflicts_report())


def test_an_unreachable_caddy_reports_unresolved_rather_than_guessing(bridge, monkeypatch):
    _conflict(bridge, NAMED_SIDE, 9517, 9083, "127.0.0.1:9083", slot="17")
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({NAMED_SIDE: 9517}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({NAMED_SIDE: 9083}, set(), 0, []),
    )

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(bridge.httpx, "get", _boom)
    report = bridge._ownership_conflicts_report()
    assert report["routes_error"]
    c = report["conflicts"][0]
    assert c["resolution"] == "unresolved"
    assert c["retire_named_available"] is False
    assert c["release_pool_slot_available"] is False


# ── 3. the operator actions ────────────────────────────────────────


def _routed(bridge, monkeypatch, prefix, pool_port, named_port, dial, slot):
    _conflict(bridge, prefix, pool_port, named_port, dial, slot=slot)
    monkeypatch.setattr(
        bridge, "_pool_route_owners_from_disk", lambda: ({prefix: pool_port}, set(), [])
    )
    monkeypatch.setattr(
        bridge, "_named_route_candidates_from_disk",
        lambda: ({prefix: named_port}, set(), 0, []),
    )
    routes = [] if dial is None else [{
        "@id": f"tenant_{prefix}",
        "match": [{"host": [f"agent-{prefix}.agents.toup.ai"]}],
        "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": dial}]}],
    }]
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp(routes))


def test_retire_named_stops_and_moves_out_of_the_tree(bridge, monkeypatch):
    _routed(bridge, monkeypatch, POOL_SIDE, 9573, 9055, "127.0.0.1:9573", "81")
    c = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: c)

    out = bridge._retire_named_side(POOL_SIDE, POOL_SIDE)
    assert out["container_stopped"] is True
    assert c.removed == 0                       # stopped, never removed
    assert not (bridge._test_agents / POOL_SIDE).exists()
    moved = list(bridge._test_retired.iterdir())
    assert len(moved) == 1 and moved[0].name.startswith(f"{POOL_SIDE}.retired-")
    assert (moved[0] / ".env").is_file()        # nothing deleted
    # And the destination is OUTSIDE the scanned tree.
    assert bridge._test_retired not in bridge._test_agents.parents
    assert out["database_untouched"] == f"toup_agent_{POOL_SIDE}"


def test_retire_named_refuses_when_the_named_side_is_the_routed_one(bridge, monkeypatch):
    from fastapi import HTTPException

    _routed(bridge, monkeypatch, NAMED_SIDE, 9517, 9083, "127.0.0.1:9083", "17")
    c = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: c)
    with pytest.raises(HTTPException) as e:
        bridge._retire_named_side(NAMED_SIDE, NAMED_SIDE)
    assert e.value.status_code == 409
    assert e.value.detail["error"] == "named_side_is_routed_or_unknown"
    assert c.stopped == 0
    assert (bridge._test_agents / NAMED_SIDE).is_dir()


def test_retire_named_refuses_without_a_route_to_read(bridge, monkeypatch):
    from fastapi import HTTPException

    _routed(bridge, monkeypatch, POOL_SIDE, 9573, 9055, None, "81")
    c = _Container()
    monkeypatch.setattr(bridge.docker_client.containers, "get", lambda n: c)
    with pytest.raises(HTTPException) as e:
        bridge._retire_named_side(POOL_SIDE, POOL_SIDE)
    assert e.value.status_code == 409
    assert c.stopped == 0


def test_retire_named_refuses_a_prefix_that_is_not_ambiguous(bridge, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(bridge, "_pool_route_owners_from_disk", lambda: ({}, set(), []))
    monkeypatch.setattr(bridge, "_named_route_candidates_from_disk",
                        lambda: ({POOL_SIDE: 9055}, set(), 0, []))
    monkeypatch.setattr(bridge.httpx, "get", lambda *a, **k: _Resp([]))
    with pytest.raises(HTTPException) as e:
        bridge._retire_named_side(POOL_SIDE, POOL_SIDE)
    assert e.value.detail["error"] == "not_ambiguous"


def test_retire_named_requires_the_confirm_token(bridge, monkeypatch):
    from fastapi import HTTPException

    _routed(bridge, monkeypatch, POOL_SIDE, 9573, 9055, "127.0.0.1:9573", "81")
    with pytest.raises(HTTPException) as e:
        bridge._retire_named_side(POOL_SIDE, "")
    assert e.value.status_code == 400


# ── 4. the pool half: release-pool-slot, and the summary nobody logged ──


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """`pool_addon` under a stub `main`, with its registry in a temp dir."""
    import importlib.util
    import types

    fake_main = types.ModuleType("main")
    for hook in ("_caddy_add_tenant_route", "_caddy_remove_tenant_route",
                 "_caddy_swap_upstream"):
        setattr(fake_main, hook, lambda *a, **k: None)
    fake_main.CADDY_ADMIN = "http://127.0.0.1:1"
    fake_main._ownership_conflicts_report = lambda: {"count": 0, "conflicts": []}
    monkeypatch.setitem(sys.modules, "main", fake_main)

    path = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
    spec = importlib.util.spec_from_file_location("pool_addon_ownership_uut", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)

    pool_dir = tmp_path / "_pool"
    (pool_dir / "binds").mkdir(parents=True)
    mod.POOL_DIR = pool_dir
    mod.MEMBERS_FILE = pool_dir / "members.json"
    mod.STATE_FILE = pool_dir / "state.json"
    mod.BINDS_DIR = pool_dir / "binds"
    mod._fake_main = fake_main
    return mod


def _register(pool, slot, prefix, port, db):
    pool.MEMBERS_FILE.write_text(json.dumps([{
        "slot": slot, "state": "ASSIGNED", "port": port,
        "container_name": f"toup-agent-pool-{slot}",
        "assigned_prefix": prefix,
        "assigned_user_id": f"{prefix}-u",
        "db_name": db,
    }]))
    (pool.BINDS_DIR / f"{slot}.json").write_text(
        json.dumps({"user_id": f"{prefix}-u", "agent_api_key": "k"})
    )


def _report(pool, prefix, slot, routed_side, db="toup_agent_feed0017"):
    pool._fake_main._ownership_conflicts_report = lambda: {
        "count": 1,
        "conflicts": [{
            "prefix": prefix,
            "routed_side": routed_side,
            "route_dial": "127.0.0.1:9083" if routed_side == "named" else "127.0.0.1:9517",
            "pool": {"slot": slot, "container_name": f"toup-agent-pool-{slot}",
                     "port": 9517, "db_name": db},
            "named": {"container_name": f"toup-agent-{prefix}", "port": 9083,
                      "db_name": f"toup_agent_{prefix}"},
        }],
    }


def test_release_pool_slot_quarantines_without_destroying_anything(pool):
    _register(pool, "17", NAMED_SIDE, 9517, "toup_agent_feed0017")
    _report(pool, NAMED_SIDE, "17", "named")

    out = pool.release_pool_slot_for_conflict(NAMED_SIDE, confirm=NAMED_SIDE)
    assert out["state"] == "QUARANTINED"
    assert out["database_untouched"] == "toup_agent_feed0017"

    member = json.loads(pool.MEMBERS_FILE.read_text())[0]
    # The assignment is what made the prefix a pool route owner; clearing it is
    # what ends the ambiguity.
    assert member["assigned_prefix"] is None
    assert member["assigned_user_id"] is None
    assert member["quarantined_prefix"] == NAMED_SIDE
    # QUARANTINED is none of the reaper's inputs, and not claimable either.
    assert member["state"] not in ("DEAD", "DRAINING", "TEARDOWN", "GENERIC")


def test_release_pool_slot_moves_the_bind_aside_rather_than_deleting_it(pool):
    _register(pool, "17", NAMED_SIDE, 9517, "toup_agent_feed0017")
    _report(pool, NAMED_SIDE, "17", "named")
    pool.release_pool_slot_for_conflict(NAMED_SIDE, confirm=NAMED_SIDE)

    assert not (pool.BINDS_DIR / "17.json").exists()
    kept = [p for p in pool.BINDS_DIR.iterdir() if ".quarantined-" in p.name]
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["agent_api_key"] == "k"


def test_a_quarantined_slot_is_invisible_to_claim_and_to_the_reaper(pool):
    _register(pool, "17", NAMED_SIDE, 9517, "toup_agent_feed0017")
    _report(pool, NAMED_SIDE, "17", "named")
    pool.release_pool_slot_for_conflict(NAMED_SIDE, confirm=NAMED_SIDE)

    assert pool._claim_one(user_id="someone-new") is None
    members = pool._load_members()
    assert [m for m in members
            if m["state"] in (pool.STATE_DEAD, pool.STATE_DRAINING,
                              pool.STATE_TEARDOWN)] == []
    # And the slot id itself stays reserved, so no spawn recycles its DB.
    assert pool._allocate_slot() != "17"


def test_release_pool_slot_refuses_when_the_pool_side_is_routed(pool):
    from fastapi import HTTPException

    _register(pool, "81", POOL_SIDE, 9573, "toup_agent_feed0081")
    _report(pool, POOL_SIDE, "81", "pool")
    with pytest.raises(HTTPException) as e:
        pool.release_pool_slot_for_conflict(POOL_SIDE, confirm=POOL_SIDE)
    assert e.value.status_code == 409
    assert json.loads(pool.MEMBERS_FILE.read_text())[0]["state"] == "ASSIGNED"


def test_release_pool_slot_refuses_without_confirm(pool):
    from fastapi import HTTPException

    _register(pool, "17", NAMED_SIDE, 9517, "toup_agent_feed0017")
    _report(pool, NAMED_SIDE, "17", "named")
    with pytest.raises(HTTPException) as e:
        pool.release_pool_slot_for_conflict(NAMED_SIDE, confirm="yes")
    assert e.value.status_code == 400


def test_the_reconciler_summary_reaches_the_journal(pool, caplog):
    """It was collected, parked in `_last_tick_summary` and never read out —
    so a condition true for 22 hours produced three log lines, all from a
    different code path."""
    import logging

    with caplog.at_level(logging.WARNING, logger="bridge.pool"):
        pool._log_tick_errors({"errors": [
            f"route prefix={NAMED_SIDE}: named_and_pool_ownership_ambiguous"
        ]})
    assert NAMED_SIDE in caplog.text
    assert "named_and_pool_ownership_ambiguous" in caplog.text


def test_a_clean_tick_logs_nothing_and_a_flood_is_capped(pool, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="bridge.pool"):
        pool._log_tick_errors({"errors": []})
        assert caplog.text == ""
        pool._log_tick_errors({"errors": [f"e{i}" for i in range(200)]})
    assert "200" in caplog.text
    assert "e199" not in caplog.text          # capped, not dumped


def test_health_reports_the_installed_file_digests(pool):
    import hashlib

    digests = pool._installed_bridge_digests()
    assert set(digests) == {"main.py", "pool_addon.py", "tenant_health.py"}
    real = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
    assert digests["pool_addon.py"] == hashlib.sha256(
        real.read_bytes()
    ).hexdigest()[:16]


def test_missing_files_do_not_break_the_digest_read(pool, monkeypatch, tmp_path):
    monkeypatch.setattr(pool, "_BRIDGE_FILES", ("main.py", "nope.py"))
    assert pool._installed_bridge_digests()["nope.py"] is None


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {"etag": '"t"'}
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Container:
    def __init__(self):
        self.stopped = 0
        self.removed = 0

    def stop(self, timeout=10):
        self.stopped += 1

    def remove(self, **kw):
        self.removed += 1
