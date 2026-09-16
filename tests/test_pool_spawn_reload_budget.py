"""One broken pool slot may not take the whole fleet's database down.

2026-09-15, the mechanism this file pins. Slot 78 became permanently
unspawnable at 14:45:16Z. The reconciler retried it every ~30-40 s for 25+
minutes, and every retry ran `sudo create_tenant_db` — a helper that rewrites
the SHARED `/etc/pgbouncer/userlist.txt` and reloads the SHARED pooler. Four
consecutive pgbouncer instances died with `got SIGHUP, re-reading config` as
their last line (15:00:48, 15:04:05, 15:05:09 after 1.8 s of life, 15:08:13);
the unit is `Restart=no` and the cron watchdog had spent its budget, so every
tenant database on the host was unreachable 15:08:13 -> 15:19:33Z. Eleven
minutes and twenty seconds, ended by a human.

The four properties that make that loop impossible, each asserted below:

  1. a slot that keeps failing is BACKED OFF and then QUARANTINED — and the
     quarantine is loud exactly once, and durable across a bridge restart;
  2. a quarantined slot is not handed out again, and the pool refills from
     the NEXT slot rather than stalling;
  3. cheap slot-local preconditions (image present, pooler answering) run
     BEFORE the fleet-global side effect, so a fault that has nothing to do
     with credentials never costs the pooler a reload;
  4. a process-global token bucket bounds pooler reloads even when many slots
     fail at once — and a refusal DEFERS rather than scoring a slot.

Neither a dead pooler nor a spent budget may be recorded as a slot fault:
quarantining healthy slots during an outage would empty the pool exactly when
signups need it.

Run:
    cd backend && RUN_MODE=platform PYTHONPATH=. \
        pytest tests/test_pool_spawn_reload_budget.py -q
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import pathlib
import sys
import types

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
IMAGE = "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"


@pytest.fixture(scope="module")
def loop():
    """ONE loop for the module: `_spawn_lock` and `_docker_sem` are created at
    import and bind to the first loop that awaits them, so a per-test
    `asyncio.run` would raise 'bound to a different event loop'."""
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


@pytest.fixture()
def mod(tmp_path, monkeypatch):
    """A fresh pool_addon with its registry in tmp_path and every host tool
    stubbed. Loaded under its own module name so it cannot collide with the
    other bridge harnesses in a shared pytest process."""
    fake_main = types.ModuleType("main")
    fake_main.CADDY_ADMIN = "http://127.0.0.1:1"
    for hook in ("_caddy_add_tenant_route", "_caddy_remove_tenant_route",
                 "_caddy_swap_upstream"):
        setattr(fake_main, hook, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "main", fake_main)

    name = "pool_addon_spawn_uut"
    spec = importlib.util.spec_from_file_location(name, BRIDGE)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)

    pool = tmp_path / "_pool"
    pool.mkdir()
    m.POOL_DIR = pool
    m.MEMBERS_FILE = pool / "members.json"
    m.STATE_FILE = pool / "state.json"
    m.BINDS_DIR = pool / "binds"
    m.WORKSPACE_HOST_BASE = tmp_path / "agents"

    # Host tools that do not exist here. Each test overrides what it is about.
    m._container_name_exists = lambda name_: False
    m._next_pool_port = lambda: 9500
    m._pooler_reachable = lambda: True
    m._image_present = lambda tag: True
    m._create_pool_db_via_helper = lambda slot: "pw_" + "z" * 30
    m._wipe_pool_db = lambda slot, pw: None
    m._spawn_pool_container = lambda slot, tag, port, pw: "c" * 12

    async def _healthy(port, timeout_s=None):
        return True
    m._wait_for_lobby_health = _healthy
    # A brand-new bucket per test; the module-level one is shared state.
    m._reload_budget = m._ReloadBudget(m.RELOAD_BUDGET_MAX, m.RELOAD_BUDGET_WINDOW_S)
    return m


def _spawn(loop, m) -> bool:
    return loop.run_until_complete(m._spawn_one_pool_member(IMAGE))


def _rows(m, state=None):
    return [r for r in m._load_members() if state is None or r.get("state") == state]


def _force_retry(m, slot="01"):
    """Stand in for the passage of the backoff.

    A slot is parked on its FIRST failure, so the pool refills from the next
    slot instead of hammering the broken one — which means a second attempt on
    the SAME slot only happens once its backoff has expired. Rewinding the
    clock is the honest way to reach the third consecutive failure here.
    """
    led = m._spawn_ledger()
    if slot in led:
        led[slot]["blocked_until"] = 0
        m._state_set("spawn_failures", led)
    for r in m._load_members():
        if r["slot"] == slot and r.get("quarantine_reason") == "spawn_failure":
            m._update_member(slot, spawn_blocked_until=0)


# ── anti-vacuity ─────────────────────────────────────────────────


def test_the_harness_drives_the_real_module(mod, loop):
    """Every assertion below reads this module's state. If the load silently
    produced a stub they would pass while proving nothing."""
    assert hasattr(mod, "_spawn_one_pool_member")
    assert mod.SPAWN_FAIL_QUARANTINE_AFTER >= 2
    assert mod.RELOAD_BUDGET_MAX >= 1
    assert _spawn(loop, mod) is True, "the happy path must still spawn"
    assert len(_rows(mod, mod.STATE_GENERIC)) == 1


# ── 1. backoff + quarantine ──────────────────────────────────────


def test_a_slot_that_keeps_failing_is_quarantined_and_alerts_once(mod, loop, caplog):
    mod._spawn_pool_container = lambda *a: (_ for _ in ()).throw(RuntimeError("docker run failed"))
    caplog.set_level(logging.INFO, logger="bridge.pool")

    for i in range(mod.SPAWN_FAIL_QUARANTINE_AFTER):
        assert _spawn(loop, mod) is False
        if i < mod.SPAWN_FAIL_QUARANTINE_AFTER - 1:
            _force_retry(mod, "01")   # the backoff expires; the quarantine does not

    ledger = mod._spawn_ledger()
    assert "01" in ledger, "the failure ledger is not recording the slot"
    assert ledger["01"]["consecutive"] == mod.SPAWN_FAIL_QUARANTINE_AFTER
    assert ledger["01"]["quarantined"] is True

    held = [r for r in _rows(mod, mod.STATE_QUARANTINED)
            if r.get("quarantine_reason") == "spawn_failure"]
    assert [r["slot"] for r in held] == ["01"]

    # Keep failing past the quarantine: the hold expires (6 h by default, and
    # it MUST expire — capacity that shrinks for good with nobody watching is
    # its own outage), the slot fails again, and that must NOT re-alert.
    for _ in range(2):
        _force_retry(mod, "01")
        assert _spawn(loop, mod) is False
    assert mod._spawn_ledger()["01"]["consecutive"] == mod.SPAWN_FAIL_QUARANTINE_AFTER + 2

    alerts = [r for r in caplog.records
              if r.levelno >= logging.ERROR and "QUARANTINED after" in r.getMessage()]
    assert len(alerts) == 1, (
        f"{len(alerts)} alerts across {mod.SPAWN_FAIL_QUARANTINE_AFTER + 2} "
        "failures — a quarantine must be loud exactly ONCE per episode, or it "
        "becomes another log line nobody reads"
    )


def test_the_backoff_grows_and_is_persisted_outside_members_json(mod, loop):
    """A bridge restart rewrites members.json from the host; the ledger lives
    in state.json precisely so a restart cannot reset the retry loop."""
    mod._spawn_pool_container = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    holds = []
    for _ in range(2):
        _spawn(loop, mod)
        e = mod._spawn_ledger()["01"]
        holds.append(e["blocked_until"] - e["last_at"])
        _force_retry(mod, "01")
    assert holds[1] > holds[0], f"backoff did not grow: {holds}"
    assert mod._state_get("spawn_failures", {}).get("01"), (
        "the ledger must be in state.json, not only in the registry row"
    )


def test_a_successful_spawn_clears_the_episode(mod, loop):
    mod._spawn_pool_container = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    _spawn(loop, mod)
    assert "01" in mod._spawn_ledger()
    # Release the hold the way time would, then let the slot succeed.
    _force_retry(mod, "01")
    mod._spawn_pool_container = lambda slot, tag, port, pw: "c" * 12
    assert _spawn(loop, mod) is True
    assert mod._spawn_ledger() == {}, "a good spawn must reset the counter AND the alert latch"


def test_expiry_releases_a_spawn_hold_but_never_an_operator_quarantine(mod, loop):
    """An operator quarantine (an ownership conflict) may hold the only copy
    of a user's messages. It carries no `spawn_failure` reason and must
    survive every sweep this module does."""
    mod._add_member({"slot": "07", "port": 9507, "state": mod.STATE_QUARANTINED,
                     "quarantined_prefix": "abcd1234",
                     "last_error": "ownership_conflict_quarantine"})
    mod._add_member({"slot": "08", "port": 9508, "state": mod.STATE_QUARANTINED,
                     "quarantine_reason": "spawn_failure",
                     "spawn_blocked_until": 0})
    released = mod._expire_spawn_holds()
    assert released == 1
    slots = {r["slot"] for r in _rows(mod)}
    assert "07" in slots and "08" not in slots


# ── 2. the pool refills from the next slot ───────────────────────


def test_a_quarantined_slot_is_skipped_and_the_next_one_is_used(mod, loop):
    calls: list[str] = []

    def _docker(slot, tag, port, pw):
        calls.append(slot)
        if slot == "01":
            raise RuntimeError("slot 01 is permanently broken")
        return "c" * 12

    mod._spawn_pool_container = _docker
    for i in range(mod.SPAWN_FAIL_QUARANTINE_AFTER):
        _spawn(loop, mod)
        if i < mod.SPAWN_FAIL_QUARANTINE_AFTER - 1:
            _force_retry(mod, "01")   # the backoff expires; the quarantine does not
    assert mod._spawn_ledger()["01"]["quarantined"] is True
    assert _spawn(loop, mod) is True, "a quarantined slot must not stall the pool"
    generic = _rows(mod, mod.STATE_GENERIC)
    assert [r["slot"] for r in generic] == ["02"]
    assert calls[-1] == "02"


# ── 3. preconditions before the fleet-global side effect ─────────


def test_a_missing_image_never_reaches_the_pooler(mod, loop):
    ran: list[str] = []
    mod._create_pool_db_via_helper = lambda slot: ran.append(slot) or "pw_" + "z" * 30
    mod._image_present = lambda tag: False

    assert _spawn(loop, mod) is False
    assert ran == [], (
        "create_tenant_db rewrites the shared pgbouncer auth file and reloads "
        "the pooler — a missing image must not cost a reload"
    )
    assert mod._spawn_ledger()["01"]["last_error"].startswith("image_missing")


def test_a_dead_pooler_defers_and_is_never_scored_as_a_slot_fault(mod, loop):
    ran: list[str] = []
    mod._create_pool_db_via_helper = lambda slot: ran.append(slot) or "pw_" + "z" * 30
    mod._pooler_reachable = lambda: False

    assert _spawn(loop, mod) is False
    assert ran == [], "the pooler must not be signalled while it is down"
    assert mod._spawn_ledger() == {}, (
        "an outage is infrastructure, not a broken slot — scoring it would "
        "quarantine every healthy slot exactly when signups need them"
    )
    assert _rows(mod) == [], "no registry row may be left behind by a defer"


def _fresh(name: str):
    """A second, unstubbed copy of the module — the `mod` fixture replaces the
    very helpers these two tests are about."""
    spec = importlib.util.spec_from_file_location(name, BRIDGE)
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[name] = fresh
    spec.loader.exec_module(fresh)
    return fresh


def test_wipe_probes_the_pooler_before_running_psql(monkeypatch):
    """`_wipe_pool_db` connects THROUGH the pooler. With pgbouncer dead every
    wipe failed, every spawn was scored as a slot fault, and the retry rewrote
    the dead pooler's auth file again — the loop that produced the outage."""
    import subprocess as _sp
    real = _fresh("pool_addon_wipe_uut")
    calls: list = []
    monkeypatch.setattr(_sp, "run", lambda *a, **k: calls.append(a) or None)
    real._pooler_reachable = lambda: False
    with pytest.raises(real.PoolerUnavailable):
        real._wipe_pool_db("01", "pw")
    assert calls == [], "psql must not be attempted against a dead pooler"


def test_a_wipe_timeout_is_infrastructure_not_a_slot_fault(monkeypatch):
    """2026-09-16 03:09-05:26Z: a 30 s wipe budget under a host at load 40-60
    scored every generic spawn `db_wipe:TimeoutExpired`, three strikes each,
    and the ledger held ten spare slots for six hours. A slow TRUNCATE is the
    host's fault, never the slot's — it must come out as PoolerUnavailable
    (deferred, retried next tick, no strike), exactly like a dead pooler."""
    import subprocess as _sp
    real = _fresh("pool_addon_wipe_timeout_uut")
    real._pooler_reachable = lambda: True

    def _hang(*a, **k):
        raise _sp.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))

    monkeypatch.setattr(_sp, "run", _hang)
    with pytest.raises(real.PoolerUnavailable) as ei:
        real._wipe_pool_db("07", "pw")
    assert "slot 07" in str(ei.value) and "not the slot" in str(ei.value)


def test_the_wipe_budget_is_generous_and_tunable(monkeypatch):
    """A TRUNCATE of ~60 tables through the pooler on a loaded host takes
    tens of seconds; 30 s scored it as broken. The default must leave room,
    and an operator with a slower Postgres must be able to widen it without
    a code change."""
    import subprocess as _sp
    real = _fresh("pool_addon_wipe_budget_uut")
    assert real.WIPE_TIMEOUT_S >= 120
    monkeypatch.setenv("BRIDGE_POOL_WIPE_TIMEOUT_S", "7")
    tuned = _fresh("pool_addon_wipe_budget_tuned_uut")
    assert tuned.WIPE_TIMEOUT_S == 7
    seen: dict = {}

    class _R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(_sp, "run", lambda *a, **k: seen.update(k) or _R())
    tuned._pooler_reachable = lambda: True
    tuned._wipe_pool_db("07", "pw")
    assert seen.get("timeout") == 7, "the env budget is not what psql is given"


def test_a_wipe_timeout_never_reaches_the_spawn_ledger():
    """The classification above is only worth something if the spawn path
    keeps its PoolerUnavailable branch AHEAD of the generic one that strikes
    the ledger. Source pin on that order."""
    src = BRIDGE.read_text()
    i_wipe = src.index("await _offload(_wipe_pool_db, slot, db_pw)")
    i_defer = src.index("except PoolerUnavailable as e:", i_wipe)
    i_strike = src.index('_fail_spawn(slot, port, f"db_wipe:{type(e).__name__}")', i_wipe)
    assert i_wipe < i_defer < i_strike, (
        "a wipe timeout must be caught as PoolerUnavailable before the generic "
        "branch scores it against the slot"
    )
    assert "except subprocess.TimeoutExpired" in src[src.index("def _wipe_pool_db"):src.index("def _wipe_pool_db") + 4000]


# ── 4. the reload token bucket ───────────────────────────────────


def test_pooler_reloads_are_capped_per_window(mod, monkeypatch):
    import subprocess as _sp
    ran: list = []

    class _R:
        returncode = 0
        stdout = "pw_" + "z" * 30
        stderr = ""

    monkeypatch.setattr(_sp, "run", lambda *a, **k: ran.append(a) or _R())
    # Reach the real helper (the fixture stubbed the module attribute).
    real = _fresh("pool_addon_budget_uut")
    real._pooler_reachable = lambda: True
    real._reload_budget = real._ReloadBudget(3, 300)

    for _ in range(3):
        assert real._create_pool_db_via_helper("01").startswith("pw_")
    with pytest.raises(real.ReloadBudgetExceeded):
        real._create_pool_db_via_helper("01")
    assert len(ran) == 3, (
        f"{len(ran)} calls to create_tenant_db against a budget of 3 — the "
        "bucket is not bounding the fleet-global side effect"
    )
    snap = real._reload_budget.snapshot()
    assert snap["used"] == 3 and snap["refused_total"] == 1


def test_a_refused_reload_is_a_defer_not_a_slot_fault(mod, loop):
    mod._create_pool_db_via_helper = lambda slot: (_ for _ in ()).throw(
        mod.ReloadBudgetExceeded("reload budget spent (15/15)")
    )
    assert _spawn(loop, mod) is False
    assert mod._spawn_ledger() == {}
    assert _rows(mod) == []


# ── the host self-check ──────────────────────────────────────────


def test_the_dropin_selfcheck_is_silent_off_a_pooler_host(mod, caplog):
    mod.PGBOUNCER_CONF_DIR = pathlib.Path("/nonexistent-pgbouncer")
    caplog.set_level(logging.INFO, logger="bridge.pool")
    out = mod.pgbouncer_dropin_selfcheck()
    assert out["checked"] is False
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_the_dropin_selfcheck_alerts_when_restart_always_is_missing(mod, tmp_path, caplog):
    conf = tmp_path / "pgbouncer"
    conf.mkdir()
    dropin = tmp_path / "dropin"
    dropin.mkdir()
    (dropin / "other.conf").write_text("[Service]\nRestartSec=2\n")
    mod.PGBOUNCER_CONF_DIR = conf
    mod.PGBOUNCER_DROPIN_DIR = dropin
    caplog.set_level(logging.INFO, logger="bridge.pool")

    out = mod.pgbouncer_dropin_selfcheck()
    assert out["ok"] is False
    assert [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "Restart=no is why a 1-second process death became an 11-minute fleet "
        "outage; its absence has to be loud"
    )

    (dropin / "toup-restart.conf").write_text("[Service]\nRestart=always\n")
    assert mod.pgbouncer_dropin_selfcheck()["ok"] is True
