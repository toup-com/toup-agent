"""The bridge answers a health probe WHILE it is running a docker command.

The dynamic half of the guard in `test_bridge_pool_addon_offload.py`. That
file proves no blocking call is written on the event loop; this one proves
the consequence — that a cheap GET stays fast while the pool addon does its
slowest work — by running the real router under a real uvicorn with a real
event loop, and measuring.

Nothing here touches production or a docker daemon. It builds a throwaway
state dir, puts a fake `docker` / `psql` / `sudo` on PATH (a Python script
that sleeps for a configured number of seconds and prints plausible output),
stubs the `main` module `pool_addon` imports its Caddy helpers from, and
serves the router on a loopback port.

Measured with this harness on 2026-09-06, sampling `/v1/health` every 50 ms
with the platform's own 5s probe timeout
(`app/services/bridge_supervisor.py`):

    scenario                       before p95   before max   after p95
    reconciler tick, 10s docker cp     5.00 s       5.00 s     0.005 s
    claim in its bind phase            4.02 s       5.00 s     0.003 s
    upgrade-assigned                   3.05 s       4.11 s     0.003 s
    snapshot loop over 20 members      5.00 s       5.00 s     0.003 s

"before" completed 11-12 probes across a 10s window and timed out on two of
them; "after" completed ~200 and timed out on none. The work itself takes
the same wall time either way — this buys availability, not throughput.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_pool_addon_event_loop.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pathlib
import socket
import statistics
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

uvicorn = pytest.importorskip("uvicorn")

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"

# The platform probes /v1/health every 60s with a 5s timeout and declares the
# bridge unreachable after 3 consecutive failures
# (app/services/bridge_supervisor.py:36-38). A probe that takes longer than
# this is, to the platform, an outage.
PROBE_TIMEOUT_S = 5.0

# How long the fake docker command blocks. Long enough that the pre-fix code
# cannot possibly answer a probe inside PROBE_TIMEOUT_S, short enough to keep
# the test quick.
BLOCK_S = 6.0

# Deliberately loose: the point is a ~1000x gap (0.005s vs 5.0s), so a CI
# runner under load has three orders of magnitude of slack before this
# flakes, while the pre-fix behaviour misses by 5x.
MAX_P95_S = 1.0
MIN_PROBES = 15


FAKE_BIN = r'''#!/usr/bin/env python3
import json, os, sys, time
argv = sys.argv[1:]
prog = os.path.basename(sys.argv[0])
if prog == "sudo":
    while argv and argv[0].startswith("-"):
        argv.pop(0)
    prog = os.path.basename(argv[0]) if argv else "sudo"
    argv = argv[1:]
sub = argv[0] if argv else ""
delays = {}
try:
    delays = json.load(open(os.environ["FAKE_DELAYS"]))
except Exception:
    pass
d = delays.get(f"{prog} {sub}".strip(), delays.get(prog, 0.0))
log = os.environ.get("FAKE_LOG")
if log:
    open(log, "a").write(f"{time.time():.4f}\t{prog} {sub}\n")
if d:
    time.sleep(float(d))
if prog == "docker":
    if sub == "run":
        print("f" * 64)
    elif sub == "inspect":
        j = " ".join(argv)
        print("running" if "State.Status" in j
              else json.dumps(["PATH=/usr/bin", "DATABASE_URL=postgresql://x"]))
elif prog == "create_tenant_db":
    print("pw_" + "z" * 30)
sys.exit(0)
'''


class _FakeAgent(BaseHTTPRequestHandler):
    """Stands in for a pool container's /agent/health and /api/admin/bind."""
    protocol_version = "HTTP/1.1"

    def _reply(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply(b'{"ready": true}')

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply(b'{"ok": true}')

    def log_message(self, *a):
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    """A live bridge: fake host tools, stub `main`, real router, real uvicorn."""
    root = tmp_path_factory.mktemp("bridge")

    binp = root / "bin"
    binp.mkdir()
    for name in ("docker", "psql", "sudo"):
        f = binp / name
        f.write_text(FAKE_BIN)
        f.chmod(0o755)
    delays = root / "delays.json"
    delays.write_text("{}")
    calls = root / "calls.log"
    calls.write_text("")
    os.environ["PATH"] = f"{binp}{os.pathsep}{os.environ['PATH']}"
    os.environ["FAKE_DELAYS"] = str(delays)
    os.environ["FAKE_LOG"] = str(calls)
    # Keep the sizer from wanting a spawn wave we did not ask for.
    os.environ["BRIDGE_POOL_MIN_K"] = "0"
    os.environ["BRIDGE_POOL_BUFFER"] = "0"

    # pool_addon does `from main import _caddy_add_tenant_route` inside its
    # functions; main.py is not a repo file (it is embedded in
    # new-vps/08-provisioning-bridge.sh), so stub it.
    fake_main = types.ModuleType("main")
    fake_main.CADDY_ADMIN = "http://127.0.0.1:1"          # unreachable on purpose
    for hook in ("_caddy_add_tenant_route", "_caddy_remove_tenant_route",
                 "_caddy_swap_upstream"):
        setattr(fake_main, hook, lambda *a, **k: None)
    sys.modules["main"] = fake_main

    spec = importlib.util.spec_from_file_location("pool_addon_uut", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pool_addon_uut"] = mod
    spec.loader.exec_module(mod)

    pool = root / "_pool"
    pool.mkdir()
    mod.POOL_DIR = pool
    mod.MEMBERS_FILE = pool / "members.json"
    mod.STATE_FILE = pool / "state.json"
    mod.BINDS_DIR = pool / "binds"
    token = root / "token"
    token.write_text("harness")
    mod.ADMIN_TOKEN_FILE = token
    mod._admin_token_cache = None
    ws = root / "agents"
    ws.mkdir()
    mod.WORKSPACE_HOST_BASE = ws

    agent_port = _free_port()
    threading.Thread(
        target=ThreadingHTTPServer(("127.0.0.1", agent_port), _FakeAgent).serve_forever,
        daemon=True,
    ).start()

    from fastapi import FastAPI
    app = FastAPI()
    loop_box: dict = {}

    # Byte-identical to new-vps/08-provisioning-bridge.sh:482-484 — the exact
    # route bridge_supervisor probes. A static dict: if THIS is slow, only the
    # event loop can be to blame.
    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.on_event("startup")
    async def _capture():
        loop_box["loop"] = asyncio.get_running_loop()

    # include_router, NOT attach_pool_routes: the reconciler and snapshot
    # loops are driven explicitly below so each measurement is deterministic.
    app.include_router(mod.router)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error", access_log=False))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2).read()
            break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("harness bridge never came up")

    yield types.SimpleNamespace(
        mod=mod, port=port, agent_port=agent_port, loop=loop_box["loop"],
        delays=delays, calls=calls, ws=ws,
    )
    server.should_exit = True


# ── helpers ──────────────────────────────────────────────────────


def _set_delays(bridge, **kw):
    bridge.delays.write_text(json.dumps({k.replace("_", " "): v for k, v in kw.items()}))


def _member(slot, port, state, **kw):
    m = {"slot": slot, "port": port, "container_name": f"toup-agent-pool-{slot}",
         "db_name": f"toup_agent_feed00{slot}", "state": state,
         "image_tag": "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa",
         "created_at": int(time.time()), "state_changed_at": int(time.time())}
    m.update(kw)
    return m


def _workspace(bridge, prefix):
    d = bridge.ws / prefix / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    (d / "f.txt").write_text("x")


class _Probe(threading.Thread):
    """Hammers a cheap GET the way bridge_supervisor does, and records it."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.stop = threading.Event()
        self.latencies: list[float] = []
        self.timeouts = 0

    def run(self):
        while not self.stop.is_set():
            t0 = time.perf_counter()
            try:
                urllib.request.urlopen(self.url, timeout=PROBE_TIMEOUT_S).read()
                self.latencies.append(time.perf_counter() - t0)
            except Exception:
                self.timeouts += 1
                self.latencies.append(PROBE_TIMEOUT_S)
            self.stop.wait(0.05)

    def p95(self) -> float:
        s = sorted(self.latencies)
        return s[max(0, int(len(s) * 0.95) - 1)] if s else 0.0


def _measure(bridge, work, url_path="/v1/health"):
    """Run `work()` on the server; probe `url_path` throughout."""
    probe = _Probe(f"http://127.0.0.1:{bridge.port}{url_path}")
    probe.start()
    time.sleep(0.3)
    t0 = time.time()
    result = work()
    elapsed = time.time() - t0
    time.sleep(0.2)
    probe.stop.set()
    probe.join(timeout=10)
    return probe, elapsed, result


def _assert_stayed_responsive(probe, elapsed, what):
    assert elapsed >= BLOCK_S * 0.8, (
        f"{what} finished in {elapsed:.1f}s — the fake docker delay did not "
        f"apply, so this measured nothing"
    )
    assert probe.timeouts == 0, (
        f"{len(probe.latencies)} probes of /v1/health during {what}: "
        f"{probe.timeouts} exceeded the platform's {PROBE_TIMEOUT_S}s budget. "
        f"The bridge's event loop is blocked by docker work again."
    )
    assert len(probe.latencies) >= MIN_PROBES, (
        f"only {len(probe.latencies)} probes completed during {elapsed:.1f}s of "
        f"{what} — the loop was not accepting connections"
    )
    assert probe.p95() < MAX_P95_S, (
        f"/v1/health p95 was {probe.p95():.2f}s during {what} "
        f"(median {statistics.median(probe.latencies):.3f}s)"
    )


# ── the three incident scenarios ─────────────────────────────────


def test_health_stays_fast_during_a_reconciler_tick(bridge):
    """(a) The reconciler runs drains and destroys INLINE every 30s.

    On 2026-09-06 that was enough to time out the platform's health probe
    more than a third of the time, all day.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.1, docker_cp=BLOCK_S)
    _workspace(bridge, "aaaa1111")
    m._save_members([_member("90", bridge.agent_port, m.STATE_DEAD,
                             assigned_prefix="aaaa1111", assigned_user_id="u-a")])
    m._save_state({"current_image_tag": "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"})

    probe, elapsed, summary = _measure(
        bridge,
        lambda: asyncio.run_coroutine_threadsafe(
            m.reconciler_tick(), bridge.loop).result(timeout=120),
    )
    assert summary["drained"] == 1, f"the tick did no work: {summary}"
    _assert_stayed_responsive(probe, elapsed, "a reconciler tick")


def test_health_stays_fast_during_a_claim(bridge):
    """(b) A claim runs `_restore_workspace_for_pool_bind` (docker cp) and
    main.py's `_caddy_add_tenant_route` before it can answer."""
    m = bridge.mod
    _set_delays(bridge, docker=0.2, docker_cp=BLOCK_S)
    _workspace(bridge, "bbbb2222")
    m._save_members([_member("91", bridge.agent_port, m.STATE_GENERIC)])

    def claim():
        req = urllib.request.Request(
            f"http://127.0.0.1:{bridge.port}/v1/pool/claim",
            data=json.dumps({"user_id": "u-b", "prefix": "bbbb2222",
                             "agent_api_key": "k-b"}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    probe, elapsed, body = _measure(bridge, claim)
    assert body["ok"] is True, body
    assert [x for x in m._load_members()
            if x["slot"] == "91"][0]["state"] == m.STATE_ASSIGNED
    _assert_stayed_responsive(probe, elapsed, "a claim")


def test_health_stays_fast_during_the_workspace_snapshot_loop(bridge):
    """(c) Every 300s the snapshot loop `docker cp`s EVERY assigned member's
    workspace to the host — serially, with no yield. At ~74 assigned members
    on a host where each cp takes seconds, that is minutes of freeze."""
    m = bridge.mod
    _set_delays(bridge, docker=0.1, docker_cp=BLOCK_S / 6)
    members = []
    for i in range(6):
        pfx = f"snap{i:04d}"
        _workspace(bridge, pfx)
        members.append(_member(f"{60 + i}", bridge.agent_port, m.STATE_ASSIGNED,
                               assigned_prefix=pfx, assigned_user_id=f"u-{i}"))
    m._save_members(members)

    async def _tick():
        # Fall back to the pre-extraction inline body so this measures the OLD
        # code too — an AttributeError would tell us the refactor is missing,
        # not what it cost.
        if hasattr(m, "_snapshot_tick"):
            return await m._snapshot_tick()
        n = 0
        for row in m._load_members():
            if row.get("state") in (m.STATE_ASSIGNED, m.STATE_DEAD):
                m._save_workspace_for_pool_release(row["assigned_prefix"],
                                                   row["container_name"])
                n += 1
        return n

    probe, elapsed, saved = _measure(
        bridge,
        lambda: asyncio.run_coroutine_threadsafe(_tick(), bridge.loop).result(timeout=120),
    )
    assert saved == 6, f"snapshot tick saved {saved}, expected 6"
    _assert_stayed_responsive(probe, elapsed, "the workspace snapshot loop")


# ── the interactive lane must not starve ─────────────────────────


def test_a_claim_does_not_queue_behind_background_docker_work(bridge):
    """Why there are TWO semaphores and not one.

    Bounding docker concurrency is necessary — an unbounded fan-out on an
    already-saturated host is its own outage — but a single shared bound
    lets the reconciler's backlog delay the one request a human is waiting
    on, which is the incident all over again with a different mechanism.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.2, docker_cp=2.0)
    _workspace(bridge, "cccc3333")
    m._save_members([_member("92", bridge.agent_port, m.STATE_GENERIC)])

    assert hasattr(m, "DOCKER_CONCURRENCY") and hasattr(m, "_offload_ux"), (
        "the two-lane offload is gone — every docker call is back on the "
        "event loop and every other test in this file explains what that costs"
    )

    # Saturate the BACKGROUND lane with far more work than it has permits.
    async def _flood():
        await asyncio.gather(*[m._offload(time.sleep, 4.0) for _ in range(8)])

    fut = asyncio.run_coroutine_threadsafe(_flood(), bridge.loop)
    time.sleep(0.4)

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{bridge.port}/v1/pool/claim",
        data=json.dumps({"user_id": "u-c", "prefix": "cccc3333",
                         "agent_api_key": "k-c"}).encode(),
        headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req, timeout=120).read())
    claim_s = time.perf_counter() - t0
    backlog_s = 8 * 4.0 / m.DOCKER_CONCURRENCY

    assert body["ok"] is True, body
    assert claim_s < backlog_s / 2, (
        f"the claim took {claim_s:.1f}s while the background lane held "
        f"~{backlog_s:.0f}s of queued work — it is sharing a semaphore with "
        f"background docker ops instead of using its own lane"
    )
    fut.result(timeout=120)


# ── work must outlive the caller that gave up on it ──────────────


def test_a_bind_completes_even_if_the_client_disconnects(bridge):
    """The incident's exact shape: the platform's httpx client gave up at its
    30s budget and the bridge finished the bind 2s later.

    That only stays true if the offload's awaits are not cancellation points
    that abort a half-done claim. If a disconnect ever DID cancel post_claim,
    the slot would be left ASSIGNING — reaped as stale five minutes later,
    with the user's container bound to nothing.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.1, docker_cp=4.0)
    _workspace(bridge, "dddd4444")
    m._save_members([_member("93", bridge.agent_port, m.STATE_GENERIC)])

    payload = json.dumps({"user_id": "u-d", "prefix": "dddd4444",
                          "agent_api_key": "k-d"}).encode()
    s = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    s.sendall(b"POST /v1/pool/claim HTTP/1.1\r\nHost: h\r\n"
              b"Content-Type: application/json\r\n"
              + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    time.sleep(0.8)
    s.close()                                    # the caller gives up here

    deadline = time.time() + 30
    state = None
    while time.time() < deadline:
        rows = [x for x in m._load_members() if x["slot"] == "93"]
        state = rows[0]["state"] if rows else "GONE"
        if state == m.STATE_ASSIGNED:
            break
        time.sleep(0.25)

    assert state == m.STATE_ASSIGNED, (
        f"slot 93 ended {state}, not ASSIGNED — the client's disconnect "
        f"cancelled the claim mid-bind, which strands the user"
    )
