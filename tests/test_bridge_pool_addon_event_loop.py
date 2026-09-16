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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
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
        missing = [x for x in os.environ.get("FAKE_MISSING", "").split(",") if x]
        if any(mname in argv for mname in missing):
            sys.stderr.write("Error: No such object\n")
            sys.exit(1)
        if ".Config.Image" in j:
            # the /v1/tenants/{prefix}/stats probe: id, image, env
            print(os.environ.get("FAKE_CID", "c" * 64))
            print("ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa")
            print(json.dumps(["PATH=/usr/bin",
                              "DATABASE_URL=postgresql://u:p@host.docker.internal:6432/d"]))
        else:
            print("running" if "State.Status" in j
                  else json.dumps(["PATH=/usr/bin", "DATABASE_URL=postgresql://x"]))
elif prog == "create_tenant_db":
    print("pw_" + "z" * 30)
sys.exit(0)
'''


class _FakeAgent(BaseHTTPRequestHandler):
    """Stands in for a pool container's /agent/health and /api/admin/bind."""
    protocol_version = "HTTP/1.1"
    hits = 0
    _hit_lock = threading.Lock()

    @classmethod
    def reset_hits(cls) -> int:
        with cls._hit_lock:
            n, cls.hits = cls.hits, 0
        return n

    def _count(self):
        with _FakeAgent._hit_lock:
            _FakeAgent.hits += 1

    def _reply(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._count()
        self._reply(b'{"ready": true}')

    def do_POST(self):
        self._count()
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

    # A spawn now TCP-probes the pooler before it runs `create_tenant_db`
    # (R46: that helper rewrites the shared pgbouncer auth file and reloads
    # the pooler; signalling a dead one is how the fleet went down on
    # 2026-09-15). There is no pgbouncer in this harness, so point the probe
    # at the fake agent's listener — a real connect, no protocol assumed.
    mod.POOLER_PORT = agent_port

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
    app.include_router(mod.tenants_router)

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


# ── the reconciler's own CPU on the loop thread ───────────────────
#
# Offloading docker was necessary and not sufficient. The tick still runs
# `asyncio.gather(_health_check(...))` over every active member and
# `asyncio.gather(_authenticated_probe(...))` over every bound one, and each
# of those helpers opened its OWN `httpx.AsyncClient`. Constructing one costs
# a full `ssl.create_default_context()` — MEASURED on the bridge host's own
# interpreter (/opt/toup-bridge/venv, Python 3.12.3, httpx 0.28.1) on
# 2026-09-07:
#
#     httpx.AsyncClient(timeout=3.0)                 40.3 ms CPU each
#     httpx.AsyncClient(timeout=3.0, verify=False)    0.3 ms CPU each
#     ssl.create_default_context()                   37.7 ms CPU each
#
# — and every one of these probes is `http://127.0.0.1:<port>`, so the
# context it builds is never used. `asyncio.gather` schedules all the first
# steps in ONE loop iteration, so the loop runs all 84 constructions
# back-to-back before it services a single socket.
#
# Measured on production the same day (84 members: 10 GENERIC + 74 ASSIGNED,
# host load ~17), 129 samples of the loopback static `/v1/health` over 149 s
# against /proc/<bridge>/task/<bridge>/stat:
#
#     p50 7 ms · p95 2.89 s · max 4.60 s · 9 % over 1 s
#     main-thread CPU per reconciler tick: 5.07 / 7.79 / 8.62 / 9.15 s
#     every slow sample coincides with a 2.8-4.0 s main-thread CPU burst
#
# 84 + 74 = 158 constructions x 40.3 ms = 6.37 s, which is the bulk of it.
# The registry is not the cause and was ruled out on the same host:
# members.json is 38 539 bytes, json.loads 0.40 ms, json.dumps 1.60 ms.
HOST_CLIENT_BUILD_S = 0.040  # the 40.3 ms measured above, on the real bridge

FLEET_GENERIC = 10
FLEET_ASSIGNED = 74


def _fleet(bridge, mod):
    """The production registry shape: 10 GENERIC + 74 ASSIGNED with binds."""
    members = []
    for i in range(FLEET_GENERIC):
        members.append(_member(f"g{i:02d}", bridge.agent_port, mod.STATE_GENERIC))
    for i in range(FLEET_ASSIGNED):
        slot = f"a{i:02d}"
        members.append(_member(slot, bridge.agent_port, mod.STATE_ASSIGNED,
                               assigned_prefix=f"pfx{i:05d}",
                               assigned_user_id=f"u-{i}"))
        mod._persist_bind(slot, {"agent_api_key": f"k-{i}", "user_id": f"u-{i}"})
    mod._save_members(members)
    mod._save_state({"current_image_tag": "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"})
    return members


def _burn(seconds: float) -> None:
    """Hold the GIL for `seconds` — what building an SSL context does."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_a_tick_builds_one_http_client_not_one_per_member(bridge, monkeypatch):
    """The deterministic half: COUNT the clients a single tick constructs.

    84 active members + 74 authenticated probes is 158 `httpx.AsyncClient`
    constructions per tick on the pre-fix file, plus one synchronous
    `httpx.get` per assigned prefix inside the Caddy route re-assert. On the
    bridge host that is ~6.4 s of event-loop CPU and ~3 s of GIL-held CPU in
    a worker thread, every 30 s, for probes that never use TLS.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.01)
    _fleet(bridge, m)
    _FakeAgent.reset_hits()

    built = {"async": 0, "sync_get": 0}
    real_async = httpx.AsyncClient
    real_get = httpx.get

    class _Counting(real_async):
        def __init__(self, *a, **kw):
            built["async"] += 1
            super().__init__(*a, **kw)

    def _counted_get(*a, **kw):
        built["sync_get"] += 1
        return real_get(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    monkeypatch.setattr(httpx, "get", _counted_get)

    summary = asyncio.run_coroutine_threadsafe(
        m.reconciler_tick(), bridge.loop).result(timeout=180)

    probes = _FakeAgent.reset_hits()
    assert probes >= FLEET_GENERIC + 2 * FLEET_ASSIGNED - 10, (
        f"the tick only reached the fake agent {probes} times over "
        f"{FLEET_GENERIC + FLEET_ASSIGNED} members — it did not run the "
        f"health + authenticated passes, so this measures nothing: {summary}"
    )
    assert built["async"] <= 2, (
        f"the tick built {built['async']} httpx.AsyncClient objects for "
        f"{probes} loopback probes. Each is a full ssl.create_default_context() "
        f"— 40.3 ms of CPU on the event-loop THREAD, measured on the bridge "
        f"host — for a context an http:// request never uses. Build one and "
        f"reuse it."
    )
    assert built["sync_get"] == 0, (
        f"{built['sync_get']} module-level httpx.get() calls in one tick: each "
        f"builds and throws away its own Client (same 40 ms), and they run "
        f"one-per-assigned-prefix inside the offloaded Caddy re-assert, where "
        f"holding the GIL blocks the event loop just as effectively"
    )


def test_health_stays_fast_during_a_full_fleet_tick(bridge, monkeypatch):
    """The measured half, with the production cost dialled in.

    Client construction is priced at the 40.3 ms measured on the bridge
    host rather than at whatever this runner's OpenSSL costs, so the number
    this asserts is production's number and not the runner's.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.01)
    _fleet(bridge, m)

    real_async_init = httpx.AsyncClient.__init__
    real_get = httpx.get

    def _slow_init(self, *a, **kw):
        _burn(HOST_CLIENT_BUILD_S)
        real_async_init(self, *a, **kw)

    def _slow_get(*a, **kw):
        _burn(HOST_CLIENT_BUILD_S)
        return real_get(*a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _slow_init)
    monkeypatch.setattr(httpx, "get", _slow_get)

    probe, elapsed, summary = _measure(
        bridge,
        lambda: asyncio.run_coroutine_threadsafe(
            m.reconciler_tick(), bridge.loop).result(timeout=300),
    )

    assert summary["assigned"] == FLEET_ASSIGNED, (
        f"the tick did not see the fleet: {summary}"
    )
    assert probe.timeouts == 0, (
        f"{probe.timeouts} of {len(probe.latencies)} /v1/health probes "
        f"exceeded the platform's {PROBE_TIMEOUT_S}s budget during one tick"
    )
    # MAX, not p95. At 20 Hz a 3 s stall produces ONE sample, so the fast
    # samples either side of it dominate every percentile — which is why the
    # pre-fix file passes a p95 assertion while freezing the loop for seconds
    # (the same reason production reads "p50 7 ms" on a bridge that is
    # unavailable 9 % of the time). The question this asks is the operator's
    # question: how long was the bridge unable to answer anything?
    # 1.0 s, not the 0.02 s this measures on an idle machine: the CI
    # runner is a cpu-capped container on the incident host itself. The
    # pre-fix file misses this by 3.7x (3.73 s measured), so the bar is
    # loose and the guard is still decisive.
    worst = max(probe.latencies)
    over = sum(1 for x in probe.latencies if x > 1.0)
    assert worst < 1.0 and over == 0, (
        f"/v1/health went unanswered for {worst:.2f}s during a {elapsed:.1f}s "
        f"tick over {FLEET_GENERIC + FLEET_ASSIGNED} members "
        f"({over}/{len(probe.latencies)} probes over 1 s; p50 "
        f"{statistics.median(probe.latencies):.3f}s). The loop thread is "
        f"doing per-member CPU again — on production that is 5-9 s of "
        f"main-thread CPU per tick and a 4.6 s worst probe."
    )


# ── a claim interrupted between bind and ASSIGNED ────────────────
#
# `bridge/ci-deploy.sh` restarts toup-bridge on every merge that touches
# pool_addon.py (#712 did, at 22:53:10Z on 2026-09-06). A restart — or a
# crash — that lands between `_claim_one` (ASSIGNING, user stamped) and the
# `_update_member(state=ASSIGNED)` at the end of `post_claim` leaves the slot
# ASSIGNING forever:
#
#   * the platform's 180 s `reclaim_stranded_users` re-claims the user
#     (pool_service.py:1180) with no state check;
#   * this file's idempotent branch matches `state in (ASSIGNED, ASSIGNING)`,
#     re-binds, and returns 200 WITHOUT promoting — nothing between the
#     re-bind and the return writes state;
#   * the platform records the slot as running from that 200;
#   * at ASSIGNING_STALE_S the reconciler marks it DEAD and step 4 removes
#     the Caddy route and destroys the container.
#
# The user then has a healthy-looking managed_containers row pointing at a
# hostname with no route and no container, and nothing on either side ever
# reconsiders it.


def _assigning(bridge, mod, slot, age_s, *, with_bind=True, user="u-stuck"):
    # 8 lowercase hex: `post_claim` validates the shape the platform's only
    # producer (`str(user_id)[:8]`) can make, so a fixture prefix that could
    # never come from a real claim is now refused at the boundary (R46 D24).
    m = _member(slot, bridge.agent_port, mod.STATE_ASSIGNING,
                assigned_prefix=f"deadbee{slot[-1]}", assigned_user_id=user)
    m["state_changed_at"] = int(time.time()) - age_s
    mod._save_members([m])
    mod._save_state({"current_image_tag": "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"})
    if with_bind:
        mod._persist_bind(slot, {"agent_api_key": f"k-{slot}", "user_id": user})
    else:
        mod._delete_bind(slot)
    return m


def _state_of(mod, slot):
    rows = [x for x in mod._load_members() if x["slot"] == slot]
    return rows[0]["state"] if rows else "GONE"


def test_an_idempotent_claim_promotes_a_slot_left_assigning(bridge):
    """The platform's reclaim is the ONLY thing that revisits a stuck slot,
    and it arrives through this branch. It must leave the slot ASSIGNED, and
    it must say so in the body so the platform can refuse to record anything
    else."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    _assigning(bridge, m, "s1", age_s=200)

    req = urllib.request.Request(
        f"http://127.0.0.1:{bridge.port}/v1/pool/claim",
        data=json.dumps({"user_id": "u-stuck", "prefix": "deadbee1",
                         "agent_api_key": "k-s1"}).encode(),
        headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req, timeout=120).read())

    assert body["ok"] is True and body["idempotent"] is True, body
    assert body.get("rebound") is True, body
    assert _state_of(m, "s1") == m.STATE_ASSIGNED, (
        f"slot s1 is still {_state_of(m, 's1')} after a successful "
        f"idempotent re-bind — the reconciler reaps it at "
        f"ASSIGNING_STALE_S and destroys a container the platform has "
        f"already recorded as running"
    )
    assert body.get("state") == m.STATE_ASSIGNED, (
        f"the claim response carries state={body.get('state')!r}; without it "
        f"the platform cannot tell a bound slot from one stuck mid-bind"
    )


def test_the_reconciler_recovers_an_assigning_slot_that_is_really_bound(bridge):
    """Nobody may have to call /claim again for this to heal: the persisted
    bind plus an authenticated probe with that member's REAL key is proof
    the bind landed, and the only thing missing is the registry write the
    restart interrupted."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    _assigning(bridge, m, "s2", age_s=m.ASSIGNING_RECOVER_S + 20)

    asyncio.run_coroutine_threadsafe(
        m.reconciler_tick(), bridge.loop).result(timeout=180)

    assert _state_of(m, "s2") == m.STATE_ASSIGNED, (
        f"slot s2 is {_state_of(m, 's2')} — a slot with a persisted bind "
        f"whose agent answers an authenticated probe is BOUND; leaving it "
        f"ASSIGNING means destroying it at ASSIGNING_STALE_S"
    )


def test_an_assigning_slot_with_no_bind_is_still_reaped(bridge):
    """The counterweight. Recovery must not become 'never reap': a claim
    that died BEFORE the bind has no persisted payload, so there is nothing
    to prove it landed and the stale reaper stays in charge."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    _assigning(bridge, m, "s3", age_s=m.ASSIGNING_STALE_S + 20, with_bind=False)

    asyncio.run_coroutine_threadsafe(
        m.reconciler_tick(), bridge.loop).result(timeout=180)

    assert _state_of(m, "s3") in (m.STATE_DEAD, "GONE"), (
        f"slot s3 is {_state_of(m, 's3')} — an ASSIGNING slot with no "
        f"persisted bind, {m.ASSIGNING_STALE_S}s old, must still be reaped"
    )


# ── the post-rollout hole in the pool ────────────────────────────
#
# `notify_pool_image_refresh` fires at the end of every non-[skip rollout]
# backend merge. Reconciler step 3 then marked EVERY stale GENERIC DRAINING
# in one tick, step 4 destroyed them all, and step 5 started ONE serial
# `_spawn_wave` that `break`s on the first failure — so the pool held ZERO
# claimable members for as long as the replacements took, and every signup
# in that window fell to the cold NAMED path (`POST /v1/tenants` on a 30 s
# budget, the path whose residue is still on the host as five orphan tnt_*
# networks).
#
# Measured from the live registry on 2026-09-07 (GET /v1/pool/list) plus
# `docker inspect` on the same containers:
#
#   * container start -> GENERIC: 61, 62, 65, 67, 69, 71, 72, 77, 81 s
#     (median 69) against GENERIC_HEALTH_TIMEOUT_S=90 — 9-29 s of margin,
#     on a host now at load 9-17 rather than the 44 of the incident;
#   * registry-row creation -> GENERIC: 64-105 s (median 95), so the
#     pre-health steps (slot alloc, create_tenant_db, wipe, docker run)
#     account for ~25 s that the 90 s budget does not cover;
#   * and the slots that replaced the 16:17Z refresh reached GENERIC at
#     16:35:48, 16:40:03, 16:48:40, 16:50:19, 16:55:14 — gaps of 4:15,
#     8:37, 1:39 and 4:55 for boots that each took ~70 s. That gap is the
#     serial wave plus its `break` plus the 30-44 s tick period.


def test_a_stale_image_recycle_keeps_the_pool_claimable(bridge):
    """The falsifier: an image refresh must never empty the pool.

    Ten GENERIC members on the old image, `current_image_tag` moved. The
    tick may drain, but what it may not do is leave a signup with nothing
    to claim — a one-image-old container that the assigned-upgrade walk
    will move later is strictly better than the cold named path.
    """
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    old = "ghcr.io/toup-com/toup-agent:oldoldoldold"
    new = "ghcr.io/toup-com/toup-agent:newnewnewnew"
    members = [_member(f"r{i:02d}", bridge.agent_port, m.STATE_GENERIC)
               for i in range(10)]
    for i, x in enumerate(members):
        x["image_tag"] = old
        x["created_at"] = int(time.time()) - (100 - i)
    m._save_members(members)
    m._save_state({"current_image_tag": new})

    summary = asyncio.run_coroutine_threadsafe(
        m.reconciler_tick(), bridge.loop).result(timeout=180)

    claimable = [x for x in m._load_members() if x.get("state") == m.STATE_GENERIC]
    assert summary["drained"] >= 1, (
        f"nothing was recycled, so this proves nothing: {summary}"
    )
    assert len(claimable) >= m.RECYCLE_MIN_READY, (
        f"{len(claimable)} claimable GENERIC members left after one refresh "
        f"tick over 10 stale ones (need >= {m.RECYCLE_MIN_READY}). Every "
        f"signup in this window falls to the cold named provision path, "
        f"which is the 30 s budget the incident blew."
    )


def test_the_spawn_wave_runs_more_than_one_boot_at_a_time(bridge):
    """A serial wave is ~95 s per replacement; ten of them is the ~20
    minutes of thin pool measured after the 16:17Z refresh."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05, docker_run=2.0)
    m._save_members([])
    m._save_state({"current_image_tag": "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"})

    assert m.SPAWN_WAVE_CONCURRENCY >= 2, (
        "the spawn wave is serial again; after an image refresh the pool "
        "refills one ~95 s boot at a time"
    )

    # A spawned member listens on a freshly allocated port, so nothing in
    # this harness can answer its lobby probe. Stub the wait — what is
    # under test is whether the BOOTS overlap, and the `docker run` delay
    # is what stands in for a boot.
    async def _ready(port, timeout_s=None):
        return True

    real_wait = m._wait_for_lobby_health
    m._wait_for_lobby_health = _ready
    try:
        t0 = time.perf_counter()
        asyncio.run_coroutine_threadsafe(
            m._spawn_wave(3, "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"),
            bridge.loop).result(timeout=180)
        took = time.perf_counter() - t0
    finally:
        m._wait_for_lobby_health = real_wait
    spawned = [x for x in m._load_members() if x.get("state") == m.STATE_GENERIC]
    assert len(spawned) == 3, f"the wave produced {len(spawned)} members"
    assert took < 3 * 2.0 * 0.8, (
        f"a 3-member wave took {took:.1f}s against a 2.0s-per-`docker run` "
        f"delay — the boots are still serialized"
    )


def test_one_failed_spawn_does_not_end_the_wave(bridge):
    """`break` on the first failure meant one poisoned slot cost the pool
    every remaining replacement until the NEXT tick — 30-44 s later, which
    is most of the multi-minute gaps in the refresh timeline."""
    m = bridge.mod
    calls = {"n": 0}

    async def _flaky(image_tag):
        calls["n"] += 1
        return calls["n"] != 1          # the first one fails

    real = m._spawn_one_pool_member
    m._spawn_one_pool_member = _flaky
    try:
        asyncio.run_coroutine_threadsafe(
            m._spawn_wave(3, "img"), bridge.loop).result(timeout=60)
    finally:
        m._spawn_one_pool_member = real

    assert calls["n"] == 3, (
        f"the wave stopped after {calls['n']} of 3 spawns because one "
        f"failed; the pool then waits a whole tick for the next attempt"
    )


def test_the_lobby_budget_still_fits_inside_the_spawning_reaper(bridge):
    """These two numbers are one number. `_spawn_one_pool_member` holds a
    row in SPAWNING for the pre-health work plus GENERIC_HEALTH_TIMEOUT_S;
    if SPAWNING_STALE_S is not comfortably above that sum, step 2b marks
    the slot DEAD while its own health wait is still running and step 4
    destroys a container that was about to come up."""
    m = bridge.mod
    pre_health_s = 30   # measured: row creation -> container start, 2026-09-07
    assert m.SPAWNING_STALE_S > m.GENERIC_HEALTH_TIMEOUT_S + pre_health_s, (
        f"SPAWNING_STALE_S={m.SPAWNING_STALE_S} is not above "
        f"GENERIC_HEALTH_TIMEOUT_S={m.GENERIC_HEALTH_TIMEOUT_S} + ~{pre_health_s}s "
        f"of pre-health work — the reaper races the spawn it is meant to "
        f"backstop"
    )
    assert m.GENERIC_HEALTH_TIMEOUT_S >= 120, (
        f"GENERIC_HEALTH_TIMEOUT_S={m.GENERIC_HEALTH_TIMEOUT_S}: the live "
        f"fleet's container-start-to-GENERIC times are 61-81 s at load 9-17 "
        f"(median 69). That is 9-29 s of margin, and the incident ran at "
        f"load 44."
    )


# ── what a rollout gate needs to see ─────────────────────────────
#
# The 2026-09-06 rollout put a 1 Hz polling supervisor on 41 real users'
# containers and walked all 41 with lobby health as its only check. The
# canary passed (3x200 + a 60 s hold + one turn probe) because nothing in
# the gate looks at RESOURCES: the difference between the pre-fix and
# post-fix images was 18.0 % vs 3-5 % CPU and 6.99 vs ~1.5 tx/s per tenant
# DB, and neither number was reachable from the bridge at all.
#
# Nor could an operator see the fleet: `/v1/pool/health` reported
# `image_lag_seconds` (time since the tag was SET, 83 424 s and counting)
# while 30 of 74 assigned members ran a two-generations-old image, and
# `assigned_stale` was written into the tick summary ONLY inside the
# auto-upgrade branch — so with BRIDGE_POOL_AUTO_UPGRADE_ASSIGNED=0 the
# platform read stale=0 and the pool looked quiet.


def _stats(bridge, prefix, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"http://127.0.0.1:{bridge.port}/v1/tenants/{prefix}/stats"
    if q:
        url += "?" + q
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def _fake_cgroup(bridge, cid, usage_usec, mem_bytes):
    d = bridge.ws.parent / "cgroup" / "system.slice" / f"docker-{cid}.scope"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpu.stat").write_text(f"usage_usec {usage_usec}\nuser_usec 1\nsystem_usec 1\n")
    (d / "memory.current").write_text(str(mem_bytes))
    bridge.mod.CGROUP_BASE = bridge.ws.parent / "cgroup"
    return d


def test_pool_health_exposes_the_fleet_split(bridge):
    """`image_lag_seconds` measures when the tag was SET, not what is
    running. A fleet that is 45/30 across two images reads as one number
    with no split anywhere."""
    m = bridge.mod
    cur = "ghcr.io/toup-com/toup-agent:currentcurr"
    old = "ghcr.io/toup-com/toup-agent:oldoldoldol"
    members = []
    for i in range(4):
        x = _member(f"f{i:02d}", bridge.agent_port, m.STATE_ASSIGNED,
                    assigned_prefix=f"fp{i:05d}", assigned_user_id=f"u{i}")
        x["image_tag"] = cur if i < 3 else old
        members.append(x)
    for i in range(2):
        x = _member(f"fg{i}", bridge.agent_port, m.STATE_GENERIC)
        x["image_tag"] = cur if i == 0 else old
        members.append(x)
    m._save_members(members)
    m._save_state({"current_image_tag": cur})

    h = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{bridge.port}/v1/pool/health", timeout=30).read())
    fleet = h.get("fleet")
    assert fleet, "/v1/pool/health has no fleet block"
    assert fleet["current_image_tag"] == cur
    assert fleet["assigned_on_current"] == 3, fleet
    assert fleet["assigned_on_other"] == 1, fleet
    assert fleet["generic_on_other"] == 1, fleet
    assert fleet["images"][cur] == 4 and fleet["images"][old] == 2, fleet
    assert fleet["auto_upgrade_assigned"] is bool(m.AUTO_UPGRADE_ASSIGNED)
    assert "last_rollout_tick" in fleet


def test_the_fleet_split_does_not_wedge_the_platforms_quiescence_gate(bridge):
    """`pool_service.pool_is_busy` reads `assigned_stale` out of
    `last_reconciler_summary`, and the reconciler writes it only when a
    batch is actually selected. Reporting a permanent backlog THERE would
    make every rollout wait out `wait_for_pool_quiescence`. The fleet block
    is a new key on purpose."""
    m = bridge.mod
    cur = "ghcr.io/toup-com/toup-agent:currentcurr"
    x = _member("fz0", bridge.agent_port, m.STATE_ASSIGNED,
                assigned_prefix="fzp", assigned_user_id="uz")
    x["image_tag"] = "ghcr.io/toup-com/toup-agent:oldoldoldol"
    m._save_members([x])
    m._save_state({"current_image_tag": cur})
    m._last_tick_summary.clear()
    m._delete_bind("fz0")

    # The production configuration: the walk is paused, so a standing
    # backlog exists and no batch is ever selected.
    was = m.AUTO_UPGRADE_ASSIGNED
    m.AUTO_UPGRADE_ASSIGNED = False
    try:
        asyncio.run_coroutine_threadsafe(
            m.reconciler_tick(), bridge.loop).result(timeout=180)
        h = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{bridge.port}/v1/pool/health", timeout=30).read())
    finally:
        m.AUTO_UPGRADE_ASSIGNED = was
    assert h["fleet"]["assigned_on_other"] == 1
    assert h["fleet"]["auto_upgrade_assigned"] is False
    assert "assigned_stale" not in (h.get("last_reconciler_summary") or {}), (
        "assigned_stale appeared in the tick summary; the platform's "
        "quiescence gate treats any non-zero value as 'the pool is busy'"
    )


def test_tenant_stats_answers_for_a_pool_slot(bridge):
    """The signal the canary gate never had. Resolves through the pool
    registry first (the 74 members `/v1/tenants/*` could not reach at all)
    and falls back to the named container."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    cid = "a" * 64
    os.environ["FAKE_CID"] = cid
    _fake_cgroup(bridge, cid, usage_usec=1_000_000, mem_bytes=512 * 1024 * 1024)
    mem = _member("t1", bridge.agent_port, m.STATE_ASSIGNED,
                  assigned_prefix="tprefix1", assigned_user_id="ut1")
    m._save_members([mem])

    body = _stats(bridge, "tprefix1", window_s=0.5)
    for k in ("prefix", "container_name", "image_tag", "cpu_pct",
              "cpu_window_s", "mem_mb", "pg_backends", "xact_per_s",
              "sampled_at"):
        assert k in body, f"{k} missing from {sorted(body)}"
    assert body["container_name"] == "toup-agent-pool-t1"
    assert body["image_tag"] == "ghcr.io/toup-com/toup-agent:aaaaaaaaaaaa"
    assert body["cpu_pct"] == 0.0, body           # static fake cgroup
    assert body["mem_mb"] == 512, body
    assert body["pg_backends"] is None and body["pg_reason"], body
    assert "DATABASE_URL" not in json.dumps(body) and "u:p@" not in json.dumps(body), (
        "the tenant's DATABASE_URL leaked into the response"
    )


def test_tenant_stats_measures_a_real_cpu_delta(bridge):
    """cgroup `usage_usec` over a window, not a `docker stats` sample. The
    P0 walk's own numbers were misread once by a single one-second sample
    that caught a synchronized top-of-hour burst."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05)
    cid = "b" * 64
    os.environ["FAKE_CID"] = cid
    d = _fake_cgroup(bridge, cid, usage_usec=0, mem_bytes=256 * 1024 * 1024)
    m._save_members([_member("t2", bridge.agent_port, m.STATE_ASSIGNED,
                             assigned_prefix="tprefix2", assigned_user_id="ut2")])

    def bump():
        time.sleep(0.5)
        # 0.25 core-seconds burned during a 1.0 s window == 25 %
        (d / "cpu.stat").write_text("usage_usec 250000\nuser_usec 1\nsystem_usec 1\n")

    threading.Thread(target=bump, daemon=True).start()
    body = _stats(bridge, "tprefix2", window_s=1.0)
    assert 15.0 < body["cpu_pct"] < 40.0, (
        f"cpu_pct={body['cpu_pct']} for 0.25 core-seconds over a 1.0 s window"
    )


def test_tenant_stats_does_not_block_the_event_loop(bridge):
    """It holds a sampling window open. Doing that on the loop would be the
    original defect with a new caller."""
    m = bridge.mod
    _set_delays(bridge, docker=0.05, docker_inspect=BLOCK_S)
    cid = "c" * 64
    os.environ["FAKE_CID"] = cid
    _fake_cgroup(bridge, cid, usage_usec=5, mem_bytes=1024 * 1024)
    m._save_members([_member("t3", bridge.agent_port, m.STATE_ASSIGNED,
                             assigned_prefix="tprefix3", assigned_user_id="ut3")])

    probe, elapsed, body = _measure(
        bridge, lambda: _stats(bridge, "tprefix3", window_s=1.0))
    assert body["container_name"] == "toup-agent-pool-t3"
    _assert_stayed_responsive(probe, elapsed, "a tenant stats sample")


def test_tenant_stats_404s_for_an_unknown_prefix(bridge):
    m = bridge.mod
    m._save_members([])
    os.environ["FAKE_MISSING"] = "toup-agent-nobodyhere"
    try:
        _stats(bridge, "nobodyhere")
    except urllib.error.HTTPError as e:
        assert e.code == 404, e.code
    else:
        raise AssertionError("an unknown prefix must 404, not invent a container")
    finally:
        os.environ.pop("FAKE_MISSING", None)


def test_tenant_db_url_is_psql_speakable_and_pg_sample_never_raises(tmp_path, monkeypatch):
    """The canary's real env carries `postgresql+asyncpg://…@host.docker.internal:6432/…`.
    psql refuses the `+asyncpg` scheme, so the stats route would have answered
    `pg_reason` on every tenant — or 500 if psql were missing. Both halves pinned."""
    import importlib.util, sys, types
    fake_main = types.ModuleType("main"); fake_main.CADDY_ADMIN = "http://127.0.0.1:1"
    for hook in ("_caddy_add_tenant_route", "_caddy_remove_tenant_route", "_caddy_swap_upstream"):
        setattr(fake_main, hook, lambda *a, **k: None)
    sys.modules["main"] = fake_main
    spec = importlib.util.spec_from_file_location("pool_addon_urltest", BRIDGE)
    mod = importlib.util.module_from_spec(spec); sys.modules["pool_addon_urltest"] = mod
    spec.loader.exec_module(mod)
    url = mod._tenant_db_url(["FOO=bar",
                              "DATABASE_URL=postgresql+asyncpg://u:p@host.docker.internal:6432/toup_agent_feed0047"])
    assert url == "postgresql://u:p@127.0.0.1:6432/toup_agent_feed0047"
    monkeypatch.setenv("PATH", str(tmp_path))          # no psql anywhere
    assert mod._pg_sample(url) is None
