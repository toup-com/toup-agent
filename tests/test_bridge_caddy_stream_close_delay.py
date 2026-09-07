"""Does a Caddy admin mutation on ONE route kill the WebSockets on all the others?

The bridge expresses tenant routing as per-tenant Caddy config: one signup adds
a route, one blue-green upgrade PATCHes an upstream. Caddy re-provisions the
whole config on every admin mutation and calls `Cleanup()` on every old
`reverse_proxy` handler, and `reverseproxy/streaming.go:421` says

    func (h *Handler) cleanupConnections() error {
        if h.StreamCloseDelay == 0 { return h.closeConnections() }

so with the key absent every hijacked (WebSocket) connection on every tenant
route dies. The 2026-09-06 P0 walk performed 41 blue-green swaps in 44 minutes.

That is an argument from vendor source. This file is the experiment, against a
real `caddy:2.9.1`, with a real WebSocket, using the bridge's own route shapes:

    A  shipped template, shipped add-route   3 mutations   ws survived: False
    B  + stream_close_delay=10m              3 mutations   ws survived: True
    C  + insert-before-wildcard              1 mutation    ws survived: True
    D  insert-before-wildcard, no delay      1 mutation    ws survived: False

D is the control that matters: cutting three reloads to one does NOT fix it.
`stream_close_delay` is the fix; the reload count is a separate, real saving.
C also proves `PUT /config/.../routes/<idx>` INSERTS at the index (final order
[tenant_a, tenant_b, *.agents]) — which is what lets one call replace the
append-then-rebuild-the-wildcard dance.

**Opt-in: set `TOUP_TEST_CADDY=1`.** It pulls an image, binds two ports and
needs `host.docker.internal` to reach a server on the machine running pytest —
none of which belongs in the CI sweep, which runs every file in `tests/` in its
own process. The invariant itself IS guarded in CI, by
`test_bridge_host_main_guard.py::test_every_reverse_proxy_template_sets_stream_close_delay`;
this file is the evidence that guard's premise is true, and it is re-run by hand
whenever the premise is doubted (a Caddy upgrade, most obviously).

Run it deliberately:
    cd backend && PYTHONPATH=. TOUP_TEST_CADDY=1 pytest -q -s tests/test_bridge_caddy_stream_close_delay.py
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
import uuid

import pytest

CADDY_IMAGE = "caddy:2.9.1"
UPSTREAM_PORT = 8798


def _docker_ok() -> bool:
    try:
        return subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=15,
        ).returncode == 0
    except Exception:
        return False


# NOT a module-level `pytest.skip(allow_module_level=True)`, and not
# `importorskip` either: both stop collection, and pytest then exits 5 ("no
# tests collected"). CI's sweep runs one process per file and treats any
# non-zero exit as a failure —
#   xargs ... 'pytest "tests/{}" ... || echo "tests/{}" >> /tmp/failed.txt'
# — so a module-level skip here would fail the whole Backend tests job. A
# `skipif` mark lets the tests be COLLECTED and reported as skipped: exit 0.
try:  # pragma: no cover - import guard
    import httpx
    import websockets
except ImportError:  # pragma: no cover
    httpx = None
    websockets = None

if os.environ.get("TOUP_TEST_CADDY") != "1":
    _SKIP = (
        "vendor-behaviour probe: set TOUP_TEST_CADDY=1 to run it (it pulls an "
        "image, binds two ports and needs host.docker.internal). The invariant "
        "is guarded in CI by test_bridge_host_main_guard.py."
    )
elif websockets is None or httpx is None:
    _SKIP = "TOUP_TEST_CADDY=1 but websockets/httpx are not installed"
elif not _docker_ok():
    _SKIP = "TOUP_TEST_CADDY=1 but the docker daemon is not reachable"
else:
    _SKIP = ""

pytestmark = pytest.mark.skipif(bool(_SKIP), reason=_SKIP or "runnable")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def harness():
    """A WS echo server on the host + caddy:2.9.1 in a container proxying to it."""
    from websockets.asyncio.server import serve  # noqa: PLC0415

    admin_port, http_port = _free_port(), _free_port()
    name = f"toup-caddy-test-{uuid.uuid4().hex[:8]}"
    stop = asyncio.Event()

    async def echo(ws):
        try:
            async for msg in ws:
                await ws.send("echo:" + msg)
        except Exception:
            pass

    loop = asyncio.new_event_loop()

    async def _serve():
        async with serve(echo, "0.0.0.0", UPSTREAM_PORT):
            await stop.wait()

    import threading
    t = threading.Thread(target=lambda: loop.run_until_complete(_serve()), daemon=True)
    t.start()
    time.sleep(1.0)

    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-e", "CADDY_ADMIN=0.0.0.0:2019",
         # Docker Desktop defines host.docker.internal; Linux does not, and the
         # upstream in every route below is a server on the pytest machine.
         "--add-host", "host.docker.internal:host-gateway",
         "-p", f"127.0.0.1:{admin_port}:2019", "-p", f"127.0.0.1:{http_port}:80",
         CADDY_IMAGE, "caddy", "run", "--config", "/dev/null", "--adapter", ""],
        check=True, capture_output=True, timeout=180,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{admin_port}/config/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        pytest.skip("caddy admin API never came up")

    yield {"admin": f"http://127.0.0.1:{admin_port}", "http_port": http_port}

    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    loop.call_soon_threadsafe(stop.set)


# ── the bridge's own route shapes ────────────────────────────────

WILDCARD = {
    "@id": "wildcard",
    "match": [{"host": ["*.agents.test"]}],
    "handle": [{"handler": "static_response", "status_code": 404, "body": "Unknown tenant"}],
}


def _route(prefix: str, extra: dict) -> dict:
    handler = {
        "handler": "reverse_proxy",
        "upstreams": [{"dial": f"host.docker.internal:{UPSTREAM_PORT}"}],
        "transport": {"protocol": "http", "read_timeout": "300s",
                      "write_timeout": "300s", "dial_timeout": "10s"},
    }
    handler.update(extra)
    return {"@id": f"tenant_{prefix}", "match": [{"host": [f"agent-{prefix}.agents.test"]}],
            "handle": [handler]}


def _load(admin: str, extra: dict) -> None:
    cfg = {
        "admin": {"listen": "0.0.0.0:2019"},
        "apps": {"http": {"servers": {"srv0": {
            "listen": [":80"], "routes": [_route("aaaa1111", extra), WILDCARD]}}}},
    }
    httpx.post(f"{admin}/load", json=cfg, timeout=20).raise_for_status()


def _routes(admin: str) -> list:
    return httpx.get(f"{admin}/config/apps/http/servers/srv0/routes", timeout=20).json()


def _add_route_shipped(admin: str, prefix: str, extra: dict) -> int:
    """What the installed `_caddy_add_tenant_route` did: append, then rebuild the wildcard."""
    httpx.post(f"{admin}/config/apps/http/servers/srv0/routes",
               json=_route(prefix, extra), timeout=20).raise_for_status()
    routes = _routes(admin)
    idx = next(i for i, rt in enumerate(routes)
               if "*.agents.test" in (rt.get("match") or [{}])[0].get("host", []))
    if idx == len(routes) - 1:
        return 1
    httpx.delete(f"{admin}/config/apps/http/servers/srv0/routes/{idx}", timeout=20).raise_for_status()
    httpx.post(f"{admin}/config/apps/http/servers/srv0/routes",
               json=routes[idx], timeout=20).raise_for_status()
    return 3


def _add_route_insert(admin: str, prefix: str, extra: dict) -> int:
    """The fix: PUT inserts at the index, so the wildcard is never rebuilt."""
    routes = _routes(admin)
    idx = next((i for i, rt in enumerate(routes)
                if "*.agents.test" in (rt.get("match") or [{}])[0].get("host", [])), None)
    if idx is None:
        httpx.post(f"{admin}/config/apps/http/servers/srv0/routes",
                   json=_route(prefix, extra), timeout=20).raise_for_status()
    else:
        httpx.put(f"{admin}/config/apps/http/servers/srv0/routes/{idx}",
                  json=_route(prefix, extra), timeout=20).raise_for_status()
    return 1


async def _survives(admin: str, http_port: int, extra: dict, adder) -> tuple[bool, int, bool]:
    _load(admin, extra)
    await asyncio.sleep(0.5)
    sock = socket.create_connection(("127.0.0.1", http_port))
    ws = await websockets.connect("ws://agent-aaaa1111.agents.test/", sock=sock, open_timeout=15)
    await ws.send("hello")
    assert await asyncio.wait_for(ws.recv(), 10) == "echo:hello"

    n = adder(admin, "bbbb2222", extra)      # a DIFFERENT tenant's signup
    await asyncio.sleep(1.5)

    try:
        await ws.send("after")
        survived = await asyncio.wait_for(ws.recv(), 5) == "echo:after"
    except Exception:
        survived = False
    hosts = [(rt.get("match") or [{}])[0].get("host") for rt in _routes(admin)]
    ordered = hosts.index(["*.agents.test"]) == len(hosts) - 1
    try:
        await ws.close()
    except Exception:
        pass
    return survived, n, ordered


def _run(harness, extra, adder):
    return asyncio.run(_survives(harness["admin"], harness["http_port"], extra, adder))


DELAY = {"stream_close_delay": "10m"}


def test_without_stream_close_delay_a_signup_kills_another_tenants_websocket(harness):
    survived, n, ordered = _run(harness, {}, _add_route_shipped)
    assert n == 3, "the shipped add-route should perform three config mutations"
    assert ordered
    assert survived is False, (
        "this test's premise has changed: Caddy no longer closes hijacked "
        "connections on a config reload, so stream_close_delay may not be needed"
    )


def test_stream_close_delay_keeps_it_alive(harness):
    survived, n, ordered = _run(harness, DELAY, _add_route_shipped)
    assert (n, ordered, survived) == (3, True, True)


def test_the_fixed_shape_is_one_mutation_and_the_socket_lives(harness):
    survived, n, ordered = _run(harness, DELAY, _add_route_insert)
    assert n == 1, "PUT at the wildcard index should be the only mutation"
    assert ordered, "PUT must INSERT at the index, leaving the wildcard last"
    assert survived is True


def test_fewer_reloads_alone_does_not_save_the_socket(harness):
    """The control. One reload still runs Cleanup() on every handler."""
    survived, n, ordered = _run(harness, {}, _add_route_insert)
    assert (n, ordered) == (1, True)
    assert survived is False, (
        "if this passes, the reload-count reduction alone is sufficient and the "
        "stream_close_delay reasoning in the guard needs revisiting"
    )
