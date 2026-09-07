"""The host half of the bridge — the part CI never deploys and no one reviews.

`bridge/pool_addon.py` is CI-deployed and guarded by
`test_bridge_pool_addon_offload.py`. Its other half is not: `/opt/toup-bridge/main.py`
and `/opt/toup-bridge/tenant_health.py` are hand-installed, and the repo's only
copy of them is the pair of heredocs in `new-vps/08-provisioning-bridge.sh`.
Between 2026-04-27 and 2026-09-07 nobody mirrored a host edit back, and the
embedded copy fell 1,360 lines behind — three routes, the whole blue-green
upgrade path and the L-4b flag strip-and-reinject lived only on a VPS.

This file makes the embedded copy testable, and guards the two properties that
the 2026-09-06 incident showed the host half gets wrong:

1. **Every `reverse_proxy` handler this script installs must set
   `stream_close_delay`.** Caddy 2.9.1 calls `Cleanup()` on every provisioned
   handler on every admin-API config mutation, and
   `reverseproxy/streaming.go:421` reads

       if h.StreamCloseDelay == 0 { return h.closeConnections() }

   — which closes every hijacked (i.e. WebSocket) connection, on every tenant
   route, host-wide. `_caddy_add_tenant_route` performs three such mutations per
   signup and `_caddy_swap_upstream` one per blue-green upgrade, so a rollout
   that walks 41 slots cut every open chat socket on the host 41 times.

2. **Nothing blocking may be reached from an `async def`.** The bridge is one
   uvicorn process with no `--workers`, so a `subprocess.run(docker …)` on the
   loop is not a slow function — it is a full stop of the bridge. Starlette runs
   a *sync* `def` route handler in its threadpool, so those are fine and
   deliberately not flagged; what is not fine is an `async def` middleware, an
   `async` task loop, or an `async def` route.

   Both offenders this guard was written for were measured on production
   2026-09-07: `_audit_middleware` (async) → `_audit()` →
   `subprocess.run(["docker","exec",…], timeout=2)` made an audited 404 take
   p50 142 ms / max 2 349 ms against 54 ms / 159 ms for an unaudited one; and
   `tenant_health.healthcheck_tick` (async) → `_list_assigned_tenants()` →
   docker-py `containers.list()` measured 2.2–11.2 s against 100 containers, on
   the loop, every 30 s.

Parsed by AST out of the shell script rather than imported: this is bridge-host
code whose imports (`docker`, `psycopg2`, `pool_addon`) do not exist here.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_host_main_guard.py
"""
from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

INSTALLER = (
    pathlib.Path(__file__).resolve().parents[2] / "new-vps" / "08-provisioning-bridge.sh"
)

# Calls that block the calling thread outright.
BLOCKING_PRIMITIVES = {
    "subprocess.run",
    "subprocess.check_output",
    "subprocess.call",
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.patch",
    "httpx.delete",
    "httpx.request",
    "time.sleep",
    "psycopg2.connect",
}

# docker-py's synchronous SDK. `cli.containers.list()` is not "a method call",
# it is one `GET /containers/json` plus one inspect per container: measured
# 2.2-11.2 s (median ~4.7 s) against the 100 containers on the production host
# at load 18 on 2026-09-07. Matched by collection prefix so `cli.containers.get`,
# `docker_client.images.pull` and `cli.api.containers` are all caught.
DOCKER_SDK_COLLECTIONS = ("containers", "images", "networks", "volumes", "api")

# Anything reached through these is already off the loop.
OFFLOADERS = {"run_in_executor", "to_thread"}


def _is_docker_sdk(dotted: str) -> bool:
    parts = dotted.split(".")
    return len(parts) >= 3 and parts[-2] in DOCKER_SDK_COLLECTIONS


def _extract(marker: str) -> str:
    """Pull one heredoc body out of the installer script."""
    text = INSTALLER.read_text("utf-8")
    m = re.search(
        r"^sudo -u \"\$BRIDGE_USER\" tee \"\$BRIDGE_DIR/%s\" > /dev/null <<'(\w+)'\n(.*?)\n\1$"
        % re.escape(marker),
        text,
        re.S | re.M,
    )
    assert m, f"no heredoc for {marker} in {INSTALLER}"
    return m.group(2)


@pytest.fixture(scope="module")
def main_src() -> str:
    return _extract("main.py")


@pytest.fixture(scope="module")
def health_src() -> str:
    return _extract("tenant_health.py")


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _functions(tree: ast.Module) -> dict[str, dict]:
    out: dict[str, dict] = {}

    class V(ast.NodeVisitor):
        def _add(self, node, kind):
            calls = []
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                name = _dotted(sub.func)
                if not name:
                    continue
                # A call handed to run_in_executor/to_thread is off the loop;
                # so is everything inside the lambda that wraps it.
                calls.append((name, sub.lineno))
            out[node.name] = {"kind": kind, "calls": calls, "node": node}
            for child in node.body:
                self.visit(child)

        def visit_FunctionDef(self, n):
            self._add(n, "sync")

        def visit_AsyncFunctionDef(self, n):
            self._add(n, "async")

    V().visit(tree)
    return out


def _offloaded_linenos(tree: ast.Module) -> set[int]:
    """Line numbers of every call that sits inside a run_in_executor/to_thread arg."""
    safe: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if not name or name.split(".")[-1] not in OFFLOADERS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Call):
                    safe.add(sub.lineno)
    return safe


def _blocking_sync_functions(funcs: dict[str, dict], safe: set[int]) -> set[str]:
    blocking: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, info in funcs.items():
            if info["kind"] != "sync" or name in blocking:
                continue
            for call, lineno in info["calls"]:
                if lineno in safe:
                    continue
                base = call.split(".")[-1]
                if (
                    call in BLOCKING_PRIMITIVES
                    or _is_docker_sdk(call)
                    or (base in blocking and base != name)
                ):
                    blocking.add(name)
                    changed = True
                    break
    return blocking


def _offenders(src: str, filename: str) -> list[str]:
    tree = ast.parse(src)
    funcs = _functions(tree)
    safe = _offloaded_linenos(tree)
    blocking = _blocking_sync_functions(funcs, safe)
    out: list[str] = []
    for name, info in funcs.items():
        if info["kind"] != "async":
            continue
        for call, lineno in info["calls"]:
            if lineno in safe:
                continue
            base = call.split(".")[-1]
            if call in BLOCKING_PRIMITIVES or _is_docker_sdk(call) or base in blocking:
                out.append(f"{filename}:{lineno} async {name}() -> {call}()")
    return sorted(set(out))


# ── anti-vacuity ─────────────────────────────────────────────────
# Every assertion below iterates a parsed collection. If the extraction
# silently returned nothing they would all pass while proving nothing.


def test_the_installer_still_carries_both_host_files(main_src, health_src):
    assert len(main_src.splitlines()) >= 2000, (
        "the embedded main.py is smaller than the file installed on the host — "
        "somebody replaced the re-synced copy with an older one"
    )
    assert len(health_src.splitlines()) >= 250
    ast.parse(main_src)
    ast.parse(health_src)


def test_the_parse_finds_the_real_bridge(main_src):
    funcs = _functions(ast.parse(main_src))
    assert len(funcs) >= 40, f"only parsed {len(funcs)} functions"
    for expected in ("_caddy_add_tenant_route", "_caddy_swap_upstream", "create_tenant"):
        assert expected in funcs, f"{expected} missing — the extraction is wrong"


def test_the_blocking_analysis_is_not_vacuous(main_src, health_src):
    tree = ast.parse(main_src)
    blocking = _blocking_sync_functions(_functions(tree), _offloaded_linenos(tree))
    for expected in ("_caddy_add_tenant_route", "_caddy_remove_tenant_route", "_audit"):
        assert expected in blocking, (
            f"{expected} is no longer detected as blocking — the transitive "
            "analysis has broken and the main guard is vacuous"
        )
    htree = ast.parse(health_src)
    hblocking = _blocking_sync_functions(_functions(htree), _offloaded_linenos(htree))
    assert "_list_assigned_tenants" in hblocking, (
        "_list_assigned_tenants is no longer detected as blocking — the "
        "docker-py rule has broken and the tenant_health guard is vacuous"
    )


# ── guard 1: Caddy must not close every WebSocket on every reload ─


def _reverse_proxy_handlers(src: str) -> list[dict]:
    """Every dict literal in the source whose "handler" is "reverse_proxy"."""
    found: list[dict] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "handler" not in keys:
            continue
        idx = keys.index("handler")
        # keys/values line up only when every key is a Constant
        if len(node.keys) != len(keys):
            continue
        val = node.values[idx]
        if isinstance(val, ast.Constant) and val.value == "reverse_proxy":
            found.append({"keys": keys, "lineno": node.lineno})
    return found


def test_every_reverse_proxy_template_sets_stream_close_delay(main_src):
    """The falsifier for the Caddy half.

    Caddy 2.9.1, reverseproxy/streaming.go:421-423:

        func (h *Handler) cleanupConnections() error {
            if h.StreamCloseDelay == 0 { return h.closeConnections() }

    `Cleanup()` runs for EVERY provisioned reverse_proxy handler on EVERY
    admin-API config mutation, and `closeConnections()` closes every hijacked
    connection. Without `stream_close_delay` on the handler this script installs,
    one signup (3 mutations) or one blue-green swap (1) severs every in-flight
    chat WebSocket on all 164 tenant routes.
    """
    handlers = _reverse_proxy_handlers(main_src)
    assert handlers, "no reverse_proxy handler found — the guard is vacuous"
    missing = [
        f"main.py:{h['lineno']}"
        for h in handlers
        if "stream_close_delay" not in h["keys"]
    ]
    assert not missing, (
        "these reverse_proxy handlers have no `stream_close_delay`, so Caddy "
        "closes every WebSocket on every tenant route each time the bridge "
        "touches the admin API (3x per signup, 1x per blue-green swap):\n  "
        + "\n  ".join(missing)
    )


def test_adding_a_tenant_route_costs_one_config_load(main_src):
    """Three config loads per signup was two more than necessary.

    Each load is a fleet-wide `Cleanup()` of every reverse_proxy handler (see
    above). The wildcard `*.agents.toup.ai` only has to be LAST in the array;
    re-appending it on every signup — DELETE + POST — is two extra loads for a
    route that has not changed. `PUT /config/.../routes/<idx>` INSERTS at the
    index (verified on caddy:2.9.1 2026-09-07), so one call is enough.

    Counted as mutating admin calls in the function body: GET does not reload.
    """
    tree = ast.parse(main_src)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "_caddy_add_tenant_route"),
        None,
    )
    assert fn is not None, "_caddy_add_tenant_route not found"
    mutating = [
        f"main.py:{sub.lineno} {_dotted(sub.func)}"
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and _dotted(sub.func) in {"httpx.post", "httpx.put", "httpx.patch", "httpx.delete"}
    ]
    assert len(mutating) <= 2, (
        "adding one tenant route performs %d mutating Caddy admin calls; each "
        "one is a config reload that runs Cleanup() on all ~164 tenant routes. "
        "Insert ahead of the wildcard with PUT instead of appending and then "
        "rebuilding the wildcard:\n  %s" % (len(mutating), "\n  ".join(mutating))
    )


# ── guard 2: nothing blocking on the event loop ───────────────────


def test_no_blocking_call_reaches_the_bridge_event_loop(main_src):
    offenders = _offenders(main_src, "main.py")
    assert not offenders, (
        "these run a blocking call on the bridge's single event loop, freezing "
        "every request for its duration — hand them to a worker thread "
        "(`run_in_executor` / `asyncio.to_thread`):\n  " + "\n  ".join(offenders)
    )


def test_no_blocking_call_reaches_the_tenant_health_loop(health_src):
    offenders = _offenders(health_src, "tenant_health.py")
    assert not offenders, (
        "the tenant health watchdog runs as an asyncio task in the bridge "
        "process; a blocking call in its tick is a bridge-wide freeze every "
        "HEALTHCHECK_INTERVAL_S:\n  " + "\n  ".join(offenders)
    )


def test_probe_clients_are_not_constructed_per_call(health_src):
    """`httpx.AsyncClient()` costs ~53 ms of CPU on the bridge's interpreter.

    Measured 2026-09-07 on /opt/toup-bridge/venv (Python 3.12, httpx 0.28.1):
    10 constructions = 0.534 s, because each one builds a default SSL context
    from certifi — for probes that are plain `http://127.0.0.1:<port>`.
    """
    tree = ast.parse(health_src)
    ctors = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _dotted(n.func) == "httpx.AsyncClient"
    ]
    assert ctors, "no httpx.AsyncClient at all — the probe is gone, guard vacuous"
    assert len(ctors) == 1, (
        "tenant_health.py constructs httpx.AsyncClient at %d places (lines %s); "
        "there must be exactly one, reused" % (len(ctors), ctors)
    )
    inside_async = [
        f"tenant_health.py:{sub.lineno} in async {n.name}()"
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        for sub in ast.walk(n)
        if isinstance(sub, ast.Call) and _dotted(sub.func) == "httpx.AsyncClient"
    ]
    assert not inside_async, (
        "an httpx.AsyncClient built inside an `async def` is built per call, on "
        "the event-loop thread, at ~53 ms of CPU each (measured on the bridge's "
        "own interpreter) — for a probe that is plain http://127.0.0.1:<port> "
        "and never uses the SSL context that cost buys:\n  "
        + "\n  ".join(inside_async)
    )
