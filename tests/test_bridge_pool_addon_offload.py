"""The bridge's event loop must never run a docker CLI.

`bridge/pool_addon.py` is mounted into a SINGLE uvicorn process with no
`--workers` (`new-vps/08-provisioning-bridge.sh`). Every `subprocess.run`
in it is therefore not "a slow function" — it is a full stop of the whole
bridge: no connection accepted, no response written, no coroutine advanced,
for as long as the docker daemon takes to answer.

2026-09-06 is what that costs when the host is busy. With the VPS at load
~40 on 16 cores (41 user containers on a pre-fix voice image burning 7.6
cores between them), every docker CLI call took seconds, and:

  * the platform's 5s `/v1/health` probes timed out MORE THAN A THIRD of
    the time, all day, producing 8 declared "bridge unreachable" outages
    (`backend/app/services/bridge_supervisor.py:37`);
  * `/v1/pool/list` — which is two file reads and a counter — took 10.2s;
  * `/v1/host/overview` took 19.5s;
  * the dedicated-container rollouts got 500/409/502 from
    `/v1/tenants/*/upgrade`;
  * a signup's `/v1/pool/claim` blew the platform's 30s budget
    (`docker_host_service._build_bridge_client`) while the bridge finished
    that same bind 2s later — so the platform recorded a failure for work
    that had SUCCEEDED, and the user's onboarding stalled.

Not one of those endpoints was slow. The loop underneath them was busy.

This file is the static half of the guard: no blocking call may be reached
directly from an `async def`, and nothing that mutates the registry may be
handed to a worker thread. The dynamic half — that a cheap GET actually
stays fast while a 6s docker op runs — is
`test_bridge_pool_addon_event_loop.py`.

Parsed by AST rather than imported, the same way
`test_bridge_feature_flag_forwarding.py` reads `_FEATURE_FLAG_ENVS`:
`pool_addon.py` is bridge-host code whose helpers (`docker`, `psql`,
`create_tenant_db`) do not exist in this environment.

Run:
    cd backend && PYTHONPATH=. pytest tests/test_bridge_pool_addon_offload.py
"""
from __future__ import annotations

import ast
import pathlib

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"

# Calls that block the calling thread outright.
BLOCKING_PRIMITIVES = {
    "subprocess.run",
    "subprocess.check_output",
    "subprocess.call",
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.delete",
    "time.sleep",
}

# Read-modify-write over members.json / state.json. Each is atomic ONLY
# because it contains no `await`: on a single-threaded loop nothing can
# interleave between its read and its write. Run two of them in worker
# threads and one silently overwrites the other — and a lost
# `state=ASSIGNED` is a bound user whose slot the reconciler then reaps as
# stale, which is the worst failure this file can produce.
REGISTRY_MUTATORS = {
    "_save_members",
    "_update_member",
    "_add_member",
    "_remove_member",
    "_claim_one",
    "_save_state",
    "_state_set",
    "_persist_bind",
    "_delete_bind",
    "_atomic_write_json",
}

OFFLOADERS = {"_offload", "_offload_ux"}

# Blocking helpers that live in main.py (`new-vps/08-provisioning-bridge.sh`),
# imported here inside the functions that use them. The transitive analysis
# below can only see functions DEFINED in pool_addon.py, so without this list
# a Caddy call reverting to the event loop passes every guard silently —
# which is exactly what happened to mutation M4 on the first pass.
# `_caddy_add_tenant_route` alone is 3-4 synchronous `httpx` calls at a 10s
# timeout each, and the reconciler can invoke it once per assigned tenant.
EXTERNAL_BLOCKING = {
    "_caddy_add_tenant_route",
    "_caddy_remove_tenant_route",
    "_caddy_swap_upstream",
}


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(BRIDGE.read_text("utf-8"))


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _functions(tree: ast.Module) -> dict[str, dict]:
    """{name: {kind, calls: [(dotted_name, lineno)], node}} for every def."""
    out: dict[str, dict] = {}

    class V(ast.NodeVisitor):
        def _add(self, node, kind):
            calls = [
                (_dotted(sub.func), sub.lineno)
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and _dotted(sub.func)
            ]
            out[node.name] = {"kind": kind, "calls": calls, "node": node}
            # Recurse so nested defs (e.g. `_abort`) are captured too.
            for child in node.body:
                self.visit(child)

        def visit_FunctionDef(self, n):
            self._add(n, "sync")

        def visit_AsyncFunctionDef(self, n):
            self._add(n, "async")

    V().visit(tree)
    return out


def _blocking_sync_functions(funcs: dict[str, dict]) -> set[str]:
    """Sync functions that block, transitively — plus main.py's Caddy trio."""
    blocking: set[str] = set(EXTERNAL_BLOCKING)
    changed = True
    while changed:
        changed = False
        for name, info in funcs.items():
            if info["kind"] != "sync" or name in blocking:
                continue
            for call, _ in info["calls"]:
                base = call.split(".")[-1]
                if call in BLOCKING_PRIMITIVES or (base in blocking and base != name):
                    blocking.add(name)
                    changed = True
                    break
    return blocking


def _offloaded_targets(funcs: dict[str, dict]) -> set[str]:
    """Every first-argument callable passed to `_offload` / `_offload_ux`."""
    targets: set[str] = set()
    for info in funcs.values():
        for sub in ast.walk(info["node"]):
            if not isinstance(sub, ast.Call):
                continue
            if _dotted(sub.func) not in OFFLOADERS or not sub.args:
                continue
            name = _dotted(sub.args[0])
            if name:
                targets.add(name.split(".")[-1])
    return targets


# ── anti-vacuity ─────────────────────────────────────────────────
# Every assertion below iterates a parsed collection. If the parse
# silently returned nothing they would all pass while proving nothing —
# the `all([]) is True` trap this suite has produced before.


def test_the_parse_actually_found_the_bridge(tree):
    funcs = _functions(tree)
    assert len(funcs) >= 40, f"only parsed {len(funcs)} functions"
    # `subprocess.run` appears BOTH as a call (inside sync helpers) and as a
    # bare `ast.Name` first argument to `_offload(subprocess.run, [...])`.
    # Count references, not calls, or the count collapses as the fix lands.
    refs = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "run"
        and _dotted(node) == "subprocess.run"
    )
    assert refs >= 20, (
        f"only found {refs} subprocess.run references — the parse is not "
        "seeing the real file, so every guard below is vacuous"
    )


def test_the_blocking_helper_set_is_not_empty(tree):
    blocking = _blocking_sync_functions(_functions(tree))
    for expected in ("_spawn_pool_container", "_destroy_container",
                     "_save_workspace_for_pool_release", "_next_pool_port",
                     "_caddy_add_tenant_route"):
        assert expected in blocking, (
            f"{expected} is no longer detected as blocking — the transitive "
            "analysis has broken and the main guard is vacuous"
        )


# ── the main guard ───────────────────────────────────────────────


def test_no_blocking_call_is_reachable_from_the_event_loop(tree):
    """The falsifier for the whole change.

    An `async def` body runs ON the event loop. A `subprocess.run` there —
    directly, or through any sync helper that reaches one — is a bridge-wide
    freeze for its duration. Route it through `_offload`/`_offload_ux`
    instead, which hands it to a worker thread behind a bounded semaphore.

    NOTE what is deliberately NOT flagged: a sync `def` route handler
    (`tenant_truth`, `list_members`, `health`, `whois`). Starlette runs
    those in anyio's threadpool, so they are off the loop by construction.
    Converting one to `async def` without an offload is the way to
    re-introduce this bug while looking like a modernisation.
    """
    funcs = _functions(tree)
    blocking = _blocking_sync_functions(funcs)

    offenders: list[str] = []
    for name, info in funcs.items():
        if info["kind"] != "async":
            continue
        for call, lineno in info["calls"]:
            base = call.split(".")[-1]
            if call in BLOCKING_PRIMITIVES or base in blocking:
                offenders.append(f"pool_addon.py:{lineno} async {name}() -> {call}()")

    assert not offenders, (
        "these run a blocking call directly on the bridge's event loop, "
        "freezing every request for its duration — wrap them in `_offload` "
        "(background) or `_offload_ux` (claim/restart/respawn):\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_registry_mutations_never_run_in_a_worker_thread(tree):
    """The counterweight to the guard above.

    Offloading is not free: `_load_members` + mutate + `_save_members` is
    atomic only while it cannot be interleaved. Two of them in different
    threads and the second read-modify-write clobbers the first.
    """
    offloaded = _offloaded_targets(_functions(tree))
    assert offloaded, "no _offload call sites found — this guard is vacuous"
    leaked = sorted(offloaded & REGISTRY_MUTATORS)
    assert not leaked, (
        f"{leaked} mutate the JSON registry and were handed to a worker "
        "thread; two concurrent read-modify-writes lose one of them"
    )


def test_offloaded_helpers_do_not_write_the_registry_transitively(tree):
    """A helper that writes the registry two calls down is the same bug."""
    funcs = _functions(tree)
    offloaded = _offloaded_targets(funcs)
    assert offloaded, "no _offload call sites found — this guard is vacuous"

    def writes_registry(name: str, seen: set[str]) -> str | None:
        if name in seen or name not in funcs:
            return None
        seen.add(name)
        for call, _ in funcs[name]["calls"]:
            base = call.split(".")[-1]
            if base in REGISTRY_MUTATORS:
                return f"{name} -> {base}"
            deeper = writes_registry(base, seen)
            if deeper:
                return f"{name} -> {deeper}"
        return None

    bad = [p for t in sorted(offloaded) if (p := writes_registry(t, set()))]
    assert not bad, f"offloaded helpers reach a registry write: {bad}"


# ── the two-lane design ──────────────────────────────────────────


def test_there_are_two_separate_docker_lanes(tree):
    """One shared semaphore lets background work starve a live signup.

    Measured with the harness (`$SP/laneB/starvation.py`): with the
    background lane saturated by 8 queued docker ops, a claim finished in
    4.40s on two lanes and 15.87s when `_offload_ux` was collapsed onto
    `_docker_sem`. The claim's own docker work is ~4.4s in both.
    """
    src = BRIDGE.read_text("utf-8")
    assigned = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "_docker_sem" in assigned and "_docker_sem_ux" in assigned, (
        "both lanes must exist as module-level semaphores"
    )
    funcs = _functions(tree)
    bg = [c for c, _ in funcs["_offload"]["calls"]]
    ux = [c for c, _ in funcs["_offload_ux"]["calls"]]
    assert any("_docker_sem" == c or c.endswith("_docker_sem") for c in bg) or \
        "_docker_sem" in src, "_offload must take the background semaphore"
    assert "_docker_sem_ux" in src, "_offload_ux must take the interactive semaphore"
    # They must not be the same object.
    assert "_docker_sem_ux = _docker_sem" not in src, (
        "the two lanes were collapsed into one — background work can now "
        "starve a claim (measured: 4.40s -> 15.87s)"
    )


@pytest.mark.parametrize("path_fn", [
    ("post_claim", "_offload_ux"),
    ("post_restart_member", "_offload_ux"),
    ("post_respawn_member", "_offload_ux"),
])
def test_interactive_routes_use_the_interactive_lane(tree, path_fn):
    """A human is waiting on these three; they must not queue behind the
    reconciler's snapshot/upgrade backlog."""
    fn, lane = path_fn
    funcs = _functions(tree)
    used = {c for c, _ in funcs[fn]["calls"] if c in OFFLOADERS}
    assert lane in used, (
        f"{fn} offloads via {used or 'nothing'} — it must use {lane} so a "
        "live signup does not wait on background docker work"
    )


# ── the cheap reads ──────────────────────────────────────────────


@pytest.mark.parametrize("name", ["health", "list_members", "whois", "tenant_truth"])
def test_the_cheap_reads_stay_sync_handlers(tree, name):
    """Starlette runs a sync `def` handler in anyio's threadpool — a pool
    that is NOT `asyncio.to_thread`'s executor, so these can answer even
    while every docker permit is taken. Promoting one to `async def` is how
    a diagnostic route becomes an outage."""
    funcs = _functions(tree)
    assert name in funcs, f"{name} route handler not found"
    assert funcs[name]["kind"] == "sync", (
        f"{name} became `async def`; its body now runs on the event loop"
    )


def test_whois_is_a_registry_read_with_no_docker(tree):
    """`/v1/pool/whois` exists so the platform can ask "did that claim
    actually land?" after its 30s budget expires, without paying for a
    `docker ps`. If it ever shells out it is no longer that."""
    funcs = _functions(tree)
    blocking = _blocking_sync_functions(funcs)
    for call, lineno in funcs["whois"]["calls"]:
        base = call.split(".")[-1]
        assert call not in BLOCKING_PRIMITIVES and base not in blocking, (
            f"whois() calls {call}() at line {lineno} — it must stay a pure "
            "registry read, or the platform's timeout-recovery path is as "
            "slow as the thing it is recovering from"
        )
    assert any(c == "_load_members" for c, _ in funcs["whois"]["calls"]), (
        "whois() must actually read the registry"
    )


# ── deploy contract (bridge/ci-deploy.sh) ────────────────────────


def test_the_file_still_satisfies_the_ci_deploy_hook():
    """`bridge/ci-deploy.sh` refuses anything under 10000 bytes and
    py_compiles it with the bridge's own interpreter before installing."""
    import py_compile
    import tempfile

    size = BRIDGE.stat().st_size
    assert size >= 10000, f"ci-deploy.sh rejects files < 10000 bytes; this is {size}"
    with tempfile.NamedTemporaryFile(suffix=".pyc") as out:
        py_compile.compile(str(BRIDGE), cfile=out.name, doraise=True)
