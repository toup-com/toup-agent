"""R46 F9c/F9d — WhatsApp credentials do not outlive the tenant that made them.

Two holes, one class:

  * The pool's workspace save/restore is a `docker cp` MERGE. A slot whose
    previous tenant linked WhatsApp still holds their `.whatsapp_auth` inside
    the container, and an incoming user whose own host tree has none inherits
    it (`_restore_workspace_for_pool_bind`, pool_addon.py).
  * Nothing logs the device out on release, so the saved host tree carries a
    live session forward too.

And one asymmetry that decides the shape of the fix: doing this on an IMAGE
REFRESH would force every tenant on the fleet to re-pair, straight into
WhatsApp's 30–60 minute rate limiter. So `_drain_member` takes `reason` as a
REQUIRED keyword with NO default, and the expensive direction is the one that
cannot be reached by accident.

Parsed rather than imported: `pool_addon.py` is bridge-host code whose imports
(docker, psql helpers) do not exist in the backend test environment — the same
constraint `test_pool_upgrade_fairness.py` works around. The functions below
are THE SHIPPED ONES, exec'd against injected fakes.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_bridge_pool_release_wa_creds.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import pathlib

import pytest
from fastapi import HTTPException

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
_SRC = BRIDGE.read_text()


def _load(name: str, ns: dict):
    """Exec ONE shipped function from the bridge source into `ns`."""
    tree = ast.parse(_SRC)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(BRIDGE), "exec"), ns)  # noqa: S102
            return ns[name]
    raise AssertionError(f"{name} not found in bridge/pool_addon.py")


def _constant(name: str):
    tree = ast.parse(_SRC)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return eval(compile(ast.Expression(node.value), "<c>", "eval"))  # noqa: S307
    raise AssertionError(f"{name} not found")


def _drain_ns(record: dict) -> dict:
    """Namespace for `_drain_member`: every side effect recorded, none real."""
    async def _offload(fn, *a):
        record.setdefault("offload", []).append((fn.__name__, a))
        return fn(*a)

    def _save_workspace_for_pool_release(prefix, cn):
        record.setdefault("saved", []).append(prefix)
        return True

    def _strip_whatsapp_auth_from_host(prefix):
        record.setdefault("stripped", []).append(prefix)
        return True

    def _destroy_container(cn):
        record.setdefault("destroyed", []).append(cn)

    async def _whatsapp_logout_member(member):
        record.setdefault("logged_out", []).append(member.get("slot"))

    class _Http:
        async def post(self, *a, **kw):
            record.setdefault("drain_posts", []).append(a[0] if a else None)

            class _R:
                status_code = 200
            return _R()

    return {
        "Dict": dict, "Any": object,
        "logger": logging.getLogger("test.pool"),
        "_DRAIN_REASONS_RELEASING": _constant("_DRAIN_REASONS_RELEASING"),
        "_DRAIN_REASONS_KNOWN": _constant("_DRAIN_REASONS_KNOWN"),
        "_offload": _offload,
        "_save_workspace_for_pool_release": _save_workspace_for_pool_release,
        "_strip_whatsapp_auth_from_host": _strip_whatsapp_auth_from_host,
        "_destroy_container": _destroy_container,
        "_whatsapp_logout_member": _whatsapp_logout_member,
        "_ahttp": lambda: _Http(),
        "get_admin_token": lambda: "tok",
        "DRAIN_TIMEOUT_S": 1,
    }


def _member(**kw):
    base = {
        "slot": 82, "port": 9082, "container_name": "toup-agent-pool-82",
        "assigned_prefix": "feed0082",
    }
    base.update(kw)
    return base


# ── the required keyword ──────────────────────────────────────────────


def test_reason_is_a_required_keyword_with_no_default():
    """A default would make the destructive/non-destructive choice implicit at
    every future call site. It has to be said out loud."""
    ns: dict = _drain_ns({})
    fn = _load("_drain_member", ns)
    sig = inspect.signature(fn)
    p = sig.parameters["reason"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty


def test_calling_without_a_reason_raises():
    ns: dict = _drain_ns({})
    fn = _load("_drain_member", ns)
    with pytest.raises(TypeError):
        asyncio.run(fn(_member()))


# ── the two directions ────────────────────────────────────────────────


def test_user_release_logs_out_and_strips_before_the_save_is_kept():
    rec: dict = {}
    fn = _load("_drain_member", _drain_ns(rec))
    asyncio.run(fn(_member(), reason="user_release"))

    assert rec.get("logged_out") == [82]
    assert rec.get("saved") == ["feed0082"]
    assert rec.get("stripped") == ["feed0082"]
    assert rec.get("destroyed") == ["toup-agent-pool-82"]


def test_the_logout_happens_before_the_workspace_is_copied_out():
    """Order: a save taken before the logout copies a LIVE session to the
    host, and the strip afterwards is then the only thing standing between
    that session and the next slot."""
    order: list = []
    rec: dict = {}
    ns = _drain_ns(rec)

    async def logout(member):
        order.append("logout")
    prev_save = ns["_save_workspace_for_pool_release"]

    def save(prefix, cn):
        order.append("save")
        return prev_save(prefix, cn)
    prev_strip = ns["_strip_whatsapp_auth_from_host"]

    def strip(prefix):
        order.append("strip")
        return prev_strip(prefix)

    ns["_whatsapp_logout_member"] = logout
    ns["_save_workspace_for_pool_release"] = save
    ns["_strip_whatsapp_auth_from_host"] = strip
    fn = _load("_drain_member", ns)
    asyncio.run(fn(_member(), reason="user_release"))

    assert order == ["logout", "save", "strip"]


def test_image_refresh_neither_logs_out_nor_strips():
    """THE expensive direction. A rollout that logged every tenant out would
    force the whole fleet to re-pair."""
    rec: dict = {}
    fn = _load("_drain_member", _drain_ns(rec))
    asyncio.run(fn(_member(), reason="image_refresh"))

    assert "logged_out" not in rec, "an image refresh logged the tenant out"
    assert "stripped" not in rec, "an image refresh deleted the tenant's creds"
    assert rec.get("saved") == ["feed0082"], "the workspace was not preserved"
    assert rec.get("destroyed") == ["toup-agent-pool-82"]


def test_an_unknown_reason_is_treated_as_non_releasing():
    """Fail SAFE: the mistake that costs a fleet-wide re-pair must not be the
    one a typo makes."""
    rec: dict = {}
    fn = _load("_drain_member", _drain_ns(rec))
    asyncio.run(fn(_member(), reason="something_new"))

    assert "logged_out" not in rec
    assert "stripped" not in rec


def test_a_generic_member_does_nothing_whatsapp_shaped():
    """No assigned_prefix = never bound = no tenant, no creds."""
    rec: dict = {}
    fn = _load("_drain_member", _drain_ns(rec))
    asyncio.run(fn(_member(assigned_prefix=None), reason="user_release"))

    assert "logged_out" not in rec
    assert "stripped" not in rec
    assert "saved" not in rec


# ── the reason mapping ────────────────────────────────────────────────


@pytest.mark.parametrize("cause,expected", [
    ("image_stale", "image_refresh"),
    ("surplus_downscale", "downscale"),
    ("released", "user_release"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_the_recorded_cause_maps_to_the_right_reason(cause, expected):
    fn = _load("_drain_reason_for", {"Dict": dict, "Any": object})
    assert fn({"last_error": cause}) == expected


def test_only_a_release_maps_onto_the_releasing_set():
    releasing = _constant("_DRAIN_REASONS_RELEASING")
    assert releasing == frozenset({"user_release"})


# ── F9d: the restore wipe ─────────────────────────────────────────────


def test_the_restore_wipes_the_container_creds_before_the_copy():
    """The inheriting case is precisely the one where the incoming user's host
    tree is EMPTY — so the wipe cannot be gated on the tree existing, and it
    must precede the `docker cp` merge."""
    body = None
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_restore_workspace_for_pool_bind":
            body = ast.unparse(node)
    assert body, "_restore_workspace_for_pool_bind not found"

    wipe = body.find("_wipe_container_whatsapp_auth(container_name)")
    early_return = body.find("src.mkdir(parents=True, exist_ok=True)")
    cp = body.find("['docker', 'cp'")
    assert wipe != -1, "a re-bound slot still inherits the last tenant's WhatsApp creds"
    assert early_return != -1 and cp != -1, "the restore body no longer matches"
    assert wipe < early_return, (
        "the wipe is skipped for a user with no saved workspace — which is "
        "exactly the inheriting case"
    )
    assert wipe < cp, "the wipe runs after the merge it exists to precede"


def test_the_restore_runs_the_wipe_against_the_workspace_path():
    ns = {
        "logger": logging.getLogger("test.pool"),
        "subprocess": __import__("subprocess"),
        "WORKSPACE_CONTAINER_PATH": "/app/workspace",
        "_WHATSAPP_AUTH_DIRNAME": _constant("_WHATSAPP_AUTH_DIRNAME"),
    }
    seen: list = []

    class _Sub:
        @staticmethod
        def run(cmd, **kw):
            seen.append(cmd)

            class _R:
                returncode = 0
                stderr = ""
            return _R()
    ns["subprocess"] = _Sub
    fn = _load("_wipe_container_whatsapp_auth", ns)
    fn("toup-agent-pool-82")

    assert seen and seen[0][:3] == ["docker", "exec", "toup-agent-pool-82"]
    assert "/app/workspace/.whatsapp_auth" in seen[0]


def test_an_empty_container_name_is_a_no_op():
    ns = {
        "logger": logging.getLogger("test.pool"),
        "WORKSPACE_CONTAINER_PATH": "/app/workspace",
        "_WHATSAPP_AUTH_DIRNAME": _constant("_WHATSAPP_AUTH_DIRNAME"),
    }
    calls: list = []

    class _Sub:
        @staticmethod
        def run(cmd, **kw):
            calls.append(cmd)
    ns["subprocess"] = _Sub
    fn = _load("_wipe_container_whatsapp_auth", ns)
    fn("")
    assert calls == []


def test_the_host_strip_removes_only_the_auth_dir(tmp_path):
    root = tmp_path / "feed0082" / "workspace"
    (root / ".whatsapp_auth").mkdir(parents=True)
    (root / ".whatsapp_auth" / "creds.json").write_text("{}")
    (root / "notes.md").write_text("keep me")

    ns = {
        "logger": logging.getLogger("test.pool"),
        "shutil": __import__("shutil"),
        "_workspace_host_path": lambda prefix: root,
        "_WHATSAPP_AUTH_DIRNAME": _constant("_WHATSAPP_AUTH_DIRNAME"),
    }
    fn = _load("_strip_whatsapp_auth_from_host", ns)

    assert fn("feed0082") is True
    assert not (root / ".whatsapp_auth").exists()
    assert (root / "notes.md").read_text() == "keep me", (
        "the strip took the user's files with it"
    )
    # Idempotent: a second release of the same slot must not raise.
    assert fn("feed0082") is False


# ── the call site ─────────────────────────────────────────────────────


def test_every_drain_call_site_passes_a_reason():
    calls = [
        ln for ln in _SRC.splitlines()
        if "_drain_member(" in ln and "async def" not in ln
    ]
    assert calls, "no _drain_member call sites found"
    for ln in calls:
        assert "reason=" in ln, f"call site without a reason: {ln.strip()}"


def test_the_snapshot_tick_logs_one_measurable_line():
    """The R46 F7 half this lane owns. On success this loop logged NOTHING, so
    its ~100 serial `docker cp`s could never be lined up against the bridge's
    `docker ps` timeouts — the 300 s cadence claim could be neither confirmed
    nor refuted from the trail."""
    assert "[pool] snapshot_tick members=%d saved=%d errors=%d ms=%d" in _SRC
    body = _SRC.split("async def _snapshot_tick", 1)[1].split("\nasync def ", 1)[0]
    assert "save_ms_p95" in body
    for token in ("prefix=%s", "assigned_prefix"):
        seg = body.split("[pool] snapshot_tick", 1)[1]
        assert token not in seg, "the tick line names a tenant"


# ── F9b: the route the bridge calls must be REACHABLE ─────────────────
#
# `/api/admin/whatsapp-logout` enforces its own `X-Pool-Admin-Token` inside
# the handler, exactly like bind/drain/status — but `AgentAPIKeyMiddleware`
# runs FIRST and is an EXACT-membership test against `_PUBLIC_PATHS`. The
# bridge holds no agent key, so without the entry every force-logout on a
# release answered 401 and the whole privacy fix was inert while
# `_whatsapp_logout_member` logged the status code as an outcome.


def _agent_main_public_paths() -> set:
    """Parsed, not imported: `agent_main` pulls the entire agent runtime."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "agent_main.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_PUBLIC_PATHS" for t in node.targets)
        ):
            call = node.value
            assert isinstance(call, ast.Call), "the _PUBLIC_PATHS literal moved"
            return {
                el.value for el in call.args[0].elts
                if isinstance(el, ast.Constant)
            }
    raise AssertionError("_PUBLIC_PATHS not found in agent_main.py")


def test_the_logout_route_bypasses_the_agent_key_middleware():
    assert "/api/admin/whatsapp-logout" in _agent_main_public_paths()


def test_the_bridge_sends_only_the_pool_admin_token():
    """Which is why the allowlist entry is the fix rather than "send the agent
    key too": the bridge does not hold the tenant's key."""
    i = _SRC.index('/api/admin/whatsapp-logout"')  # the POST url, not the docstring
    block = _SRC[i:i + 400]
    assert "X-Pool-Admin-Token" in block, block
    assert "X-Agent-Key" not in block, block


# ── fix lane A: the seams the cases above do not cross ────────────────


def _fn_src(name: str) -> str:
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found in bridge/pool_addon.py")


def test_the_agent_route_the_bridge_calls_is_public_to_the_key_middleware():
    """The bridge holds NO agent key — only `X-Pool-Admin-Token`.

    `AgentAPIKeyMiddleware` is an EXACT-membership test against
    `_PUBLIC_PATHS`, so a route missing from that frozenset answers 401 to
    every bridge call — and `_whatsapp_logout_member` swallows the status, so
    the F9 privacy fix would be inert while logging `http=401`. Parsed, not
    imported: `agent_main` is agent-mode and this file is platform-lane.
    """
    body = _fn_src("_whatsapp_logout_member")
    m = __import__("re").search(r"127\.0\.0\.1:\{port\}(/[A-Za-z0-9/_-]+)", body)
    assert m, "the bridge no longer posts to a literal agent path"
    path = m.group(1)

    agent_main_src = (pathlib.Path(__file__).resolve().parents[1] / "agent_main.py").read_text()
    public = None
    for node in ast.parse(agent_main_src).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_PUBLIC_PATHS" for t in node.targets
        ):
            public = eval(compile(ast.Expression(node.value), "<c>", "eval"))  # noqa: S307
    assert public is not None, "_PUBLIC_PATHS not found in agent_main.py"
    assert path in public, (
        f"the bridge posts {path} with only X-Pool-Admin-Token, but the agent's "
        f"X-Agent-Key middleware will 401 it — add it to _PUBLIC_PATHS"
    )
    # Also assert the header posture the route relies on.
    assert "X-Pool-Admin-Token" in body
    assert "X-Agent-Key" not in body


def test_the_drain_reasons_are_read_from_their_producers_not_retyped():
    """`_drain_reason_for` is pinned by hand-written literals everywhere else.

    Renaming a producer's `last_error` string routes every user release to
    `unknown` → non-releasing → no logout and no `.whatsapp_auth` strip, with
    every test green. So: take the literals from the CODE on both sides.
    """
    import re as _re

    reason_for = _load("_drain_reason_for", {"Dict": dict, "Any": object})

    # (a) what post_release actually records must be the releasing cause.
    rel = _re.findall(r"last_error=['\"]([a-z_]+)['\"]", _fn_src("post_release"))
    assert rel, "post_release no longer records a last_error"
    assert any(reason_for({"last_error": c}) == "user_release" for c in rel), (
        f"post_release records {rel}, none of which _drain_reason_for maps to "
        "user_release — every deletion would drain as non-releasing"
    )

    # (b) every cause the mapper knows must still be produced somewhere.
    known_causes = _re.findall(
        r"cause == ['\"]([a-z_]+)['\"]", _fn_src("_drain_reason_for")
    )
    assert known_causes, "_drain_reason_for compares against no literals"
    produced = set(_re.findall(r"last_error=['\"]([a-z_]+)['\"]", _SRC))
    for cause in known_causes:
        assert cause in produced, (
            f"_drain_reason_for still branches on {cause!r} but nothing writes "
            "it — the producer was renamed and the mapping is now dead"
        )


def test_the_release_route_saves_the_workspace_before_the_courtesy_logout():
    """ORDER and BUDGET.

    The platform calls `/v1/pool/release` under a 30 s client timeout whose
    failure aborts a user deletion ("the user's data staying on disk after
    they asked for it to be erased"). A 15 s blocking logout in FRONT of the
    workspace save — inside the same `try` — can push the route past that and
    can skip the save of an account about to be destroyed. The durable half
    (`_strip_whatsapp_auth_from_host` on the reap) does not need this order.
    """
    body = _fn_src("post_release")
    save = body.find("_save_workspace_for_pool_release")
    logout = body.find("_whatsapp_logout_member")
    assert save != -1 and logout != -1
    assert save < logout, (
        "the release route logs WhatsApp out before saving the workspace — a "
        "wedged agent then delays, and can abort, a deletion"
    )
    import re as _re
    m = _re.search(r"_whatsapp_logout_member\([^)]*timeout_s=([0-9.]+)", body)
    assert m, "the release path uses the reaper's full logout budget"
    assert float(m.group(1)) <= 5.0, "the courtesy logout budget is too generous"


def test_the_workspace_path_refuses_a_prefix_that_is_not_a_path_segment():
    """`prefix` reaches the registry straight from the `/v1/pool/claim` body
    with a truthiness check and nothing else, and since R46 it feeds a
    recursive delete (`_strip_whatsapp_auth_from_host`), not only `docker cp`.
    """
    import pathlib as _pl

    ns = {"WORKSPACE_HOST_BASE": _pl.Path("/data/agents"), "Path": _pl.Path}
    fn = _load("_workspace_host_path", ns)

    assert fn("feed0082") == _pl.Path("/data/agents/feed0082/workspace")
    for bad in ("../../etc", "a/b", "", None, ".", "..", "x" * 65, "a;rm -rf /"):
        with pytest.raises(ValueError):
            fn(bad)


def test_the_claim_body_is_where_the_prefix_shape_is_enforced():
    """The registry's copy of `prefix` is what every later consumer joins onto
    a host path — including the R46 recursive delete. The platform's only
    producer is `str(user_id)[:8]`, so the boundary can be exact."""
    import re as _re

    class _Router:
        def post(self, *_a, **_kw):
            return lambda fn: fn

    class _Req:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    ns = {
        "router": _Router(),
        "HTTPException": HTTPException,
        "Request": object,
        "Dict": dict,
        "Any": object,
        "_CLAIM_PREFIX_RE": _re.compile(r"[0-9a-f]{8}"),
    }
    fn = _load("post_claim", ns)

    for bad in ("../../etc", "FEED0082", "0ff148e", "feed00829", "a;rm -rf /", "zzzzzzzz"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(fn(_Req({"user_id": "u-1", "prefix": bad})))
        assert exc.value.status_code == 400, bad

    # And the shape the platform actually sends gets PAST the check (it fails
    # later, on the fakes this namespace does not provide — never on the shape).
    with pytest.raises(Exception) as exc2:
        asyncio.run(fn(_Req({"user_id": "u-1", "prefix": "feed0082"})))
    assert not (
        isinstance(exc2.value, HTTPException) and exc2.value.status_code == 400
    ), "a real platform prefix was refused by the boundary check"


def test_a_refused_logout_is_logged_as_a_failure_not_an_outcome():
    """401 is the shape of "the allowlist entry was dropped again": the agent's
    key middleware refusing the bridge before the route's own admin-token check
    runs. Logged at info as `http=401` it reads as success in the trail, which
    is how the F9 privacy fix could be inert and unnoticed."""
    seen: list[tuple[int, str]] = []

    class _Log:
        def info(self, msg, *a):
            seen.append((logging.INFO, msg % a if a else msg))

        def warning(self, msg, *a):
            seen.append((logging.WARNING, msg % a if a else msg))

        def exception(self, msg, *a):
            seen.append((logging.ERROR, msg % a if a else msg))

    def _http(code):
        class _Http:
            async def post(self, *_a, **_kw):
                class _R:
                    status_code = code
                return _R()
        return lambda: _Http()

    for code, level in ((200, logging.INFO), (401, logging.WARNING), (500, logging.WARNING)):
        seen.clear()
        ns = {
            "Dict": dict, "Any": object,
            "logger": _Log(),
            "_ahttp": _http(code),
            "get_admin_token": lambda: "tok",
        }
        fn = _load("_whatsapp_logout_member", ns)
        asyncio.run(fn(_member(), timeout_s=1.0))
        assert seen, f"http={code} logged nothing at all"
        assert seen[0][0] == level, f"http={code} logged at {seen[0][0]}: {seen[0][1]}"
