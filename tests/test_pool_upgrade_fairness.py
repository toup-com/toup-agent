"""`/upgrade-assigned` must not starve the tail of the member list.

MEASURED ON PRODUCTION, 2026-08-07
----------------------------------
Seven ASSIGNED pool members sat on a 24-hour-old image across two consecutive
rollouts while the other 48 containers converged. They were indices 33-39 of
40 in `_load_members()` order — the contiguous tail, not a scatter — and all
seven reported `healthy`, so nothing alarmed. Agent-side fixes (including a
whole memory-remediation arc) simply were not reaching those users.

Mechanism: `post_upgrade_assigned` took `candidates[:limit]` in registry file
order. Within ONE rollout that self-advances, because an upgraded member stops
matching (`image_tag == image_tag`). But a NEW rollout retargets the tag, every
member becomes a candidate again, and the walk restarts at index 0. The tail is
therefore only reached in the gaps between rollouts — and it takes ~30 minutes
while merges arrive roughly every 20.

This is indefinite DEFERRAL, not permanent starvation: once merges paused, the
7 drained. That is what makes it hard to see — it self-heals whenever the repo
goes quiet, so it never looks broken, and it returns exactly when shipping is
busiest. The simulation below therefore models sustained retargeting, which is
the condition under which it actually bites.

The fix orders candidates least-recently-upgraded first, using a
`last_upgraded_at` stamp written only on a SUCCESSFUL upgrade.

Parsed rather than imported: `pool_addon.py` is bridge-host code whose imports
(docker, psql helpers) do not exist in the backend test environment — the same
constraint `test_bridge_feature_flag_forwarding.py` works around.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"


def _load_orderer():
    """Extract `order_upgrade_candidates` from the bridge source and exec it.

    This runs THE SHIPPED FUNCTION, not a reimplementation of it — a test that
    re-derives the ordering would pass even if the bridge stopped calling it.
    """
    tree = ast.parse(BRIDGE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "order_upgrade_candidates":
            module = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(module, str(BRIDGE), "exec"), ns)  # noqa: S102
            return ns["order_upgrade_candidates"]
    raise AssertionError("order_upgrade_candidates not found in bridge/pool_addon.py")


@pytest.fixture(scope="module")
def order():
    return _load_orderer()


def _member(slot: str, attempted=None, upgraded=None) -> dict:
    m = {"slot": slot, "state": "ASSIGNED",
         "image_tag": "ghcr.io/toup-com/toup-agent:old"}
    if attempted is not None:
        m["last_upgrade_attempt_at"] = attempted
    if upgraded is not None:
        m["last_upgraded_at"] = upgraded
    return m


def test_never_attempted_members_come_first(order):
    """A member with no stamp outranks every member that has one."""
    members = [_member("01", 5000), _member("02", 9000), _member("03")]
    assert [m["slot"] for m in order(members)] == ["03", "01", "02"]


def test_oldest_stamp_first(order):
    members = [_member("a", 300), _member("b", 100), _member("c", 200)]
    assert [m["slot"] for m in order(members)] == ["b", "c", "a"]


def test_legacy_success_stamp_is_honoured(order):
    """Members upgraded before the attempt clock existed carry only
    `last_upgraded_at`. They must not all pin to 0 and re-run ahead of
    members that were attempted more recently — the key is the LATER of
    the two clocks."""
    members = [
        _member("01", attempted=None, upgraded=9000),   # legacy, recent
        _member("02", attempted=100),                   # attempted long ago
    ]
    assert [m["slot"] for m in order(members)] == ["02", "01"]


def test_attempt_clock_wins_when_later_than_success(order):
    """A member tried recently but which FAILED (attempt newer than success)
    must sort behind one whose last success is newer than any attempt."""
    members = [
        _member("fails", attempted=9000, upgraded=100),
        _member("ok", attempted=200, upgraded=500),
    ]
    assert [m["slot"] for m in order(members)] == ["ok", "fails"]


def test_ties_break_deterministically_on_slot(order):
    """Before this ships every member has no stamp, so ALL of them tie.
    The order must still be stable and not registry-serialization order."""
    members = [_member("30"), _member("07"), _member("19")]
    assert [m["slot"] for m in order(members)] == ["07", "19", "30"]


@pytest.mark.parametrize("bad", [None, "", "not-a-number", {}, []])
def test_unparseable_stamp_is_treated_as_never_upgraded(order, bad):
    """A corrupt stamp must sort to the FRONT, never crash the endpoint.
    Sorting it to the back would silently starve exactly the member whose
    registry row is damaged."""
    members = [_member("01", 5000), _member("02", bad)]
    assert [m["slot"] for m in order(members)][0] == "02"


def test_tail_of_a_retargeting_fleet_is_eventually_served(order):
    """The production failure, simulated.

    40 members, 2 upgraded per call, and the target tag CHANGES before the
    tail is reached — the burst-merge pattern that starved slots 33-39.

    Under file order this loops forever without ever serving the tail. Under
    least-recently-upgraded order every member is served.
    """
    clock = [1000]

    return _simulate(order, clock, members=[_member(f"{i:02d}") for i in range(40)],
                     retarget_every=5, calls=100, always_fail=frozenset(),
                     expect_all=True)


def _simulate(order, clock, members, retarget_every, calls, always_fail,
              expect_all):
    """Replay the production loop.

    IMPORTANT — candidates are rebuilt from the registry on EVERY call, which
    is what `post_upgrade_assigned` actually does: it re-reads `_load_members()`
    and filters on `image_tag == image_tag`. An earlier version of this test
    kept a local candidate list and popped served members from it, which
    quietly made head-of-line blocking impossible to reproduce — a failing
    member disappeared from the local list even though in production it stays
    a candidate forever (its image_tag never becomes the target).

    That modelling error made the test pass against the WRONG implementation.
    Rebuilding per call is the whole point.
    """
    target = "new"
    for m in members:
        m["image_tag"] = "old"
    served = set()
    for i in range(calls):
        if i % retarget_every == 0:
            target = f"new{i}"                  # a fresh merge retargets
        # Exactly post_upgrade_assigned's filter, re-evaluated per call.
        candidates = [m for m in members if m["image_tag"] != target]
        if not candidates:
            continue
        for m in order(candidates)[:2]:
            clock[0] += 1
            m["last_upgrade_attempt_at"] = clock[0]      # finally: always
            if m["slot"] not in always_fail:
                m["last_upgraded_at"] = clock[0]         # success only
                m["image_tag"] = target
                served.add(m["slot"])
    expected = {m["slot"] for m in members} - set(always_fail)
    if expect_all:
        assert served == expected, (
            "never served: " + ", ".join(sorted(expected - served)))
    return served


def test_permanently_failing_members_do_not_block_everyone_else(order):
    """The reason the key is last-ATTEMPT and not last-SUCCESS.

    Two members whose upgrade always fails stay candidates forever, because
    their `image_tag` never reaches the target. Under success-ordering their
    stamp never advances either, so they sort first on EVERY call, consume
    both slots, and nothing else ever converges.

    Attempt-ordering rotates them to the back the moment they are tried, so
    they are retried after everyone else rather than instead of everyone else.
    """
    clock = [1000]
    members = [_member(f"{i:02d}") for i in range(20)]
    served = _simulate(order, clock, members, retarget_every=5, calls=60,
                       always_fail={"00", "01"}, expect_all=False)
    healthy = {f"{i:02d}" for i in range(20)} - {"00", "01"}
    assert served == healthy, (
        "two permanently-failing members blocked: " +
        ", ".join(sorted(healthy - served)))


def test_file_order_would_starve_the_tail():
    """Anti-vacuity control for the test above.

    Same simulation with the OLD behaviour (take the first N in list order,
    no staleness rotation) must FAIL to serve the tail. Without this, the
    test above could pass for reasons unrelated to the fix.
    """
    members = [{"slot": f"{i:02d}"} for i in range(40)]
    served = set()
    for i in range(100):
        if i % 5 == 0:
            candidates = list(members)
        batch = candidates[:2]                      # file order, no sort
        for m in batch:
            served.add(m["slot"])
        candidates = candidates[2:]
    assert served != {f"{i:02d}" for i in range(40)}
    assert "39" not in served, "the tail should be starved under file order"


def test_successful_upgrade_stamps_the_success_clock():
    """Without this the ordering has nothing to order by and degrades to a
    permanent tie broken on slot."""
    src = BRIDGE.read_text()
    assert "last_upgraded_at=int(time.time())" in src, (
        "the successful-upgrade path must stamp last_upgraded_at")


def test_attempt_clock_is_stamped_in_a_finally():
    """The attempt clock MUST be written on every exit path.

    If it were stamped only on success, a member that always fails would sort
    first forever and burn a slot every call. If it were stamped in the
    endpoint rather than the guarded entry point, the reconciler-driven
    upgrade path would never stamp at all. Both are silent failures, so this
    asserts the structure and not just the string.
    """
    tree = ast.parse(BRIDGE.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_upgrade_assigned_member"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Try) and sub.finalbody:
                if "last_upgrade_attempt_at" in ast.unparse(ast.Module(
                        body=sub.finalbody, type_ignores=[])):
                    return
        raise AssertionError(
            "_upgrade_assigned_member must stamp last_upgrade_attempt_at in "
            "its finally block, so failures and timeouts rotate too")
    raise AssertionError("_upgrade_assigned_member not found")


def test_endpoint_actually_calls_the_orderer():
    """The ordering helper is only worth testing if the endpoint uses it."""
    tree = ast.parse(BRIDGE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "post_upgrade_assigned":
            called = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "order_upgrade_candidates" in called
            return
    raise AssertionError("post_upgrade_assigned not found")
