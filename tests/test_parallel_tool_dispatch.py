"""Ticket 1 — parallelize intra-turn tool dispatch.

`agent_runner.run()` used to execute every tool call in a turn strictly
sequentially (`for tc in pending_tool_calls: await self.tools.execute(...)`),
so three `web_fetch`es ran back-to-back and a research turn paid the SUM of
their latencies. This change pre-executes the idempotent read-only tools
(`PARALLEL_SAFE_TOOLS`) concurrently under a semaphore, while the ordered
loop, stateful tools, single-tool turns, and result ordering are unchanged.

Behavioral tests exercise the real `_execute_tools_parallel` helper (it only
touches `self.tools.execute`, so a stub `self` is enough). Structural tests
match the project's source-grep convention (see test_agent_runner_tool_events)
to lock the loop-integration invariants without booting the runtime.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from types import SimpleNamespace

from app.agent.agent_runner import AgentRunner, PARALLEL_SAFE_TOOLS


_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()


def _stub_self(execute_fn):
    """Minimal object exposing only `self.tools.execute`, which is all
    `_execute_tools_parallel` reads off `self`."""
    return SimpleNamespace(tools=SimpleNamespace(execute=execute_fn))


def _tcs(n, name="web_fetch"):
    return [{"id": f"id{i}", "name": name, "input": {"u": i}} for i in range(n)]


# ── Behavioral: the gate ────────────────────────────────────────────────

def test_three_fetches_run_concurrently_not_serially():
    """GATE: a turn issuing 3 web_fetches completes in ~max(individual),
    not the sum, and all three are in flight at once."""
    active = {"now": 0, "max": 0}

    async def fake_execute(name, inp):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.2)
        active["now"] -= 1
        return f"result:{inp['u']}"

    t0 = time.perf_counter()
    res = asyncio.run(AgentRunner._execute_tools_parallel(_stub_self(fake_execute), _tcs(3), 6))
    elapsed = time.perf_counter() - t0

    assert active["max"] == 3, "all 3 fetches should be in flight simultaneously"
    assert elapsed < 0.45, f"expected ~0.2s (max), not ~0.6s (sum); got {elapsed:.2f}s"
    assert set(res.keys()) == {"id0", "id1", "id2"}, "every tool_use_id must get a result"
    assert res["id1"]["result"] == "result:1"
    for v in res.values():
        assert v["completed_ms"] >= v["started_ms"]


def test_one_failing_fetch_yields_error_result_others_succeed():
    """GATE: a single raising tool produces an ERROR result for ITS
    tool_use_id while the rest of the batch succeeds — never drop a block,
    never fail the whole batch."""
    async def fake_execute(name, inp):
        if inp["u"] == 1:
            raise RuntimeError("boom")
        return f"ok:{inp['u']}"

    res = asyncio.run(AgentRunner._execute_tools_parallel(_stub_self(fake_execute), _tcs(3), 6))

    assert res["id0"]["result"] == "ok:0"
    assert res["id2"]["result"] == "ok:2"
    assert res["id1"]["result"].startswith("ERROR: Tool crashed: RuntimeError")
    assert set(res.keys()) == {"id0", "id1", "id2"}, "failing task must still yield its block"


def test_concurrency_cap_is_respected():
    """The semaphore bounds in-flight calls so we don't hammer backends."""
    active = {"now": 0, "max": 0}

    async def fake_execute(name, inp):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        return "x"

    asyncio.run(AgentRunner._execute_tools_parallel(_stub_self(fake_execute), _tcs(10), 3))
    assert active["max"] <= 3, f"cap=3 must bound concurrency; saw {active['max']}"


def test_result_keys_cover_every_tool_use_id_in_order():
    """Result dict is keyed by tool_use_id for every input, so the unchanged
    in-order loop can pair results back to their tool_use blocks."""
    async def fake_execute(name, inp):
        return f"r{inp['u']}"

    tcs = _tcs(4)
    res = asyncio.run(AgentRunner._execute_tools_parallel(_stub_self(fake_execute), tcs, 6))
    assert [tc["id"] for tc in tcs] == list(res.keys())


# ── Structural: lock the loop-integration invariants ────────────────────

def test_parallel_safe_set_is_read_only_web_tools_only():
    assert PARALLEL_SAFE_TOOLS == frozenset({
        "web_search", "web_fetch",
        "extension_search", "extension_read", "extension_research",
    })
    # Stateful / mutating / legacy tools must NOT be parallelized.
    for stateful in (
        "browser", "browser_action", "browser_screenshot",
        "browser_session_start", "browser_session_end",
        "exec", "write_file", "edit_file",
    ):
        assert stateful not in PARALLEL_SAFE_TOOLS


def test_single_tool_turn_skips_parallel_path():
    """`> 1` guard means a 0- or 1-parallel-safe-tool turn pre-executes
    nothing → byte-identical to the old sequential behavior. Round 4 adds
    ONE exception: a lone web call is still batched when there is job
    bookkeeping (create_job/update_job) in the same round to overlap with."""
    assert "_run_batch = len(_parallel_tcs) > 1 or (bool(_parallel_tcs) and bool(_bk_tcs))" in _SRC


def test_loop_still_builds_results_in_tool_use_order():
    """Ordering invariant: the combined user-role message is still assembled
    by the in-order `for tc in pending_tool_calls` loop, paired by id."""
    assert "for tc in pending_tool_calls:" in _SRC
    assert '"tool_use_id": tc["id"]' in _SRC


def test_loop_consumes_precomputed_parallel_results():
    # Round 4: the batch coroutine is built once and either awaited alone or
    # overlapped with the bookkeeping chain; the ordered loop consumes both.
    # Round 8: the overlap is "batch as a Task, bookkeeping in the turn's own
    # coroutine" — never a gather over both (see test_round4_speed).
    assert "_batch_coro = self._execute_tools_parallel(" in _SRC
    assert "_batch_task = asyncio.ensure_future(_batch_coro)" in _SRC
    assert "await _run_bookkeeping()" in _SRC
    assert '_parallel_results.get(tc["id"])' in _SRC


def test_helper_uses_gather_and_bounded_semaphore():
    # Bound the slice by the NEXT method rather than a fixed character count:
    # the old `idx + 1500` window broke the moment the helper grew (adding the
    # voice tool-event emissions pushed `asyncio.gather(` to offset ~2700),
    # failing on a change that preserved the very invariant being asserted.
    idx = _SRC.index("async def _execute_tools_parallel(")
    nxt = re.search(r"\n    (?:async )?def ", _SRC[idx + 10:])
    body = _SRC[idx: idx + 10 + nxt.start()] if nxt else _SRC[idx:]
    assert "asyncio.gather(" in body
    assert "asyncio.Semaphore(" in body
