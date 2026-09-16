"""The instrument the 15 Sep incident had none of.

Container `toup-agent-pool-82` stopped logging ENTIRELY twice inside one turn —
9.47 s at 14:49:37.705, right after `[OPENAI] Client rebuilt`, and 20.86 s at
14:49:55.059, right after `llm_total` — while ~40 sibling containers on the
same host logged normally throughout. The only evidence either block left was
an ABSENCE of lines, including the ~10 s `/agent/health` poll, because nothing
in the codebase measures whether the loop is running.

Three properties decide whether `loop_health` is worth shipping:

  * it MEASURES the block (the sampler's overshoot), and
  * it TRIES to name it, and says how much to trust the name. The capture is
    a Python call from the watcher thread, so it needs the GIL: a
    GIL-RELEASING blocker is photographed mid-block and `top=` is the cause,
    while a GIL-HOLDING one (a lazy import, pydantic core-schema
    construction — the leading hypothesis for these freezes) freezes the
    watcher too and `top=` ends up naming the loop's RESUME point. `cap_ms` /
    `top_is_blocker` are that distinction, measured per stall; `lag_ms` is
    correct in both cases. And
  * it can never become the outage. Bounded memory, rate-limited logging, and
    `module:lineno` only — the raw faulthandler text carries absolute paths and
    function names and must never reach a log line.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_loop_health_sampler.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import re
import time

import pytest


def lh():
    # A plain import, never `importorskip`: FIRST-PARTY code that fails to
    # import must be a RED sweep, not silent skips reading green.
    from app.services import loop_health as mod

    return mod


@pytest.fixture(autouse=True)
def _clean():
    m = lh()
    m._reset_for_tests()
    yield
    asyncio.run(m.stop())
    m._reset_for_tests()


def test_snapshot_shape_before_start():
    """`/agent/health` renders this unconditionally. It must answer on a
    process where the sampler never started, and answer with zeros rather than
    None — a diagnostic that can raise in the health handler is worse than no
    diagnostic."""
    s = lh().snapshot()
    assert set(s) == {"running", "samples", "blocked_ms_max_30s", "blocked_ms_p99",
                      "stalls", "last_stall"}
    assert s["running"] is False
    assert s["samples"] == 0
    assert s["blocked_ms_max_30s"] == 0.0
    assert s["last_stall"] is None


def test_a_real_on_loop_block_is_measured():
    async def main():
        m = lh()
        assert m.start() is True
        await asyncio.sleep(0.6)
        time.sleep(1.2)                 # a genuine block, on the loop thread
        await asyncio.sleep(0.6)
        return m.snapshot()

    s = asyncio.run(main())
    assert s["running"] is True
    assert s["samples"] > 0
    # The sampler's overshoot is the block minus the sleep it asked for.
    # The LOWER bound is the assertion; the upper one only catches a sampler
    # that measures something unrelated. It is generous because the workflow's
    # own comments record that the toup-vps runners share the box with the
    # agent fleet and wall-clock doubles under contention — a tight ceiling
    # here is a flaky test, not a stronger one.
    assert 900 <= s["blocked_ms_max_30s"] <= 6000, s


def test_a_stall_names_itself_with_module_and_line_only():
    async def main():
        m = lh()
        m.start()
        await asyncio.sleep(0.4)
        time.sleep(2.6)                 # past loop_stall_warn_s (2.0)
        await asyncio.sleep(0.6)
        return m.snapshot()

    s = asyncio.run(main())
    assert s["stalls"] >= 1, s
    last = s["last_stall"]
    assert last is not None
    assert last["lag_ms"] >= 1900
    # THE PRIVACY SHAPE. `module:lineno`, nothing else — never a path, never a
    # function name, never a source line.
    assert re.fullmatch(r"[A-Za-z0-9_.\-]+:\d+", last["top"]), last["top"]
    assert "/" not in last["top"] and ".py" not in last["top"]
    # …and this test's own frame is what blocked, so the sampler must not be
    # blaming itself.
    assert not last["top"].startswith("loop_health:")
    assert last["at"].endswith("Z")


def test_start_is_idempotent_and_never_raises():
    async def main():
        m = lh()
        first = m.start()
        second = m.start()
        return first, second, m.snapshot()

    first, second, snap = asyncio.run(main())
    assert first is True and second is False, "a second start must not add a second sampler"
    assert snap["running"] is True


def test_stop_is_idempotent():
    async def main():
        m = lh()
        m.start()
        await asyncio.sleep(0.1)
        await m.stop()
        await m.stop()
        return m.snapshot()

    assert asyncio.run(main())["running"] is False


def test_the_sample_ring_is_bounded():
    """A process that runs for days must not grow a sample list."""
    m = lh()
    assert m._samples.maxlen is not None
    assert m._samples.maxlen <= 200
    for _ in range(10_000):
        m._samples.append((time.monotonic(), 1.0))
    assert len(m._samples) == m._samples.maxlen


def test_the_warning_is_rate_limited(caplog):
    """A wedged process must not flood the log store. Two stalls inside one
    report interval count twice and log once."""
    import logging

    m = lh()

    async def main():
        m.start()
        await asyncio.sleep(0.4)
        time.sleep(2.4)
        await asyncio.sleep(0.6)
        time.sleep(2.4)
        await asyncio.sleep(0.6)

    with caplog.at_level(logging.WARNING, logger="app.services.loop_health"):
        asyncio.run(main())
    lines = [r for r in caplog.records if "[LOOP_STALL]" in r.getMessage()]
    assert m.snapshot()["stalls"] >= 2, m.snapshot()
    assert len(lines) == 1, [r.getMessage() for r in lines]


def test_the_log_line_carries_counts_and_frames_only(caplog):
    import logging

    m = lh()

    async def main():
        m.start()
        await asyncio.sleep(0.4)
        time.sleep(2.4)
        await asyncio.sleep(0.6)

    with caplog.at_level(logging.WARNING, logger="app.services.loop_health"):
        asyncio.run(main())
    msgs = [r.getMessage() for r in caplog.records if "[LOOP_STALL] lag_ms=" in r.getMessage()]
    assert msgs, caplog.text
    line = msgs[0]
    # Forbidden shapes: absolute paths, .py filenames, quoted source.
    assert "/" not in line.split("stack=")[0]
    assert ".py" not in line
    assert "'" not in line and '"' not in line
    for key in ("lag_ms=", "threshold_ms=", "top=", "stalls="):
        assert key in line, line


def test_loop_ok_is_not_a_gate():
    """The critic's condition for shipping this at all: the normal
    distribution of loop lag on a 1.0-CPU cgroup has never been measured, so
    nothing may branch on it. `agent_main` renders `loop` as diagnostics and
    pins `turn_ready_detail.loop_ok` to True."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "agent_main.py"
    text = src.read_text()
    i = text.index('"loop_ok"')
    # The value on that key must be the literal True, not a comparison.
    assert re.match(r'"loop_ok":\s*True', text[i:i + 40]), text[i:i + 60]
    # …and `serving` must not mention it. Bounded to the ASSIGNMENT — the
    # payload built a few lines below legitimately carries a `loop` key, and a
    # fixed character window walked onto it the moment `serving` got shorter.
    j = text.index("serving = runner and")
    assert "loop" not in text[j:text.index("\n", j)]


def test_the_sampler_is_actually_started_by_the_agent_lifespan():
    """Nine green tests over a module nothing imports at runtime would be an
    instrument that never runs. Remove the two lines in agent_main's lifespan
    and every other assertion in this file still passes."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "agent_main.py").read_text()
    assert "from app.services import loop_health as _loop_health" in text
    assert "_loop_health.start()" in text
    # Inside the lifespan, not at import time: starting a sampler at import
    # would attach it to whatever loop happened to be running (or none).
    i = text.index("_loop_health.start()")
    assert "async def lifespan" in text[:i], "the start moved out of the lifespan"


def test_a_stall_says_whether_its_top_frame_is_the_blocker():
    """MEASURED, not assumed: `faulthandler.dump_traceback` is a PYTHON call,
    so the watcher thread must take the GIL to make it. For a GIL-HOLDING
    blocker — a lazy import, pydantic core-schema construction, the leading
    hypothesis for 15 Sep — the watcher is frozen too and the frames it
    finally returns name the loop's RESUME point. `cap_ms` is how a reader
    tells the two apart, and `top_is_blocker` is that judgement made once.

    This case blocks with a GIL-RELEASING `time.sleep`, so the capture is
    fast and the top frame really is the blocker."""
    async def main():
        m = lh()
        m.start()
        await asyncio.sleep(0.4)
        time.sleep(2.6)
        await asyncio.sleep(0.6)
        return m.snapshot()

    s = asyncio.run(main())
    last = s["last_stall"]
    assert last is not None
    assert "cap_ms" in last and "top_is_blocker" in last, last
    assert last["cap_ms"] >= 0.0
    assert last["top_is_blocker"] is True, last
