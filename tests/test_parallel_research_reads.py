"""Ticket 3 — parallelize server-side research page reads.

`ToolExecutor._fallback_research` (the extension_research server-side fallback)
read each result URL one-at-a-time in a `for` loop awaiting `toup_read_page`,
so a depth-N research call paid the SUM of page-read latencies. Behind the
`research_parallel_reads` kill-switch (default on), the pages are now read
concurrently under a semaphore (`research_read_concurrency`), with the per-URL
formatting and assembled output byte-identical to the legacy path.

Behavioral tests monkeypatch `toup_search`/`toup_read_page` (re-imported inside
the method at call time) and call the method with a dummy `self` (it touches no
instance state). Structural tests match the project's source-grep convention.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import app.agent.smart_fetch.reader as Sreader
import app.agent.smart_fetch.search as Ssearch
import app.agent.tool_executor as TE
from app.agent.tool_executor import ToolExecutor


_SRC = (Path(__file__).resolve().parent.parent
        / "app" / "agent" / "tool_executor.py").read_text()


def _serp(n):
    return "\n".join(f"{i}. Title {i}\n   http://site/{i}\n   snippet {i}" for i in range(n))


def _patch_search(monkeypatch, n=5):
    async def fake_search(query, count):
        return _serp(n)
    monkeypatch.setattr(Ssearch, "toup_search", fake_search)


def _patch_reader(monkeypatch, delay=0.2, *, fail_on=None, tracker=None):
    async def fake_read(url, per_page):
        if tracker is not None:
            tracker["now"] += 1
            tracker["max"] = max(tracker["max"], tracker["now"])
        await asyncio.sleep(delay)
        if tracker is not None:
            tracker["now"] -= 1
        if fail_on is not None and fail_on in url:
            raise RuntimeError("boom")
        return f"BODY {url}"
    monkeypatch.setattr(Sreader, "toup_read_page", fake_read)


def _run(inp):
    return asyncio.run(ToolExecutor._fallback_research(SimpleNamespace(), inp))


# ── Behavioral: the gate ────────────────────────────────────────────────

def test_depth5_reads_pages_concurrently(monkeypatch):
    """GATE: a depth=5 research call reads pages concurrently — all 5 in flight,
    total ~max(page) not sum."""
    tracker = {"now": 0, "max": 0}
    _patch_search(monkeypatch, 5)
    _patch_reader(monkeypatch, delay=0.2, tracker=tracker)
    monkeypatch.setattr(TE.settings, "research_parallel_reads", True)
    monkeypatch.setattr(TE.settings, "research_read_concurrency", 6)

    t0 = time.perf_counter()
    out = _run({"query": "q", "depth": 5, "per_page_chars": 4000})
    elapsed = time.perf_counter() - t0

    assert tracker["max"] == 5, "all 5 reads should be in flight at once"
    assert elapsed < 0.5, f"expected ~0.2s (max), not ~1.0s (sum); got {elapsed:.2f}s"
    for i in range(5):
        assert f"BODY http://site/{i}" in out


def test_concurrency_bounded_by_semaphore(monkeypatch):
    tracker = {"now": 0, "max": 0}
    _patch_search(monkeypatch, 10)  # depth caps at 10
    _patch_reader(monkeypatch, delay=0.05, tracker=tracker)
    monkeypatch.setattr(TE.settings, "research_parallel_reads", True)
    monkeypatch.setattr(TE.settings, "research_read_concurrency", 3)
    _run({"query": "q", "depth": 10, "per_page_chars": 4000})
    assert tracker["max"] <= 3, f"cap=3 must bound concurrency; saw {tracker['max']}"


def test_order_preserved_and_failures_isolated(monkeypatch):
    """Output order matches URL order; a single failing read yields its
    '(read failed: ...)' marker while the others succeed."""
    _patch_search(monkeypatch, 4)
    _patch_reader(monkeypatch, delay=0.0, fail_on="site/2")
    monkeypatch.setattr(TE.settings, "research_parallel_reads", True)
    out = _run({"query": "q", "depth": 4, "per_page_chars": 4000})

    positions = [out.index(f"## [{i}] http://site/{i - 1}") for i in range(1, 5)]
    assert positions == sorted(positions), "sections must stay in URL order"
    assert "(read failed: boom)" in out
    assert "BODY http://site/0" in out and "BODY http://site/3" in out


def test_flag_off_is_sequential_and_byte_identical(monkeypatch):
    """Kill-switch off → sequential reads (max concurrency 1) AND byte-identical
    output to the parallel path."""
    # parallel output
    _patch_search(monkeypatch, 3)
    _patch_reader(monkeypatch, delay=0.0)
    monkeypatch.setattr(TE.settings, "research_parallel_reads", True)
    out_parallel = _run({"query": "q", "depth": 3, "per_page_chars": 4000})

    # sequential output
    tracker = {"now": 0, "max": 0}
    _patch_search(monkeypatch, 3)
    _patch_reader(monkeypatch, delay=0.02, tracker=tracker)
    monkeypatch.setattr(TE.settings, "research_parallel_reads", False)
    out_seq = _run({"query": "q", "depth": 3, "per_page_chars": 4000})

    assert tracker["max"] == 1, "flag off must read one page at a time"
    assert out_parallel == out_seq, "output must be byte-identical regardless of the flag"


def test_no_urls_returns_search_summary(monkeypatch):
    async def fake_search(query, count):
        return "No search results found across all engines."
    monkeypatch.setattr(Ssearch, "toup_search", fake_search)
    out = _run({"query": "q", "depth": 5})
    assert "No search results" in out


# ── Structural: lock invariants ─────────────────────────────────────────

def test_reads_use_gather_and_semaphore():
    idx = _SRC.index("async def _fallback_research(")
    body = _SRC[idx: idx + 2500]
    assert "asyncio.gather(" in body
    assert "asyncio.Semaphore(" in body
    assert "settings.research_read_concurrency" in body


def test_reads_are_flag_gated():
    assert "if settings.research_parallel_reads:" in _SRC


def test_flag_defaults_on_killswitch():
    from app.config import Settings
    assert Settings.model_fields["research_parallel_reads"].default is True
