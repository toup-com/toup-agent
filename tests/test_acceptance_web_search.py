"""Ticket 8 — end-to-end acceptance benchmark for the web-search overhaul.

Measures the MECHANICAL latency the overhaul removes, exercising the REAL code
paths (engine race, parallel intra-turn dispatch, parallel research reads, cache)
under baseline (flags off / legacy sequential) vs final (flags on), with a fixed
synthetic per-network-op latency so the numbers are deterministic and the
speedup is attributable to the change, not the network or the model.

Each scenario asserts a material speedup (acceptance gate) and prints a row that
feeds docs/web-search/acceptance-report.md. Model-round-trip reduction (Ticket 7)
is steering and can't be measured without a live model — verified by construction
and flagged for a live canary.

Run `pytest tests/test_acceptance_web_search.py -s` to see the numbers.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import app.agent.agent_runner as AR
import app.agent.smart_fetch.reader as R
import app.agent.smart_fetch.search as S
import app.agent.tool_executor as TE
from app.agent.agent_runner import AgentRunner

PER_OP = 0.15  # synthetic latency of one search/fetch network op (seconds)
_ROWS = []


def _record(scenario, baseline_s, final_s, note=""):
    speedup = baseline_s / final_s if final_s > 1e-6 else float("inf")
    _ROWS.append((scenario, baseline_s, final_s, speedup, note))


# ── Scenario A: intra-turn tool batch (3 web_fetch) ─────────────────────

def test_a_parallel_intra_turn_dispatch(monkeypatch):
    async def slow_exec(name, inp):
        await asyncio.sleep(PER_OP)
        return "word " * 20

    n = 3
    # baseline: sequential (pre-Ticket-1)
    async def baseline():
        t0 = time.perf_counter()
        for _ in range(n):
            await slow_exec("web_fetch", {})
        return time.perf_counter() - t0

    # final: real _execute_tools_parallel
    monkeypatch.setattr(AR.settings, "web_token_budget_enabled", False)  # isolate dispatch timing
    stub = SimpleNamespace(tools=SimpleNamespace(execute=slow_exec))
    tcs = [{"id": f"id{i}", "name": "web_fetch", "input": {}} for i in range(n)]

    base = asyncio.run(baseline())
    t0 = time.perf_counter()
    asyncio.run(AgentRunner._execute_tools_parallel(stub, tcs, 6))
    final = time.perf_counter() - t0

    _record("A. 3 web_fetch in one turn", base, final, "sequential→parallel")
    assert final < base * 0.55, f"parallel batch should be ~max not sum: base={base:.2f} final={final:.2f}"


# ── Scenario B: search engine race (dead Whoogle + slow DDG + fast Bing) ─

def test_b_engine_race(monkeypatch):
    def eng(delay, source, *, dead=False):
        async def fn(query, count, client=None):
            if dead:
                # Real engines CATCH their network error and return empty, never
                # raise (the sequential path relies on this). Whoogle absent =
                # instant empty.
                return S.SearchResponse(results=[], source=source, error="refused")
            await asyncio.sleep(delay)
            return S.SearchResponse(
                results=[S.SearchResult(f"{source}", f"http://{source}", "snip", source)],
                source=source,
            )
        return fn

    monkeypatch.setattr(S, "_search_google_whoogle", eng(0, "whoogle", dead=True))
    monkeypatch.setattr(S, "_search_duckduckgo", eng(PER_OP * 3, "ddg"))   # slow primary
    monkeypatch.setattr(S, "_search_bing", eng(PER_OP, "bing"))            # fast
    monkeypatch.setattr(S, "_search_mojeek", eng(PER_OP, "mojeek"))
    monkeypatch.setattr(S.settings, "search_cache_enabled", False)

    seq_out = {}
    race_out = {}
    base = _time(lambda: seq_out.update(r=asyncio.run(S._toup_search_sequential("q", 5))))
    final = _time(lambda: race_out.update(r=asyncio.run(S._toup_search_race("q", 5))))

    _record("B. search w/ dead+slow engine", base, final, "sequential→race")

    # Assert the CLAIM, not the clock.
    #
    # This used to be `final < base * 0.6`, and it went red in CI on
    # 2026-08-03 with base=0.45 final=0.31 — nothing was broken, the runner was
    # just busy (the sweep runs three pytest processes at once, and 0.16s of
    # scheduling noise is enough to eat a 0.15s-vs-0.45s margin). A wall-clock
    # ratio measured under parallel load is a coin toss, and a test that goes
    # red when the machine is busy teaches people to ignore it.
    #
    # What "race should return the fastest engine" actually MEANS is that the
    # winner is bing (0.15s) and not ddg (0.45s) — which is deterministic.
    assert race_out["r"][1] == "bing", (
        f"race must return the FASTEST engine's result, got {race_out['r'][1]!r}"
    )
    assert seq_out["r"][1] == "ddg", (
        "the sequential baseline must still be the slow primary — otherwise "
        f"this scenario is not measuring what it claims, got {seq_out['r'][1]!r}"
    )
    # Timing kept as a soft floor only: the race must not WAIT for the slow
    # engine. Bounded well away from scheduling noise rather than at 0.6x.
    assert final < base, f"race should not be slower: base={base:.2f} final={final:.2f}"


# ── Scenario C: cache (repeat query) ────────────────────────────────────

def test_c_cache_repeat_query(monkeypatch):
    calls = {"n": 0}

    async def fake_race(query, count=5):
        calls["n"] += 1
        await asyncio.sleep(PER_OP)
        # (formatted, engine) — the tier contract since web-tool metering
        # needed to attribute each result to a concrete upstream.
        return "1. R\n   http://x\n   snip\n", "duckduckgo"

    monkeypatch.setattr(S, "_toup_search_race", fake_race)
    monkeypatch.setattr(S.settings, "search_engine_race", True)
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    S._SEARCH_CACHE.clear()

    base = _time(lambda: asyncio.run(S.toup_search("repeat me", 5)))    # miss → network
    final = _time(lambda: asyncio.run(S.toup_search("repeat me", 5)))   # hit → no network

    _record("C. repeat query (cache)", base, final, "network→cache hit")
    assert calls["n"] == 1, "second identical query must be served from cache"
    assert final < base * 0.2, f"cache hit should be ~free: base={base:.3f} final={final:.4f}"


# ── Scenario D: server-side research reads (5 pages) ─────────────────────

def test_d_parallel_research_reads(monkeypatch):
    async def fake_search(query, count):
        return "\n".join(f"{i}. T\n   http://site/{i}\n   s" for i in range(5))

    async def fake_read(url, per_page):
        await asyncio.sleep(PER_OP)
        return f"BODY {url}"

    monkeypatch.setattr(S, "toup_search", fake_search)
    monkeypatch.setattr(R, "toup_read_page", fake_read)
    self_stub = SimpleNamespace()

    monkeypatch.setattr(TE.settings, "research_parallel_reads", False)
    base = _time(lambda: asyncio.run(TE.ToolExecutor._fallback_research(self_stub, {"query": "q", "depth": 5})))
    monkeypatch.setattr(TE.settings, "research_parallel_reads", True)
    final = _time(lambda: asyncio.run(TE.ToolExecutor._fallback_research(self_stub, {"query": "q", "depth": 5})))

    _record("D. 5-page research reads", base, final, "sequential→parallel")
    assert final < base * 0.5, f"parallel reads should be ~slowest page: base={base:.2f} final={final:.2f}"


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_z_print_summary():
    """Print the acceptance table (visible with -s). Runs last (z prefix)."""
    print("\n\n=== Web-search acceptance: baseline (legacy) vs final (overhaul) ===")
    print(f"{'scenario':<34}{'baseline':>10}{'final':>10}{'speedup':>10}   note")
    for sc, b, f, sp, note in _ROWS:
        print(f"{sc:<34}{b*1000:>8.0f}ms{f*1000:>8.0f}ms{sp:>9.1f}x   {note}")
    print(f"(synthetic per-op latency = {PER_OP*1000:.0f}ms; real network ops are typically 300–2000ms, so absolute gaps scale up)")
    assert _ROWS, "scenarios must have populated the summary"
