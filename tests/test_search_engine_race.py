"""Ticket 2 — stop the dead-engine tax & parallelize search backends.

Legacy `toup_search` tried 4 engines strictly sequentially (Whoogle ->
DuckDuckGo -> Bing -> Mojeek), each on a fresh client, so a slow engine
blocked the chain and a hung Whoogle could stall before DuckDuckGo. Behind
the `search_engine_race` flag, the primary engines now race concurrently over
a shared client and the first non-empty result wins (losers cancelled), with
Mojeek as a last resort. The flag defaults OFF (legacy behavior unchanged).

Behavioral tests monkeypatch the engine module-globals (resolved at call time
inside the race path) to prove concurrency, loser-cancellation, dead-engine
non-blocking, and the Mojeek fallback — no network. Structural tests match the
project's source-grep convention (see test_agent_runner_tool_events).
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

import app.agent.smart_fetch.search as S
from app.agent.smart_fetch.search import (
    SearchResponse,
    SearchResult,
    _first_with_results,
    _format_results,
    _toup_search_race,
    _toup_search_sequential,
    toup_search,
)


_SRC = (Path(__file__).resolve().parent.parent
        / "app" / "agent" / "smart_fetch" / "search.py").read_text()


def _resp(source, n=2):
    return SearchResponse(
        results=[SearchResult(f"{source}-{i}", f"http://{source}/{i}", "snip", source) for i in range(n)],
        source=source,
    )


def _engine(delay, source=None, *, raises=None, empty=False, completed=None):
    async def fn(query, count, client=None):
        await asyncio.sleep(delay)
        if completed is not None:
            completed.add(source)
        if raises is not None:
            raise raises
        if empty:
            return SearchResponse(results=[], source=source)
        return _resp(source)
    return fn


# ── Behavioral: the gate ────────────────────────────────────────────────

def test_race_returns_fastest_engine_and_cancels_slow_losers(monkeypatch):
    """GATE: the first non-empty engine wins in ~its own latency, not the sum;
    slower engines are cancelled (never run to completion)."""
    completed = set()
    monkeypatch.setattr(S, "_search_google_whoogle", _engine(2.0, "whoogle", completed=completed))
    monkeypatch.setattr(S, "_search_duckduckgo", _engine(0.05, "ddg", completed=completed))
    monkeypatch.setattr(S, "_search_bing", _engine(2.0, "bing", completed=completed))

    t0 = time.perf_counter()
    out = asyncio.run(_toup_search_race("q", 5))
    elapsed = time.perf_counter() - t0

    assert "ddg-0" in out, "fastest engine's results should be returned"
    assert elapsed < 0.5, f"should finish in ~0.05s (winner), not ~2s (losers); got {elapsed:.2f}s"
    assert "ddg" in completed
    assert "whoogle" not in completed and "bing" not in completed, "losers must be cancelled"


def test_dead_whoogle_does_not_block_the_race(monkeypatch):
    """GATE: a dead/refusing Whoogle no longer stalls the chain — a healthy
    engine racing alongside still returns promptly."""
    monkeypatch.setattr(S, "_search_google_whoogle", _engine(0.0, "whoogle", raises=httpx.ConnectError("refused")))
    monkeypatch.setattr(S, "_search_duckduckgo", _engine(0.05, "ddg"))
    monkeypatch.setattr(S, "_search_bing", _engine(2.0, "bing"))

    t0 = time.perf_counter()
    out = asyncio.run(_toup_search_race("q", 5))
    elapsed = time.perf_counter() - t0

    assert "ddg-0" in out
    assert elapsed < 0.5, f"dead Whoogle must not add latency; got {elapsed:.2f}s"


def test_mojeek_is_last_resort_when_all_primaries_empty(monkeypatch):
    """All primaries empty/error -> fall through to Mojeek (sequential)."""
    monkeypatch.setattr(S, "_search_google_whoogle", _engine(0.0, "whoogle", empty=True))
    monkeypatch.setattr(S, "_search_duckduckgo", _engine(0.0, "ddg", raises=RuntimeError("boom")))
    monkeypatch.setattr(S, "_search_bing", _engine(0.0, "bing", empty=True))
    monkeypatch.setattr(S, "_search_mojeek", _engine(0.02, "mojeek"))

    out = asyncio.run(_toup_search_race("q", 5))
    assert "mojeek-0" in out, "Mojeek should be used when primaries yield nothing"


def test_race_returns_no_results_when_everything_fails(monkeypatch):
    for name in ("_search_google_whoogle", "_search_duckduckgo", "_search_bing", "_search_mojeek"):
        monkeypatch.setattr(S, name, _engine(0.0, name, empty=True))
    out = asyncio.run(_toup_search_race("q", 5))
    assert out == "No search results found across all engines."


def test_first_with_results_skips_empty_and_errored(monkeypatch):
    """_first_with_results returns the first engine WITH results, skipping
    ones that error or return empty even if they finish first."""
    async def run():
        async with httpx.AsyncClient() as client:
            engines = [
                _engine(0.0, "a", empty=True),
                _engine(0.0, "b", raises=RuntimeError("x")),
                _engine(0.03, "c"),
            ]
            return await _first_with_results(engines, "q", 5, client)
    winner = asyncio.run(run())
    assert winner is not None and winner.source == "c"


def test_flag_off_uses_sequential_path(monkeypatch):
    """GATE: default (flag off) preserves the legacy sequential behavior."""
    calls = []

    async def fake_seq(q, c=5):
        calls.append("seq")
        return "SEQ"

    async def fake_race(q, c=5):
        calls.append("race")
        return "RACE"

    monkeypatch.setattr(S, "_toup_search_sequential", fake_seq)
    monkeypatch.setattr(S, "_toup_search_race", fake_race)
    # Isolate routing from the Ticket-4 result cache (which would otherwise
    # serve the 2nd identical query from cache and mask the race-path routing).
    monkeypatch.setattr(S.settings, "search_cache_enabled", False)

    monkeypatch.setattr(S.settings, "search_engine_race", False)
    assert asyncio.run(toup_search("q", 5)) == "SEQ"
    monkeypatch.setattr(S.settings, "search_engine_race", True)
    assert asyncio.run(toup_search("q", 5)) == "RACE"
    assert calls == ["seq", "race"]


def test_format_results_shape_is_unchanged():
    """Result shape (numbered title/url/snippet) must be byte-identical to the
    legacy formatter so the model contract is unchanged."""
    results = [SearchResult("T1", "http://u1", "S1", "x"), SearchResult("T2", "http://u2", "", "x")]
    out = _format_results(results, 5)
    assert out == "1. T1\n   http://u1\n   S1\n\n2. T2\n   http://u2\n"


def test_whoogle_connect_timeout_only_on_race_path(monkeypatch):
    """Flag-OFF parity: the 0.5s connect timeout is applied ONLY when a shared
    client is passed (race path); the legacy own-client path keeps the 15s
    default so sequential behavior is byte-identical."""
    class _Resp:
        text = "<html></html>"
        headers: dict = {}
        def raise_for_status(self):
            pass

    class _Capture:
        def __init__(self):
            self.calls = []
        async def get(self, url, **kwargs):
            self.calls.append(kwargs)
            return _Resp()

    cap = _Capture()

    @asynccontextmanager
    async def fake_cm(borrowed):
        yield cap

    monkeypatch.setattr(S, "_client", fake_cm)

    asyncio.run(S._search_google_whoogle("q", 5, client=None))
    assert "timeout" not in cap.calls[-1], "sequential path must not override the connect timeout"

    asyncio.run(S._search_google_whoogle("q", 5, client=cap))
    assert "timeout" in cap.calls[-1], "race path must apply the sub-second connect timeout"
    assert cap.calls[-1]["timeout"].connect == 0.5


def test_priority_winner_among_cocompleting_engines(monkeypatch):
    """When multiple engines complete in the same tick with results, the
    higher-priority engine (Whoogle > DDG > Bing) deterministically wins."""
    monkeypatch.setattr(S, "_search_google_whoogle", _engine(0.0, "whoogle"))
    monkeypatch.setattr(S, "_search_duckduckgo", _engine(0.0, "ddg"))
    monkeypatch.setattr(S, "_search_bing", _engine(0.0, "bing"))
    out = asyncio.run(_toup_search_race("q", 5))
    assert "whoogle-0" in out and "ddg-0" not in out


# ── Structural: lock invariants ─────────────────────────────────────────

def test_stale_priority_docstring_removed():
    assert "Priority: DuckDuckGo" not in _SRC, "stale toup_search docstring should be fixed"


def test_whoogle_has_subsecond_connect_timeout():
    assert "_WHOOGLE_CONNECT_TIMEOUT_S = 0.5" in _SRC
    assert "connect=_WHOOGLE_CONNECT_TIMEOUT_S" in _SRC


def test_race_path_uses_first_completed():
    assert "return_when=asyncio.FIRST_COMPLETED" in _SRC


def test_behavior_is_flag_gated():
    assert "if settings.search_engine_race:" in _SRC


def test_flag_defaults_on_killswitch():
    """The race is a kill-switch: default ON (flip off to disable)."""
    from app.config import Settings
    assert Settings.model_fields["search_engine_race"].default is True
