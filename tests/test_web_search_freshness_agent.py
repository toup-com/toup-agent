"""Agent side of the freshness fix (incident 2026-08-18).

  * cache policy by intent class — key carries the class, recency TTL ≤ 15
    min or bypass, and the spec's regression: a seeded stale (2023) entry for
    the same words can NEVER be served to a recency query.
  * `_gateway_search` forwards the class and renders page dates + header.
  * `_tool_web_search` prepends the cache verdict line and never returns a
    stale cached recency block.
  * `_brave_search_fallback` (container key) mirrors the ladder + stale filter.
  * the citation gate as applied by the runner (`apply_citation_gate`).

Run: cd backend && python3 -m pytest tests/test_web_search_freshness_agent.py -q
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

import app.agent.smart_fetch.search as S
import app.agent.tool_executor as TE
from app.agent.smart_fetch._cache import TTLCache


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    S._SEARCH_CACHE.clear()
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    monkeypatch.setattr(S.settings, "search_cache_ttl_s", 420)
    monkeypatch.setattr(S.settings, "search_cache_recency_ttl_s", 900)
    monkeypatch.setattr(S.settings, "search_cache_recency_bypass", False)
    monkeypatch.setattr(TE.settings, "search_freshness_enabled", True)
    yield
    S._SEARCH_CACHE.clear()


# ── Cache policy ─────────────────────────────────────────────────────

def test_cache_key_carries_the_freshness_class():
    assert S.cache_key("Anthropic  Newest Model", 8) == ("anthropic newest model", 8, "evergreen")
    assert S.cache_key("x", 8, "recent") != S.cache_key("x", 8, "evergreen")
    assert S.cache_key("x", 8, None) == S.cache_key("x", 8, "evergreen")


def test_ttlcache_per_entry_ttl_override():
    clock = {"t": 0.0}
    c = TTLCache(maxsize=8, ttl_s=1000, clock=lambda: clock["t"])
    c.set("long", 1)
    c.set("short", 2, ttl_s=10)
    clock["t"] = 11
    assert c.get("long") == 1 and c.get("short") is None


def test_recency_entries_expire_within_15_minutes(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(S, "_SEARCH_CACHE", TTLCache(maxsize=8, ttl_s=420, clock=lambda: clock["t"]))
    S.cache_set("anthropic newest model", 8, "1. R\n   http://x\n", "recent")
    S.cache_set("history of rome", 8, "1. R\n   http://y\n", "evergreen")
    clock["t"] = 419
    assert S.cache_get("anthropic newest model", 8, "recent") and S.cache_get("history of rome", 8, "evergreen")
    clock["t"] = 421
    assert S.cache_get("history of rome", 8, "evergreen") is None, "evergreen keeps the 7-min TTL"
    assert S.cache_get("anthropic newest model", 8, "recent"), "recent gets its own (longer, ≤15 min) TTL"
    clock["t"] = 901
    assert S.cache_get("anthropic newest model", 8, "recent") is None


def test_recency_ttl_never_exceeds_the_configured_bound(monkeypatch):
    monkeypatch.setattr(S.settings, "search_cache_recency_ttl_s", 99999)
    assert S._ttl_for("recent") <= 900 * 2, "bounded by 4× the evergreen TTL even if misconfigured"


def test_recency_bypass_flag_neither_reads_nor_writes(monkeypatch):
    monkeypatch.setattr(S.settings, "search_cache_recency_bypass", True)
    S.cache_set("anthropic newest model", 8, "payload", "recent")
    assert S.cache_get("anthropic newest model", 8, "recent") is None
    S.cache_set("history of rome", 8, "payload", "evergreen")
    assert S.cache_get("history of rome", 8, "evergreen") == "payload"


def test_seeded_stale_evergreen_result_cannot_surface_for_a_recency_query():
    """Spec regression: seed a fake cached 2023 result under the same words
    (as an older agent would have stored it) → a recency query must miss it."""
    stale = "1. Chatbot Arena Leaderboard Updates (Week 4) - LMSYS Org\n   https://www.lmsys.org/blog/2023-05-25-leaderboard/\n   2023\n"
    # legacy key shape (no class) and legacy value shape (raw string)
    S._SEARCH_CACHE.set(("lmsys chatbot arena leaderboard latest models", 8), stale)
    S._SEARCH_CACHE.set(("lmsys chatbot arena leaderboard latest models", 8, "evergreen"), (stale, 0.0))
    assert S.cache_get("LMSYS Chatbot Arena leaderboard latest models", 8, "recent") is None
    assert S.cache_get_meta("LMSYS Chatbot Arena leaderboard latest models", 8, "recent") is None


def test_cache_get_meta_returns_stored_at():
    S.cache_set("q", 8, "payload", "evergreen")
    val, stored_at = S.cache_get_meta("q", 8, "evergreen")
    assert val == "payload" and isinstance(stored_at, float) and stored_at > 0


def test_empty_sentinels_are_never_cached():
    S.cache_set("q", 8, "No results found.", "recent")
    S.cache_set("q2", 8, "No search results found across all engines.", "recent")
    assert S.cache_get("q", 8, "recent") is None and S.cache_get("q2", 8, "recent") is None


# ── _with_cache_line ─────────────────────────────────────────────────

def test_cache_line_is_prepended_and_parser_safe():
    out = TE._with_cache_line("1. T\n   https://a/1\n   s\n", "miss")
    assert out.startswith("cache: miss\n1. T")
    hit = TE._with_cache_line("1. T\n   https://a/1\n", "hit", stored_at=1.0)
    assert hit.startswith("cache: hit (stored ")
    assert TE._with_cache_line("", "hit") == ""


# ── _gateway_search renders dates ────────────────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status; self._p = payload
    def json(self): return self._p


class _Client:
    def __init__(self, result): self._r = result; self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, **kw):
        self.calls.append((url, kw)); return self._r


def _wire(monkeypatch, payload):
    c = _Client(_Resp(200, payload))
    monkeypatch.setattr(TE.settings, "toup_token", "tok", raising=False)
    monkeypatch.setattr(TE.settings, "platform_api_url", "https://toup.ai/api", raising=False)
    monkeypatch.setattr(TE.httpx, "AsyncClient", lambda **kw: c)
    TE._gw_fails = 0; TE._gw_skip_until = 0.0
    return c


_GW = {
    "served": True, "freshness_class": "recent", "freshness_applied": "pm", "dropped_stale": 1, "news_count": 1,
    "results": [
        {"title": "Introducing Claude Opus 5", "url": "https://www.anthropic.com/news/claude-opus-5",
         "description": "Today we're announcing Opus 5.", "extra_snippets": ["s1"],
         "age": "3 weeks ago", "page_age": "2026-07-24T00:00:00", "source": "web"},
        {"title": "News item", "url": "https://news.example/a", "description": "n", "age": "2 days ago", "source": "news"},
        {"title": "Old-style result (older gateway)", "url": "https://x.example/1", "description": "d"},
    ],
}


@pytest.mark.asyncio
async def test_gateway_search_forwards_the_class_and_renders_dates(monkeypatch):
    c = _wire(monkeypatch, _GW)
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    served, text, reason = await ex._gateway_search("anthropic newest model", 8, freshness_class="recent")
    assert served and reason is None
    assert c.calls[0][1]["json"]["freshness_class"] == "recent"
    assert 'Web results for "anthropic newest model" — 3 results · freshness: recent (Brave freshness=pm)' in text
    assert "1 result older than 18 months dropped" in text and "1 from news index" in text
    assert "1. Introducing Claude Opus 5\n   published: 2026-07-24 (" in text
    assert "https://www.anthropic.com/news/claude-opus-5" in text and "s1" in text
    assert "· news" in text
    assert "3. Old-style result (older gateway)\n   published: date unknown" in text


# ── _tool_web_search: cache hit path ─────────────────────────────────

@pytest.mark.asyncio
async def test_tool_web_search_serves_cached_recency_block_with_hit_line(monkeypatch):
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    S.cache_set("anthropic newest model", 8, "1. R\n   https://a/1\n   s\n", "recent")
    out = await ex._tool_web_search({"query": "anthropic newest model", "count": 8})
    assert out.startswith("cache: hit (stored ")
    assert "1. R" in out


@pytest.mark.asyncio
async def test_tool_web_search_does_not_serve_evergreen_cache_to_a_recency_query(monkeypatch):
    """Same words, different class → different key → the gateway is asked."""
    S.cache_set("anthropic newest model", 8, "1. STALE\n   https://old/1\n", "evergreen")
    c = _wire(monkeypatch, _GW)
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)   # channel/user come from ContextVars (unset here)
    out = await ex._tool_web_search({"query": "anthropic newest model", "count": 8})
    assert "STALE" not in out and "Introducing Claude Opus 5" in out
    assert out.startswith("cache: miss\n")
    assert c.calls, "gateway must have been consulted"


# ── _brave_search_fallback mirrors the ladder + filter ───────────────

class _GetClient:
    def __init__(self, pages): self.pages = pages; self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, **kw):
        self.calls.append(kw["params"])
        fr = kw["params"].get("freshness")
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: self.pages.get(fr, {"web": {"results": []}}))


def _page(*items):
    return {"web": {"results": [
        {"title": t, "url": u, "description": "d", "page_age": pa} for (t, u, pa) in items
    ]}}


@pytest.mark.asyncio
async def test_brave_fallback_widens_and_drops_stale(monkeypatch):
    pages = {
        "pm": _page(("thin", "https://a/1", "2026-08-01T00:00:00")),
        "py": _page(("fresh", "https://a/1", "2026-08-01T00:00:00"),
                    ("also", "https://a/2", "2026-07-01T00:00:00"),
                    ("stale-2023", "https://a/3", "2023-05-25T00:00:00"),
                    ("undated", "https://a/4", None)),
    }
    c = _GetClient(pages)
    monkeypatch.setattr(TE.httpx, "AsyncClient", lambda **kw: c)
    monkeypatch.setattr(TE.settings, "brave_api_key", "k")
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    out = await ex._brave_search_fallback("anthropic newest model", 8)
    assert [p.get("freshness") for p in c.calls] == ["pm", "py"]
    assert "stale-2023" not in out and "https://a/3" not in out
    assert "undated" in out and "date unknown" in out
    assert "freshness: recent (Brave freshness=py)" in out
    assert "1 result older than 18 months dropped" in out


@pytest.mark.asyncio
async def test_brave_fallback_evergreen_is_a_single_unfiltered_call(monkeypatch):
    c = _GetClient({None: _page(("r", "https://a/1", "2023-05-25T00:00:00"))})
    monkeypatch.setattr(TE.httpx, "AsyncClient", lambda **kw: c)
    monkeypatch.setattr(TE.settings, "brave_api_key", "k")
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    out = await ex._brave_search_fallback("history of rome", 8)
    assert c.calls == [{"q": "history of rome", "count": 8, "extra_snippets": "true"}]
    assert "https://a/1" in out and "freshness: evergreen" in out


# ── Citation gate as applied by the runner ───────────────────────────

def test_apply_citation_gate_rewrites_only_in_scope(monkeypatch):
    import app.agent.agent_runner as AR
    from app.websearch.citations import CitationGate
    monkeypatch.setattr(AR.settings, "citation_gate_scope", "web_turns", raising=False)
    monkeypatch.setattr(AR.settings, "citation_gate_mode", "mark", raising=False)
    g = CitationGate(); g.add_text("1. x\n   https://www.anthropic.com/news/claude-opus-5\n")
    ans = "See [Fable 5](https://www.anthropic.com/news/claude-fable-5) and [Opus 5](https://www.anthropic.com/news/claude-opus-5)."
    before = dict(AR._CITATION_GATE_COUNTERS)
    # web turn → rewritten
    out = AR.apply_citation_gate(g, ans, used_web_tool=True, user_id="u", channel="chat")
    assert "Fable 5 (unverified: https://www.anthropic.com/news/claude-fable-5)" in out
    assert "[Opus 5](https://www.anthropic.com/news/claude-opus-5)" in out
    # non-web turn → logged + counted, NOT rewritten
    out2 = AR.apply_citation_gate(g, ans, used_web_tool=False, user_id="u", channel="chat")
    assert out2 == ans
    assert AR._CITATION_GATE_COUNTERS["violations"] == before["violations"] + 2
    assert AR._CITATION_GATE_COUNTERS["turns_rewritten"] == before["turns_rewritten"] + 1
    # scope=all → rewritten regardless
    monkeypatch.setattr(AR.settings, "citation_gate_scope", "all", raising=False)
    assert "(unverified:" in AR.apply_citation_gate(g, ans, used_web_tool=False)


def test_apply_citation_gate_never_raises(monkeypatch):
    import app.agent.agent_runner as AR
    class Broken:
        size = 0
        def apply(self, *a, **k): raise RuntimeError("boom")
    assert AR.apply_citation_gate(Broken(), "text https://x/y", used_web_tool=True) == "text https://x/y"


def test_runner_wires_the_gate_structurally():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
    assert "_cite_gate.add_text(result" in src, "every tool result must ground the gate"
    assert "apply_citation_gate(" in src
    # applied BEFORE persistence and the done frame
    assert src.index("apply_citation_gate(\n                _cite_gate") < src.index("# ── Phase 3: Save to DB")
    assert "STALENESS RULE" in src and "CITATIONS: link only to URLs" in src


def test_flags_default_on_and_documented():
    from app.config import Settings
    f = Settings.model_fields
    assert f["search_freshness_enabled"].default is True
    assert f["search_news_blend_enabled"].default is True
    assert f["search_stale_filter_enabled"].default is True and f["search_stale_max_days"].default == 548
    assert f["citation_gate_enabled"].default is True and f["citation_gate_mode"].default == "mark"
    assert f["search_cache_recency_ttl_s"].default <= 900
    assert f["search_token_budget"].default >= 3000


# ── All stale: agent says so instead of falling through ──────────────

@pytest.mark.asyncio
async def test_all_stale_from_gateway_stops_the_ladder(monkeypatch):
    _wire(monkeypatch, {"served": False, "degraded_reason": "all_stale", "dropped_stale": 3, "freshness_class": "recent"})
    async def _boom(*a, **k):
        raise AssertionError("undated tiers must not run after all_stale")
    monkeypatch.setattr(S, "toup_search_meta", _boom)
    monkeypatch.setattr(TE.settings, "brave_api_key", None)
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    out = await ex._tool_web_search({"query": "LMSYS Chatbot Arena leaderboard latest models", "count": 8})
    assert out.startswith('No results newer than 18 months for "LMSYS Chatbot Arena leaderboard latest models"')
    assert "could not verify" in out
    assert S.cache_get("LMSYS Chatbot Arena leaderboard latest models", 8, "recent") is None, "never cached"


@pytest.mark.asyncio
async def test_brave_fallback_all_stale_says_so(monkeypatch):
    c = _GetClient({fr: _page(("old", "https://a/1", "2023-05-25T00:00:00")) for fr in ("pm", "py", None)})
    monkeypatch.setattr(TE.httpx, "AsyncClient", lambda **kw: c)
    monkeypatch.setattr(TE.settings, "brave_api_key", "k")
    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    out = await ex._brave_search_fallback("anthropic newest model", 8)
    assert out.startswith("No results newer than 18 months")
