"""Ticket 5 — rerank & filter search results.

Search results were emitted in raw engine order with no dedup/filter/rerank.
`_rerank.rerank_filter` (pure Python, no network/LLM) now dedups near-duplicate
URLs, drops useless (no title/url) results, and BM25-reranks by relevance to the
query before the model sees them, behind the `search_rerank_enabled` kill-switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import app.agent.smart_fetch.search as S
from app.agent.smart_fetch._rerank import _norm_url, rerank_filter

_RERANK_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "smart_fetch" / "_rerank.py").read_text()


@dataclass
class _R:
    title: str
    url: str
    snippet: str = ""


# ── dedup / filter ──────────────────────────────────────────────────────

def test_dedup_collapses_near_duplicate_urls():
    rs = [
        _R("A", "http://x.com/page"),
        _R("A dup", "https://www.x.com/page/"),   # https + www + trailing slash
        _R("A frag", "http://x.com/page#section"), # fragment
        _R("B", "http://y.com/other"),
    ]
    out = rerank_filter(rs, "")
    urls = [r.url for r in out]
    assert "http://x.com/page" in urls
    assert "http://y.com/other" in urls
    assert len(out) == 2, "x.com variants collapse to one; y.com kept"


def test_query_string_keeps_distinct_pages():
    rs = [_R("p1", "http://x.com/a?id=1"), _R("p2", "http://x.com/a?id=2")]
    out = rerank_filter(rs, "")
    assert len(out) == 2, "different query strings are distinct pages"


def test_drops_results_missing_title_or_url():
    rs = [_R("", "http://a.com"), _R("ok", ""), _R("good", "http://b.com")]
    out = rerank_filter(rs, "")
    assert [r.title for r in out] == ["good"]


# ── rerank ──────────────────────────────────────────────────────────────

def test_bm25_orders_relevant_first():
    rs = [
        _R("random page", "http://a.com", "nothing relevant here"),
        _R("python tutorial", "http://b.com", "learn python programming fast"),
        _R("cooking", "http://c.com", "recipes and food"),
    ]
    out = rerank_filter(rs, "python tutorial")
    assert out[0].url == "http://b.com", "most relevant result ranks first"


def test_ties_are_stable():
    # none of these contain the query terms -> all score 0 -> original order kept
    rs = [_R("one", "http://a"), _R("two", "http://b"), _R("three", "http://c")]
    out = rerank_filter(rs, "zzzz nonexistent")
    assert [r.title for r in out] == ["one", "two", "three"]


def test_empty_query_preserves_cleaned_order():
    rs = [_R("b", "http://b"), _R("a", "http://a")]
    out = rerank_filter(rs, "")
    assert [r.title for r in out] == ["b", "a"]


def test_single_result_passes_through():
    rs = [_R("only", "http://a", "x")]
    assert rerank_filter(rs, "anything") == rs


# ── integration with toup_search ────────────────────────────────────────

def _patch_engine(monkeypatch, results):
    async def fake_race(query, count=5):
        from app.agent.smart_fetch.search import _format_results, _ranked
        return _format_results(_ranked(results, query), count)
    monkeypatch.setattr(S, "_toup_search_race", fake_race)
    monkeypatch.setattr(S.settings, "search_engine_race", True)
    monkeypatch.setattr(S.settings, "search_cache_enabled", False)


def test_ranked_flag_off_preserves_order(monkeypatch):
    rs = [_R("random", "http://a", "x"), _R("python guide", "http://b", "python")]
    monkeypatch.setattr(S.settings, "search_rerank_enabled", False)
    out = S._ranked(rs, "python")
    assert out == rs, "flag off => raw engine order"


def test_ranked_flag_on_reranks(monkeypatch):
    rs = [_R("random", "http://a", "x"), _R("python guide", "http://b", "python tutorial")]
    monkeypatch.setattr(S.settings, "search_rerank_enabled", True)
    out = S._ranked(rs, "python tutorial")
    assert out[0].url == "http://b"


# ── structural ──────────────────────────────────────────────────────────

def test_rerank_is_pure_no_network_or_llm():
    for bad in ("httpx", "requests", "openai", "anthropic", "asyncio"):
        assert bad not in _RERANK_SRC, f"rerank must stay pure-Python; found {bad}"


def test_rerank_flag_defaults_on_killswitch():
    from app.config import Settings
    assert Settings.model_fields["search_rerank_enabled"].default is True


def test_norm_url_collapses_scheme_www_slash():
    assert _norm_url("http://www.x.com/a/") == _norm_url("https://x.com/a")
    assert _norm_url("http://x.com/a") != _norm_url("http://x.com/b")
