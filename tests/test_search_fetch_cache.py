"""Ticket 4 — per-tenant search/fetch TTL+LRU cache.

The smart_fetch search/fetch path had no caching, so identical queries/URLs
re-hit the network every time. This adds a tiny in-process TTL+LRU cache
(`smart_fetch/_cache.py`) wired into `toup_search` (keyed on normalized
query+count) and `toup_read_page` (keyed on requested url + final post-redirect
url, with max_chars), both behind default-on kill-switches. A generic pool
container can be re-bound to a new identity in-place via /admin/bind, so tenant
isolation is enforced by clear_caches() in the bind handler (run before the
bind commits); entries also TTL-expire and never persist across a restart.

Behavioral tests prove hit-avoids-network, TTL expiry, LRU eviction, empties
not cached, and flag-off behavior. Structural tests lock the PERF logging and
the kill-switch defaults.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.agent.smart_fetch.reader as R
import app.agent.smart_fetch.search as S
from app.agent.smart_fetch._cache import TTLCache

_SEARCH_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "smart_fetch" / "search.py").read_text()
_READER_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "smart_fetch" / "reader.py").read_text()


@pytest.fixture(autouse=True)
def _clear_module_caches():
    S._SEARCH_CACHE.clear()
    R._PAGE_CACHE.clear()
    yield
    S._SEARCH_CACHE.clear()
    R._PAGE_CACHE.clear()


# ── TTLCache unit ───────────────────────────────────────────────────────

def test_ttlcache_get_set_and_ttl_expiry():
    clock = {"t": 0.0}
    c = TTLCache(maxsize=10, ttl_s=5, clock=lambda: clock["t"])
    c.set("k", "v")
    assert c.get("k") == "v"
    clock["t"] = 4.9
    assert c.get("k") == "v", "still fresh just before TTL"
    clock["t"] = 5.0
    assert c.get("k") is None, "expired at TTL"
    assert len(c) == 0, "expired entry is dropped on access"


def test_ttlcache_lru_eviction():
    c = TTLCache(maxsize=2, ttl_s=1000, clock=lambda: 0.0)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # touch a -> b becomes LRU
    c.set("c", 3)           # evicts b
    assert c.get("a") == 1 and c.get("c") == 3
    assert c.get("b") is None


def test_ttlcache_clear():
    c = TTLCache(maxsize=4, ttl_s=1000, clock=lambda: 0.0)
    c.set("a", 1)
    c.clear()
    assert c.get("a") is None and len(c) == 0


# ── web_search cache ────────────────────────────────────────────────────

def _fake_search(calls, payload="1. R\n   http://x\n   snip\n"):
    async def fn(query, count=5):
        calls["n"] += 1
        return payload
    return fn


def test_search_cache_hit_avoids_network(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(S, "_toup_search_race", _fake_search(calls))
    monkeypatch.setattr(S.settings, "search_engine_race", True)
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)

    out1 = asyncio.run(S.toup_search("Hello   World", 5))
    out2 = asyncio.run(S.toup_search("hello world", 5))  # normalized to same key
    assert out1 == out2
    assert calls["n"] == 1, "second identical query must be served from cache"


def test_search_cache_keys_on_count(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(S, "_toup_search_race", _fake_search(calls))
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    asyncio.run(S.toup_search("q", 5))
    asyncio.run(S.toup_search("q", 8))  # different count -> different key -> miss
    assert calls["n"] == 2


def test_search_empty_results_not_cached(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(S, "_toup_search_race",
                        _fake_search(calls, "No search results found across all engines."))
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    asyncio.run(S.toup_search("q", 5))
    asyncio.run(S.toup_search("q", 5))
    assert calls["n"] == 2, "empty/no-result responses must not be cached"


def test_search_cache_flag_off(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(S, "_toup_search_race", _fake_search(calls))
    monkeypatch.setattr(S.settings, "search_cache_enabled", False)
    asyncio.run(S.toup_search("q", 5))
    asyncio.run(S.toup_search("q", 5))
    assert calls["n"] == 2, "flag off must disable caching"


# ── web_fetch / toup_read_page cache ────────────────────────────────────

_ARTICLE = (
    "<html><head><title>Cache Test</title></head><body><article><p>"
    + ("This is a sufficiently long article body to clear the JS-render heuristic. " * 4)
    + "</p></article></body></html>"
)


def _patch_fetch(monkeypatch, counter, *, final_url=None):
    class _Resp:
        def __init__(self, url):
            self.text = _ARTICLE
            self.headers = {"content-type": "text/html"}
            self.url = final_url or url
        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url, headers=None):
            counter["n"] += 1
            return _Resp(url)

    monkeypatch.setattr(R.httpx, "AsyncClient", _Client)


def test_fetch_cache_hit_avoids_network(monkeypatch):
    counter = {"n": 0}
    _patch_fetch(monkeypatch, counter)
    monkeypatch.setattr(R.settings, "fetch_cache_enabled", True)
    out1 = asyncio.run(R.toup_read_page("http://example.com/a", 10000))
    out2 = asyncio.run(R.toup_read_page("http://example.com/a", 10000))
    assert "article body" in out1 and out1 == out2
    assert counter["n"] == 1, "second fetch of same url must be served from cache"


def test_fetch_cache_dedups_final_redirect_url(monkeypatch):
    counter = {"n": 0}
    _patch_fetch(monkeypatch, counter, final_url="http://example.com/final")
    monkeypatch.setattr(R.settings, "fetch_cache_enabled", True)
    asyncio.run(R.toup_read_page("http://example.com/start", 10000))
    # the final post-redirect url is also cached -> a direct fetch hits
    asyncio.run(R.toup_read_page("http://example.com/final", 10000))
    assert counter["n"] == 1


def test_fetch_cache_flag_off(monkeypatch):
    counter = {"n": 0}
    _patch_fetch(monkeypatch, counter)
    monkeypatch.setattr(R.settings, "fetch_cache_enabled", False)
    asyncio.run(R.toup_read_page("http://example.com/a", 10000))
    asyncio.run(R.toup_read_page("http://example.com/a", 10000))
    assert counter["n"] == 2


# ── Structural ──────────────────────────────────────────────────────────

def test_perf_logging_present():
    assert "[PERF] web_search cache=hit" in _SEARCH_SRC
    assert "[PERF] web_search cache=miss" in _SEARCH_SRC
    assert "[PERF] web_fetch cache=hit" in _READER_SRC
    assert "[PERF] web_fetch cache=miss" in _READER_SRC


def test_caches_are_flag_gated():
    assert "if settings.search_cache_enabled:" in _SEARCH_SRC
    assert "if settings.fetch_cache_enabled:" in _READER_SRC


def test_cache_flags_default_on_killswitch():
    from app.config import Settings
    assert Settings.model_fields["search_cache_enabled"].default is True
    assert Settings.model_fields["fetch_cache_enabled"].default is True


def test_cache_module_documents_tenant_isolation():
    src = (Path(__file__).resolve().parent.parent / "app" / "agent" / "smart_fetch" / "_cache.py").read_text()
    assert "tenant" in src.lower()


def test_clear_caches_empties_both():
    from app.agent.smart_fetch import clear_caches
    S._SEARCH_CACHE.set(("q", 5), "x")
    R._PAGE_CACHE.set(("u", 1), "y")
    clear_caches()
    assert len(S._SEARCH_CACHE) == 0 and len(R._PAGE_CACHE) == 0


def test_bind_clears_caches_for_tenant_isolation():
    """Identity (re)bind must drop the per-process caches so a re-bound pool
    container can't serve the previous tenant's cached content."""
    bind_src = (Path(__file__).resolve().parent.parent / "app" / "api" / "admin_pool.py").read_text()
    assert "from app.agent.smart_fetch import clear_caches" in bind_src
    assert "clear_caches()" in bind_src
    # ordered after the identity apply
    assert bind_src.index("apply_to_settings") < bind_src.index("clear_caches()")
