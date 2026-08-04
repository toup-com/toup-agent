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

def _fake_search(calls, payload="1. R\n   http://x\n   snip\n", engine="duckduckgo"):
    # The internal tier functions return (formatted, engine) since the
    # web-tool metering work — `toup_search_meta` needs to attribute a result
    # to a concrete upstream. `toup_search` still returns text only.
    async def fn(query, count=5):
        calls["n"] += 1
        return payload, engine
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
    """Stub the HTTP client AND the name resolution the SSRF guard performs.

    `_assert_public_url` calls `socket.getaddrinfo` on the initial URL and on
    every redirect hop, and raises if the host will not resolve. Patching only
    `httpx.AsyncClient` therefore was not enough: the guard rejected
    `example.com` before the stubbed client was ever reached, the fetch
    returned "", and the counter stayed at 0 — so three cache tests failed for
    a reason that has nothing to do with caching.

    A unit test of an in-process cache must not perform DNS. Resolve to a
    fixed PUBLIC address so the guard's real logic still runs: a private or
    loopback answer must still be rejected, which is what
    test_ssrf_guard_rejects_internal_addresses at the bottom of this file
    pins.
    """
    import socket as _socket

    def _fake_getaddrinfo(host, port, *a, **k):
        return [(_socket.AF_INET, _socket.SOCK_STREAM, _socket.IPPROTO_TCP, "",
                 ("93.184.216.34", port or 80))]

    monkeypatch.setattr(R.socket, "getaddrinfo", _fake_getaddrinfo)

    class _Resp:
        def __init__(self, url, *, redirect_to=None):
            self.text = _ARTICLE
            self.headers = {"content-type": "text/html"}
            if redirect_to:
                self.headers["location"] = redirect_to
            self.url = url
            # `_guarded_get` follows redirects BY HAND (the client is built
            # with follow_redirects=False so the SSRF guard can run on every
            # hop). It reads `.is_redirect`, which this stub never had — the
            # AttributeError was swallowed by toup_read_page's handler and
            # came back as an empty string, so the cache tests failed with
            # "assert 'article body' in ''" and no hint of the real cause.
            self.is_redirect = bool(redirect_to)

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
            if final_url and url != final_url:
                return _Resp(url, redirect_to=final_url)
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
    after_first = counter["n"]
    assert after_first >= 1, "the first fetch must actually hit the network"

    # The final post-redirect url is cached too, so fetching it DIRECTLY costs
    # nothing more. Asserted as a delta rather than an absolute count: the
    # redirect chain is walked one hop at a time now, so pinning a total would
    # be pinning the number of hops, which is not what this test is about.
    asyncio.run(R.toup_read_page("http://example.com/final", 10000))
    assert counter["n"] == after_first, (
        f"direct fetch of the final url should have been served from cache, "
        f"but requests went {after_first} -> {counter['n']}"
    )


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


def test_caches_are_flag_gated(monkeypatch):
    # The search gate moved out of an inline `if` in toup_search and into the
    # cache_get/cache_set helpers when the cache was hoisted above the Brave
    # API tier, so assert the gate BEHAVES rather than grepping for a literal
    # that a refactor can move. The reader still gates inline.
    assert "settings.search_cache_enabled" in _SEARCH_SRC
    assert "if settings.fetch_cache_enabled:" in _READER_SRC

    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    S.cache_set("gated query", 5, "1. R\n   http://x\n   snip\n")
    assert S.cache_get("gated query", 5) is not None

    monkeypatch.setattr(S.settings, "search_cache_enabled", False)
    assert S.cache_get("gated query", 5) is None, "flag off must bypass reads"
    S.cache_set("other query", 5, "payload")
    monkeypatch.setattr(S.settings, "search_cache_enabled", True)
    assert S.cache_get("other query", 5) is None, "flag off must bypass writes"


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


# ── SSRF guard — behaviour, not spelling ────────────────────────────────
#
# Until 2026-08-04 the only "coverage" of this guard was
# `assert "_assert_public_url" in rd` in test_security_builder_attribution
# — a source-text assertion that the guard is MENTIONED. It would pass
# against a body of `def _assert_public_url(url): return`. This is a control
# that stops injected content from pointing the agent at cloud metadata, the
# docker-bridge pgbouncer, the bridge admin API, or another tenant's
# container, so it deserves tests that would fail if it stopped working.

def _resolving_to(monkeypatch, ip: str):
    """Force name resolution to `ip` so the guard's real logic is exercised."""
    import socket as _socket

    def _gai(host, port, *a, **k):
        fam = _socket.AF_INET6 if ":" in ip else _socket.AF_INET
        return [(fam, _socket.SOCK_STREAM, _socket.IPPROTO_TCP, "", (ip, port or 80))]

    monkeypatch.setattr(R.socket, "getaddrinfo", _gai)


@pytest.mark.parametrize("ip,label", [
    ("127.0.0.1", "loopback"),
    ("10.0.0.5", "private/8"),
    ("172.16.4.4", "private/12"),
    ("192.168.1.1", "private/16"),
    ("169.254.169.254", "cloud metadata (link-local)"),
    ("100.64.1.1", "CGNAT / Tailscale"),
    ("0.0.0.0", "unspecified"),
    ("224.0.0.1", "multicast"),
    ("::1", "IPv6 loopback"),
])
def test_ssrf_guard_rejects_internal_addresses(monkeypatch, ip, label):
    _resolving_to(monkeypatch, ip)
    with pytest.raises(ValueError) as e:
        R._assert_public_url("http://totally-innocent.example/x")
    assert "internal address" in str(e.value), (
        f"{label} ({ip}) must be refused; got {e.value!r}"
    )


def test_ssrf_guard_allows_a_public_address(monkeypatch):
    _resolving_to(monkeypatch, "93.184.216.34")
    R._assert_public_url("https://example.com/x")  # must not raise


@pytest.mark.parametrize("scheme", ["file", "gopher", "ftp", "data"])
def test_ssrf_guard_rejects_non_http_schemes(monkeypatch, scheme):
    _resolving_to(monkeypatch, "93.184.216.34")
    with pytest.raises(ValueError) as e:
        R._assert_public_url(f"{scheme}://example.com/x")
    assert "unsupported URL scheme" in str(e.value)


def test_ssrf_guard_rejects_a_host_that_will_not_resolve(monkeypatch):
    import socket as _socket

    def _boom(*a, **k):
        raise _socket.gaierror("nope")

    monkeypatch.setattr(R.socket, "getaddrinfo", _boom)
    with pytest.raises(ValueError) as e:
        R._assert_public_url("http://nx.example/x")
    assert "cannot resolve host" in str(e.value)


def test_ssrf_guard_runs_on_every_redirect_hop(monkeypatch):
    """The attack this exists to stop: a PUBLIC url that redirects inward.

    Guarding only the initial URL would let `http://evil.example/` 302 to
    `http://169.254.169.254/latest/meta-data/` and hand the agent cloud
    credentials. `_guarded_get` re-runs the guard on each hop, so the second
    hop must raise even though the first was fine.
    """
    import socket as _socket
    hops = []

    def _gai(host, port, *a, **k):
        hops.append(host)
        ip = "169.254.169.254" if host == "169.254.169.254" else "93.184.216.34"
        return [(_socket.AF_INET, _socket.SOCK_STREAM, _socket.IPPROTO_TCP, "", (ip, port or 80))]

    monkeypatch.setattr(R.socket, "getaddrinfo", _gai)

    class _Redirect:
        is_redirect = True
        headers = {"location": "http://169.254.169.254/latest/meta-data/"}

    class _Client:
        async def get(self, url, headers=None):
            return _Redirect()

    with pytest.raises(ValueError) as e:
        asyncio.run(_guarded_get_probe(_Client(), "http://evil.example/start"))
    assert "internal address" in str(e.value), (
        f"a redirect to the metadata address must be refused; got {e.value!r}"
    )
    assert "169.254.169.254" in hops, "the guard never ran on the redirect target"


async def _guarded_get_probe(client, url):
    return await R._guarded_get(client, url, {}, max_redirects=3)
