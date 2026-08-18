"""
Toup Search — Multi-engine search without a browser.

Queries multiple search engines — Google first (best results), then fallbacks:
  1. Google (via local Whoogle instance — self-hosted Google proxy, no CAPTCHA)
  2. DuckDuckGo HTML (no CAPTCHA, no API key)
  3. Bing Web Search (lighter bot detection)
  4. Mojeek (independent engine, no CAPTCHA)

Whoogle is a self-hosted Google frontend that queries Google server-side
without triggering CAPTCHAs. It is present only on self-hosted-VPS agents
(see ssh_deploy_service); managed pool/containers have no Whoogle, so its
attempt fails instantly (connection refused) and is skipped — when racing is
enabled it never blocks the other engines either way.

All engines are queried via simple HTTP requests with proper headers.
No browser, no headless Chrome, no Playwright — just httpx.
"""

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.agent.smart_fetch._cache import TTLCache
from app.agent.smart_fetch._rerank import rerank_filter

logger = logging.getLogger(__name__)

# A dead or hung local Whoogle must never stall a search. A refused connection
# on localhost already fails instantly; this only caps the rarer "listening but
# hung" case so the race below can't be held up by it.
_WHOOGLE_CONNECT_TIMEOUT_S = 0.5

# Per-tenant TTL+LRU cache of formatted results. Cleared on /admin/bind
# (smart_fetch.clear_caches) so an in-place re-bind can't leak across tenants.
_SEARCH_CACHE = TTLCache(maxsize=settings.search_cache_max, ttl_s=settings.search_cache_ttl_s)


@asynccontextmanager
async def _client(borrowed: Optional[httpx.AsyncClient]):
    """Yield a shared pooled client when one is passed (race path), else a
    short-lived own client (legacy sequential path). A borrowed client is
    never closed here — the caller owns its lifecycle."""
    if borrowed is not None:
        yield borrowed
    else:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as own:
            yield own

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""  # which engine returned this


@dataclass
class SearchResponse:
    results: List[SearchResult] = field(default_factory=list)
    source: str = ""
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Google via Whoogle (primary — best results, self-hosted proxy)
# ──────────────────────────────────────────────────────────────

WHOOGLE_URL = "http://127.0.0.1:5000"


async def _search_google_whoogle(
    query: str, count: int = 10, client: Optional[httpx.AsyncClient] = None
) -> SearchResponse:
    """Search Google via local Whoogle instance (self-hosted, no CAPTCHA).

    Whoogle is installed during agent provisioning and runs on port 5000.
    It queries Google server-side, strips tracking, returns clean results.
    """
    try:
        # The sub-second connect timeout applies only on the race path (a
        # shared client is passed there); the legacy own-client path keeps the
        # original 15s default so flag-OFF behavior stays byte-identical.
        get_kwargs = {}
        if client is not None:
            get_kwargs["timeout"] = httpx.Timeout(15.0, connect=_WHOOGLE_CONNECT_TIMEOUT_S)
        async with _client(client) as c:
            resp = await c.get(
                f"{WHOOGLE_URL}/search",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                **get_kwargs,
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        # Whoogle wraps results in div.ZINbbc or standard Google-like structure
        for div in soup.select("div.ZINbbc, div.g, div.tF2Cxc, div.fP1Qef")[:count]:
            link = div.select_one("a[href]")
            if not link:
                continue
            url = link.get("href", "")
            # Skip internal whoogle links
            if not url.startswith("http"):
                continue

            # Title: first <h3> or <a> text
            title_el = div.select_one("h3") or link
            title = title_el.get_text(strip=True) if title_el else ""

            # Snippet: various possible containers
            snippet_el = (
                div.select_one("div.BNeawe.s3v9rd, div.VwiC3b, span.aCOpRe, div.s3v9rd")
                or div.select_one("div:not(:has(a)):not(:has(h3))")
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            if title and url:
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet, source="google"
                ))

        return SearchResponse(results=results, source="google")
    except httpx.ConnectError:
        logger.info("[ToupSearch] Whoogle not running on port 5000, skipping Google")
        return SearchResponse(error="Whoogle not available", source="google")
    except Exception as e:
        logger.warning("[ToupSearch] Google/Whoogle failed: %s", e)
        return SearchResponse(error=str(e), source="google")


# ──────────────────────────────────────────────────────────────
# DuckDuckGo HTML search (secondary — most reliable fallback)
# ──────────────────────────────────────────────────────────────

async def _search_duckduckgo(
    query: str, count: int = 10, client: Optional[httpx.AsyncClient] = None
) -> SearchResponse:
    """Search DuckDuckGo via HTML endpoint (no API key, no CAPTCHA)."""
    try:
        async with _client(client) as c:
            resp = await c.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.select(".result")[:count]:
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            # DDG wraps URLs in a redirect — extract the real URL
            raw_url = title_el.get("href", "")
            url = raw_url
            # Parse DDG redirect: //duckduckgo.com/l/?uddg=ENCODED_URL&...
            if "uddg=" in raw_url:
                from urllib.parse import unquote, urlparse, parse_qs
                parsed = parse_qs(urlparse(raw_url).query)
                url = unquote(parsed.get("uddg", [raw_url])[0])
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo"))

        return SearchResponse(results=results, source="duckduckgo")
    except Exception as e:
        logger.warning("[ToupSearch] DuckDuckGo failed: %s", e)
        return SearchResponse(error=str(e), source="duckduckgo")


# ──────────────────────────────────────────────────────────────
# Bing search (secondary — lighter bot detection than Google)
# ──────────────────────────────────────────────────────────────

async def _search_bing(
    query: str, count: int = 10, client: Optional[httpx.AsyncClient] = None
) -> SearchResponse:
    """Search Bing via HTML scrape (no API key needed)."""
    try:
        async with _client(client) as c:
            resp = await c.get(
                "https://www.bing.com/search",
                params={"q": query, "count": count},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for li in soup.select("#b_results > li.b_algo")[:count]:
            link = li.select_one("h2 a")
            snippet_el = li.select_one(".b_caption p")
            if not link:
                continue
            title = link.get_text(strip=True)
            url = link.get("href", "")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="bing"))

        return SearchResponse(results=results, source="bing")
    except Exception as e:
        logger.warning("[ToupSearch] Bing failed: %s", e)
        return SearchResponse(error=str(e), source="bing")


# ──────────────────────────────────────────────────────────────
# Mojeek search (tertiary — independent engine, no tracking)
# ──────────────────────────────────────────────────────────────

async def _search_mojeek(
    query: str, count: int = 10, client: Optional[httpx.AsyncClient] = None
) -> SearchResponse:
    """Search Mojeek via HTML scrape (independent engine, no CAPTCHA)."""
    try:
        async with _client(client) as c:
            resp = await c.get(
                "https://www.mojeek.com/search",
                params={"q": query},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for li in soup.select("ul.results-standard li")[:count]:
            link = li.select_one("a.title")
            snippet_el = li.select_one("p.s")
            if not link:
                continue
            title = link.get_text(strip=True)
            url = link.get("href", "")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="mojeek"))

        return SearchResponse(results=results, source="mojeek")
    except Exception as e:
        logger.warning("[ToupSearch] Mojeek failed: %s", e)
        return SearchResponse(error=str(e), source="mojeek")


# ──────────────────────────────────────────────────────────────
# Public API: toup_search
# ──────────────────────────────────────────────────────────────

def _format_results(results: List[SearchResult], count: int) -> str:
    """Render results as the numbered title/url/snippet block the model sees.
    Identical output shape for both the sequential and race paths."""
    lines: List[str] = []
    for i, r in enumerate(results[:count], 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        lines.append("")
    return "\n".join(lines)


def _ranked(results: List[SearchResult], query: str) -> List[SearchResult]:
    """Dedup near-duplicate URLs, drop empties, and BM25-rerank by relevance to
    the query (kill-switch ``settings.search_rerank_enabled``). Off → unchanged
    engine order. Pure-Python and best-effort: never raises, never blocks."""
    if not settings.search_rerank_enabled:
        return results
    try:
        return rerank_filter(results, query)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("[ToupSearch] rerank failed, using raw order: %s", exc)
        return results


async def _first_with_results(
    engines, query: str, count: int, client: httpx.AsyncClient
) -> Optional[SearchResponse]:
    """Run ``engines`` concurrently over a shared client and return the first
    SearchResponse that actually has results, cancelling the still-pending
    losers. Engines that error or return empty are skipped. None if none of
    them produced results."""
    tasks = [asyncio.create_task(engine(query, count, client)) for engine in engines]
    pending = set(tasks)
    winner: Optional[SearchResponse] = None
    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # Inspect the just-completed tasks in the engines' priority order
            # (not asyncio.wait's unordered `done` set) so that when two engines
            # finish in the same tick the higher-priority one deterministically wins.
            for t in (task for task in tasks if task in done):
                if t.cancelled() or t.exception() is not None:
                    continue
                resp = t.result()
                if isinstance(resp, SearchResponse) and resp.results:
                    winner = resp
                    break
    finally:
        # Cancel the still-pending losers, then await every task so cancellations
        # settle and the exceptions of done-but-uninspected losers are retrieved
        # (avoids "Task exception was never retrieved" log noise) before the
        # caller closes the shared client.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return winner


_NO_RESULTS = "No search results found across all engines."


def cache_key(query: str, count: int, freshness_class: str = "evergreen") -> tuple:
    """Canonical cache key. Exported so callers that front this cache with
    their own earlier tier (see ``ToolExecutor._tool_web_search``) hash the
    query exactly the way ``toup_search`` does — a divergent key would make
    the two tiers cache-miss each other forever.

    The freshness class is part of the key (incident 2026-08-18, F2): a
    recency-intent search is served with a Brave date filter and a short TTL,
    an evergreen one without either, and the same words must never let one
    be served as the other."""
    return (" ".join(query.split()).lower(), count, freshness_class or "evergreen")


def _ttl_for(freshness_class: str) -> Optional[float]:
    """Recency results live at most ``search_cache_recency_ttl_s`` (15 min);
    evergreen keeps the cache-wide TTL. None → cache default."""
    if (freshness_class or "evergreen") == "recent":
        return float(min(settings.search_cache_recency_ttl_s, settings.search_cache_ttl_s * 4))
    return None


def cache_get_meta(query: str, count: int, freshness_class: str = "evergreen") -> Optional[tuple]:
    """``(result, stored_at_epoch)`` or None. Read-through for callers ahead
    of this module in the fallback chain."""
    if not settings.search_cache_enabled:
        return None
    if (freshness_class or "evergreen") == "recent" and getattr(settings, "search_cache_recency_bypass", False):
        return None
    hit = _SEARCH_CACHE.get(cache_key(query, count, freshness_class))
    if hit is None:
        return None
    if isinstance(hit, tuple) and len(hit) == 2:
        return hit
    return (hit, None)   # legacy shape (raw string) — tolerated, never written now


def cache_get(query: str, count: int, freshness_class: str = "evergreen") -> Optional[str]:
    """Read-through for callers ahead of this module in the fallback chain.
    Returns None when caching is disabled or on a miss."""
    hit = cache_get_meta(query, count, freshness_class)
    return hit[0] if hit else None


def cache_set(query: str, count: int, result: str, freshness_class: str = "evergreen") -> None:
    """Store a result produced by an earlier tier. No-ops on empty/no-result
    payloads so a transient outage can't be cached for 7 minutes. Recency
    results get the short TTL (or are not stored at all when the bypass flag
    is on)."""
    if not settings.search_cache_enabled:
        return
    if not result or result.startswith("No search results") or result.strip() == "No results found.":
        return
    fc = freshness_class or "evergreen"
    if fc == "recent" and getattr(settings, "search_cache_recency_bypass", False):
        return
    _SEARCH_CACHE.set(cache_key(query, count, fc), (result, time.time()), ttl_s=_ttl_for(fc))


async def _toup_search_sequential(query: str, count: int = 5) -> tuple[str, str]:
    """Legacy behavior: try engines in priority order, first non-empty wins.
    Returns ``(formatted, engine)``; engine is "" when nothing produced."""
    engines = [_search_google_whoogle, _search_duckduckgo, _search_bing, _search_mojeek]
    for engine in engines:
        resp = await engine(query, count)
        if resp.results:
            return _format_results(_ranked(resp.results, query), count), resp.source
    return _NO_RESULTS, ""


async def _toup_search_race(query: str, count: int = 5) -> tuple[str, str]:
    """Race the primary engines (Whoogle/DuckDuckGo/Bing) concurrently over one
    shared pooled client; first non-empty result wins and the losers are
    cancelled, so a dead or slow backend can't stall the chain. Mojeek stays a
    sequential last resort, used only if all primaries come back empty.
    Returns ``(formatted, engine)``."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        winner = await _first_with_results(
            [_search_google_whoogle, _search_duckduckgo, _search_bing],
            query, count, client,
        )
        if winner is None:
            resp = await _search_mojeek(query, count, client)
            if resp.results:
                winner = resp
        if winner is not None and winner.results:
            return _format_results(_ranked(winner.results, query), count), winner.source
    return _NO_RESULTS, ""


async def toup_search_meta(
    query: str, count: int = 5, freshness_class: str = "evergreen",
) -> tuple[str, str, bool]:
    """:func:`toup_search` plus provenance — returns
    ``(formatted, engine, cache_hit)``.

    ``engine`` is the winning engine's name ("duckduckgo", "mojeek", …) or ""
    when every engine came back empty. Callers use it to attribute usage to a
    concrete upstream; see the web-tool metering in ``tool_executor``.

    ``freshness_class`` only selects the cache key/TTL here — the scrape
    engines have no date filter of their own.
    """
    cached = cache_get(query, count, freshness_class)
    if cached is not None:
        logger.info("[PERF] web_search cache=hit q=%r class=%s", query[:60], freshness_class)
        return cached, "cache", True

    if settings.search_engine_race:
        result, engine = await _toup_search_race(query, count)
    else:
        result, engine = await _toup_search_sequential(query, count)

    if result and not result.startswith("No search results"):
        cache_set(query, count, result, freshness_class)
        logger.info("[PERF] web_search cache=miss q=%r engine=%s class=%s", query[:60], engine or "-", freshness_class)
    return result, engine, False


async def toup_search(query: str, count: int = 5) -> str:
    """
    Search using multiple engines. Returns formatted results.

    With ``settings.search_engine_race`` on (the default; it's a kill-switch),
    the primary engines (Whoogle, DuckDuckGo, Bing) are raced concurrently over
    a shared client and the first non-empty result wins (losers cancelled), with
    Mojeek as a last resort — a dead or slow Whoogle/engine can no longer block
    the chain. Flipped off, the legacy sequential priority chain is used unchanged.

    Results are served from a short-lived per-tenant TTL+LRU cache (kill-switch
    ``settings.search_cache_enabled``); a repeat query within the TTL returns
    with zero network calls. Empty/no-result responses are never cached.

    Text-only wrapper over :func:`toup_search_meta` for callers that don't
    care which engine served.
    """
    result, _engine, _cached = await toup_search_meta(query, count)
    return result
