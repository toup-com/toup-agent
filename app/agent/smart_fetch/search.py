"""
Toup Search — Multi-engine search without a browser.

Queries multiple search engines — Google first (best results), then fallbacks:
  1. Google (via local Whoogle instance — self-hosted Google proxy, no CAPTCHA)
  2. DuckDuckGo HTML (no CAPTCHA, no API key)
  3. Bing Web Search (lighter bot detection)
  4. Mojeek (independent engine, no CAPTCHA)

Whoogle is a self-hosted Google frontend that queries Google server-side
without triggering CAPTCHAs. It's installed as part of the agent provisioning.

All engines are queried via simple HTTP requests with proper headers.
No browser, no headless Chrome, no Playwright — just httpx.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

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


async def _search_google_whoogle(query: str, count: int = 10) -> SearchResponse:
    """Search Google via local Whoogle instance (self-hosted, no CAPTCHA).

    Whoogle is installed during agent provisioning and runs on port 5000.
    It queries Google server-side, strips tracking, returns clean results.
    """
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                f"{WHOOGLE_URL}/search",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
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

async def _search_duckduckgo(query: str, count: int = 10) -> SearchResponse:
    """Search DuckDuckGo via HTML endpoint (no API key, no CAPTCHA)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
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

async def _search_bing(query: str, count: int = 10) -> SearchResponse:
    """Search Bing via HTML scrape (no API key needed)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
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

async def _search_mojeek(query: str, count: int = 10) -> SearchResponse:
    """Search Mojeek via HTML scrape (independent engine, no CAPTCHA)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
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

async def toup_search(query: str, count: int = 5) -> str:
    """
    Search using multiple engines. Returns formatted results.

    Priority: DuckDuckGo → Bing → Mojeek.
    Falls through to the next engine if one fails or returns no results.
    """
    engines = [_search_google_whoogle, _search_duckduckgo, _search_bing, _search_mojeek]

    for engine in engines:
        resp = await engine(query, count)
        if resp.results:
            lines = []
            for i, r in enumerate(resp.results[:count], 1):
                lines.append(f"{i}. {r.title}")
                lines.append(f"   {r.url}")
                if r.snippet:
                    lines.append(f"   {r.snippet}")
                lines.append("")
            return "\n".join(lines)

    return "No search results found across all engines."
