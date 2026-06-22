"""
Toup Smart Fetch — API-first search, page reading, and data extraction.

Provides fast, CAPTCHA-free alternatives to browser-based search and fetching.
The browser is used only as a last resort for complex interactive tasks.

Modules:
  search  — Multi-engine search (DuckDuckGo, Bing, Mojeek) without a browser
  reader  — Article/page extraction via HTTP + readability (no browser)
  video   — YouTube search and metadata via yt-dlp (no API key)
  data    — Weather (wttr.in), finance (yfinance), Wikipedia
"""

from .search import toup_search
from .reader import toup_read_page
from .video import toup_video_search


def clear_caches() -> None:
    """Drop all cached search/page results. Called on tenant (re)bind so a
    re-bound pool container can never serve the previous identity's cached
    content — belt-and-suspenders on top of the one-process-per-tenant +
    destroy-on-release lifecycle."""
    from .search import _SEARCH_CACHE
    from .reader import _PAGE_CACHE
    _SEARCH_CACHE.clear()
    _PAGE_CACHE.clear()


__all__ = ["toup_search", "toup_read_page", "toup_video_search", "clear_caches"]
