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

__all__ = ["toup_search", "toup_read_page", "toup_video_search"]
