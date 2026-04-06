"""
Content Router — resolves "play Oppenheimer" to the best available streaming provider.

Uses TMDB Watch Providers API (free, powered by JustWatch) for content availability,
then matches against the user's connected streaming accounts.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# TMDB provider ID → our channel name
TMDB_PROVIDER_MAP = {
    8:   "netflix",
    9:   "prime",           # Amazon Prime Video
    337: "disney",          # Disney+
    230: "crave",           # Crave (Canada)
    350: "apple_tv",        # Apple TV+
    384: "hbo_max",         # HBO Max / Max
    15:  "hulu",
    386: "peacock",
    283: "paramount",       # Paramount+
    1899: "paramount",      # Paramount+ (alternate ID in some regions)
    613: "crave",           # Crave (alternate)
}

# Reverse: our channel name → TMDB provider IDs
CHANNEL_TO_TMDB = {}
for _pid, _ch in TMDB_PROVIDER_MAP.items():
    CHANNEL_TO_TMDB.setdefault(_ch, []).append(_pid)

# Provider preference order (subscription streaming prioritized)
PROVIDER_PRIORITY = ["netflix", "prime", "disney", "crave", "hbo_max", "hulu", "apple_tv", "paramount", "peacock"]

# Simple TTL cache
_cache: Dict[str, tuple] = {}  # key → (value, expires_at)
SEARCH_TTL = 86400    # 24h for title metadata
PROVIDER_TTL = 43200  # 12h for availability data


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and entry[1] > time.time():
        return entry[0]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value, ttl: int):
    _cache[key] = (value, time.time() + ttl)


@dataclass
class ContentMatch:
    """Result of content routing."""
    title: str
    tmdb_id: int
    media_type: str  # "movie" or "tv"
    year: Optional[int] = None
    provider: Optional[str] = None         # Best matched channel name
    provider_name: Optional[str] = None    # Display name (e.g. "Prime Video")
    all_providers: List[str] = field(default_factory=list)  # All available channels
    poster_path: Optional[str] = None


class ContentRouter:
    """Resolves content queries to the best streaming provider for a user."""

    def __init__(self, tmdb_api_key: str, region: str = "CA"):
        self.api_key = tmdb_api_key
        self.region = region
        self.base_url = "https://api.themoviedb.org/3"

    async def resolve(
        self,
        query: str,
        connected_channels: List[str],
        prefer_channel: Optional[str] = None,
    ) -> Optional[ContentMatch]:
        """
        Resolve a content query to the best streaming provider.

        Args:
            query: Movie/show title (e.g. "Oppenheimer", "Stranger Things S4")
            connected_channels: User's connected provider channel names
            prefer_channel: Optional preferred channel (user explicitly asked for it)

        Returns:
            ContentMatch with best provider, or None if not found on TMDB.
        """
        if not self.api_key:
            logger.warning("[ContentRouter] No TMDB API key configured")
            return None

        # 1. Search TMDB
        result = await self._search(query)
        if not result:
            return None

        # 2. Get watch providers
        providers = await self._get_providers(result.tmdb_id, result.media_type)

        # All channels where this content is available (subscription streaming)
        available_channels = []
        provider_display = {}
        for p in providers:
            pid = p.get("provider_id")
            channel = TMDB_PROVIDER_MAP.get(pid)
            if channel and channel not in available_channels:
                available_channels.append(channel)
                provider_display[channel] = p.get("provider_name", channel)

        result.all_providers = available_channels

        # 3. Match against connected channels
        if prefer_channel and prefer_channel in connected_channels and prefer_channel in available_channels:
            result.provider = prefer_channel
            result.provider_name = provider_display.get(prefer_channel, prefer_channel)
            return result

        # Find best match: available AND connected, sorted by preference
        for ch in PROVIDER_PRIORITY:
            if ch in available_channels and ch in connected_channels:
                result.provider = ch
                result.provider_name = provider_display.get(ch, ch)
                return result

        # No connected provider has it — return with all_providers for the agent to report
        return result

    async def _search(self, query: str) -> Optional[ContentMatch]:
        """Search TMDB for a title. Returns best match."""
        cache_key = f"tmdb_search:{query.lower().strip()}"
        cached = _cache_get(cache_key)
        if cached:
            return cached

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/search/multi",
                    params={
                        "api_key": self.api_key,
                        "query": query,
                        "include_adult": "false",
                        "language": "en-US",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            # Filter to movies and TV shows only
            results = [r for r in results if r.get("media_type") in ("movie", "tv")]
            if not results:
                return None

            # Pick best match (TMDB already ranks by relevance)
            best = results[0]
            media_type = best["media_type"]
            title = best.get("title") or best.get("name") or query
            year = None
            date_str = best.get("release_date") or best.get("first_air_date") or ""
            if date_str and len(date_str) >= 4:
                year = int(date_str[:4])

            match = ContentMatch(
                title=title,
                tmdb_id=best["id"],
                media_type=media_type,
                year=year,
                poster_path=best.get("poster_path"),
            )
            _cache_set(cache_key, match, SEARCH_TTL)
            return match

        except Exception as e:
            logger.warning("[ContentRouter] TMDB search failed for '%s': %s", query, e)
            return None

    async def _get_providers(self, tmdb_id: int, media_type: str) -> list:
        """Get streaming providers for a title in the configured region."""
        cache_key = f"tmdb_providers:{media_type}:{tmdb_id}:{self.region}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/{media_type}/{tmdb_id}/watch/providers",
                    params={"api_key": self.api_key},
                )
                resp.raise_for_status()
                data = resp.json()

            region_data = data.get("results", {}).get(self.region, {})
            # "flatrate" = subscription streaming (what we care about)
            providers = region_data.get("flatrate", [])

            _cache_set(cache_key, providers, PROVIDER_TTL)
            return providers

        except Exception as e:
            logger.warning("[ContentRouter] TMDB providers failed for %s/%d: %s", media_type, tmdb_id, e)
            return []

    async def get_available_text(self, match: ContentMatch, connected_channels: List[str]) -> str:
        """Generate a human-readable message about content availability."""
        if match.provider:
            return f'"{match.title}" is available on {match.provider_name}.'

        if match.all_providers:
            available_names = []
            for ch in match.all_providers:
                # Capitalize nicely
                display = {
                    "netflix": "Netflix", "prime": "Prime Video", "disney": "Disney+",
                    "crave": "Crave", "hbo_max": "Max", "hulu": "Hulu",
                    "apple_tv": "Apple TV+", "paramount": "Paramount+", "peacock": "Peacock",
                }.get(ch, ch.title())
                connected = ch in connected_channels
                available_names.append(f"{display} {'(connected)' if connected else ''}")

            providers_str = ", ".join(available_names)
            unconnected = [ch for ch in match.all_providers if ch not in connected_channels]
            if unconnected:
                return (
                    f'"{match.title}" is available on: {providers_str}. '
                    f"You don't have any of these connected. "
                    f"Connect one in Movies settings to watch."
                )
            return f'"{match.title}" is available on: {providers_str}.'

        return f'Could not find streaming availability for "{match.title}".'
