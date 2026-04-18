"""Radio Mode YouTube search + broadcast helper.

Mirrors the core of `_tool_play_media` / `_fast_media_check` but is
callable without an agent turn. Used by the auto-pick path so the
next-track broadcast doesn't create a new assistant message.

Also exposes `youtube_related` — scrapes the watch page for related
video IDs without needing an LLM. This is the primary next-track
strategy; LLM-based picking is a fallback for when related-scraping
returns nothing useful.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "Chrome/137.0.0.0 Safari/537.36"
)


# YouTube "Videos" search filter — excludes Shorts, Lives, channels, playlists.
# Verified empirically (local smoke test 2026-04-18): with this filter,
# /shorts/ URLs drop from ~22/page to 0/page on music-artist queries. That's
# the primary Shorts defense; the WEB_PAGE_TYPE_SHORTS id-set below is belt.
_YT_VIDEOS_FILTER = "EgIQAQ%3D%3D"


async def youtube_search_many(query: str, limit: int = 20, timeout: float = 8.0) -> List[Tuple[str, str]]:
    """Return [(video_id, title)] for up to `limit` non-Short search results.

    Used by the radio-mode picker to fetch a batch of candidates from a
    single query, iterate through them over successive track-end events,
    and only re-search when the batch is exhausted.

    Shorts are filtered two ways:
      1. `sp=<Videos filter>` query param — YouTube applies it server-side
      2. Any id that appears in a WEB_PAGE_TYPE_SHORTS navigation block is
         dropped (catches anything that slips past #1).
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as hc:
            resp = await hc.get(
                "https://www.youtube.com/results",
                params={"search_query": q, "sp": _YT_VIDEOS_FILTER},
                headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            )
        html = resp.text
    except Exception as e:
        print(f"[radio/player] youtube_search_many fetch failed q={q!r} err={e}", flush=True)
        return []

    # Collect Shorts ids so we can reject them regardless of where they appear.
    shorts_ids = set(re.findall(r'/shorts/([a-zA-Z0-9_-]{11})', html))

    # Pull all videoId matches + attempt to align with a nearby title.
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    title_matches = re.findall(
        r'"title":\{(?:"runs":\[\{"text":"([^"]+)"\}\]|"simpleText":"([^"]+)")',
        html,
    )
    titles = [a or b for (a, b) in title_matches]

    out: List[Tuple[str, str]] = []
    seen: set = set()
    shorts_skipped = 0
    for i, vid in enumerate(ids):
        if vid in seen:
            continue
        seen.add(vid)
        if vid in shorts_ids:
            shorts_skipped += 1
            continue
        title = titles[i] if i < len(titles) else ""
        out.append((vid, title or ""))
        if len(out) >= limit:
            break
    if shorts_skipped:
        print(
            f"[radio/player] search q={q!r} shorts_skipped={shorts_skipped} "
            f"returned={len(out)}",
            flush=True,
        )
    return out


async def youtube_search_first(query: str, timeout: float = 8.0) -> Optional[Tuple[str, str]]:
    """Return (video_id, title) for the first YouTube search hit, or None."""
    q = (query or "").strip()
    if not q:
        return None

    # Direct URL first
    m = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', q)
    if m:
        return (m.group(1), q)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as hc:
            resp = await hc.get(
                "https://www.youtube.com/results",
                params={"search_query": q},
                headers={"User-Agent": _UA},
            )
        ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
        if not ids:
            return None
        video_id = ids[0]
        title_m = re.search(r'"title":\{"runs":\[\{"text":"([^"]+)"\}', resp.text)
        title = title_m.group(1) if title_m else "YouTube Video"
        return (video_id, title)
    except Exception as e:
        logger.warning("[radio/player] youtube_search_first failed q=%r err=%s", q, e)
        return None


async def broadcast_radio_track(
    user_id: str,
    video_id: str,
    title: str,
    channel: str,
) -> bool:
    """Emit a media_play event flagged as radio_auto, and kick age-check in background."""
    try:
        from app.api.ws_chat import broadcast_to_user, _check_age_and_swap
        await broadcast_to_user(user_id, {
            "type": "media_play",
            "provider": "youtube",
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "radio_auto": True,
            "channel": channel,
        })
        asyncio.create_task(_check_age_and_swap(video_id, user_id))
        logger.info(
            "[radio/player] broadcast radio_auto user=%s channel=%s video=%s title=%r",
            user_id[:8], channel, video_id, title,
        )
        return True
    except Exception as e:
        logger.warning("[radio/player] broadcast failed: %s", e)
        return False
