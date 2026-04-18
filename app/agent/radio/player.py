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


async def youtube_search_many(query: str, limit: int = 20, timeout: float = 8.0) -> List[Tuple[str, str]]:
    """Return [(video_id, title)] for up to `limit` search results.

    Used by the radio-mode picker to fetch a batch of candidates from a
    single query, iterate through them over successive track-end events,
    and only re-search when the batch is exhausted.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as hc:
            resp = await hc.get(
                "https://www.youtube.com/results",
                params={"search_query": q},
                headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            )
        html = resp.text
    except Exception as e:
        print(f"[radio/player] youtube_search_many fetch failed q={q!r} err={e}", flush=True)
        return []

    # Pull all videoId matches + attempt to align with a nearby title.
    # Search results embed titles as {"title":{"runs":[{"text":"..."}]}} in
    # proximity to "videoId":"...". Title is best-effort; picker works with
    # just videoIds too, since broadcast_radio_track fetches its own metadata.
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    # Titles in simpleText or runs form, in page order
    title_matches = re.findall(
        r'"title":\{(?:"runs":\[\{"text":"([^"]+)"\}\]|"simpleText":"([^"]+)")',
        html,
    )
    titles = [a or b for (a, b) in title_matches]

    out: List[Tuple[str, str]] = []
    seen: set = set()
    for i, vid in enumerate(ids):
        if vid in seen:
            continue
        seen.add(vid)
        title = titles[i] if i < len(titles) else ""
        out.append((vid, title or "YouTube Video"))
        if len(out) >= limit:
            break
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
