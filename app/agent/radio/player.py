"""Radio Mode YouTube search + broadcast helper.

Mirrors the core of `_tool_play_media` / `_fast_media_check` but is
callable without an agent turn. Used by the auto-pick path so the
next-track broadcast doesn't create a new assistant message.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "Chrome/137.0.0.0 Safari/537.36"
)


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
