"""Radio Mode broadcast helper.

The picker lives in `playlist.py` now (YT Music Song Radio). This module
only holds the `media_play` broadcast that wraps a radio-auto track for
the frontend — kept separate from `broadcast_to_user` so the
`radio_auto: True` flag and the age-restriction check live in one place.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def broadcast_radio_track(
    user_id: str,
    video_id: str,
    title: str,
    channel: str,
    artist: str = "",
    thumbnail_url: str = "",
    video_type: str = "",
) -> bool:
    """Emit a media_play event flagged radio_auto + kick age-check in background.

    Extra fields (artist, thumbnail_url, video_type) let the frontend render
    the Song-mode now-playing overlay without a secondary fetch.
    """
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
            "artist": artist,
            "thumbnail_url": thumbnail_url,
            "video_type": video_type,
        })
        asyncio.create_task(_check_age_and_swap(video_id, user_id))
        logger.info(
            "[radio/player] broadcast radio_auto user=%s channel=%s video=%s "
            "title=%r artist=%r video_type=%r",
            user_id[:8], channel, video_id, title, artist, video_type,
        )
        return True
    except Exception as e:
        logger.warning("[radio/player] broadcast failed: %s", e)
        return False
