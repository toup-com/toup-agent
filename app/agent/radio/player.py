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
    reason: str = "auto_advance",
    upcoming: list | None = None,
) -> bool:
    """Emit a media_play event flagged radio_auto + kick age-check in background.

    Extra fields (artist, thumbnail_url, video_type) let the frontend render
    the Song-mode now-playing overlay without a secondary fetch.

    `reason` is metadata for the frontend / logs:
      - "auto_advance"  (default) — radio popped the next queued track.
      - "toggle_seed"             — Radio was just toggled on; this broadcast
                                    forces the iframe to load the seed so
                                    iframe and session sync by construction.
      - "topic_swap" / "mv_swap"  — mid-track variant swap.
    Kept as `radio_auto: True` so the frontend skips chat-card attachment
    (there's no new assistant message for these broadcasts — the seed card
    already exists, the swap reuses the current card).

    `upcoming` (optional) is the next 1-2 station tracks
    ([{video_id, title, artist, thumbnail_url}, ...]). Mobile pre-buffers them
    into the native player queue so lock-screen skip / auto-advance is instant
    (no on-tap cold-load). Web ignores it. Backward-compatible additive field.
    """
    try:
        from app.api.ws_chat import broadcast_to_user, _check_age_and_swap, _user_ws_queues
        # Snapshot fan-out width BEFORE the send so a concurrent disconnect
        # between the count and the send can't confuse the log. Multi-client
        # diagnosis: if num_ws>1 while session.channel=='app', every other
        # connected client also receives this frame — see SKILL.md Rule 13.
        num_ws = len(_user_ws_queues.get(user_id, []))
        frame = {
            "type": "media_play",
            "provider": "youtube",
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "radio_auto": True,
            "reason": reason,
            "channel": channel,
            "artist": artist,
            "thumbnail_url": thumbnail_url,
            "video_type": video_type,
        }
        if upcoming:
            frame["upcoming"] = upcoming
        sent = await broadcast_to_user(user_id, frame)
        asyncio.create_task(_check_age_and_swap(video_id, user_id))
        logger.info(
            "[radio/player] broadcast radio_auto user=%s channel=%s num_ws_connections=%d sent=%d "
            "video=%s title=%r artist=%r video_type=%r reason=%s",
            user_id[:8], channel, num_ws, sent, video_id, title, artist, video_type, reason,
        )
        return True
    except Exception as e:
        logger.warning("[radio/player] broadcast failed: %s", e)
        return False
