"""Radio Mode broadcast helper.

The picker lives in `playlist.py` now (YT Music Song Radio). This module
only holds the `media_play` broadcast that wraps a radio-auto track for
the frontend — kept separate from `broadcast_to_user` so the
`radio_auto: True` flag and the age-restriction check live in one place.
"""
from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,16}$")


# ── "Did the user ask to WATCH this?" ──────────────────────────────────
# Audio-first means the phone starts every play as native audio unless the
# frame says otherwise, so the *producer* of the frame has to answer this — and
# there are two producers: the fast path in ws_chat (which handles the majority
# of typed "play …" messages and forbids the agent from calling play_media at
# all) and the play_media tool itself. Both call this, so a request phrased the
# same way behaves the same way whichever one catches it.
#
# Deliberately conservative about the word "watch": it is a verb of intent only
# at the START of the request. Mid-sentence it is far more likely to be part of
# a title ("Watch Me", "I'll Be Watching You"), and mistaking one for the other
# puts a song on a screen the user isn't looking at.
_VIDEO_INTENT_RE = re.compile(
    r"^\s*(?:show\s+me|show|watch|let'?s\s+watch)\b"
    r"|\bmusic\s+video\b"
    r"|\b(?:the\s+)?(?:video|mv)\s+(?:of|for|version)\b"
    r"|\bplay\s+(?:the\s+|a\s+)?video\b"
    r"|\bon\s+(?:the\s+)?(?:screen|tv)\b",
    re.I,
)

# Content that is not music at all. Audio-only is the right default for a song;
# it is plainly wrong for a documentary or a trailer, and the user should not
# have to say "video" to watch something that has nothing to listen to.
_NON_MUSIC_RE = re.compile(
    r"\b(?:documentar(?:y|ies)|trailer|movie|film|episode|series|docuseries|"
    r"interview|podcast|lecture|tutorial|walkthrough|gameplay|stand[\s-]?up|"
    r"comedy\s+special|ted\s+talk|highlights|full\s+match|news\s+report)\b",
    re.I,
)


def infer_requested_mode(text: str) -> str | None:
    """'video' when the request is explicitly to WATCH (or is non-music
    content), else None — meaning "no signal", so the caller applies its
    channel default. Never returns 'song': absence of a video signal is not
    the same as an explicit audio request, and only the caller knows what its
    channel's default surface is."""
    if not text:
        return None
    if _VIDEO_INTENT_RE.search(text) or _NON_MUSIC_RE.search(text):
        return "video"
    return None


def warm_audio_cache(video_ids: list) -> None:
    """Fire-and-forget: ask the PLATFORM to pre-build the audio remux for these
    tracks (L1/L2 cache) the moment they are broadcast/queued, so the phone's
    native play lands on a warm cache instead of a cold yt-dlp extraction.

    This is the server half of audio-first: the agent knows the video_id
    seconds before the phone can ask for bytes, and the platform's build is
    deduped (`_remux_tasks`) + concurrency-capped (`_build_sem`), so warming
    here is free when the phone's own request arrives first and decisive when
    it doesn't. Never awaited, never raises — a failed warm degrades to
    exactly today's cold path.
    """
    ids = [v for v in (video_ids or []) if isinstance(v, str) and _VIDEO_ID_RE.match(v)][:3]
    if not ids:
        return

    async def _go() -> None:
        try:
            from app.config import settings
        except Exception:
            return
        if getattr(settings, "run_mode", "") != "agent":
            # Platform/monolith: media_proxy is in-process — warm directly.
            try:
                from app.api.media_proxy import _ensure_remux_bg
                for vid in ids:
                    asyncio.create_task(_ensure_remux_bg(vid))
            except Exception:
                pass
            return
        base = (getattr(settings, "platform_api_url", "") or "").rstrip("/")
        key = getattr(settings, "agent_api_key", "") or ""
        uid = getattr(settings, "user_id", "") or ""
        if not (base and key and uid):
            return
        # Same both-layouts idiom as _yt_remote: PLATFORM_API_URL is not
        # reliably set with the /api suffix.
        candidates = [f"{base}/internal/media/warm"]
        if not base.endswith("/api"):
            candidates.append(f"{base}/api/internal/media/warm")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as hc:
                for url in candidates:
                    try:
                        resp = await hc.post(
                            url,
                            headers={"X-Agent-Key": key},
                            json={"user_id": uid, "video_ids": ids},
                        )
                    except Exception:
                        continue
                    if resp.status_code == 200:
                        logger.info("[radio/player] warm ok ids=%s", ids)
                        return
                logger.info("[radio/player] warm unreachable ids=%s", ids)
        except Exception as e:
            logger.debug("[radio/player] warm failed: %s", e)

    try:
        asyncio.create_task(_go())
    except RuntimeError:
        pass  # no running loop (sync test context) — the warm is best-effort


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
        # Warm the platform's remux cache for what is playing NOW and what
        # plays NEXT — audio-first means the phone asks for these bytes within
        # seconds on the 'app' channel.
        if channel == "app":
            # UPCOMING ONLY — never the track being broadcast. A warm is a
            # full itag-18 pull through the SAME single residential proxy the
            # live progressive stream is about to use, so warming the now-
            # playing id starves the very playback it was meant to accelerate
            # (the media proxy already defers its own background build to
            # AFTER the response body for exactly this reason). The live track
            # gains nothing from it either: its extraction is already coalesced
            # server-side.
            warm_audio_cache([u.get("video_id", "") for u in (upcoming or [])[:2]])
        logger.info(
            "[radio/player] broadcast radio_auto user=%s channel=%s num_ws_connections=%d sent=%d "
            "video=%s title=%r artist=%r video_type=%r reason=%s",
            user_id[:8], channel, num_ws, sent, video_id, title, artist, video_type, reason,
        )
        return True
    except Exception as e:
        logger.warning("[radio/player] broadcast failed: %s", e)
        return False
