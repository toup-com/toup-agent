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


# Does the platform this agent talks to have the single-flight SPOOL
# (media_proxy, 2026-08-09)? None = not yet known. Learned from the warm
# endpoint's response ({"spool": true/false}) and cached for the process.
# Until confirmed, a NOW-PLAYING build warm is downgraded to extract: on a
# pre-spool platform the build is a second full download racing the very
# stream it was meant to accelerate (the 2026-08-05 regression class — and
# the old live-first gate cannot save it, because at warm time no live start
# exists yet, so the gate waves the build through). Upcoming-track builds are
# not downgraded; they were always safe.
_platform_spool: "bool | None" = None


def warm_audio_cache(video_ids: list, mode: str = "build", now_playing: bool = False) -> None:
    """Fire-and-forget: ask the PLATFORM to warm these tracks the moment they are
    broadcast/queued, so the phone's native play lands warm instead of cold.

    This is the server half of audio-first: the agent knows the video_id
    seconds before the phone can ask for bytes. Never awaited, never raises —
    a failed warm degrades to exactly today's cold path.

    TWO MODES, and the distinction is the whole point:

    * `build` (default) — download + remux the entire ~11.8MB itag-18. Deduped
      (`_remux_tasks`) and concurrency-capped (`_build_sem`), but it saturates
      the single residential proxy, so it is only ever correct for UPCOMING
      tracks. Aimed at the track playing right now it competes with that
      track's own stream, which is the regression this argument exists to keep
      from being re-introduced.

    * `extract` — the yt-dlp metadata handshake only, no media bytes. Cheap
      enough to aim at the track playing RIGHT NOW, and that is where the wait
      actually is: measured against the production proxy on 2026-08-04,
      extraction alone is a median 3.4s, a mean 7.1s and a worst case of 20.8s,
      all of it in front of the first byte of audio. Run at broadcast time it
      overlaps the agent's reply and the card render, so the phone's request
      finds the extraction cached or already in flight.
    """
    ids = [v for v in (video_ids or []) if isinstance(v, str) and _VIDEO_ID_RE.match(v)][:3]
    if not ids:
        return
    mode = mode if mode in ("build", "extract") else "build"

    async def _go() -> None:
        global _platform_spool
        try:
            from app.config import settings
        except Exception:
            return
        if getattr(settings, "run_mode", "") != "agent":
            # Platform/monolith: media_proxy is in-process — warm directly,
            # and the spool capability is directly readable.
            try:
                from app.api.media_proxy import _ensure_extract_bg, _ensure_remux_bg, _SPOOL_ENABLED
                _platform_spool = bool(_SPOOL_ENABLED)
                eff = mode
                if eff == "build" and now_playing and not _platform_spool:
                    eff = "extract"
                warmer = _ensure_extract_bg if eff == "extract" else _ensure_remux_bg
                for vid in ids:
                    asyncio.create_task(warmer(vid))
            except Exception:
                pass
            return
        eff = mode
        if eff == "build" and now_playing and _platform_spool is not True:
            eff = "extract"
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
                            json={"user_id": uid, "video_ids": ids, "mode": eff},
                        )
                    except Exception:
                        continue
                    if resp.status_code == 200:
                        # Learn the platform's spool capability from the ack so
                        # the NEXT now-playing warm can be a real build.
                        try:
                            body = resp.json()
                            if isinstance(body, dict) and "spool" in body:
                                _platform_spool = bool(body["spool"])
                        except Exception:
                            pass
                        logger.info("[radio/player] warm ok mode=%s ids=%s", eff, ids)
                        return
                logger.info("[radio/player] warm unreachable mode=%s ids=%s", eff, ids)
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
    duration: int = 0,
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
        if duration and duration > 0:
            # Seconds. Lets the card render a real length the instant the frame
            # lands, instead of '--:--' (or a stale previous-track value) until
            # the player reports one. Additive; older clients ignore it.
            frame["duration"] = int(duration)
        if upcoming:
            frame["upcoming"] = upcoming
        sent = await broadcast_to_user(user_id, frame)
        asyncio.create_task(_check_age_and_swap(video_id, user_id))
        # Warm the platform's remux cache for what is playing NOW and what
        # plays NEXT — audio-first means the phone asks for these bytes within
        # seconds on the 'app' channel.
        if channel == "app":
            # The now-playing track gets a BUILD warm — a deliberate reversal
            # of the old "build is upcoming-only" rule, enabled by the
            # platform's single-flight SPOOL (media_proxy, 2026-08-09).
            #
            # The old rule existed because a build was a SECOND download of the
            # same itag-18 through the same residential proxy the live stream
            # was using — it starved the playback it was meant to accelerate.
            # With the spool, a build STARTS (or joins) the ONE shared upstream
            # download that the phone's own /audio_stream request reads from.
            # At an advance the phone usually plays its prebuffered LOCAL file,
            # so this build runs on an idle proxy and its real value is filling
            # R2 — every future replay of the track, platform-wide, goes warm.
            #
            # now_playing=True is the deploy-skew guard: until the platform has
            # confirmed it HAS the spool ({"spool":true} on the warm ack), this
            # is sent as an extract — on a pre-spool image a now-playing build
            # is the 2026-08-05 regression, and its live-first gate cannot save
            # it (checked before any live start exists).
            warm_audio_cache([video_id], mode="build", now_playing=True)
            # Explicit, for the same reason as the ws_chat call site: the
            # dangerous mode is the DEFAULT one, so an omitted argument reads
            # as "cheap" and is not.
            warm_audio_cache(
                [u.get("video_id", "") for u in (upcoming or [])[:2]], mode="build"
            )
        logger.info(
            "[radio/player] broadcast radio_auto user=%s channel=%s num_ws_connections=%d sent=%d "
            "video=%s title=%r artist=%r video_type=%r reason=%s",
            user_id[:8], channel, num_ws, sent, video_id, title, artist, video_type, reason,
        )
        return True
    except Exception as e:
        logger.warning("[radio/player] broadcast failed: %s", e)
        return False
