"""
WebSocket Chat Endpoint — Real-time streaming chat via WebSocket.

Protocol:
  Client sends JSON:
    { "type": "message", "text": "...", "session_id": "..." }
    { "type": "ping" }

  Server sends JSON:
    { "type": "text_chunk", "text": "..." }
    { "type": "tool_start", "tool": "..." }
    { "type": "tool_end", "tool": "...", "summary": "..." }
    { "type": "attachment", "message_id": "...", "attachment_id": "...",
      "filename": "...", "mime_type": "...", "size_bytes": 1234,
      "download_url": "/api/files/{message_id}/{attachment_id}",
      "preview_url": "/api/files/.../preview?format=html" }
    { "type": "done", "session_id": "...", "tokens": {...}, "model": "..." }
    { "type": "error", "message": "..." }
    { "type": "pong" }

Authentication:
  Connect with token as query param: ws://host/api/ws/chat?token=JWT_TOKEN
  Or send as first message: { "type": "auth", "token": "JWT_TOKEN" }
"""

import asyncio
import json
import logging
from datetime import datetime
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.config import settings

logger = logging.getLogger(__name__)


def _friendly_error(exc: Exception) -> str:
    """Convert raw exceptions into user-friendly, actionable error messages."""
    name = type(exc).__name__
    msg = str(exc)
    msg_lower = msg.lower()

    # ── Quota / billing exhausted (must check before generic 429) ──
    if any(kw in msg_lower for kw in ("insufficient_quota", "quota", "billing", "credit", "balance", "exceeded your current")):
        return (
            "Your API credit limit has been reached. To continue:\n"
            "• Add more credits at platform.openai.com/account/billing\n"
            "• Or add a different API key in Settings"
        )

    # ── Rate limit (temporary, not billing) ──
    status = getattr(exc, "status_code", None)
    if status == 429 or "rate_limit" in msg_lower:
        return "Rate limit reached — too many requests. Please wait a moment and try again."

    # ── Anthropic overloaded ──
    if status == 529 or "overloaded" in msg_lower:
        return "The AI service is temporarily overloaded. Please try again in a few seconds."

    # ── Auth errors — distinguish invalid key from expired token ──
    if status == 401 or "authentication" in msg_lower or "unauthorized" in msg_lower:
        if "expired" in msg_lower or "oauth" in msg_lower:
            return "Your API token has expired. Please reconnect or update your API key in Settings."
        return "Your API key is invalid. Please check your API key in Settings."

    # ── Permission errors ──
    if status == 403 or "permission" in msg_lower or "forbidden" in msg_lower:
        return "Your API key doesn't have access to this model. Please check your plan or switch models in Settings."

    # ── Bad request ──
    if status == 400 or "bad_request" in name.lower():
        return "There was an issue with the request. Please try rephrasing your message."

    # ── Server errors ──
    if status and status >= 500:
        return "The AI service is temporarily unavailable. Please try again in a moment."

    # ── Timeout ──
    if "timeout" in name.lower() or "timeout" in msg_lower:
        return "The request timed out. Please try again."

    # ── Connection ──
    if "connection" in name.lower() or "connection" in msg_lower:
        return "Couldn't reach the AI service. Please check your connection and try again."

    # ── Generic fallback — don't expose internals ──
    logger.error(f"[WS] Unhandled error type for friendly message: {name}: {msg}")
    return "Something went wrong. Please try again in a moment."


# ── User WebSocket broadcast registry ────────────────────────────────
# Maps user_id → list of asyncio.Queues (one per active WS connection).
# Background tasks (e.g. app builder) push events here, and the WS
# handler forwards them to the client.
_user_ws_queues: Dict[str, List[asyncio.Queue]] = {}


async def _resolve_day_chat_id_for_now(db_session, user_id: str, tz_override: str = None):
    """Thin wrapper around the shared helper. See app.db.message_helpers."""
    from app.db.message_helpers import resolve_day_chat_id_for_now
    return await resolve_day_chat_id_for_now(db_session, user_id, tz_override=tz_override)


async def broadcast_to_user(user_id: str, event: dict) -> int:
    """Push an event to all WebSocket connections for a user.
    Returns number of connections that received the event."""
    queues = _user_ws_queues.get(user_id, [])
    sent = 0
    for q in queues:
        try:
            q.put_nowait(event)
            sent += 1
        except asyncio.QueueFull:
            logger.warning(f"[BROADCAST] Queue full for user {user_id}, dropping event type={event.get('type')}")
    etype = event.get("type", "?")
    print(f"[BROADCAST] user={user_id[:8]} type={etype} queues={len(queues)} sent={sent}", flush=True)
    return sent


def _register_ws_queue(user_id: str, queue: asyncio.Queue) -> None:
    """Register a queue for a user's WebSocket connection."""
    if user_id not in _user_ws_queues:
        _user_ws_queues[user_id] = []
    _user_ws_queues[user_id].append(queue)


def _unregister_ws_queue(user_id: str, queue: asyncio.Queue) -> None:
    """Unregister a queue when WebSocket disconnects."""
    queues = _user_ws_queues.get(user_id, [])
    try:
        queues.remove(queue)
    except ValueError:
        pass
    if not queues:
        _user_ws_queues.pop(user_id, None)

# ── Onboarding prompt ────────────────────────────────────────────────
_ONBOARDING_TRIGGER = (
    "[SYSTEM: ONBOARDING — FIRST CONVERSATION]\n"
    "The user just set up their AI agent and is opening the chat for the very first time. "
    "You are coming alive for the first time. Greet them warmly.\n\n"
    "Your FIRST question MUST be: 'What is your name, and what is my name?'\n"
    "Wait for their answer, then:\n"
    "- Store user's name: memory_store(brain_type='user', category='identity', "
    "content='User name: <name>')\n"
    "- Store your name: memory_store(brain_type='agent', category='agent_soul', "
    "content='My name is <name>')\n\n"
    "Then continue naturally, ONE question at a time:\n"
    "- What they primarily need you for — goals, work domain. "
    "Store: brain_type='user', category='goals'\n"
    "- Their preferred language. Store: brain_type='user', category='preferences'\n"
    "- How they want you to communicate — formal/casual, concise/detailed. "
    "Store: brain_type='agent', category='agent_soul'\n"
    "- Any behavioral rules (always/never do). Store: brain_type='agent', category='agent_soul'\n\n"
    "After gathering core info (names, goals, language, personality), summarize what you learned. "
    "Then store: memory_store(brain_type='agent', category='agent_decisions', "
    "content='Onboarding complete. I know the user and they know me.')\n\n"
    "Be warm, enthusiastic, conversational. Ask ONE question at a time."
)

# ── ANSI helpers for terminal activity display ────────────────────────
_CYAN_BOLD = "\033[1;36m"
_GREEN_BOLD = "\033[1;32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_IS_TTY = sys.stdout.isatty()


def _tprint(msg: str) -> None:
    """Print to terminal only when stdout is a TTY (not piped/redirected)."""
    if _IS_TTY:
        print(msg, flush=True)

router = APIRouter(tags=["WebSocket Chat"])


@router.get("/debug/broadcast-test")
async def debug_broadcast_test():
    """DEBUG: Send a test job_update event to all connected WS clients."""
    from app.config import settings
    user_id = settings.user_id
    queues_count = len(_user_ws_queues.get(user_id, []))
    event = {
        "type": "job_update",
        "job_id": "test-debug-job-001",
        "name": "Debug Test Build",
        "status": "running",
        "step": "Testing broadcast...",
        "total_steps": 5,
        "completed_steps": 2,
    }
    sent = await broadcast_to_user(user_id, event)
    return {"user_id": user_id[:8], "queues": queues_count, "sent": sent, "event_type": "job_update"}

# ── Fast-path media detection ────────────────────────────────────────
# Detects play/music requests via regex and fires media_play BEFORE LLM.
# YouTube search takes ~1-2s vs 10-20s for full LLM pipeline.
#
# Age-restricted videos cannot play in YouTube embeds (iframe blocks them).
# We detect restriction via yt-dlp metadata and fall back to Piped embed.

import re as _re_mod

_PLAY_PATTERNS = [
    _re_mod.compile(r'^\s*play\s+(?:me\s+)?(?:a\s+)?(?:song\s+(?:of\s+|by\s+|called\s+)?|video\s+(?:of\s+|by\s+|called\s+)?|music\s+(?:of\s+|by\s+)?)?(.+?)(?:\s+(?:on|from|in)\s+(?:youtube|yt))?\s*$', _re_mod.I),
    _re_mod.compile(r'^\s*(?:put on|play me|play)\s+["\u201c]?(.+?)["\u201d]?\s*$', _re_mod.I),
]

_NETFLIX_KEYWORDS = _re_mod.compile(r'\b(?:netflix|disney|hulu|prime video|hbo)\b', _re_mod.I)


# Piped is an open-source YouTube frontend that plays age-restricted content
_PIPED_EMBED_BASE = "https://piped.video/embed"


async def _check_age_and_swap(video_id: str, user_id: str) -> None:
    """Background task: check if video is age-restricted via yt-dlp.
    If yes, send a media_embed_swap event so frontend hot-swaps to Piped embed.
    This runs AFTER the video is already playing — zero latency impact."""
    import shutil
    ytdlp = shutil.which("yt-dlp") or "/opt/toup-agent/venv/bin/yt-dlp"
    try:
        proc = await asyncio.create_subprocess_exec(
            ytdlp, f"https://www.youtube.com/watch?v={video_id}",
            "--dump-json", "--no-download", "--no-warnings", "--quiet",
            "--skip-download",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0 or not stdout:
            return
        import json
        data = json.loads(stdout.decode().strip())
        age_limit = data.get("age_limit", 0)
        if not age_limit or age_limit <= 0:
            return  # Not restricted — YouTube embed works fine

        # Age-restricted → tell frontend to swap embed to Piped
        embed_url = f"{_PIPED_EMBED_BASE}/{video_id}?autoplay=1"
        logger.info("[AGE-SWAP] Video %s is age-restricted (age_limit=%s), swapping to Piped", video_id, age_limit)
        await broadcast_to_user(user_id, {
            "type": "media_embed_swap",
            "video_id": video_id,
            "embed_url": embed_url,
        })
    except asyncio.TimeoutError:
        logger.warning("[AGE-SWAP] Timeout checking %s — YouTube embed stays", video_id)
    except Exception as e:
        logger.warning("[AGE-SWAP] Error checking %s: %s — YouTube embed stays", video_id, e)


# ── Task intent detection ──────────────────────────────────────────
# Keyword/regex heuristic: detects imperative task requests in chat.
# Matches phrases like "research X", "find me Y", "monitor Z", "set up A".
import re as _re
_TASK_INTENT_PATTERN = _re.compile(
    r"^(research|find\s+(me\s+)?|look\s+up|monitor|track|set\s+up|remind\s+me|"
    r"schedule|analyze|investigate|summarize|compile|gather|collect|prepare|"
    r"write\s+(me\s+)?(a\s+)?|draft\s+(me\s+)?(a\s+)?|create\s+(a\s+)?report|"
    r"compare|review|check\s+(if|whether)|scan|audit|benchmark|evaluate)",
    _re.IGNORECASE,
)


async def _detect_and_create_task(
    text: str, user_id: str, session_id: Optional[str],
    broadcast_queue: asyncio.Queue,
) -> Optional[str]:
    """If text looks like a task request, create an agent_task BuildJob.

    Returns job_id if a task was created, None otherwise.
    """
    if not _TASK_INTENT_PATTERN.search(text.strip()):
        return None

    import uuid
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    job_id = str(uuid.uuid4())
    title = text.strip()[:60]

    job = BuildJob(
        id=job_id,
        user_id=user_id,
        job_type="agent_task",
        title=title,
        prompt=text[:2000],
        status="running",
        steps_json="[]",
        model="",
        layer=0,
    )
    try:
        async with async_session_maker() as db:
            db.add(job)
            await db.commit()

        # Notify frontend: "Created task → view in Dashboard"
        broadcast_queue.put_nowait({
            "type": "task_created",
            "job_id": job_id,
            "title": title,
        })
        logger.info(f"[TASK] Detected task intent, created agent_task job {job_id[:8]}: {title}")
        return job_id
    except Exception as e:
        logger.warning(f"[TASK] Failed to create task job: {e}")
        return None


async def _fast_media_check(text: str, user_id: str, broadcast_queue: asyncio.Queue) -> Optional[tuple]:
    """If text looks like a play request, search YouTube and broadcast immediately.
    Returns (modified_text, media_meta_dict) if handled, or None if not a media request."""
    # Skip if it mentions streaming services (Netflix etc.) — those go through the agent
    if _NETFLIX_KEYWORDS.search(text):
        return None

    query = None
    for pat in _PLAY_PATTERNS:
        m = pat.match(text)
        if m:
            query = m.group(1).strip()
            break

    if not query or len(query) < 2 or len(query) > 200:
        return None

    logger.info("[FAST-MEDIA] Detected play request: %r → query=%r", text, query)

    try:
        import httpx
        video_id = None
        video_title = "YouTube Video"

        # Quick YouTube search via httpx (~1s)
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as hc:
            resp = await hc.get(
                "https://www.youtube.com/results",
                params={"search_query": query},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/137.0.0.0 Safari/537.36"},
            )
            id_matches = _re_mod.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
            if id_matches:
                video_id = id_matches[0]
                title_m = _re_mod.search(r'"title":\{"runs":\[\{"text":"([^"]+)"\}', resp.text)
                if title_m:
                    video_title = title_m.group(1)

        if not video_id:
            logger.warning("[FAST-MEDIA] No video found for: %s", query)
            return None

        # Broadcast media_play immediately — frontend opens YouTube embed (zero delay)
        event = {
            "type": "media_play",
            "provider": "youtube",
            "video_id": video_id,
            "title": video_title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
        broadcast_queue.put_nowait(event)
        logger.info("[FAST-MEDIA] Broadcast media_play in fast-path: %s - %s", video_id, video_title)

        # Fire-and-forget: check age restriction in background, swap embed if needed
        asyncio.create_task(_check_age_and_swap(video_id, user_id))

        # Radio session: record this as a user-driven seed. Channel is unknown
        # at this helper layer — caller records after resolving channel.

        # Return modified text so agent knows media is already playing
        modified_text = (
            f"{text}\n\n[SYSTEM: The video \"{video_title}\" (https://www.youtube.com/watch?v={video_id}) "
            f"is already playing in the user's browser. Do NOT call play_media. "
            f"Just respond conversationally about what's now playing.]"
        )
        media_meta = {"type": "youtube", "video_id": video_id, "title": video_title}
        return (modified_text, media_meta)
    except Exception as e:
        logger.warning("[FAST-MEDIA] Fast-path failed (agent will handle): %s", e)
        return None


# ── Radio Mode orchestration ─────────────────────────────────────────
# Toggle: enable / disable a per-channel radio session (seed = last played).
# Track-ended: when current track ends AND radio is ON, pick next via Haiku
# and broadcast a radio_auto media_play.

async def _handle_radio_toggle(user_id: str, msg: dict) -> None:
    from app.agent.radio import (
        get_radio_manager,
        RadioSessionManager,
        SeedTrack,
        build_station,
    )

    channel = (msg.get("channel") or "").strip().lower()
    enabled = bool(msg.get("enabled"))

    print(f"[radio] toggle entry user={user_id[:8]} channel={channel!r} enabled={enabled} msg={msg}", flush=True)

    if not RadioSessionManager.is_channel_allowed(channel):
        print(f"[radio] toggle REJECT channel_not_supported channel={channel!r}", flush=True)
        await broadcast_to_user(user_id, {
            "type": "radio_state",
            "channel": channel,
            "enabled": False,
            "error": "channel_not_supported",
        })
        return

    mgr = get_radio_manager()
    if not enabled:
        sess = mgr.disable(user_id, channel)
        if sess is None:
            sess_dict = {"type": "radio_state", "channel": channel, "enabled": False}
        else:
            sess_dict = sess.to_broadcast_dict()
        await broadcast_to_user(user_id, sess_dict)
        return

    # Resolve the seed — payload wins; falls back to existing session or last
    # played track so a toggle-on without explicit seed still works.
    seed_video_id = (msg.get("video_id") or "").strip()
    seed_title = (msg.get("title") or "").strip()
    seed_intent = (msg.get("seed_intent") or "").strip()
    seed_source = "payload" if seed_video_id else None

    if not seed_video_id:
        existing = mgr.get(user_id, channel)
        if existing and existing.seed_track:
            seed_video_id = existing.seed_track.video_id
            seed_title = seed_title or existing.seed_track.title
            seed_intent = seed_intent or existing.seed_intent
            seed_source = "session"

    if not seed_video_id and _agent_runner and getattr(_agent_runner, "tools", None):
        last = getattr(_agent_runner.tools, "_last_media", None)
        if last and last.get("video_id"):
            seed_video_id = last["video_id"]
            seed_title = seed_title or last.get("title", "") or "Now Playing"
            seed_source = "last_media"

    if not seed_video_id:
        print(
            f"[radio] toggle REJECT no_seed_track user={user_id[:8]} channel={channel!r}",
            flush=True,
        )
        await broadcast_to_user(user_id, {
            "type": "radio_state",
            "channel": channel,
            "enabled": False,
            "error": "no_seed_track",
        })
        return

    # Build the YT Music station BEFORE we mark the session enabled — if YT
    # Music rejects the seed (non-music, region-locked, etc.) we don't want
    # the UI pinned ON with no queue behind it.
    station = await build_station(seed_video_id, limit=50)
    if not station:
        print(
            f"[radio] toggle REJECT build_station returned empty "
            f"user={user_id[:8]} seed_video_id={seed_video_id}",
            flush=True,
        )
        await broadcast_to_user(user_id, {
            "type": "radio_state",
            "channel": channel,
            "enabled": False,
            "error": "station_unavailable",
        })
        await broadcast_to_user(user_id, {
            "type": "radio_notice",
            "channel": channel,
            "message": "Couldn't build a radio station from this track — try a different song.",
        })
        return

    sess = mgr.enable(
        user_id=user_id,
        channel=channel,
        seed_intent=seed_intent or seed_title or "music",
        seed_track=SeedTrack(video_id=seed_video_id, title=seed_title or "Now Playing"),
        station=station,
    )
    top = station[0] if station else None
    print(
        f"[radio] toggle OK user={user_id[:8]} channel={channel} "
        f"seed_source={seed_source} seed_vid={seed_video_id} "
        f"station_size={len(station)} next={top.display_title() if top else None!r}",
        flush=True,
    )
    await broadcast_to_user(user_id, sess.to_broadcast_dict())


async def _advance_and_broadcast_next(user_id: str, channel: str, sess, trigger: str) -> bool:
    """Shared path for media_ended + skip_next: advance the playlist (possibly
    extending it first), apply Song-mode Topic lookup if set, record in history,
    broadcast media_play. Returns True on success, False on exhaustion.

    `trigger` is a log label: 'media_ended' | 'skip_next'.
    """
    from app.agent.radio import build_station, PLAYLIST_REFILL_THRESHOLD
    from app.agent.radio.playlist import find_topic_version
    from app.agent.radio.player import broadcast_radio_track

    mgr = get_radio_manager()

    # Step forward in history first — if the user walked back via skip_prev,
    # skip_next should replay the tape before advancing the playlist.
    if trigger == "skip_next":
        stepped = mgr.step_forward_in_history(sess)
        if stepped is not None:
            print(
                f"[radio] skip_next cursor_step_forward history_cursor="
                f"{sess.history_cursor}/{len(sess.played_history) - 1} "
                f"video_id={stepped.video_id}",
                flush=True,
            )
            await _broadcast_track_for_mode(user_id, channel, sess, stepped, trigger, record=False)
            return True

    # Pop next from playlist; extend if within threshold.
    next_track = mgr.pop_next_from_playlist(sess)
    remaining = len(sess.playlist) - sess.playlist_cursor
    if next_track is None or remaining <= PLAYLIST_REFILL_THRESHOLD:
        extend_seed = sess.current_track_id or (sess.seed_track.video_id if sess.seed_track else "")
        print(
            f"[radio] playlist extending ({trigger}) user={user_id[:8]} seed={extend_seed} "
            f"cursor={sess.playlist_cursor}/{len(sess.playlist)} remaining={remaining}",
            flush=True,
        )
        if extend_seed:
            new_tracks = await build_station(extend_seed, limit=50)
            if new_tracks:
                mgr.extend_playlist(sess, new_tracks)
        if next_track is None:
            next_track = mgr.pop_next_from_playlist(sess)

    if next_track is None:
        disabled = mgr.record_failure(sess)
        print(
            f"[radio] pick failed ({trigger}) user={user_id[:8]} — playlist exhausted, "
            f"extension produced no new tracks. disabled={disabled}",
            flush=True,
        )
        if disabled:
            await broadcast_to_user(user_id, {
                "type": "radio_state",
                "channel": channel,
                "enabled": False,
                "error": "no_more_tracks",
            })
            await broadcast_to_user(user_id, {
                "type": "radio_notice",
                "channel": channel,
                "message": "Ran out of tracks in that vibe — try a different song.",
            })
        return False

    # Song-mode Topic lookup — best-effort swap to the ATV variant when the
    # playlist track is an OMV (music video). Failure falls back to original.
    if sess.display_mode == "song" and next_track.video_type and next_track.video_type != "MUSIC_VIDEO_TYPE_ATV":
        alt = await find_topic_version(next_track)
        if alt is not None:
            print(
                f"[radio] topic_lookup_ok track={next_track.video_id} → {alt.video_id} "
                f"title={alt.title!r}",
                flush=True,
            )
            next_track = alt

    mgr.record_auto_play(sess, next_track)
    print(
        f"[radio] playlist pop ({trigger}) cursor={sess.playlist_cursor - 1}/{len(sess.playlist)} "
        f"track={next_track.title!r} by={next_track.artist!r} "
        f"length={next_track.length!r} video_id={next_track.video_id} "
        f"history_cursor={sess.history_cursor}/{len(sess.played_history) - 1}",
        flush=True,
    )
    await _broadcast_track_for_mode(user_id, channel, sess, next_track, trigger, record=False)
    return True


async def _broadcast_track_for_mode(user_id, channel, sess, track, trigger: str, record: bool) -> None:
    """Broadcast a media_play for `track` plus a radio_state update. `record`
    is False for skip_prev and history-step-forward (track already in tape)."""
    from app.agent.radio.player import broadcast_radio_track
    await broadcast_radio_track(
        user_id=user_id,
        video_id=track.video_id,
        title=track.display_title(),
        channel=channel,
        artist=track.artist,
        thumbnail_url=track.thumbnail_url,
        video_type=track.video_type,
    )
    await broadcast_to_user(user_id, sess.to_broadcast_dict())


async def _handle_media_ended(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager

    channel = (msg.get("channel") or "").strip().lower()
    ended_video_id = (msg.get("video_id") or "").strip()

    print(
        f"[radio] media_ended entry user={user_id[:8]} channel={channel!r} video={ended_video_id}",
        flush=True,
    )

    if not RadioSessionManager.is_channel_allowed(channel):
        print(f"[radio] media_ended skip — channel_not_allowed channel={channel!r}", flush=True)
        return

    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None or not sess.enabled:
        print(
            f"[radio] media_ended — no active session; broadcasting OFF "
            f"user={user_id[:8]} sess_none={sess is None} "
            f"enabled={getattr(sess, 'enabled', None)}",
            flush=True,
        )
        await broadcast_to_user(user_id, {
            "type": "radio_state",
            "channel": channel,
            "enabled": False,
        })
        return

    # Defensive: tolerate end-event mismatch (frontend may report a prior
    # track on a fast double-end from YT's replay glitch).
    if ended_video_id and sess.current_track_id and ended_video_id != sess.current_track_id:
        logger.info(
            "[radio] ended_video_id=%s != current=%s — continuing anyway",
            ended_video_id, sess.current_track_id,
        )

    await _advance_and_broadcast_next(user_id, channel, sess, trigger="media_ended")


async def _handle_radio_skip_next(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager

    channel = (msg.get("channel") or "").strip().lower()
    print(f"[radio] skip_next entry user={user_id[:8]} channel={channel!r}", flush=True)

    if not RadioSessionManager.is_channel_allowed(channel):
        return
    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None or not sess.enabled:
        print(f"[radio] skip_next — no active session user={user_id[:8]}", flush=True)
        await broadcast_to_user(user_id, {
            "type": "radio_state",
            "channel": channel,
            "enabled": False,
            "error": "not_enabled",
        })
        return
    print(
        f"[radio] skip_next clicked cursor={sess.playlist_cursor}/{len(sess.playlist)} "
        f"history={sess.history_cursor}/{len(sess.played_history) - 1}",
        flush=True,
    )
    await _advance_and_broadcast_next(user_id, channel, sess, trigger="skip_next")


async def _handle_radio_skip_prev(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager

    channel = (msg.get("channel") or "").strip().lower()
    print(f"[radio] skip_prev entry user={user_id[:8]} channel={channel!r}", flush=True)

    if not RadioSessionManager.is_channel_allowed(channel):
        return
    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None or not sess.enabled:
        print(f"[radio] skip_prev — no active session user={user_id[:8]}", flush=True)
        return
    print(
        f"[radio] skip_prev clicked history_cursor={sess.history_cursor}/{len(sess.played_history) - 1}",
        flush=True,
    )
    prev_track = mgr.skip_prev(sess)
    if prev_track is None:
        print(f"[radio] skip_prev — at start of tape, no-op user={user_id[:8]}", flush=True)
        # Re-broadcast state so frontend knows can_prev is now false.
        await broadcast_to_user(user_id, sess.to_broadcast_dict())
        return
    print(
        f"[radio] skip_prev replaying history[{sess.history_cursor}] "
        f"video_id={prev_track.video_id} title={prev_track.title!r}",
        flush=True,
    )
    await _broadcast_track_for_mode(user_id, channel, sess, prev_track, "skip_prev", record=False)


async def _handle_radio_display_mode(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager

    channel = (msg.get("channel") or "").strip().lower()
    mode = (msg.get("mode") or "").strip().lower()
    print(f"[radio] display_mode entry user={user_id[:8]} channel={channel!r} mode={mode!r}", flush=True)

    if not RadioSessionManager.is_channel_allowed(channel):
        return
    if mode not in ("song", "video"):
        return
    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None:
        # No active session yet — create one in disabled state just to remember
        # the user's mode preference for when they next toggle radio on.
        sess = mgr.get_or_create(user_id, channel)
    mgr.set_display_mode(sess, mode)
    print(
        f"[radio] display_mode set user={user_id[:8]} channel={channel} mode={sess.display_mode}",
        flush=True,
    )
    await broadcast_to_user(user_id, sess.to_broadcast_dict())


# References set at startup
_agent_runner = None
_skill_loader = None


def set_ws_refs(agent_runner, skill_loader=None):
    """Set references to the agent runner and skill loader (called from main.py lifespan)."""
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate a JWT token and return the user_id, or None."""
    try:
        from app.services import decode_access_token, get_user_by_id
        from app.db.database import async_session_maker

        user_id = decode_access_token(token)
        if not user_id:
            return None

        async with async_session_maker() as db:
            user = await get_user_by_id(db, user_id)
            if user and user.is_active:
                return user.id
        return None
    except Exception as e:
        logger.warning(f"WS auth failed: {e}")
        return None


@router.websocket("/ws/chat")
async def ws_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time chat with the agent.

    Supports streaming text chunks, tool call indicators, and session management.
    Auth: JWT token (query param or first message) OR agent_key (for platform proxy).
    """
    await websocket.accept()
    user_id: Optional[str] = None

    # Try agent_key auth first (platform proxy mode)
    if agent_key and settings.agent_api_key and agent_key == settings.agent_api_key:
        user_id = settings.user_id
        if user_id:
            # Ensure stub user exists (same as auth.py agent mode)
            from app.db.database import async_session_maker as _sm
            from app.db.models import User
            async with _sm() as _db:
                from app.services.auth_service import get_user_by_id
                u = await get_user_by_id(_db, user_id)
                if not u:
                    u = User(id=user_id, email=f"{user_id[:8]}@agent.local", hashed_password="", name="Agent Owner")
                    _db.add(u)
                    await _db.commit()

    # Try JWT query-param auth
    if not user_id and token:
        user_id = await _authenticate_ws(token)

    try:
        # If not authenticated via query param, expect auth message
        if not user_id:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                msg = json.loads(raw)
                if msg.get("type") == "auth" and msg.get("token"):
                    user_id = await _authenticate_ws(msg["token"])
            except asyncio.TimeoutError:
                pass

        if not user_id:
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            await websocket.close(code=4001, reason="Unauthorized")
            return

        if not _agent_runner:
            await websocket.send_json({"type": "error", "message": "Agent not available"})
            await websocket.close(code=4503, reason="Service unavailable")
            return

        logger.info(f"[WS] Authenticated user: {user_id}")

        # Register broadcast queue for this connection
        broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _register_ws_queue(user_id, broadcast_queue)

        async def _broadcast_reader():
            """Forward broadcast events to this WebSocket."""
            _persisted_job_ids: set = set()

            try:
                while True:
                    event = await broadcast_queue.get()
                    etype = event.get("type", "?")
                    print(f"[BROADCAST_READER] Forwarding event type={etype} to WS for user={user_id[:8]}", flush=True)
                    try:
                        await websocket.send_json(event)
                        print(f"[BROADCAST_READER] Sent OK: type={etype}", flush=True)
                    except Exception as e:
                        print(f"[BROADCAST_READER] Send FAILED: type={etype} error={e}", flush=True)
                        break

                    # Persist job cards to DB so they survive page reload
                    if etype == "job_update" and event.get("job_id"):
                        _jid = event["job_id"]
                        if _jid not in _persisted_job_ids:
                            _persisted_job_ids.add(_jid)
                            try:
                                from app.db.database import async_session_maker as _sm
                                from app.db.models import Message as _Msg
                                from sqlalchemy import select as _sel
                                async with _sm() as _jdb:
                                    _existing = await _jdb.execute(
                                        _sel(_Msg).where(_Msg.id == f"job-{_jid}")
                                    )
                                    if not _existing.scalar_one_or_none():
                                        _job_dc = await _resolve_day_chat_id_for_now(_jdb, user_id)
                                        _jdb.add(_Msg(
                                            id=f"job-{_jid}",
                                            conversation_id=f"build-{_jid[:8]}",
                                            day_chat_id=_job_dc,
                                            role="job",
                                            content=json.dumps({
                                                "job_id": _jid,
                                                "job_name": event.get("name", "App Build"),
                                            }),
                                        ))
                                        await _jdb.commit()
                            except Exception as _pe:
                                print(f"[BROADCAST_READER] Job persist failed: {_pe}", flush=True)
            except asyncio.CancelledError:
                print(f"[BROADCAST_READER] Cancelled for user={user_id[:8]}", flush=True)

        broadcast_task = asyncio.create_task(_broadcast_reader())

        # Server-side keepalive: send periodic pings to prevent proxy idle timeouts.
        # Railway and Cloudflare drop idle WebSocket connections after ~30s.
        # Research subagent can take 15-30s with no data flowing — this keeps the WS alive.
        WS_KEEPALIVE_INTERVAL = 15  # seconds

        async def _keepalive():
            try:
                while True:
                    await asyncio.sleep(WS_KEEPALIVE_INTERVAL)
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass

        keepalive_task = asyncio.create_task(_keepalive())

        # Main message loop
        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info(f"[WS] Client disconnected: {user_id}")
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type in (
                    "radio_toggle", "media_ended",
                    "radio_skip_next", "radio_skip_prev", "radio_display_mode",
                ):
                    print(f"[WS IN] user={user_id[:8]} type={msg_type} keys={list(msg.keys())}", flush=True)

                # ── Radio Mode: toggle / track-ended / skip / display-mode ──
                if msg_type == "radio_toggle":
                    await _handle_radio_toggle(user_id, msg)
                    continue

                if msg_type == "media_ended":
                    asyncio.create_task(_handle_media_ended(user_id, msg))
                    continue

                if msg_type == "radio_skip_next":
                    asyncio.create_task(_handle_radio_skip_next(user_id, msg))
                    continue

                if msg_type == "radio_skip_prev":
                    asyncio.create_task(_handle_radio_skip_prev(user_id, msg))
                    continue

                if msg_type == "radio_display_mode":
                    await _handle_radio_display_mode(user_id, msg)
                    continue

                if msg_type != "message":
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                    continue

                text = msg.get("text", "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "message": "Empty message"})
                    continue

                # ── Onboarding trigger (DEPRECATED — Soul page is now the onboarding)
                is_onboarding_msg = False
                if text == "__ONBOARDING_START__":
                    from app.db.database import async_session_maker
                    from app.db.models import AgentConfig
                    async with async_session_maker() as _db:
                        _cfg = (await _db.execute(
                            select(AgentConfig).where(AgentConfig.user_id == user_id)
                        )).scalar_one_or_none()
                        if _cfg and _cfg.onboarding_completed:
                            text = "Hello!"
                        else:
                            # Redirect to Soul page instead of running text onboarding
                            await websocket.send_json({"type": "redirect", "url": "/agent/soul?onboarding=true"})
                            continue

                session_id = msg.get("session_id")
                model = msg.get("model")
                channel = msg.get("channel")  # e.g. "mobile", "web", "app"
                client_tz = msg.get("tz")  # IANA timezone, e.g. "America/Toronto"
                force_new_session = bool(msg.get("force_new"))  # Skip session reuse (New Thread in app workspace)
                # system_action: true when a structured action (e.g. customize_app) triggers
                # an agent turn without a real user message. Skip presave + user message save.
                _is_system_action = bool(msg.get("system_action"))

                # ── Persist timezone to User if changed ──
                # Self-healing: frontend sends tz on every message, we persist it once.
                # If timezone changes from one real value to another, queue a re-bucket.
                if client_tz and isinstance(client_tz, str) and len(client_tz) < 50:
                    try:
                        from app.db.database import async_session_maker as _tz_sm
                        from app.db.models import User
                        async with _tz_sm() as _tz_db:
                            _user = (await _tz_db.execute(
                                select(User).where(User.id == user_id)
                            )).scalar_one_or_none()
                            if _user:
                                _old_tz = _user.timezone
                                if _old_tz != client_tz:
                                    _user.timezone = client_tz
                                    await _tz_db.commit()
                                    logger.info("[WS] Updated timezone for %s: %s → %s", user_id[:8], _old_tz, client_tz)

                                    # Auto-rebucket if timezone changed from one real value to another
                                    # (not just NULL → real, which is the initial backfill case)
                                    if _old_tz and _old_tz != "UTC" and _old_tz != client_tz:
                                        import os as _tz_os
                                        async def _trigger_rebucket():
                                            try:
                                                import importlib.util as _ilu
                                                _spec = _ilu.spec_from_file_location(
                                                    "backfill_day_chats",
                                                    _tz_os.path.join(_tz_os.path.dirname(_tz_os.path.dirname(__file__)), "services", "backfill_day_chats.py"),
                                                )
                                                _bmod = _ilu.module_from_spec(_spec)
                                                _spec.loader.exec_module(_bmod)

                                                # Reset migration status to not_started
                                                from app.db.models.day_chat import MigrationStatus
                                                async with _tz_sm() as _rb_db:
                                                    ms = (await _rb_db.execute(
                                                        select(MigrationStatus).where(
                                                            MigrationStatus.migration_name == "day_chat_backfill"
                                                        )
                                                    )).scalar_one_or_none()
                                                    if ms:
                                                        ms.status = "not_started"
                                                        ms.started_at = None
                                                        ms.completed_at = None
                                                        ms.progress_json = None
                                                        ms.error_message = None
                                                        await _rb_db.commit()

                                                result = await _bmod.run_backfill(_tz_sm)
                                                logger.info("day_chat_backfill.rebucket_completed tz_change=%s→%s result=%s", _old_tz, client_tz, result)
                                            except Exception as _rbe:
                                                logger.error("day_chat_backfill.rebucket_failed error=%s", _rbe)

                                        # NOTE: If the WS connection closes mid-rebucket, this task may be
                                        # garbage-collected and MigrationStatus left in 'in_progress'.
                                        # This is acceptable — the next agent restart resumes it via
                                        # the standard backfill startup path (in_progress → resume).
                                        asyncio.create_task(_trigger_rebucket())
                                        logger.info("day_chat_backfill.rebucket_queued tz_change=%s→%s", _old_tz, client_tz)
                    except Exception as _tz_err:
                        logger.debug("[WS] Timezone persistence skipped: %s", _tz_err)

                # If message comes from inside a built app, prepend context
                # Uses the consolidated build_layer2_context (Checkpoint 5 Part 2, Risk 5)
                app_id_from_msg = msg.get("app_id")
                _original_user_text = text  # Preserve before context injection
                if channel == "app" and app_id_from_msg:
                    try:
                        from app.db.database import async_session_maker
                        from app.services.layer2_context import build_layer2_context
                        _is_layer2 = msg.get("layer2") or False
                        async with async_session_maker() as _db:
                            _l2_ctx = await build_layer2_context(app_id_from_msg, _db, is_layer2=_is_layer2)
                            if _l2_ctx:
                                text = f"{_l2_ctx.render(is_layer2=_is_layer2)}\n\n{text}"
                    except Exception as e:
                        logger.warning(f"[WS] Failed to load app context: {e}")

                # Terminal activity: show user message
                _tprint(f"\n{_CYAN_BOLD} user {_RESET} {text}")

                # ── Pre-save user message so it survives stream failures ──
                # Skip if no session_id yet (first message — agent_runner creates session)
                # Skip for system actions (customize_app) — no user message to save
                _user_msg_presaved = False
                if session_id and not _is_system_action:
                    try:
                        from app.db.database import async_session_maker
                        from app.db.models import Message as DbMessage, Conversation
                        async with async_session_maker() as _presave_db:
                            _presave_dc_id = await _resolve_day_chat_id_for_now(_presave_db, user_id, tz_override=client_tz)
                            _presave_db.add(DbMessage(
                                conversation_id=session_id,
                                day_chat_id=_presave_dc_id,
                                role="user",
                                content=_original_user_text,
                            ))
                            # Update conversation timestamp
                            _conv = (await _presave_db.execute(
                                __import__('sqlalchemy').select(Conversation).where(Conversation.id == session_id)
                            )).scalar_one_or_none()
                            if _conv:
                                _conv.message_count = (_conv.message_count or 0) + 1
                                _conv.updated_at = __import__('datetime').datetime.utcnow()
                            await _presave_db.commit()
                        _user_msg_presaved = True
                    except Exception as _pse:
                        logger.warning(f"[WS] Failed to pre-save user message: {_pse}")

                # Accumulate streamed text for partial-save on error
                _streamed_chunks: list = []

                # Stream callbacks
                async def on_text_chunk(chunk: str):
                    _streamed_chunks.append(chunk)
                    try:
                        await websocket.send_json({"type": "text_chunk", "text": chunk})
                    except Exception:
                        pass

                async def on_attachment(message_id: str, att: dict):
                    """Emit `attachment` event when a generate_* tool produces a file.

                    The frontend uses this to open the DocumentSplit pane (or inline
                    fallback on narrow viewports) and attach the file reference to
                    the currently-streaming assistant message.
                    """
                    mime = att.get("mime_type", "")
                    aid = att.get("id", "")
                    preview_mimes = {
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    }
                    payload = {
                        "type": "attachment",
                        "message_id": message_id,
                        "attachment_id": aid,
                        "filename": att.get("filename", ""),
                        "mime_type": mime,
                        "size_bytes": att.get("size_bytes", 0),
                        "download_url": f"{settings.api_prefix}/files/{message_id}/{aid}",
                    }
                    if mime in preview_mimes or mime == "application/pdf" or mime.startswith("image/"):
                        payload["preview_url"] = f"{settings.api_prefix}/files/{message_id}/{aid}/preview?format=html"
                    try:
                        await websocket.send_json(payload)
                    except Exception:
                        pass

                async def on_tool_start(tool_name: str):
                    _tprint(f"{_DIM}  ⚙ {tool_name}{_RESET}")
                    try:
                        await websocket.send_json({"type": "tool_start", "tool": tool_name})
                    except Exception:
                        pass

                # Collect build job info during tool execution for later persistence
                _pending_job_cards: list = []

                async def on_tool_end(tool_name: str, summary: str, tool_input: dict = None):
                    short = summary[:120] + "..." if len(summary) > 120 else summary
                    # Collapse to single line for terminal readability
                    short = short.replace("\n", " ")
                    _tprint(f"{_DIM}  ✓ {tool_name}: {short}{_RESET}")
                    try:
                        event: dict = {"type": "tool_end", "tool": tool_name, "summary": summary}
                        # Enrich with file data for Layer 2 editor UI
                        if tool_input:
                            _tn = tool_name.lower()
                            if "write_file" in _tn or "edit_file" in _tn:
                                event["file_path"] = tool_input.get("file_path", "")
                                event["file_action"] = "write"
                                if "write_file" in _tn and tool_input.get("content"):
                                    content = tool_input["content"]
                                    event["code"] = content[:50000]
                                    event["lines"] = content.count("\n") + 1
                                    event["size"] = len(content)
                            elif "read_file" in _tn:
                                event["file_path"] = tool_input.get("file_path", "")
                                event["file_action"] = "read"
                                # Include the read content from the summary (tool output)
                                if summary and len(summary) > 10:
                                    event["code"] = summary[:50000]
                                    event["lines"] = summary.count("\n") + 1
                                    event["size"] = len(summary)
                        await websocket.send_json(event)
                    except Exception:
                        pass

                    # Capture build job info for persistence after agent completes
                    if tool_name == "app_builder__build_app" and summary:
                        import re
                        _jid_m = re.search(r"Job ID:\s*([a-f0-9-]+)", summary, re.I)
                        _jnm_m = re.search(r"Building '([^']+)'", summary)
                        if _jid_m:
                            _pending_job_cards.append({
                                "job_id": _jid_m.group(1),
                                "job_name": _jnm_m.group(1) if _jnm_m else "App Build",
                            })

                # Handle media attachments (images/files from frontend)
                _media_paths = []
                _media_items = msg.get("media", [])
                if _media_items and isinstance(_media_items, list):
                    import tempfile, base64 as _b64, os as _os
                    for _mi in _media_items[:5]:  # Max 5 attachments
                        try:
                            _mtype = _mi.get("type", "image/png")
                            _mdata = _mi.get("data", "")
                            _mname = _mi.get("name", "")
                            if not _mdata:
                                continue
                            # Prefer extension from filename, fall back to MIME mapping
                            _ext = ""
                            if _mname:
                                _ext = _os.path.splitext(_mname.lower())[1]
                            if not _ext:
                                _ext = {
                                    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
                                    "image/webp": ".webp", "application/pdf": ".pdf",
                                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                                    "application/zip": ".zip",
                                    "application/x-zip-compressed": ".zip",
                                }.get(_mtype, ".bin")
                            # Write to temp dir with original filename so agent runner sees the real name
                            _tmpdir = tempfile.mkdtemp(prefix="toup_media_")
                            _fname = _mname or f"attachment{_ext}"
                            # Sanitize filename (keep only the basename, no path traversal)
                            _fname = _os.path.basename(_fname)
                            _fpath = _os.path.join(_tmpdir, _fname)
                            with open(_fpath, "wb") as _wf:
                                _wf.write(_b64.b64decode(_mdata))
                            _media_paths.append(_fpath)
                        except Exception as _me:
                            logger.warning("[WS] Failed to process media attachment: %s", _me)

                # ── Fast-path: detect play/music requests and fire media_play immediately ──
                # Returns (modified_text, media_meta) if media was found, so agent skips play_media
                _fast_result = await _fast_media_check(text, user_id, broadcast_queue)
                _fast_text = _fast_result[0] if _fast_result else None
                _agent_text = _fast_text or text
                # Pre-set _last_media so the inline card persists on the saved message
                if _fast_result and hasattr(_agent_runner, 'tools'):
                    _agent_runner.tools._last_media = _fast_result[1]

                # Radio: record this as a user-driven seed for the current channel.
                # New unrelated intent while radio is ON → session flips to OFF (per spec).
                if _fast_result:
                    try:
                        from app.agent.radio import get_radio_manager, RadioSessionManager
                        from app.agent.radio.session import SeedTrack
                        if RadioSessionManager.is_channel_allowed(channel):
                            _mgr = get_radio_manager()
                            _meta = _fast_result[1] or {}
                            _mgr.record_user_seed(
                                user_id=user_id,
                                channel=channel,
                                seed_intent=text.strip(),
                                seed_track=SeedTrack(
                                    video_id=_meta.get("video_id", ""),
                                    title=_meta.get("title", "") or "YouTube Video",
                                ),
                            )
                            # Sync the toggle UI (toggle stays OFF for a fresh seed,
                            # unless the user re-enables it explicitly).
                            _sess = _mgr.get(user_id, channel)
                            if _sess:
                                await broadcast_to_user(user_id, _sess.to_broadcast_dict())
                    except Exception as _re:
                        logger.warning("[radio] seed-record failed: %s", _re)

                # Task intent detection — detect imperative task requests in regular chat
                _chat_task_job_id = None
                if channel != "vibecoding" and not _fast_result:
                    _chat_task_job_id = await _detect_and_create_task(
                        text, user_id, session_id, broadcast_queue
                    )

                # Run agent — use the modified text for LLM but save the original user text
                # display_user_message: the clean text to save to DB (no context injection).
                # For app-channel messages, text has [CONTEXT:...] prepended — always
                # save the original. For fast-path (music), save the modified fast text.
                _display_text = _original_user_text if (_original_user_text != text) else (text if _fast_text else None)
                agent_task = asyncio.create_task(_agent_runner.run(
                    user_message=_agent_text,
                    display_user_message=_display_text,
                    user_id=user_id,
                    session_id=session_id,
                    channel=channel,
                    on_text_chunk=on_text_chunk,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    on_attachment=on_attachment,
                    model_override=model,
                    save_user_message=not is_onboarding_msg and not _user_msg_presaved and not _is_system_action,
                    media_paths=_media_paths if _media_paths else None,
                    client_tz=client_tz,
                    app_id=app_id_from_msg,
                    force_new_session=force_new_session,
                ))

                # Wait for agent to finish, but also listen for stop via a receiver task
                async def _wait_for_stop():
                    """Read WS messages while agent runs. Cancel agent on 'stop'."""
                    try:
                        while not agent_task.done():
                            raw2 = await websocket.receive_text()
                            try:
                                m2 = json.loads(raw2)
                            except json.JSONDecodeError:
                                continue
                            if m2.get("type") == "stop":
                                agent_task.cancel()
                                logger.info(f"[WS] Agent stopped by user: {user_id}")
                                return
                            elif m2.get("type") == "ping":
                                await websocket.send_json({"type": "pong"})
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass

                stop_task = asyncio.create_task(_wait_for_stop())

                # Helper to safely send on WS (may be closed)
                async def _safe_send(data: dict):
                    try:
                        await websocket.send_json(data)
                    except RuntimeError:
                        pass  # WS already closed

                try:
                    response = await agent_task
                    stop_task.cancel()
                    try:
                        await stop_task  # Wait for stop task to actually finish
                    except (asyncio.CancelledError, Exception):
                        pass

                    # Finalize chat task job if one was created
                    if _chat_task_job_id:
                        try:
                            from app.db.database import async_session_maker as _sm
                            from app.db.models import BuildJob as _BJ
                            async with _sm() as _tdb:
                                _tj = await _tdb.get(_BJ, _chat_task_job_id)
                                if _tj and _tj.status == "running":
                                    _tj.status = "completed"
                                    _tj.completed_at = datetime.utcnow()
                                    _tj.total_tokens = response.tokens_total or 0
                                    _tj.model = response.model or ""
                                    await _tdb.commit()
                            await broadcast_queue.put({
                                "type": "job_update",
                                "job_id": _chat_task_job_id,
                                "status": "completed",
                            })
                        except Exception as _te:
                            logger.warning(f"[TASK] Failed to finalize task job: {_te}")

                    # Terminal activity: show agent response summary
                    resp_preview = response.text[:200].replace("\n", " ")
                    if len(response.text) > 200:
                        resp_preview += "..."
                    _tprint(f"{_GREEN_BOLD} agent {_RESET} {resp_preview}")
                    _tprint(
                        f"{_DIM}  ({response.tokens_total or 0} tokens, "
                        f"{response.processing_time_ms or 0}ms, "
                        f"{response.model or '?'}){_RESET}"
                    )

                    # Check if onboarding just completed (agent stored the signal memory)
                    for tc in response.tool_calls:
                        if tc.get("name") == "memory_store":
                            tc_content = (tc.get("input") or {}).get("content", "")
                            if "onboarding complete" in tc_content.lower():
                                try:
                                    from app.db.database import async_session_maker
                                    from app.db.models import AgentConfig
                                    async with async_session_maker() as _db:
                                        _cfg = (await _db.execute(
                                            select(AgentConfig).where(AgentConfig.user_id == user_id)
                                        )).scalar_one_or_none()
                                        if _cfg:
                                            _cfg.onboarding_completed = True
                                            await _db.commit()
                                            logger.info(f"[WS] Onboarding completed for user {user_id}")
                                except Exception as e:
                                    logger.warning(f"[WS] Failed to mark onboarding complete: {e}")
                                break

                    # Persist any build job card messages to the session
                    if _pending_job_cards and response.session_id:
                        try:
                            from app.db.database import async_session_maker
                            from app.db.models import Message as MsgModel
                            async with async_session_maker() as _jdb:
                                for _jc in _pending_job_cards:
                                    _jmid = f"job-{_jc['job_id']}"
                                    _existing = await _jdb.execute(
                                        select(MsgModel).where(MsgModel.id == _jmid)
                                    )
                                    if not _existing.scalar_one_or_none():
                                        _job_dc = await _resolve_day_chat_id_for_now(_jdb, user_id, tz_override=client_tz)
                                        _jdb.add(MsgModel(
                                            id=_jmid,
                                            conversation_id=response.session_id,
                                            day_chat_id=_job_dc,
                                            role="job",
                                            content=json.dumps({
                                                "job_id": _jc["job_id"],
                                                "job_name": _jc["job_name"],
                                            }),
                                        ))
                                await _jdb.commit()
                                logger.info(f"[WS] Persisted {len(_pending_job_cards)} job card(s) to session {response.session_id}")
                        except Exception as _je:
                            logger.warning(f"[WS] Failed to persist job cards: {_je}")
                        _pending_job_cards.clear()

                    _done_payload = {
                        "type": "done",
                        "text": response.text,
                        "session_id": response.session_id,
                        "tokens": {
                            "input": response.tokens_input,
                            "output": response.tokens_output,
                            "total": response.tokens_total,
                        },
                        "model": response.model,
                        "tool_calls": len(response.tool_calls),
                        "processing_time_ms": response.processing_time_ms,
                    }
                    # Day-as-Chat: include day_chat_id so frontend can group by day
                    if getattr(response, 'day_chat_id', None):
                        _done_payload["day_chat_id"] = response.day_chat_id
                    # Include any build jobs triggered during this run
                    if _pending_job_cards:
                        _done_payload["build_jobs"] = list(_pending_job_cards)
                    await _safe_send(_done_payload)

                except asyncio.CancelledError:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    # Save partial response on user-stop so it's not lost
                    _partial = "".join(_streamed_chunks).strip()
                    if _partial:
                        try:
                            async with async_session_maker() as _err_db:
                                _err_dc = await _resolve_day_chat_id_for_now(_err_db, user_id, tz_override=client_tz)
                                _err_db.add(DbMessage(
                                    conversation_id=session_id, day_chat_id=_err_dc, role="assistant",
                                    content=_partial + "\n\n*[Generation stopped by user]*",
                                ))
                                await _err_db.commit()
                        except Exception:
                            pass
                    await _safe_send({"type": "stopped", "message": "Generation stopped"})
                except Exception as e:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    logger.exception(f"[WS] Agent error for {user_id}")
                    # Mark chat task job as failed if one was created
                    if _chat_task_job_id:
                        try:
                            from app.db.database import async_session_maker as _sm2
                            from app.db.models import BuildJob as _BJ2
                            async with _sm2() as _fdb:
                                _fj = await _fdb.get(_BJ2, _chat_task_job_id)
                                if _fj and _fj.status == "running":
                                    _fj.status = "failed"
                                    _fj.error_message = str(e)[:500]
                                    _fj.completed_at = datetime.utcnow()
                                    await _fdb.commit()
                        except Exception:
                            pass
                    _tprint(f"\033[1;31m  ✗ Error: {e}{_RESET}")
                    # Save partial response so it's not lost on refresh
                    _partial = "".join(_streamed_chunks).strip()
                    if _partial and session_id:
                        try:
                            async with async_session_maker() as _err_db:
                                _err_dc = await _resolve_day_chat_id_for_now(_err_db, user_id, tz_override=client_tz)
                                _err_db.add(DbMessage(
                                    conversation_id=session_id, day_chat_id=_err_dc, role="assistant",
                                    content=_partial + "\n\n*[Response interrupted due to an error]*",
                                ))
                                await _err_db.commit()
                        except Exception as _save_err:
                            logger.warning(f"[WS] Failed to save partial response: {_save_err}")
                    # Show user-friendly error instead of raw exception
                    user_msg = _friendly_error(e)
                    await _safe_send({"type": "error", "message": user_msg})
        finally:
            # Clean up broadcast queue and task
            broadcast_task.cancel()
            keepalive_task.cancel()
            _unregister_ws_queue(user_id, broadcast_queue)

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=4500)
        except Exception:
            pass
