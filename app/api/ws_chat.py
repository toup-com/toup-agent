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

    # ── Anthropic Claude subscription quota exhausted (CLI OAuth tokens) ──
    # Example: "You're out of extra usage. Add more at claude.ai/settings/usage..."
    if "out of extra" in msg_lower or "claude.ai/settings/usage" in msg_lower:
        return (
            "Your Claude subscription usage is exhausted for this window. To continue:\n"
            "• Wait for the 5-hour rolling window to reset, or\n"
            "• Add more usage at claude.ai/settings/usage, or\n"
            "• Set a different API key in Settings"
        )

    # ── Anthropic API console billing exhausted ──────────────────────────
    # The platform's shared Claude account (bundle mode) returns a hard 400:
    #   "Your credit balance is too low to access the Anthropic API. Please
    #    go to Plans & Billing to upgrade or purchase credits."
    # This is a PLATFORM-side outage — the bundle user can't fix it, and
    # pointing them at OpenAI billing (the old behavior) is doubly wrong:
    # wrong provider AND wrong actor. Match this BEFORE the generic
    # credit/billing branch so it doesn't fall through to the OpenAI copy.
    if "anthropic api" in msg_lower or ("plans & billing" in msg_lower) or ("credit balance is too low" in msg_lower):
        return (
            "The AI service is temporarily unavailable (the platform's Claude "
            "access needs attention). We've been alerted — please try again "
            "shortly, or switch your model in Settings to keep working now."
        )

    # ── OpenAI quota / billing exhausted (must check before generic 429) ──
    if any(kw in msg_lower for kw in ("insufficient_quota", "billing", "credit", "balance", "exceeded your current")) or ("quota" in msg_lower and "openai" in msg_lower):
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

    # ── Agent not finished provisioning yet (bundle proxy auth gate) ──
    # The platform LLM proxy returns these EXACT signatures when a bundle
    # agent's credentials aren't bound yet: "Missing token" / "Invalid token"
    # (no llm_token_hash match -> 401) or "Bundle subscription is not active"
    # (bundle_status != active -> 403). For a brand-new signup whose container
    # is reachable a hair before activation lands, the OLD code mapped the 401
    # to "Your API key is invalid" - a misleading dead-end (the user has no
    # API key to fix). Surface the honest, recoverable state instead. These
    # strings are bundle-proxy-specific, so a genuine BYOK bad key still falls
    # through to the "invalid key" branch below.
    if (
        "missing token" in msg_lower
        or "invalid token" in msg_lower
        or "bundle subscription is not active" in msg_lower
    ):
        return (
            "Your agent is still finishing setup - give it a few seconds and "
            "send your message again. If this keeps happening, reload the page."
        )

    # ── Auth errors — distinguish invalid key from expired token ──
    if status == 401 or "authentication" in msg_lower or "unauthorized" in msg_lower:
        if "expired" in msg_lower or "oauth" in msg_lower:
            return "Your API token has expired. Please reconnect or update your API key in Settings."
        return "Your API key is invalid. Please check your API key in Settings."

    # ── Permission errors ──
    if status == 403 or "permission" in msg_lower or "forbidden" in msg_lower:
        return "Your API key doesn't have access to this model. Please check your plan or switch models in Settings."

    # ── Bad request — surface Anthropic's own message when it looks human-readable ──
    if status == 400 or "bad_request" in name.lower():
        import re as _re
        m = _re.search(r"'message':\s*\"([^\"]+)\"", msg) or _re.search(r"'message':\s*'([^']+)'", msg)
        if m:
            provider_msg = m.group(1).strip()
            if provider_msg and len(provider_msg) < 400:
                return provider_msg
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


# ── Session-independent duplicate-message guard ──────────────────────────
# A brand-new user's FIRST message is sent with session_id=null (the agent
# creates the session). The DB-backed replay check further down requires a
# session/conversation row, so it is SKIPPED for that first message. During
# the post-onboarding boot window the WS drops and the client REPLAYS its
# queued (un-acked) message on reconnect — to the SAME agent container
# process — producing two agent turns → duplicate replies (the 2026-06
# "HI → two greetings" incident). This in-process guard dedupes by
# (user_id, client_msg_id) regardless of session_id, dropping the replay
# BEFORE dispatch rather than after the fact. TTL-bounded + size-capped; the
# realistic replay arrives within seconds on the same process, so an
# in-process guard is sufficient and complements (does not replace) the
# DB-level idempotency used once a session exists.
_recent_client_msgs: Dict[str, float] = {}  # "user_id:client_msg_id" → expiry (monotonic)
_RECENT_MSG_TTL_S = 180.0
_RECENT_MSG_MAX = 4096


def _dedup_seen_client_msg(key: str) -> bool:
    """Return True if `key` was already dispatched recently (a duplicate);
    otherwise record it with a TTL and return False. Bounded + self-pruning."""
    import time as _t
    now = _t.monotonic()
    if len(_recent_client_msgs) > _RECENT_MSG_MAX:
        for _k in [k for k, exp in _recent_client_msgs.items() if exp <= now]:
            _recent_client_msgs.pop(_k, None)
    exp = _recent_client_msgs.get(key)
    if exp is not None and exp > now:
        return True
    _recent_client_msgs[key] = now + _RECENT_MSG_TTL_S
    return False


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
#
# IMPORTANT: reminder-shaped phrases ("remind me…", "schedule a…", "ping
# me at…", "wake me up in…") are deliberately EXCLUDED. They flow into
# the routines/reminder skill (routines__remind), not the agent_task
# BuildJob pipeline. Pre-2026-05-18 this regex matched "remind\s+me"
# which created an empty agent_task BuildJob for every reminder, leaving
# an orphan "Thinking…" indicator in chat (because the BuildJob's
# steps_json stayed "[]" forever) and a bogus "No build steps recorded"
# job card on /jobs.
import re as _re
_TASK_INTENT_PATTERN = _re.compile(
    r"^(research|find\s+(me\s+)?|look\s+up|monitor|track|set\s+up|"
    r"analyze|investigate|summarize|compile|gather|collect|prepare|"
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

    title = text.strip()[:60]

    # PR 4c (unified-jobs arc): repoint through ``JobRunner.create_job``
    # so the new columns are populated for chat-intent tasks.
    #   - source_kind='chat_intent' (versus 'manual' for dashboard input).
    #   - source_id = session_id when present — links the task back to
    #     the chat conversation that spawned it.
    #   - conversation_id = session_id too — Mission Control reads this
    #     to show "from chat with ___" attribution.
    # No idempotency_key: the same text fired twice IS a legitimate
    # two-task request from the user's perspective.
    from app.agent.job_runner import JobRunner, TaskSpec
    spec = TaskSpec(
        user_id=user_id,
        channel="chat_intent",
        source_kind="chat_intent",
        source_id=session_id,
        conversation_id=session_id,
    )
    try:
        job = await JobRunner().create_job(
            job_type="agent_task",
            spec=spec,
            title=title,
            prompt=text[:2000],
            status="running",
            layer=0,
        )
        job_id = job.id
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
        sess = mgr.disable(user_id, channel, source="toggle_off")
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
    # the UI pinned ON with no queue behind it. We also get back the seed's
    # own metadata from the watch-playlist's index 0 — used below to drive
    # the authoritative iframe swap on toggle-on (Rule 9).
    seed_meta, station = await build_station(seed_video_id, limit=50)
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
        source="toggle_on",
    )
    top = station[0] if station else None
    print(
        f"[radio] toggle OK user={user_id[:8]} channel={channel} "
        f"seed_source={seed_source} seed_vid={seed_video_id} "
        f"station_size={len(station)} next={top.display_title() if top else None!r}",
        flush=True,
    )

    # Authoritative iframe swap (Rule 9). Radio toggle-on is a user-initiated
    # command — iframe, session, and UI must all sync to the clicked card,
    # NOT wait for whatever the iframe happens to be playing to end. Without
    # this broadcast, a user clicking Radio on a card that isn't currently
    # in the iframe gets a seeded session + queue but no playback — and the
    # stray-end no-op (294e3ee) correctly refuses to advance when the
    # unrelated iframe track ends. Net: radio does nothing until manual
    # playback. Broadcast media_play for the seed so loadVideoById runs and
    # iframe ↔ session sync by construction.
    already_iframe_synced = (
        sess.current_track_id == seed_video_id and top is None
    )
    print(
        f"[radio] toggle_seed_playback seed={seed_video_id} source=toggle_on "
        f"already_playing={already_iframe_synced}",
        flush=True,
    )
    from app.agent.radio.player import broadcast_radio_track
    seed_title_full = (seed_meta.title if seed_meta else seed_title) or "Now Playing"
    await broadcast_radio_track(
        user_id=user_id,
        video_id=seed_video_id,
        title=seed_title_full,
        channel=channel,
        artist=(seed_meta.artist if seed_meta else ""),
        thumbnail_url=(seed_meta.thumbnail_url if seed_meta else ""),
        video_type=(seed_meta.video_type if seed_meta else ""),
        reason="toggle_seed",
        upcoming=_upcoming_tracks(sess),
    )
    print(
        f"[radio] iframe_force_sync from=unknown to={seed_video_id} reason=toggle_on",
        flush=True,
    )

    await broadcast_to_user(user_id, sess.to_broadcast_dict())


async def _advance_and_broadcast_next(user_id: str, channel: str, sess, trigger: str) -> bool:
    """Shared path for media_ended + skip_next: advance the playlist (possibly
    extending it first), apply Song-mode Topic lookup if set, record in history,
    broadcast media_play. Returns True on success, False on exhaustion.

    `trigger` is a log label: 'media_ended' | 'skip_next'.
    """
    from app.agent.radio import build_station, PLAYLIST_REFILL_THRESHOLD, get_radio_manager
    from app.agent.radio.playlist import find_topic_version, find_music_video
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
            _seed_meta, new_tracks = await build_station(extend_seed, limit=50)
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

    # Auto-detect display mode from the popped track's video_type unless the
    # user has explicitly overridden. Runs BEFORE the Topic-swap branch so the
    # swap only triggers when the resolved mode is 'song' AND the track is OMV.
    mode_before = sess.display_mode
    effective_mode = mgr.maybe_auto_set_mode(sess, next_track)
    source_label = _video_type_source(next_track.video_type)
    print(
        f"[radio] track_loaded id={next_track.video_id} source={source_label} "
        f"auto_mode={effective_mode} override={sess.display_mode_user_override} "
        f"mode_was={mode_before}",
        flush=True,
    )

    # Mid-track variant swaps on auto-advance fire ONLY when the user has
    # explicitly picked a mode. Default (no override) plays whatever the
    # queue gives — Video on Topic-audio ATV = same as the original feature
    # before Song/Video existed. Once the user clicks the Song/Video pill,
    # subsequent auto-advanced tracks are reshaped to match their pick.
    # See Rule 9 in docs/skills/radio-mode/SKILL.md.
    if sess.display_mode_user_override:
        # Song-mode Topic lookup — swap to ATV when the popped track is an
        # OMV and the user is in Song mode (needs clean audio variant).
        if effective_mode == "song" and next_track.video_type == "MUSIC_VIDEO_TYPE_OMV":
            print(
                f"[radio] topic_lookup query={next_track.title!r} by {next_track.artist!r} "
                f"from={next_track.video_id}",
                flush=True,
            )
            alt = await find_topic_version(next_track)
            if alt is not None:
                print(
                    f"[radio] topic_swap from={next_track.video_id} to={alt.video_id} "
                    f"title={alt.title!r}",
                    flush=True,
                )
                next_track = alt

        # Video-mode Music-Video lookup — mirror for users who picked Video.
        elif effective_mode == "video" and next_track.video_type == "MUSIC_VIDEO_TYPE_ATV":
            print(
                f"[radio] mv_lookup query={next_track.title!r} by {next_track.artist!r} "
                f"from={next_track.video_id}",
                flush=True,
            )
            mv = await find_music_video(next_track)
            if mv is not None:
                print(
                    f"[radio] mv_swap from={next_track.video_id} to={mv.video_id} "
                    f"title={mv.title!r}",
                    flush=True,
                )
                next_track = mv

    mgr.record_auto_play(sess, next_track, source=trigger)
    print(
        f"[radio] playlist pop ({trigger}) cursor={sess.playlist_cursor - 1}/{len(sess.playlist)} "
        f"track={next_track.title!r} by={next_track.artist!r} "
        f"length={next_track.length!r} video_id={next_track.video_id} "
        f"video_type={next_track.video_type!r} "
        f"history_cursor={sess.history_cursor}/{len(sess.played_history) - 1}",
        flush=True,
    )
    await _broadcast_track_for_mode(user_id, channel, sess, next_track, trigger, record=False)
    return True


def _video_type_source(video_type: str) -> str:
    """Short label for logs."""
    if video_type == "MUSIC_VIDEO_TYPE_ATV":
        return "topic"
    if video_type == "MUSIC_VIDEO_TYPE_OMV":
        return "music_video"
    if video_type == "MUSIC_VIDEO_TYPE_UGC":
        return "user_clip"
    return "other" if video_type else "unknown"


def _upcoming_tracks(sess, n: int = 2) -> list:
    """The next `n` station tracks the playlist will pop, as lightweight dicts
    for the mobile pre-load queue (instant lock-screen skip / auto-advance).
    Reads straight from the in-memory playlist at the current cursor — no fetch."""
    out = []
    try:
        for t in sess.playlist[sess.playlist_cursor:sess.playlist_cursor + n]:
            out.append({
                "video_id": t.video_id,
                "title": t.display_title(),
                "artist": t.artist,
                "thumbnail_url": t.thumbnail_url,
            })
    except Exception:
        pass
    return out


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
        upcoming=_upcoming_tracks(sess),
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

    # End-event provenance check. See Process Rule 7 / Trap 8: the iframe
    # can play videos outside the radio session (page-refresh restore,
    # user-pasted URLs, stale state post-reconnect). Narrow tolerance — we
    # only advance when ended_video_id matches:
    #   (a) the session's current track
    #   (b) the active seed
    #   (c) a recent entry in played_history (last 10)
    # Commit 1 (this commit) adds the classification log; commit 2 promotes
    # the stray-end branch to a no-op.
    recent_history_ids = {
        t.video_id for t in sess.played_history[-10:]
    }
    in_session = (
        ended_video_id
        and (
            ended_video_id == sess.current_track_id
            or (sess.seed_track and ended_video_id == sess.seed_track.video_id)
            or ended_video_id in recent_history_ids
        )
    )
    if ended_video_id and not in_session:
        logger.info(
            "[radio] iframe_out_of_sync iframe=%s session=%s detected_at=media_ended "
            "seed=%s history_size=%d",
            ended_video_id,
            sess.current_track_id,
            sess.seed_track.video_id if sess.seed_track else None,
            len(sess.played_history),
        )
        logger.info(
            "[radio] media_ended stray_end video=%s current=%s — no-op (queue intact)",
            ended_video_id, sess.current_track_id,
        )
        # Re-broadcast radio_state so the frontend stays anchored to what
        # the session actually holds (seed + current), not to whatever the
        # iframe last played. No queue mutation.
        await broadcast_to_user(user_id, sess.to_broadcast_dict())
        return
    elif ended_video_id and ended_video_id != sess.current_track_id:
        # Matches the session somewhere (seed or history) but not current —
        # typically a user replay of an earlier track. Advance.
        logger.info(
            "[radio] ended_video_id=%s != current=%s but in_session=true — advancing",
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
    from app.agent.radio.playlist import find_topic_version, find_music_video, StationTrack
    from app.agent.radio.player import broadcast_radio_track

    channel = (msg.get("channel") or "").strip().lower()
    mode = (msg.get("mode") or "").strip().lower()
    print(f"[radio] mode_toggle entry user={user_id[:8]} channel={channel!r} mode={mode!r}", flush=True)

    if not RadioSessionManager.is_channel_allowed(channel):
        return
    if mode not in ("song", "video"):
        return
    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None:
        sess = mgr.get_or_create(user_id, channel)

    prev_mode = sess.display_mode
    current_track = sess.current_station_track
    # User-initiated click: flip the override flag so auto-detect stops on
    # subsequent track loads. The pick sticks for the rest of the session.
    mgr.set_display_mode(sess, mode, user_initiated=True, source="user_mode_toggle")

    # Mid-track swap rules (bidirectional, same overwrite-history semantics):
    #   Song  + OMV current → Topic lookup → swap to ATV if found.
    #   Video + ATV current → MV lookup    → swap to OMV if found.
    #   Song  + ATV current → no swap (already audio). Overlay appears.
    #   Video + OMV current → no swap (already music video). Iframe visible.
    #   UGC / unknown video_type → no swap in either direction.
    # A failed lookup stays on the current track; the display-mode still
    # flips so the overlay / iframe chrome responds.
    reload_needed = False
    alt: Optional[StationTrack] = None

    if sess.enabled and current_track is not None:
        if mode == "song" and current_track.video_type == "MUSIC_VIDEO_TYPE_OMV":
            print(
                f"[radio] mode_toggle clicked → song reload_needed=candidate "
                f"current={current_track.video_id} video_type={current_track.video_type}",
                flush=True,
            )
            alt = await find_topic_version(current_track)
            swap_log = "topic_swap"
            fail_log = "topic_lookup_failed"
        elif mode == "video" and current_track.video_type == "MUSIC_VIDEO_TYPE_ATV":
            print(
                f"[radio] mode_toggle clicked → video reload_needed=candidate "
                f"current={current_track.video_id} video_type={current_track.video_type}",
                flush=True,
            )
            alt = await find_music_video(current_track)
            swap_log = "mv_swap"
            fail_log = "mv_lookup_failed"
        else:
            swap_log = fail_log = ""

        if alt is not None and alt.video_id != current_track.video_id:
            reload_needed = True
            print(
                f"[radio] {swap_log} from={current_track.video_id} to={alt.video_id} "
                f"title={alt.title!r} (mid-track mode toggle)",
                flush=True,
            )
            prev_current = sess.current_track_id
            sess.current_track_id = alt.video_id
            sess.current_station_track = alt
            sess.played_track_ids.add(alt.video_id)
            # Overwrite the current tape entry rather than appending — same
            # song, different format, the user didn't move forward.
            if 0 <= sess.history_cursor < len(sess.played_history):
                sess.played_history[sess.history_cursor] = alt
            logger.info(
                "[radio] current_track_mutation source=mid_toggle_swap user=%s channel=%s "
                "from=%s to=%s history_cursor=%d swap=%s",
                user_id[:8], channel, prev_current, alt.video_id,
                sess.history_cursor, swap_log,
            )
            await broadcast_radio_track(
                user_id=user_id,
                video_id=alt.video_id,
                title=alt.display_title(),
                channel=channel,
                artist=alt.artist,
                thumbnail_url=alt.thumbnail_url,
                video_type=alt.video_type,
            )
        elif swap_log:
            # Lookup was attempted but yielded nothing.
            print(
                f"[radio] {fail_log} fallback={current_track.video_id} "
                f"(staying on current track, {mode} UI still applies)",
                flush=True,
            )

    print(
        f"[radio] mode_toggle clicked → {sess.display_mode} reload_needed={reload_needed} "
        f"prev={prev_mode} override={sess.display_mode_user_override}",
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


async def _authenticate_ws_session_token(token: str) -> Optional[str]:
    """Validate a direct-to-agent session token signed by the platform.

    Used by the browser to open a WebSocket directly to the agent
    (`agent-<prefix>.agents.toup.ai/api/ws/chat`) without going through
    the Railway platform proxy. Why: platform redeploys would otherwise
    drop the chat WS for ~30s; talking to the agent directly keeps the
    chat alive across any platform deploy.

    Token format: HS256 JWT, secret = `settings.agent_api_key` (the
    same key the platform uses for X-Agent-Key proxy auth — already
    shared between platform and this agent, so no new secret bootstrap
    needed). Required claims:
        sub: user_id (string)
        exp: int unix-ts (must be in future)
        iss: "toup-platform"
        aud: "toup-agent-session"  (distinguishes from regular auth JWTs)

    Returns the user_id on valid, None otherwise."""
    if not token:
        return None
    secret = (settings.agent_api_key or "").strip()
    if not secret:
        return None
    try:
        from jose import jwt as _jose_jwt
        payload = _jose_jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="toup-agent-session",
            issuer="toup-platform",
        )
    except Exception as e:
        logger.info("[WS] session token rejected: %s", e)
        return None
    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        return None
    # Cross-check against the agent's bound user. settings.user_id is
    # set at bind-time; reject tokens for any other user even if the
    # signature checks out (defense in depth — a token valid for some
    # other agent must not authenticate against this one).
    if settings.user_id and user_id != settings.user_id:
        logger.warning(
            "[WS] session token user mismatch: token=%s bound=%s",
            user_id[:8], (settings.user_id or "?")[:8],
        )
        return None
    # Lazy-create the user row (mirrors the X-Agent-Key path).
    try:
        from app.db.database import async_session_maker as _sm
        from app.db.models import User
        async with _sm() as _db:
            from app.services.auth_service import get_user_by_id
            u = await get_user_by_id(_db, user_id)
            if not u:
                u = User(
                    id=user_id,
                    email=f"{user_id[:8]}@agent.local",
                    hashed_password="",
                    name="Agent Owner",
                )
                _db.add(u)
                await _db.commit()
    except Exception as e:
        logger.warning("[WS] session-token user-row create failed: %s", e)
    return user_id


@router.websocket("/ws/chat")
async def ws_chat(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time chat with the agent.

    Supports streaming text chunks, tool call indicators, and session management.
    Auth: agent_key (proxy mode) OR JWT (subprotocol/?token=/first-frame).
    """
    # ST-2: accept + subprotocol JWT extraction.
    # ST-3a: agent_key now preferred via X-Agent-Key request header
    # (read from the upgrade-request headers). The ?agent_key= query
    # path stays alive as a bake-window fallback and emits
    # [DEPRECATED-AGENT-KEY-URL] when used for auth.
    from app.api._ws_auth_helpers import (
        accept_with_subprotocol_auth,
        log_deprecated_query_token,
        log_deprecated_agent_key_url,
        safe_send_close_ws,
    )
    # Headers are populated from the upgrade request; available before
    # or after accept (we read after, for readability).
    header_agent_key = websocket.headers.get("x-agent-key") or ""
    subprotocol_token = await accept_with_subprotocol_auth(websocket)
    user_id: Optional[str] = None

    # Try agent_key auth first (platform proxy mode).
    # 1. X-Agent-Key header (preferred — no URL leak).
    # 2. ?agent_key= query (deprecated — fires marker on use).
    matched_agent_key: Optional[str] = None
    if header_agent_key and settings.agent_api_key and header_agent_key == settings.agent_api_key:
        matched_agent_key = header_agent_key
    elif agent_key and settings.agent_api_key and agent_key == settings.agent_api_key:
        log_deprecated_agent_key_url("/api/ws/chat")
        matched_agent_key = agent_key

    if matched_agent_key:
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

    # Try direct-to-agent session token (signed with this agent's
    # X-Agent-Key, issued by platform's /api/auth/agent-session-token).
    # Tried BEFORE the platform-JWT path because session tokens carry
    # `aud=toup-agent-session` which the regular JWT path would reject —
    # cheaper to try the right path first.
    if not user_id and subprotocol_token:
        user_id = await _authenticate_ws_session_token(subprotocol_token)

    # Try subprotocol JWT auth (ST-2 — header-based, no URL leak)
    if not user_id and subprotocol_token:
        user_id = await _authenticate_ws(subprotocol_token)

    # Try ?token= session token (signed with agent_api_key, aud=toup-agent-session).
    # The Chrome extension's sidepanel chat-ws connects via query string with
    # such a token — the path needs to recognize it BEFORE falling through to
    # the platform-JWT decoder which uses a different secret.
    if not user_id and token:
        user_id = await _authenticate_ws_session_token(token)

    # Try ?token= platform JWT (deprecated bake-window fallback)
    if not user_id and token:
        log_deprecated_query_token("/api/ws/chat (agent)")
        user_id = await _authenticate_ws(token)

    try:
        # If still not authenticated, expect first-frame auth message.
        client_disconnected = False
        if not user_id:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                msg = json.loads(raw)
                if msg.get("type") == "auth" and msg.get("token"):
                    user_id = await _authenticate_ws(msg["token"])
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass
            except WebSocketDisconnect:
                client_disconnected = True

        if not user_id:
            if client_disconnected:
                return
            await safe_send_close_ws(
                websocket, code=4001, message="Authentication required",
            )
            return

        if not _agent_runner:
            await websocket.send_json({"type": "error", "message": "Agent not available"})
            await websocket.close(code=4503, reason="Service unavailable")
            return

        logger.info(f"[WS] Authenticated user: {user_id}")

        # Phase A/B: track this WS in the drain coordinator. Decremented
        # in the finally below. The increment happens here (post-auth)
        # rather than at accept-time so failed auth doesn't pin the
        # counter at >0 forever — the bind handler's drain timer waits
        # for this counter to hit 0 before exiting.
        from app.services import drain_state as _drain_state
        _drain_state.increment_active()

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

                # Cross-channel reply-to: client passes the target message id when
                # the user used the Reply affordance. Resolve + authorize, capture
                # the target's role/content/timestamp so we can render the LLM
                # preamble even if the DB column isn't there yet (init_db ALTER
                # may not have run on a freshly-deployed tenant). Specific
                # columns only — selecting the whole ORM entity would fail when
                # `reply_to_message_id` is mapped in code but missing in DB.
                _reply_to_id_raw = msg.get("reply_to_message_id")
                reply_to_message_id: Optional[str] = None
                _reply_target_role: Optional[str] = None
                _reply_target_content: Optional[str] = None
                _reply_target_created_at = None
                if _reply_to_id_raw and isinstance(_reply_to_id_raw, str):
                    _candidate = _reply_to_id_raw.strip()
                    if 0 < len(_candidate) <= 50:
                        logger.info(
                            "[WS] reply_to received target=%s user=%s",
                            _candidate[:8], user_id[:8],
                        )
                        try:
                            from app.db.database import async_session_maker as _rt_sm
                            from app.db.models import Message as _RtMsg, Conversation as _RtConv
                            from app.db.models.day_chat import DayChat as _RtDC
                            async with _rt_sm() as _rt_db:
                                # Ownership check accepts EITHER:
                                #   - Conversation.user_id == user_id, OR
                                #   - DayChat.user_id == user_id
                                # Conversation alone is too narrow: system-channel
                                # rows (routine, trigger, radio output) historically
                                # carried service-stamped or null user_id even though
                                # the message itself sits in the user's day_chat.
                                # Day_chat ownership is the canonical user boundary
                                # per conversation.py:21-33's Reading-A invariant.
                                # Outer-joins so a missing Conversation or DayChat
                                # row (older data, race conditions) doesn't fail
                                # the whole query — we just check the side that
                                # resolved.
                                _row = (await _rt_db.execute(
                                    select(
                                        _RtMsg.id,
                                        _RtMsg.role,
                                        _RtMsg.content,
                                        _RtMsg.created_at,
                                        _RtConv.user_id.label("conv_user_id"),
                                        _RtDC.user_id.label("dc_user_id"),
                                    )
                                    .select_from(_RtMsg)
                                    .outerjoin(_RtConv, _RtMsg.conversation_id == _RtConv.id)
                                    .outerjoin(_RtDC, _RtMsg.day_chat_id == _RtDC.id)
                                    .where(_RtMsg.id == _candidate)
                                )).first()
                                if _row is None:
                                    # Loud — id from the frontend doesn't match
                                    # any row. Could be a stale optimistic id
                                    # the client never reconciled, or a typo'd
                                    # payload. Either way the user's reply
                                    # context is lost; surface it in prod logs.
                                    logger.warning(
                                        "[WS] reply_to target=%s does not exist in messages "
                                        "table (user=%s) — frontend may have sent a stale id",
                                        _candidate[:8], user_id[:8],
                                    )
                                elif (
                                    _row.conv_user_id == user_id
                                    or _row.dc_user_id == user_id
                                ):
                                    reply_to_message_id = _row.id
                                    _reply_target_role = _row.role
                                    _reply_target_content = _row.content
                                    _reply_target_created_at = _row.created_at
                                    _via = "conv" if _row.conv_user_id == user_id else "day_chat"
                                    logger.info(
                                        "[WS] reply_to authorized target=%s role=%s content_len=%d via=%s",
                                        _candidate[:8], _row.role,
                                        len(_row.content or ""), _via,
                                    )
                                else:
                                    # Real ownership failure: row exists but is
                                    # in some other user's tree. Warn (not info)
                                    # so prod alerts flag this — could be an
                                    # IDOR attempt OR a legitimate user reply
                                    # we've still got an auth-boundary gap on.
                                    logger.warning(
                                        "[WS] reply_to target=%s rejected (user=%s, "
                                        "conv_user=%s, dc_user=%s) — dropping pointer",
                                        _candidate[:8], user_id[:8],
                                        (_row.conv_user_id or "")[:8] if _row.conv_user_id else "NULL",
                                        (_row.dc_user_id or "")[:8] if _row.dc_user_id else "NULL",
                                    )
                        except Exception as _rt_err:
                            logger.warning(
                                "[WS] reply_to auth lookup failed err=%s: %s",
                                type(_rt_err).__name__, _rt_err,
                            )

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
                                    # TKT-LAT-004: drop the in-process tz
                                    # cache so the next agent turn picks
                                    # up the new value instead of stale.
                                    try:
                                        from app.agent._user_tz_cache import invalidate_cached_user_tz
                                        invalidate_cached_user_tz(user_id)
                                    except Exception:
                                        pass

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

                # Phase 4: Chrome sidepanel ambient page-context.
                # The sidepanel attaches `page_context` to every outgoing
                # message so the agent knows what the user is currently
                # looking at. Inject as a hidden context block — agent sees
                # it, user sees only their original text (via display_text).
                _page_ctx = msg.get("page_context")
                if _page_ctx and isinstance(_page_ctx, dict):
                    try:
                        from app.services.page_context_render import render_page_context
                        # Honor any privacy flags the sidepanel surfaces with
                        # the context (set per-message based on Options page
                        # toggles). Defaults remain permissive.
                        _flags = _page_ctx.get("_flags") or {}
                        _filtered = dict(_page_ctx)
                        if _flags.get("hide_readable", False):
                            _filtered["readable_content"] = ""
                        if _flags.get("hide_selection", False):
                            _filtered["selected_text"] = ""
                        if _flags.get("hide_dom", False):
                            _filtered["dom_summary"] = ""
                        _ctx_block = render_page_context(_filtered)
                        if _ctx_block:
                            text = f"{_ctx_block}\n\n{text}"
                    except Exception as _pc_err:
                        logger.debug("[WS] page_context render skipped: %s", _pc_err)

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

                # ── Session-independent duplicate guard (runs BEFORE the
                # session-gated DB replay check below) ───────────────────────
                # Drops the new-user FIRST-message reconnect-replay, which
                # carries session_id=null and would otherwise bypass all
                # idempotency and spawn a second agent turn. Same agent process
                # handles the reconnect, so an in-process guard is sufficient.
                _client_msg_id_top = msg.get("client_msg_id")
                if _client_msg_id_top and not _is_system_action:
                    if _dedup_seen_client_msg(f"{user_id}:{_client_msg_id_top}"):
                        logger.info(
                            "[WS] Duplicate dropped client_msg_id=%s session_id=%s user=%s — replay/double-send",
                            _client_msg_id_top, session_id, user_id[:8],
                        )
                        try:
                            await websocket.send_json({
                                "type": "user_message_persisted",
                                "client_msg_id": _client_msg_id_top,
                                "session_id": session_id,
                                "duplicate": True,
                            })
                        except Exception:
                            pass
                        continue

                # ── Durable, session-independent exactly-once gate ───────────
                # The in-process guard above only survives within ONE process
                # for _RECENT_MSG_TTL_S. During the post-onboarding boot window
                # the agent container is swapped (pool claim / blue-green),
                # wiping that in-memory dict — so the client's reconnect-replay
                # of its queued FIRST message (session_id=null, which bypasses
                # the session-gated DB check below) was re-dispatched on every
                # refresh: a fresh session + a fresh LLM call + a fresh credit
                # charge each time (the 2026-06 replay-loop credit burn). This
                # DB-backed ledger claims (user_id, client_msg_id) ATOMICALLY
                # before dispatch — the PK is uuid5(user_id, client_msg_id), the
                # same derivation the session-gated path uses for messages.id —
                # so a replay collides on INSERT and is dropped (acked as a
                # duplicate) and can NEVER trigger a second LLM call or charge.
                # Durable across container restarts (it lives in the tenant DB).
                # FAILS OPEN if the ledger table is missing on an un-migrated
                # tenant; the in-process guard + session-gated check still apply.
                if _client_msg_id_top and not _is_system_action:
                    import uuid as _uuid_pm
                    from sqlalchemy.exc import IntegrityError as _IntegrityError
                    _pm_id = str(_uuid_pm.uuid5(
                        _uuid_pm.NAMESPACE_OID,
                        f"toup-msg:{user_id}:{_client_msg_id_top}",
                    ))
                    _pm_replay = False
                    try:
                        from app.db.database import async_session_maker
                        from app.db.models import ProcessedMessage
                        async with async_session_maker() as _pm_db:
                            _pm_db.add(ProcessedMessage(
                                id=_pm_id,
                                user_id=user_id,
                                client_msg_id=_client_msg_id_top,
                                session_id=session_id,
                            ))
                            try:
                                await _pm_db.commit()
                            except _IntegrityError:
                                await _pm_db.rollback()
                                _pm_replay = True
                    except Exception as _pm_err:
                        # Ledger unavailable (e.g. table not yet created on an
                        # un-migrated tenant, or a transient DB error): fail OPEN
                        # so real messages are never blocked. Failing CLOSED would
                        # drop legitimate messages on any DB blip — strictly worse
                        # than the narrow window this leaves (a transient ledger
                        # error coinciding with a cross-container replay, where the
                        # in-process guard is also wiped). Logged at error level so
                        # ops can see ledger unavailability on this critical path.
                        logger.error("[WS] exactly-once ledger unavailable, failing open: %s", _pm_err)
                    # NOTE (exactly-once contract): this ledger is the SOLE
                    # per-message guard against a second LLM call / credit charge
                    # on replay — the credit gate's own idempotency key is per
                    # LLM call (uuid4), not per message, so it cannot dedupe a
                    # re-dispatch on its own. The claim is intentionally NOT
                    # released if dispatch later fails: a genuinely-failed first
                    # turn is left claimed (the user resends with a fresh
                    # client_msg_id) rather than auto-retried, because we cannot
                    # prove from the error path that no charge occurred (a
                    # tool-only turn can charge yet stream no text). No credit is
                    # lost — the charge is post-LLM-success.
                    if _pm_replay:
                        logger.info(
                            "[WS] Duplicate dropped (durable ledger) client_msg_id=%s session_id=%s user=%s — replay",
                            _client_msg_id_top, session_id, user_id[:8],
                        )
                        try:
                            await websocket.send_json({
                                "type": "user_message_persisted",
                                "client_msg_id": _client_msg_id_top,
                                "session_id": session_id,
                                "duplicate": True,
                            })
                        except Exception:
                            pass
                        continue

                # ── Pre-save user message so it survives stream failures ──
                # Skip if no session_id yet (first message — agent_runner creates session)
                # Skip for system actions (customize_app) — no user message to save
                _user_msg_presaved = False
                _persisted_user_msg_id: Optional[str] = None
                _persisted_day_chat_id: Optional[str] = None
                if session_id and not _is_system_action:
                    try:
                        from app.db.database import async_session_maker
                        from app.db.models import Message as DbMessage, Conversation
                        # Idempotency: derive a deterministic UUID from
                        # (user_id, client_msg_id). If the client retries the
                        # same message after a dropped WS, the same derived
                        # id collides with the existing row and we recognize
                        # the replay — skip re-persist + skip the LLM call
                        # (we already replied; replaying would duplicate the
                        # assistant message). Client_msg_id absent? Fall
                        # through to the normal random-uuid path; no
                        # idempotency, but we don't break older clients.
                        import uuid as _uuid
                        _client_msg_id = msg.get("client_msg_id")
                        _derived_msg_id: Optional[str] = None
                        if _client_msg_id:
                            _derived_msg_id = str(_uuid.uuid5(
                                _uuid.NAMESPACE_OID,
                                f"toup-msg:{user_id}:{_client_msg_id}",
                            ))
                        async with async_session_maker() as _presave_db:
                            # Replay check.
                            if _derived_msg_id:
                                _existing = (await _presave_db.execute(
                                    __import__('sqlalchemy').select(DbMessage)
                                    .where(DbMessage.id == _derived_msg_id)
                                )).scalar_one_or_none()
                                if _existing is not None:
                                    _persisted_user_msg_id = _existing.id
                                    _persisted_day_chat_id = _existing.day_chat_id
                                    _user_msg_presaved = True
                                    logger.info(
                                        "[WS] Idempotent replay detected client_msg_id=%s — skipping persist + LLM",
                                        _client_msg_id,
                                    )
                                    # Echo the ack so the client swaps its
                                    # optimistic id and clears its pending
                                    # queue.
                                    try:
                                        await websocket.send_json({
                                            "type": "user_message_persisted",
                                            "client_msg_id": _client_msg_id,
                                            "server_msg_id": _persisted_user_msg_id,
                                            "day_chat_id": _persisted_day_chat_id,
                                            "session_id": session_id,
                                            "duplicate": True,
                                        })
                                    except Exception:
                                        pass
                                    # Skip the LLM call: a previous handler
                                    # already responded. The client's
                                    # `messages-since` reconnect path will
                                    # pull the assistant reply if it exists.
                                    continue
                            _presave_dc_id = await _resolve_day_chat_id_for_now(_presave_db, user_id, tz_override=client_tz)
                            # Build kwargs defensively: omit reply_to_message_id
                            # when None so SQLAlchemy doesn't reference the
                            # column at all (lets regular user messages save
                            # on tenants where the ALTER hasn't run yet). The
                            # in-memory preamble below still threads the LLM
                            # turn regardless of DB persistence — the structured
                            # pointer is for future history rendering, not the
                            # current turn.
                            _msg_kwargs: dict = dict(
                                id=(_derived_msg_id or str(_uuid.uuid4())),
                                conversation_id=session_id,
                                day_chat_id=_presave_dc_id,
                                role="user",
                                content=_original_user_text,
                            )
                            if reply_to_message_id:
                                _msg_kwargs["reply_to_message_id"] = reply_to_message_id
                            _new_msg = DbMessage(**_msg_kwargs)
                            _presave_db.add(_new_msg)
                            # Update conversation timestamp
                            _conv = (await _presave_db.execute(
                                __import__('sqlalchemy').select(Conversation).where(Conversation.id == session_id)
                            )).scalar_one_or_none()
                            if _conv:
                                _conv.message_count = (_conv.message_count or 0) + 1
                                _conv.updated_at = __import__('datetime').datetime.utcnow()
                            try:
                                await _presave_db.commit()
                            except Exception as _commit_err:
                                # Defensive retry: if the ALTER hasn't reached
                                # this tenant DB yet, Postgres rejects the
                                # INSERT with "column reply_to_message_id of
                                # relation messages does not exist". Drop the
                                # pointer column and retry — the in-memory
                                # preamble still threads the current turn.
                                _err_text = str(_commit_err).lower()
                                if (
                                    reply_to_message_id
                                    and "reply_to_message_id" in _err_text
                                ):
                                    logger.warning(
                                        "[WS] reply_to_message_id column missing on this tenant; "
                                        "saving user message without the structured pointer. "
                                        "Run init_db / migration 049 to enable persistence.",
                                    )
                                    await _presave_db.rollback()
                                    _msg_kwargs.pop("reply_to_message_id", None)
                                    _new_msg = DbMessage(**_msg_kwargs)
                                    _presave_db.add(_new_msg)
                                    if _conv:
                                        _conv.message_count = (_conv.message_count or 0) + 1
                                        _conv.updated_at = __import__('datetime').datetime.utcnow()
                                        _presave_db.add(_conv)
                                    await _presave_db.commit()
                                else:
                                    raise
                            await _presave_db.refresh(_new_msg)
                            _persisted_user_msg_id = _new_msg.id
                            _persisted_day_chat_id = _presave_dc_id
                        _user_msg_presaved = True

                        # Reconcile the client's optimistic message: echo back
                        # the server-assigned id so the frontend can swap its
                        # temporary `msg-${Date.now()}` placeholder for the
                        # canonical id. Without this, the day-chat refetch
                        # dropped the optimistic entry until the page was
                        # refreshed (matin-era UX bug).
                        if _client_msg_id:
                            try:
                                await websocket.send_json({
                                    "type": "user_message_persisted",
                                    "client_msg_id": _client_msg_id,
                                    "server_msg_id": _persisted_user_msg_id,
                                    "day_chat_id": _persisted_day_chat_id,
                                    "session_id": session_id,
                                })
                            except Exception:
                                pass
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

                async def on_credential_confirm_request(payload: dict):
                    """Vault CP4: forward the confirmation-card frame to the client."""
                    try:
                        await websocket.send_json(payload)
                    except Exception:
                        logger.exception("on_credential_confirm_request send_json failed")

                async def on_credit_exhausted(payload: dict):
                    """Forward the exhausted-balance card payload to the client.

                    Carries `monthly_reset_at` / `daily_reset_at` ISO
                    timestamps so the frontend can render a live
                    countdown without trusting server clock skew.
                    """
                    try:
                        await websocket.send_json(payload)
                    except Exception:
                        logger.exception("on_credit_exhausted send_json failed")

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

                # Reply-to preamble (LLM-only): prepend a short quoted reference
                # to `_agent_text` so the model treats the new user turn as a
                # threaded reply, not a fresh question. Stored content stays
                # clean — the structured pointer (reply_to_message_id) is what
                # the frontend renders. Runs from in-memory state, so it works
                # even on tenants where the DB column hasn't landed yet.
                if _reply_target_content is not None:
                    try:
                        from app.agent.reply_quote import render_reply_preamble
                        _preamble = render_reply_preamble(
                            target_role=_reply_target_role or "assistant",
                            target_content=_reply_target_content,
                            target_created_at=_reply_target_created_at,
                            tz_name=client_tz,
                        )
                        _agent_text = f"{_preamble}\n\n{_agent_text}"
                        logger.info(
                            "[WS] reply_to preamble prepended preamble_len=%d agent_text_len=%d user=%s",
                            len(_preamble), len(_agent_text), user_id[:8],
                        )
                    except Exception as _rq_err:
                        logger.warning("[WS] reply preamble render failed: %s", _rq_err)

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
                    on_credential_confirm_request=on_credential_confirm_request,
                    on_credit_exhausted=on_credit_exhausted,
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
                        # Server-side UUID of the saved Message row.
                        # Frontend stamps it on the live bubble so a
                        # later day-chat reload finds the same id and
                        # dedupes — without this, the live render and
                        # the DB-backed render appear as two separate
                        # messages with identical content.
                        "asst_message_id": response.asst_message_id,
                        # Echo the client_msg_id so the frontend can clear its
                        # localStorage pending-replay entry on the happy path —
                        # critical for the FIRST message (session_id=null), whose
                        # success ack is otherwise gated behind `if session_id`
                        # below and never sent, leaving the entry to replay on
                        # every refresh (the replay-loop credit burn).
                        "client_msg_id": _client_msg_id_top,
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
            # Phase A/B: release this WS from the drain counter. Match
            # the increment above; if drain has been engaged and this
            # was the last in-flight WS, the drain watcher will exit
            # the process within ~1s.
            try:
                _drain_state.decrement_active()
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        logger.exception(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=4500)
        except Exception:
            pass
