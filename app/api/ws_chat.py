"""
WebSocket Chat Endpoint — Real-time streaming chat via WebSocket.

Protocol:
  Client sends JSON:
    { "type": "message", "text": "...", "session_id": "..." }
    { "type": "ping" }

  Server sends JSON:
    { "type": "status", "stage": "received" | "thinking",
      "mission_id": "chatturn:<hex>" }
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
import time
import uuid
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
        # Toup credits, NOT the user's OpenAI account. The old copy sent users
        # to platform.openai.com — wrong (it is the platform's own billing) and
        # an App Review 3.1.1 anti-steering hit on iOS, where the only
        # legitimate purchase surface is the in-app Credits screen.
        return (
            "You're out of Toup credits. Open Credits to top up or upgrade "
            "your plan and keep going."
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


async def broadcast_to_user(
    user_id: str, event: dict, exclude: Optional[asyncio.Queue] = None,
) -> int:
    """Push an event to all WebSocket connections for a user.

    `exclude` skips one connection's queue — used by the turn-mirror lane so
    the socket that OWNS the running turn (and already receives the frame
    directly) doesn't get a duplicate, while every other live socket of the
    same user — a phone that just reconnected after a background stint — does.

    Returns number of connections that received the event."""
    queues = _user_ws_queues.get(user_id, [])
    sent = 0
    for q in queues:
        if exclude is not None and q is exclude:
            continue
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


# ── In-flight turn registry (2026-07-23) ─────────────────────────────
# A turn that loses its client keeps running headless (see the client-gone
# lane below) — but nothing ever told a RECONNECTING client that work was
# still in progress. Returning to the app mid-turn therefore showed a
# thread with the user's message and no sign of life: no "thinking"
# indicator, no tool progress, nothing until the answer eventually landed
# (founder repro 2026-07-23: "left the app while it was thinking, came
# back to an empty chat").
#
# This registry is the missing signal. One entry per user for as long as
# their turn runs: announced as a `turn_active` frame the moment a socket
# connects, mirrored to every OTHER live socket as `turn_status` while the
# turn advances, and closed out with `turn_ended`. In-process by design —
# the turn itself lives in this process, so an entry can never outlive the
# work it describes (and a hard TTL covers a killed process anyway).
_active_turns: Dict[str, dict] = {}
_TURN_STALE_S = 900.0  # 15 min: longer than any real turn, short enough to self-heal


def _set_active_turn(user_id: str, **fields) -> None:
    """Create or update this user's in-flight turn entry. A different mission
    id starts a FRESH entry — the previous turn's tool/stage must never leak
    into the new one."""
    mission_id = fields.get("mission_id")
    entry = _active_turns.get(user_id)
    if entry is None or (mission_id and entry.get("mission_id") != mission_id):
        entry = {}
        _active_turns[user_id] = entry
    entry.update(fields)


def _clear_active_turn(user_id: str, mission_id: Optional[str] = None) -> None:
    """Drop the entry. `mission_id` guards against a finished turn clearing a
    NEWER turn's entry (the user sent again while the old one was wrapping up)."""
    entry = _active_turns.get(user_id)
    if entry is None:
        return
    if mission_id and entry.get("mission_id") != mission_id:
        return
    _active_turns.pop(user_id, None)


def _get_active_turn(user_id: str) -> Optional[dict]:
    """Fresh entry for this user, or None. Stale entries (a process that died
    mid-turn, a leak) are dropped rather than announced — a client must never
    be told to wait on a turn that no longer exists."""
    entry = _active_turns.get(user_id)
    if not entry:
        return None
    started = entry.get("started_at") or 0.0
    if time.time() - started > _TURN_STALE_S:
        _active_turns.pop(user_id, None)
        return None
    return entry


def _turn_frame(kind: str, entry: dict, **extra) -> dict:
    """Wire shape for turn_active / turn_status. `stage` is coarse
    ('thinking' | 'tool' | 'writing'); `tool` is the raw tool name so the
    client maps it to its own label + orb state (one vocabulary, client-side
    — see agentStates.ts / getToolInfo)."""
    frame = {
        "type": kind,
        "mission_id": entry.get("mission_id"),
        "title": entry.get("title"),
        "stage": entry.get("stage") or "thinking",
        "tool": entry.get("tool"),
        "started_at_ms": int((entry.get("started_at") or time.time()) * 1000),
    }
    frame.update(extra)
    return frame


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

        # Currency check: this task fires up to ~10s after its media_play,
        # and during a skip burst the station has moved on by the time the
        # yt-dlp probe answers. The swap frame is channel-less (it reaches
        # every device), so a late swap for a track a station has provably
        # MOVED PAST is pure churn on a live card. Only that case is dropped:
        # sessions never expire on their own, so "some session's current isn't
        # this video" is the NORMAL state for every one-off play (an enabled
        # app session from yesterday would have suppressed every age swap on
        # web — review finding). Moved-past = in a session's recent tape and
        # no longer its current, with no session still on it.
        try:
            from app.agent.radio import get_radio_manager as _grm
            _enabled = [
                s for s in list(getattr(_grm(), "_sessions", {}).values())
                if s.user_id == user_id and s.enabled
            ]
            _is_current = any(s.current_track_id == video_id for s in _enabled)
            _moved_past = any(
                s.current_track_id != video_id
                and any(t.video_id == video_id for t in s.played_history[-10:])
                for s in _enabled
            )
            if _moved_past and not _is_current:
                logger.info("[AGE-SWAP] stale swap suppressed video=%s", video_id)
                return
        except Exception:  # noqa: BLE001 — a failed check must not block the swap
            pass

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
            from app.agent import media_resolve as _mr
            _cands = _mr.scrape_results(resp.text, limit=6)
            _pick = _mr.pick_best(query, _cands, variety=False)
            if _pick:
                # RELEVANCE-RANKED, not "first id on the page". The top result
                # for a Persian song title is regularly a mix, a reaction or a
                # different artist's cover; taking it blind is half of how the
                # agent came to announce one song while another played.
                video_id, video_title, _ = _pick
            else:
                # No result carried a title of its own. Do NOT fall back to the
                # user's words — that is where the literal lowercase "ebi" card
                # came from, and a fake title does not stop at the card: it is
                # the [SYSTEM] line the model paraphrases, the persisted card,
                # and the seed the whole station is built from. Hand the request
                # to the agent's own resolver ladder (yt-dlp always returns real
                # metadata) by declining the fast path.
                logger.warning(
                    "[FAST-MEDIA] no titled result for %r — deferring to play_media", query,
                )
                return None
        if not video_id:
            logger.warning("[FAST-MEDIA] No video found for: %s", query)
            return None

        # Which surface did they ask for? This path handles most typed "play …"
        # messages AND tells the agent not to call play_media, so if it doesn't
        # answer the question nothing downstream can: the phone is audio-first,
        # and "play the music video for HUMBLE" would come out as album art.
        # See infer_requested_mode — the same call the tool makes.
        from app.agent.radio.player import infer_requested_mode
        requested_mode = infer_requested_mode(text) or "song"

        # Broadcast media_play immediately — frontend opens YouTube embed (zero delay)
        event = {
            "type": "media_play",
            "provider": "youtube",
            "video_id": video_id,
            "title": video_title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "mode": requested_mode,
            # Always send artwork — see the note in tool_executor's identical
            # broadcast. Derived from the id, so it costs nothing and can never
            # be blank.
            "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        }
        broadcast_queue.put_nowait(event)
        logger.info("[FAST-MEDIA] Broadcast media_play in fast-path: %s - %s", video_id, video_title)

        # Start the yt-dlp extraction NOW, in parallel with everything the user
        # is about to watch happen: the agent composing its reply, the card
        # rendering, the phone deciding to ask for bytes. That gap is one to
        # three seconds of otherwise dead time, and extraction is the single
        # largest item on the cold-start critical path — a median 3.4s against
        # the production proxy, a mean of 7.1s, and a worst case of 20.8s
        # (measured 2026-08-04). By the time /audio_stream asks, the result is
        # cached or the call is already in flight and coalesces onto it.
        #
        # EXTRACT, deliberately, even now that the spool exists. The arithmetic
        # (adversarial review, 2026-08-09): the phone's own /audio_stream lands
        # ~1.5s after this frame, but extraction finishes at ~3.4s — so on the
        # SAME replica the phone's request is already waiting when the spool
        # could first start, and it starts the spool itself; a build here buys
        # nothing extraction doesn't. On the OTHER replica (platform runs 2), a
        # build-warm's spool would run a full duplicate pull through the same
        # per-video sticky proxy slot DURING the phone's pre-roll — the exact
        # contention the old rule existed to prevent. Extract is the whole win
        # with none of the risk. (The flip and broadcast warms differ: there
        # the proxy is idle when they fire — see radio/player.py.)
        try:
            from app.agent.radio.player import warm_audio_cache as _warm
            _warm([video_id], mode="extract")
        except Exception as _we:
            logger.debug("[FAST-MEDIA] pre-extract warm skipped: %s", _we)

        # Fire-and-forget: check age restriction in background, swap embed if needed
        asyncio.create_task(_check_age_and_swap(video_id, user_id))

        # Radio session: record this as a user-driven seed. Channel is unknown
        # at this helper layer — caller records after resolving channel.

        # Return modified text so the agent knows media has been dispatched.
        # The model cannot see the card, so this line is the ONLY thing standing
        # between it and announcing whatever the user asked for. Bind it to the
        # resolved title, and when the resolution does not answer the request,
        # tell it to say so rather than claim success over a stranger's song.
        #
        # It says STARTING, not "already playing". All that has happened here is
        # a media_play frame going out over the socket; on the audio-first phone
        # path the device then has to resolve, fetch and buffer the track, which
        # on a cold track is seconds — 12.5s of them in the founder recording of
        # 2026-08-03. The old wording ("is ALREADY playing") was a claim the
        # server cannot make and the model repeated it verbatim: it answered a
        # fresh "Play me moein" with "Moein's already playing", over silence,
        # nine seconds before the first sound. Worse, "already" reads as "this
        # was on before you asked", so the reply denied the request it was
        # fulfilling. Playback truth lives on the device (`nowPlaying`, written
        # only when audio is genuinely audible); the agent must not assert it.
        _mismatch = _mr.describe_mismatch(query, video_title)
        modified_text = (
            f"{text}\n\n[SYSTEM: The track \"{video_title}\" "
            f"(https://www.youtube.com/watch?v={video_id}) is being STARTED on the "
            f"user's device right now — it is not audible yet. Do NOT call play_media. "
            f"Respond conversationally as though putting it on, and when you name what "
            f"is playing use this EXACT title — \"{video_title}\" — never the words the "
            f"user typed. Never imply it was audible before this message, and never "
            f"ask whether they can hear it yet."
            + (f" {_mismatch}" if _mismatch else "")
            + "]"
        )
        media_meta = {
            "type": "youtube", "video_id": video_id, "title": video_title,
            # Carried so the auto-enable that follows can build the station on
            # the SAME surface the frame just told the client to use. Without
            # it the station resolves its variants for 'song' while the phone
            # is showing the video the user asked for.
            "mode": requested_mode,
        }
        return (modified_text, media_meta)
    except Exception as e:
        logger.warning("[FAST-MEDIA] Fast-path failed (agent will handle): %s", e)
        return None


# ── Radio Mode orchestration ─────────────────────────────────────────
# Toggle: enable / disable a per-channel radio session (seed = last played).
# Track-ended: when current track ends AND radio is ON, pick next via Haiku
# and broadcast a radio_auto media_play.

_radio_toggle_locks: dict = {}


def _radio_toggle_lock(user_id: str, channel: str) -> asyncio.Lock:
    key = (user_id, channel)
    lock = _radio_toggle_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _radio_toggle_locks[key] = lock
    return lock


async def _handle_radio_toggle(user_id: str, msg: dict) -> None:
    """Serialized per (user, channel). A typed play fires the server-side
    `_auto` toggle as a create_task while the client's own reseed toggle rides
    the socket — with no lock the two BUILD CONCURRENTLY: the dup-toggle
    dedupe below requires a COMMITTED station, so a reseed landing mid-build
    sails past it, `variety` guarantees the two stations hold different
    tracks, and the user hears one station while the card and window describe
    another (the 2026-08-10 recording's mid-song identity flips). Under the
    lock the second toggle waits, then lands in the committed-station dedupe
    and re-ships the live station instead of rebuilding it."""
    channel = (msg.get("channel") or "").strip().lower()
    # Validate BEFORE creating a lock: the dict is keyed on the raw channel
    # string, so unvalidated input grew it unboundedly (review finding).
    from app.agent.radio import RadioSessionManager as _RSM
    if not _RSM.is_channel_allowed(channel):
        await _handle_radio_toggle_locked(user_id, msg)  # its own reject path
        return
    async with _radio_toggle_lock(user_id, channel):
        await _handle_radio_toggle_locked(user_id, msg)


async def _handle_radio_toggle_locked(user_id: str, msg: dict) -> None:
    from app.agent.radio import (
        get_radio_manager,
        RadioSessionManager,
        SeedTrack,
        build_station,
    )

    channel = (msg.get("channel") or "").strip().lower()
    enabled = bool(msg.get("enabled"))
    # `_auto`: this enable was triggered server-side right after a fast-path
    # "play me X" (default-radio), NOT by a user tapping the toggle. The seed is
    # ALREADY playing from the fast-path media_play, so we must NOT re-broadcast
    # it (a second media_play would cold-reload the in-flight track → restart).
    # We still build the station + flip the toggle ON; the queue prefetches from
    # the first auto-advance onward.
    auto = bool(msg.get("_auto"))
    # Optional initial display mode riding the toggle. Applied AFTER enable()'s
    # reset — this is what collapses the old two-frame race where the client
    # toggled and then sent radio_display_mode, and the station resolved its
    # first upcoming window in between, for the wrong surface.
    initial_mode = (msg.get("mode") or "").strip().lower()
    if initial_mode not in ("song", "video"):
        initial_mode = ""

    print(f"[radio] toggle entry user={user_id[:8]} channel={channel!r} enabled={enabled} auto={auto} msg={msg}", flush=True)

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

    # Resolve the seed to its YT Music "Song" (ATV) id BEFORE building the
    # station. Mobile's fast-path seeds the session with a RAW scraped YouTube
    # videoId (a lyric video / OMV / random upload), and YT Music's
    # get_watch_playlist only returns a same-mood station when fed a proper ATV
    # seed — fed a non-music id it returns a generic, unrelated mix (the
    # "radio plays unrelated songs" bug). Apply the same ATV discipline already
    # used for queue tracks (find_topic_version) to the SEED. Pure superset:
    # any miss/empty-title/exception falls back to the raw seed, so the web
    # client (whose seed is already a clean id) is unaffected. Seed-only — the
    # user still hears `seed_video_id`; only the station QUEUE derives from ATV.
    atv_seed = seed_video_id
    try:
        from app.agent.radio.playlist import StationTrack as _ST, find_topic_version as _ftv
        from app.agent.media_resolve import split_artist_title as _split
        # PASS THE ARTIST. find_topic_version holds the only artist check in the
        # media stack and it is conditional on the caller supplying one — probe
        # with "" and the guard is skipped entirely, so the first ATV hit for
        # the seed's title wins no matter WHOSE it is, and the whole 50-track
        # station is then built from that stranger. "Play me ebi" coming back as
        # Music_Afghani and Parastoo Ahmadi (recording, 2026-08-03 14:11) is this
        # line. Titles arrive as "Artist - Title" from both producers.
        _seed_label = seed_title or seed_intent or ""
        _seed_artist, _seed_song = _split(_seed_label)
        _probe = _ST(
            video_id=seed_video_id,
            title=_seed_song or _seed_label,
            artist=_seed_artist,
        )
        _resolved = await _ftv(_probe)
        if _resolved and _resolved.video_id:
            atv_seed = _resolved.video_id
            print(
                f"[radio] seed_atv_resolved raw={seed_video_id} atv={atv_seed} "
                f"title={(seed_title or seed_intent)!r}",
                flush=True,
            )
    except Exception as _se:
        print(f"[radio] seed_atv_resolve_failed seed={seed_video_id} err={_se}", flush=True)

    # DUPLICATE TOGGLE FOR A STATION THAT ALREADY EXISTS.
    #
    # A typed "play me X" on the phone produces TWO toggles: this handler's own
    # `_auto` one, fired server-side right after the fast-path broadcast, and
    # the client's reseed, which it sends for every play it sees. Building twice
    # is not merely wasteful — the second build re-announces the seed that is
    # already playing, carrying YT Music's artwork where the first carried the
    # video thumbnail (the picture visibly flipping mid-song in the 2026-08-03
    # recording), and `variety` guarantees the two stations hold DIFFERENT
    # tracks, so whichever window the phone prefetched now belongs to a station
    # the backend has thrown away.
    #
    # Within a short window, treat the second toggle as what it is — the same
    # request arriving twice — and re-ship the existing station's state instead.
    # Outside that window the same seed genuinely means "build me a fresh
    # station", which is the whole point of variety, so it rebuilds.
    _existing = mgr.get(user_id, channel)
    if (
        _existing is not None
        and _existing.enabled
        and _existing.seed_track is not None
        and _existing.seed_track.video_id == seed_video_id
        and _existing.playlist
        and (time.time() - (_existing.station_built_ts or 0)) < _DUP_TOGGLE_WINDOW_SEC
    ):
        print(
            f"[radio] toggle DEDUPED user={user_id[:8]} seed={seed_video_id} "
            f"age={time.time() - _existing.station_built_ts:.1f}s — re-shipping the live station",
            flush=True,
        )
        if initial_mode:
            mgr.set_display_mode(
                _existing, initial_mode, user_initiated=True, source="toggle_mode",
            )
        await _resolve_upcoming_variants(_existing)
        _win = _upcoming_tracks(_existing)
        if _win:
            await broadcast_to_user(user_id, {
                "type": "radio_upcoming",
                "channel": channel,
                "upcoming": _win,
                "resolved_mode": _existing.display_mode,
            })
        await broadcast_to_user(user_id, _existing.to_broadcast_dict())
        return

    # Build the YT Music station BEFORE we mark the session enabled — if YT
    # Music rejects the seed (non-music, region-locked, etc.) we don't want
    # the UI pinned ON with no queue behind it. We also get back the seed's
    # own metadata from the watch-playlist's index 0 — used below to drive
    # the authoritative iframe swap on toggle-on (Rule 9).
    seed_meta, station = await build_station(
        atv_seed,
        limit=50,
        # Seed the search-based fallback (regional / non-catalog tracks have no
        # YT Music Song-radio) — the title alone is usually "Artist - Title".
        seed_title=(seed_title or seed_intent or ""),
        # Fresh station per request: YT Music "Start Radio" variant + a light
        # order shuffle, so a repeated "play me X" never replays the same
        # songs in the same order. Saved playlists (media_playlists) are the
        # way to replay an exact list.
        variety=True,
    )
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
    if initial_mode:
        mgr.set_display_mode(sess, initial_mode, user_initiated=True, source="toggle_mode")
    # Record this station in the user's library the moment it exists, so Toup
    # Media holds everything they have actually listened to rather than only
    # the stations someone remembered to save. Fire-and-forget and failure-proof
    # — a library entry must never be on the playback path.
    try:
        from app.api.media_playlists import autosave_station as _autosave
        asyncio.create_task(_autosave(user_id, channel))
    except Exception as _ae:
        logger.warning("[toup-media] autosave hook failed: %s", _ae)
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
    # Seed playback re-broadcast — USER toggles only. For _auto (default-radio
    # after a fast-path play) the seed is already playing; skip it to avoid a
    # cold-reload race. The session already binds current_track_id=seed (enable),
    # so the seed's media_ended still auto-advances correctly.
    if not auto:
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
        # Pre-resolve the variant for the upcoming window BEFORE shipping it so mobile
        # prebuffers the exact ids the station will play. The seed audio is already
        # playing, so this delay is invisible to the user.
        await _resolve_upcoming_variants(sess)
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
            duration=_length_sec(seed_meta.length) if seed_meta else 0,
        )
        print(
            f"[radio] iframe_force_sync from=unknown to={seed_video_id} reason=toggle_on",
            flush=True,
        )
    else:
        # _auto: the seed is already playing from the fast-path broadcast, so
        # no media_play re-send — but the phone still needs the prebuffer
        # window NOW. It used to get none until the first natural advance,
        # which made the first lock-screen ⏭ after a fresh play a guaranteed
        # cold round-trip (the 1-2s silent skip, 2026-08-03 recording). The
        # dedicated radio_upcoming frame carries the window without touching
        # the in-flight track.
        await _resolve_upcoming_variants(sess)
        upcoming = _upcoming_tracks(sess)
        if upcoming:
            await broadcast_to_user(user_id, {
                "type": "radio_upcoming",
                "channel": channel,
                "upcoming": upcoming,
                "resolved_mode": sess.display_mode,
            })
        from app.agent.radio.player import warm_audio_cache
        # Explicit `build`, because the invisible default is what caused the
        # 2026-08-05 cold-start report: this fires on the `_auto` toggle,
        # seconds into a play whose first 2.5MB is still arriving, and a build
        # pulls ~11.8MB through the same proxy. `_bounded_build` now yields to
        # a live cold start, so this is safe again — but state the mode, so the
        # next reader sees which one they are in.
        warm_audio_cache([t.get("video_id", "") for t in upcoming[:2]], mode="build")

    await broadcast_to_user(user_id, sess.to_broadcast_dict())


# Radio control frames the mid-turn stop-watcher must forward rather than eat.
#
# `radio_toggle` is deliberately EXCLUDED. It rebuilds the station from a new
# seed, and mid-turn is exactly when the agent is about to broadcast its own
# `media_play` for the song it just found — letting a toggle race that produces
# two stations for one request. Both clients already defer their toggle to the
# end of the turn for this reason (mobile: `pendingReseedRef`), so forwarding it
# here would undo a deliberate client-side decision. The frames below are all
# operations on the station that ALREADY exists.
_MID_TURN_PASSTHROUGH = frozenset({
    "media_ended", "radio_skip_next", "radio_skip_prev", "radio_display_mode",
})


async def _dispatch_radio_frame(user_id: str, msg: dict) -> None:
    """Route one radio control frame to its handler.

    Shared by the main receive loop and the mid-turn stop-watcher so the two
    paths cannot drift — the drift is what produced the swallowed-frame bug in
    the first place.
    """
    t = msg.get("type")
    try:
        if t == "media_ended":
            await _handle_media_ended(user_id, msg)
        elif t == "radio_skip_next":
            await _handle_radio_skip_next(user_id, msg)
        elif t == "radio_skip_prev":
            await _handle_radio_skip_prev(user_id, msg)
        elif t == "radio_display_mode":
            await _handle_radio_display_mode(user_id, msg)
    except Exception as e:  # noqa: BLE001
        # Never let a radio frame take down the turn that is carrying it.
        logger.warning("[radio] mid-turn %s failed: %s", t, e)


async def _advance_and_broadcast_next(
    user_id: str, channel: str, sess, trigger: str, target_video_id: str = "",
) -> bool:
    """Shared path for media_ended + skip_next: advance the playlist (possibly
    extending it first), apply Song-mode Topic lookup if set, record in history,
    broadcast media_play. Returns True on success, False on exhaustion.

    `trigger` is a log label: 'media_ended' | 'skip_next'.

    `target_video_id` (skip only) is the id the PHONE already advanced its card
    to — its upcoming[0]. The phone hops optimistically, so when the pop would
    resolve to a different id (the pop-time variant swap re-searching under
    load), the user watches their card get "corrected" to a stranger a second
    later — 3 of 9 skips did this on 2026-08-06. If the target is what the pop
    would give (or sits within the next few slots after a racing advance),
    honor it verbatim and pin its variant so nothing downstream re-resolves it.
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

    # Honor the phone's optimistic hop: if the target it already shows sits in
    # the next few unplayed slots, advance TO it. Marking anything popped over
    # as played keeps the dedupe set truthful (rapid taps legitimately jump
    # slots when an in-flight advance raced this skip).
    if target_video_id and trigger == "skip_next":
        _probe = sess.playlist_cursor
        _found_at = -1
        _seen = 0
        while _probe < len(sess.playlist) and _seen < 4:
            _t = sess.playlist[_probe]
            if _t.video_id not in sess.played_track_ids:
                if _t.video_id == target_video_id:
                    _found_at = _probe
                    break
                _seen += 1
            _probe += 1
        if _found_at >= 0:
            _cursor_was = sess.playlist_cursor
            for _j in range(sess.playlist_cursor, _found_at):
                _mid = sess.playlist[_j]
                if _mid.video_id not in sess.played_track_ids:
                    sess.played_track_ids.add(_mid.video_id)
            sess.playlist_cursor = _found_at
            # The phone's surface already holds THIS id; a pop-time variant
            # re-search replacing it is exactly the divergence we're closing.
            sess.playlist[_found_at].variant_resolved_mode = sess.display_mode
            print(
                f"[radio] skip target honored id={target_video_id} "
                f"at={_found_at} (cursor was {_cursor_was})",
                flush=True,
            )
        else:
            print(
                f"[radio] skip target not in window id={target_video_id} — normal pop",
                flush=True,
            )

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
            # Thread the title/artist through, exactly as the toggle handler
            # does. build_station's search-based recovery (_build_station_fallback)
            # keys entirely on `f"{title} {artist}"` and returns (None, []) the
            # moment that string is empty — so without this the fallback is
            # unreachable on EVERY extend, and the seeds it exists to rescue
            # (regional / non-catalog music: its docstring names Persian tracks)
            # get one short search-built queue and then die at the first refill.
            # Prefer the track we're extending FROM — it carries a real artist —
            # and fall back to the session's original seed.
            _cur = sess.current_station_track
            _ext_title = (_cur.title if _cur else "") or (
                sess.seed_track.title if sess.seed_track else ""
            )
            _ext_artist = _cur.artist if _cur else ""
            # A placeholder title is WORSE than none. `_tool_play_media`
            # initialises `video_title` to "YouTube Video" and leaves it there
            # whenever its title regex misses, and that string can reach the
            # session as the seed title. Searching YT Music for "YouTube Video"
            # returns real songs — arbitrary ones — so the fallback would
            # cheerfully extend the station with tracks unrelated to anything
            # the user asked for. Empty makes the fallback decline instead,
            # which is the honest outcome.
            if _ext_title.strip().casefold() in ("", "youtube video"):
                _ext_title = ""
                _ext_artist = ""
            _seed_before = sess.seed_track.video_id if sess.seed_track else None
            _seed_meta, new_tracks = await build_station(
                extend_seed, limit=50,
                seed_title=_ext_title, seed_artist=_ext_artist,
                variety=True,
            )
            # The build can run many seconds under the advance lock while a
            # RESEED (which takes the toggle lock, not this one) replaces the
            # whole station. Extending + broadcasting the DEAD station's pick
            # over the just-requested song is an uncommanded jump (review
            # finding). A changed seed or a disable ends this advance.
            _seed_now = sess.seed_track.video_id if sess.seed_track else None
            if not sess.enabled or _seed_now != _seed_before:
                print(
                    f"[radio] advance abandoned mid-refill — station replaced "
                    f"(seed {_seed_before} → {_seed_now}, enabled={sess.enabled})",
                    flush=True,
                )
                return False
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
        # Tell the user on EVERY exhaustion, not only the third.
        #
        # `record_failure` disables at MAX_CONSECUTIVE_FAILURES (3), but from
        # natural playback that counter can never get past 1: it only advances
        # when a media_ended arrives, a media_ended only arrives after a track
        # plays, and a track only plays after a SUCCESSFUL advance — which
        # resets the counter to 0. So the first exhaustion returned False with
        # nothing broadcast at all: the player had no next track, no further
        # media_ended could ever be generated, and the session sat enabled
        # forever with the radio pill lit over a station that would never move
        # again. Silence, and no way for the user to find out why.
        #
        # The notice is unconditional; the `radio_state` above stays gated on
        # an actual disable, so a transient extend failure informs without
        # tearing the station down (a manual skip can still retry it).
        #
        # THROTTLED, because the clients render it as a modal alert and the
        # mobile skip queue does not drop repeats: hold ⏭ on an exhausted
        # station and every tap is another exhaustion, i.e. another dialog
        # stacked on the last. One notice per minute per session says the same
        # thing without making the user dismiss a queue of them.
        _now_ts = time.time()
        if _now_ts - sess.last_exhaustion_notice_ts >= 60.0:
            sess.last_exhaustion_notice_ts = _now_ts
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
    # Skip the pop-time swap when the upcoming-window pre-resolver already
    # resolved this track for the current mode (the common path) — it's already
    # the right variant, and re-searching could pick a DIFFERENT id than the one
    # we already shipped in `upcoming` (re-introducing the desync). Falls through
    # to the inline swap only for tracks the pre-resolver didn't reach in time.
    if sess.display_mode_user_override and next_track.variant_resolved_mode != effective_mode:
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

    # Mark the track we're about to play as resolved for this mode so the
    # upcoming-window pre-resolver treats it as settled and `upcoming`==pop holds.
    if sess.display_mode_user_override:
        next_track.variant_resolved_mode = effective_mode

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
    # Keep the library entry in step with what was actually played — the tape
    # grows as the station runs, and a playlist the user opens tomorrow should
    # show the songs they heard, not only the ones queued at the start.
    try:
        from app.api.media_playlists import autosave_station as _autosave
        asyncio.create_task(_autosave(user_id, channel))
    except Exception:
        pass
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


def _length_sec(length) -> int:
    """Duration in whole seconds from a YT Music "4:18" string; 0 = unknown.
    parse_length_sec is a staticmethod — reaching it through a session
    instance coupled the window builder to the session TYPE, and one bad
    slot then emptied the whole window inside its blanket except. A bad
    length may cost that slot its duration, never the caller its window."""
    try:
        from app.agent.radio.session import RadioSession

        return int(RadioSession.parse_length_sec(length))
    except Exception:
        return 0


def _upcoming_tracks(sess, n: int = 5) -> list:
    """The next `n` station tracks the playlist will pop, as lightweight dicts
    for the mobile pre-load queue. Mobile prefetches + queues these to local
    disk so even rapid back-to-back skips hop to an already-downloaded file
    (instant) instead of round-tripping. n=5 covers a burst of ~4 fast skips;
    it's just IDs+titles from the in-memory playlist, so it's ~free to send.

    Truncated at the first slot the POP would still swap. The phone skips into
    this window OPTIMISTICALLY — it advances the card to `upcoming[0]` without
    waiting for us — so a slot the pre-resolver has not settled is not a hint,
    it is a wrong card: `_advance_and_broadcast_next` re-resolves it inline and
    plays a DIFFERENT id. `_resolve_upcoming_variants` runs under a 2s budget
    over 8 slots, so under load it routinely leaves some unsettled, and the
    window promised them anyway.

    Measured 2026-08-06: the phone skipped to `mi-JD4jGVUQ` (KAROL G - Topic,
    which YouTube plays as bare album art) while the pop played `QCZZwZQ4qNs`
    (the official video) — same song, ~1s apart. Three of nine skips in that
    run did it. That is the "stale song card that heals after a second".

    Only the override case swaps at pop time, so only it truncates. With no
    override the queue plays exactly what it holds and the window is already
    true; in song mode an ATV needs no swap, so nothing truncates there either.
    """
    out = []
    try:
        strict = bool(getattr(sess, "display_mode_user_override", False))
        mode = sess.display_mode
        for t in sess.playlist[sess.playlist_cursor:sess.playlist_cursor + n]:
            if strict and t.variant_resolved_mode != mode:
                break
            out.append({
                "video_id": t.video_id,
                "title": t.display_title(),
                "artist": t.artist,
                "thumbnail_url": t.thumbnail_url,
                # Seconds; 0 = unknown. Lets an optimistic hop into this slot
                # render a real length instead of '--:--' until first buffer.
                "duration": _length_sec(t.length),
            })
    except Exception:
        pass
    return out


# Resolve a few more than we ship so a rapid skip burst stays on pre-resolved
# (correct-variant) tracks, and cap the wall-time so a slow YT Music search never
# stalls a media_play broadcast.
_VARIANT_RESOLVE_WINDOW = 8
_VARIANT_RESOLVE_BUDGET = 2.0  # seconds
# How many times a variant lookup may come back empty for one track before we
# accept that no counterpart exists. Small on purpose: the failures this is
# here for are transient (timeout / throttle), and a track that genuinely has
# no music video should stop costing searches quickly.
_VARIANT_RESOLVE_MAX_ATTEMPTS = 3
# An explicit Song/Video tap is off the audio path (the current track keeps
# playing), and the window it produces decides whether the phone can skip
# instantly for the rest of the station. Worth more than the 2s the advance
# path can afford.
_VARIANT_RESOLVE_FLIP_BUDGET = 9.0  # seconds


async def _resolve_upcoming_variants(sess, n: int = _VARIANT_RESOLVE_WINDOW,
                                     budget: float = _VARIANT_RESOLVE_BUDGET) -> None:
    """Pre-resolve the Song/Video variant of the next `n` station tracks IN PLACE,
    so the `upcoming` window shipped to mobile carries the SAME video_ids the pop
    will actually play.

    Why: the pop-time swap (below) replaces an OMV track with its ATV "Song"
    variant (and vice-versa) — a DIFFERENT video_id — but `_upcoming_tracks`
    ships the raw pre-swap ids. The mobile client prebuffers those raw ids, then
    the backend plays the swapped id → the card's title/artwork desync from the
    audio and the skip cold-loads (2026-06-13 bug). Resolving the window ahead of
    time and writing the resolved track back into the playlist makes upcoming and
    pop agree by construction; the pop-time swap then no-ops on resolved tracks.

    Web-safe: web ignores `upcoming`, and the popped/broadcast video_id is
    unchanged (still the ATV in song mode) — this only moves WHEN the swap is
    computed (window-entry instead of pop-time) and caches it per track+mode.
    No-op unless the user has an explicit display-mode override (matches the
    pop-time swap gate). Cached via StationTrack.variant_resolved_mode, run
    concurrently, and bounded by _VARIANT_RESOLVE_BUDGET so it can't stall audio:
    anything not resolved in time ships raw this frame and resolves on a later one.
    """
    # Mobile ('app') is the only client that prebuffers `upcoming`; gating here
    # keeps web/telegram/discord/slack byte-identical to before (they skip this
    # entirely and fall back to the unchanged pop-time swap).
    if getattr(sess, "channel", None) != "app":
        return
    if not getattr(sess, "display_mode_user_override", False):
        return
    mode = sess.display_mode
    targets = []  # snapshot video_ids — the cursor may advance while we await
    for t in sess.playlist[sess.playlist_cursor:sess.playlist_cursor + n]:
        if t.video_id in sess.played_track_ids or t.variant_resolved_mode == mode:
            continue
        targets.append(t.video_id)
    if not targets:
        return
    from app.agent.radio.playlist import find_topic_version, find_music_video

    def _find_in_queue(vid):
        for j in range(sess.playlist_cursor, len(sess.playlist)):
            if sess.playlist[j].video_id == vid:
                return j
        return None

    async def _resolve_one(vid: str) -> None:
        idx = _find_in_queue(vid)
        if idx is None:
            return
        tr = sess.playlist[idx]
        if tr.variant_resolved_mode == mode:
            return
        alt = None
        # A flip BACK is free. The forward swap replaced this slot and kept the
        # track it replaced (`counterpart`), so the variant this mode wants is
        # already in memory — no YT Music search, no budget spent, no slot left
        # unsettled for `_upcoming_tracks` to truncate the window at. This is
        # what keeps the phone's prebuffer window at full depth across a
        # Song<->Video flip, and with it the difference between a track that
        # starts instantly and one that pays the ~5.4s cold build.
        cp = tr.counterpart
        if cp is not None and cp.is_right_variant_for(mode):
            alt = cp
        elif mode == "song" and tr.video_type == "MUSIC_VIDEO_TYPE_OMV":
            alt = await find_topic_version(tr)
        elif mode == "video" and tr.video_type == "MUSIC_VIDEO_TYPE_ATV":
            alt = await find_music_video(tr)
        idx = _find_in_queue(vid)  # re-find: the awaited search may have raced a pop
        if idx is None:
            return
        if alt is not None and alt.video_id != vid:
            alt.variant_resolved_mode = mode
            # One direction is enough, and the asymmetry is not an oversight:
            # the track ENTERING the playlist gets a link to the one LEAVING it,
            # so the very next flip finds its counterpart on the track it is
            # looking at, and that flip re-links in the other direction in turn.
            # A back-link here would be unreachable code — a mutation test that
            # deletes it cannot make anything fail.
            alt.counterpart = tr
            sess.playlist[idx] = alt
            print(
                f"[radio] upcoming_variant_resolved mode={mode} from={vid} to={alt.video_id} "
                f"title={alt.title!r}",
                flush=True,
            )
        else:
            # "No swap NEEDED" and "no swap AVAILABLE" are not the same fact, and
            # collapsing them is what made Video mode play album art.
            #
            # `variant_resolved_mode == mode` means "already the right variant",
            # and the pop-time swap gate at _advance_and_broadcast_next reads it
            # as exactly that. Stamping it after a FAILED lookup therefore does
            # not merely skip a re-search — it permanently tells every later
            # stage this ATV id is the music video. One 6s find_music_video
            # timeout, one anti-bot-throttled YT Music search, or one artist-name
            # mismatch condemned that track to album art for the whole session,
            # with no retry anywhere.
            #
            # So: settle the track only when nothing was to be found in the first
            # place, and give a genuine failure a bounded number of retries on
            # later frames before accepting that no counterpart exists.
            tr2 = sess.playlist[idx]
            needed = (
                (mode == "song" and tr2.video_type == "MUSIC_VIDEO_TYPE_OMV")
                or (mode == "video" and tr2.video_type == "MUSIC_VIDEO_TYPE_ATV")
            )
            if not needed:
                # Already the right variant for this mode, or a UGC/unknown type
                # we have no counterpart lookup for. Settled, truthfully.
                tr2.variant_resolved_mode = mode
            else:
                tr2.variant_attempts += 1
                if tr2.variant_attempts >= _VARIANT_RESOLVE_MAX_ATTEMPTS:
                    # Tried enough times across enough frames to call it: this
                    # song has no usable counterpart. Settle it so we stop paying
                    # for the search, and let the client present it honestly.
                    tr2.variant_resolved_mode = mode
                    print(
                        f"[radio] variant_unavailable mode={mode} video_id={vid} "
                        f"attempts={tr2.variant_attempts} title={tr2.title!r}",
                        flush=True,
                    )
                else:
                    print(
                        f"[radio] variant_lookup_failed mode={mode} video_id={vid} "
                        f"attempt={tr2.variant_attempts} — will retry",
                        flush=True,
                    )

    try:
        await asyncio.wait_for(
            asyncio.gather(*[_resolve_one(v) for v in targets], return_exceptions=True),
            timeout=budget,
        )
    except asyncio.TimeoutError:
        pass  # ship what's resolved; the rest resolve on a later frame


async def _broadcast_track_for_mode(user_id, channel, sess, track, trigger: str, record: bool) -> None:
    """Broadcast a media_play for `track` plus a radio_state update. `record`
    is False for skip_prev and history-step-forward (track already in tape)."""
    from app.agent.radio.player import broadcast_radio_track
    # A re-anchor is a correction, not an advance, and the wire must say so:
    # trigger used to die at this boundary, so re-anchors went out as
    # reason="auto_advance" — indistinguishable from a real pop (observed live
    # 2026-08-11: a duplicate media_ended produced two identical auto_advance
    # frames for one advance). Genuine advances keep the default; clients that
    # branch on reason are unaffected.
    reason = "reanchor" if trigger == "reanchor" else "auto_advance"
    if trigger == "reanchor":
        # …and corrections are paced. Every stray/duplicate end earns at most
        # one media_play re-anchor per window; inside it, state alone. A
        # client looping bogus ends otherwise gets a media_play per report,
        # and on web each one can re-trigger the very report being answered.
        _now = time.time()
        if _now - getattr(sess, "last_reanchor_ts", 0.0) < _REANCHOR_MIN_INTERVAL_SEC:
            await broadcast_to_user(user_id, sess.to_broadcast_dict())
            return
        sess.last_reanchor_ts = _now
    # Keep `upcoming` in lockstep with what the pop will actually play: resolve the
    # window's variants (cached, time-bounded) before shipping the prebuffer hints.
    await _resolve_upcoming_variants(sess)
    await broadcast_radio_track(
        user_id=user_id,
        video_id=track.video_id,
        title=track.display_title(),
        channel=channel,
        artist=track.artist,
        thumbnail_url=track.thumbnail_url,
        video_type=track.video_type,
        reason=reason,
        upcoming=_upcoming_tracks(sess),
        # Every advance ships the length YT Music already gave us. Without it
        # the card sits on '--:--' (or the PREVIOUS track's length) until the
        # player has buffered enough to measure — the whole 'Starting…' window.
        duration=_length_sec(track.length),
    )
    await broadcast_to_user(user_id, sess.to_broadcast_dict())


# A track that became current less than this long ago has not ended — nothing a
# user asks for is 15 seconds long. Absolute floor, applied when we have no
# duration for the track.
# How long after a station is built a second toggle for the SAME seed counts
# as a duplicate of the same user request rather than a fresh ask.
_DUP_TOGGLE_WINDOW_SEC = 25.0
_MIN_TRACK_PLAY_SEC = 30.0
# …and when we DO know the duration, require a real fraction of it. A 4-minute
# song does not end at 45s. Kept well below 1.0 so a fade-out, a short outro or
# a client that stops reporting near the end still advances normally.
_MIN_TRACK_PLAY_FRACTION = 0.5
# Two ends for the same video inside this window are the same physical event.
_MEDIA_ENDED_DEDUP_SEC = 20.0
# At most one media_play re-anchor per this window; further corrections send
# radio_state alone (see _broadcast_track_for_mode).
_REANCHOR_MIN_INTERVAL_SEC = 5.0
# Machine advances (the client's auto-skip-on-error, radio_skip_next
# reason="auto_error"). A HUMAN ⏭ is authoritative and never throttled; a
# machine advancing because streams keep dying gets pacing and a cap — the
# 2026-08-10 recording shows what the unpaced lane does: a failure train
# walks the station one card flash at a time, each skip arriving with the
# user's authority. Past the cap the station STOPS advancing and the user is
# told, which is the honest outcome the client's own give-up already
# implements for its local player.
_AUTO_SKIP_MIN_INTERVAL_SEC = 5.0
_AUTO_SKIP_WINDOW_SEC = 90.0
_AUTO_SKIP_MAX_PER_WINDOW = 3
# One advance at a time per user. `_handle_media_ended` is dispatched with
# `asyncio.create_task` from two independent places (the main receive loop and
# the mid-turn stop-watcher), so without this two copies interleave: both read
# the pre-advance state, both pass their guards, and both pop a track — one
# physical track-end moving the cursor twice.
_media_ended_locks: dict = {}


def _media_ended_lock(user_id: str, channel: str) -> asyncio.Lock:
    key = (user_id, channel)
    lock = _media_ended_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _media_ended_locks[key] = lock
    return lock


async def _handle_media_ended(user_id: str, msg: dict) -> None:
    channel_pre = (msg.get("channel") or "").strip().lower()
    async with _media_ended_lock(user_id, channel_pre):
        await _handle_media_ended_locked(user_id, msg)


async def _handle_media_ended_locked(user_id: str, msg: dict) -> None:
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
        # Matches the session somewhere (seed or history) but NOT the current
        # track. This branch used to ADVANCE ("typically a user replay of an
        # earlier track") — but no client replays history through this lane,
        # and the 2026-08-10 recordings show what it actually caught: late
        # ends from players still holding a PREVIOUS track (a torn-down
        # WebView, a dead prebuffered queue item, a second device). Advancing
        # the CURRENT track off a previous track's end is exactly the
        # uncommanded jump the user recorded, and because the dedupe below
        # keys on the ended id (not the track it advanced), a stale end for
        # A followed by a stale end for B walks the station twice. A stray
        # end moves nothing; re-anchor the sender instead.
        logger.info(
            "[radio] media_ended for non-current session track video=%s current=%s "
            "— no advance; re-anchoring to the session's current",
            ended_video_id, sess.current_track_id,
        )
        # Re-anchor with a real media_play for the CURRENT track, not only a
        # radio_state. Web's inline cards replay history by swapping the
        # iframe LOCALLY (no backend frame), so when that replay ends, the
        # only thing that can resume the station is a play command for what
        # the session actually holds — the old code advanced here, which
        # double-moved on stray ends but was also the resume path for the
        # replay flow. A same-id media_play is a no-op on the app (hop guard /
        # active-track resume) and a clean resume on web. (Review finding:
        # a bare radio_state left web sitting on a dead end screen.)
        _cur = sess.current_station_track
        if _cur is not None and sess.enabled:
            await _broadcast_track_for_mode(
                user_id, channel, sess, _cur, "reanchor", record=False,
            )
        else:
            await broadcast_to_user(user_id, sess.to_broadcast_dict())
        return

    # ── Idempotency ──────────────────────────────────────────────────
    # One physical track-end can reach us more than once: a second connected
    # client, a reconnect replay, or the two dispatch paths (main receive loop
    # and the mid-turn stop-watcher) racing as separate tasks. Each duplicate
    # used to pop another track, which is how a single ended event could jump
    # the station by two and leave the phone's card and its lock screen showing
    # different songs.
    now = time.time()
    # Keyed on EVERY recently-honored end, not only the last one: the single
    # last_ended pair cannot see alternation (ends A, B, A each looked new and
    # each popped a track — one per card flash in the recorded rapid cycle).
    sess.recent_ended_ids = {
        v: t for v, t in sess.recent_ended_ids.items()
        if now - t < _MEDIA_ENDED_DEDUP_SEC
    }
    if ended_video_id and ended_video_id in sess.recent_ended_ids:
        logger.info(
            "[radio] media_ended duplicate video=%s within %.1fs — no-op",
            ended_video_id, now - sess.recent_ended_ids[ended_video_id],
        )
        return

    # ── Track pinning ────────────────────────────────────────────────
    # A track that became current seconds ago has not ENDED. Nothing the user
    # asks for is 15 seconds long, so an "ended" this early is a client-side
    # artefact — a stalled stream, a player replaced mid-load, a WebView torn
    # down by backgrounding — and honoring it silently replaces the song the
    # user explicitly asked for. That is exactly what happened on 2026-07-31:
    # "Love The Way You Lie" was preempted ~18s in by an advance the user never
    # requested, and a minute of failed loads walked a Kendrick Lamar request
    # down someone else's station.
    #
    # Deliberately scoped to media_ended, the only PASSIVE advance trigger. An
    # explicit `radio_skip_next` is authoritative and is never gated: the user
    # may skip whenever they like.
    # Completion evidence: the client can now prove an end (reported playhead
    # at the track's own duration). A user who SEEKS to the last seconds of a
    # track and lets it finish produced a genuine end the wall-clock floor
    # cannot see — refusing it left web/video sitting on an ENDED screen with
    # every client-side latch spent (review finding). Evidence-backed ends
    # waive the pin; unevidenced ends keep the full wall-clock discipline.
    _rep_pos = msg.get("position")
    _rep_dur = msg.get("duration")
    completed_by_evidence = (
        isinstance(_rep_pos, (int, float))
        and isinstance(_rep_dur, (int, float))
        and _rep_dur > 0
        and _rep_pos >= _rep_dur - 5
    )
    if sess.current_track_started_ts and not completed_by_evidence:
        elapsed = now - sess.current_track_started_ts
        # Proportional floor when we know the length (a 4-minute track cannot
        # end at 45s), absolute floor when we don't.
        floor = _MIN_TRACK_PLAY_SEC
        if sess.current_track_length_sec > 0:
            floor = max(floor, sess.current_track_length_sec * _MIN_TRACK_PLAY_FRACTION)
        if elapsed < floor:
            logger.info(
                "[radio] media_ended too early video=%s elapsed=%.1fs floor=%.1fs "
                "(len=%.0fs) — track pinned, not advancing",
                ended_video_id, elapsed, floor, sess.current_track_length_sec,
            )
            # Re-anchor the client on what the session actually holds, so a
            # phone that lost its stream re-syncs instead of drifting.
            await broadcast_to_user(user_id, sess.to_broadcast_dict())
            return

    sess.last_ended_video_id = ended_video_id
    sess.last_ended_ts = now
    if ended_video_id:
        sess.recent_ended_ids[ended_video_id] = now

    await _advance_and_broadcast_next(user_id, channel, sess, trigger="media_ended")


def _reconcile_cursor_to_target(mgr, sess, target_video_id: str) -> bool:
    """Adopt a move the phone's queue already made: advance the cursor TO
    `target_video_id` if it sits in the next few unplayed slots. Marks
    jumped-over slots played, records history — but broadcasts nothing (the
    phone is already audibly there; the caller ships state + a fresh window).
    False when the target is not in the window (caller falls back to the
    request rules)."""
    _probe = sess.playlist_cursor
    _seen = 0
    while _probe < len(sess.playlist) and _seen < 4:
        _t = sess.playlist[_probe]
        if _t.video_id not in sess.played_track_ids:
            if _t.video_id == target_video_id:
                for _j in range(sess.playlist_cursor, _probe):
                    _mid = sess.playlist[_j]
                    if _mid.video_id not in sess.played_track_ids:
                        sess.played_track_ids.add(_mid.video_id)
                sess.playlist_cursor = _probe + 1
                _t.variant_resolved_mode = sess.display_mode
                mgr.record_auto_play(sess, _t, source="reconcile")
                print(
                    f"[radio] cursor reconciled to phone's advance "
                    f"target={target_video_id} at={_probe}",
                    flush=True,
                )
                return True
            _seen += 1
        _probe += 1
    print(
        f"[radio] reconcile target not in window target={target_video_id}",
        flush=True,
    )
    return False


async def _handle_radio_skip_next(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager

    channel = (msg.get("channel") or "").strip().lower()
    # Who is asking. Absent ⇒ "user" (every existing client), so a human ⏭
    # keeps its full authority. "auto_error" is the client's auto-skip-on-
    # error lane declaring itself a machine — it used to arrive as a bare
    # skip, indistinguishable from a tap, which handed the failure train the
    # one verb the pinning defence deliberately exempts.
    reason = (msg.get("reason") or "user").strip().lower()
    # What the machine is skipping AWAY from (auto_error only), and what the
    # phone optimistically hopped TO (its upcoming[0]) — see the target
    # honoring in _advance_and_broadcast_next.
    from_video_id = (msg.get("video_id") or "").strip()
    target_video_id = (msg.get("target_video_id") or "").strip()
    print(
        f"[radio] skip_next entry user={user_id[:8]} channel={channel!r} "
        f"reason={reason} from={from_video_id or '-'} target={target_video_id or '-'}",
        flush=True,
    )

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
    # ONE advance at a time per (user, channel) — the same lock media_ended
    # holds. Skips are dispatched as independent create_task's from two lanes
    # (main loop + mid-turn stop-watcher), and _advance_and_broadcast_next
    # awaits network mid-flight (refill build, variant search): two unlocked
    # skip tasks interleave around those awaits and pop twice / broadcast out
    # of order. A USER skip stays UNGATED (no pinning, no dedup — an explicit
    # tap is authoritative); the lock only serializes it.
    async with _media_ended_lock(user_id, channel):
        # Re-check under the lock. The wait is not instant — the holder may be
        # a media_ended advance in the middle of a station refill (a build can
        # take tens of seconds), and a user who gives up and switches radio OFF
        # in the meantime is handled inline, unlocked. Popping and broadcasting
        # after that restarts music the user just stopped.
        if not sess.enabled:
            print(f"[radio] skip_next abandoned — radio turned off while queued user={user_id[:8]}", flush=True)
            return
        if reason == "auto_error":
            now = time.time()
            sess.auto_advance_ts = [
                t for t in sess.auto_advance_ts if now - t < _AUTO_SKIP_WINDOW_SEC
            ]
            # RECONCILE vs REQUEST — the distinction the whole lane hangs on.
            # A machine report that carries a TARGET describes a move the
            # phone's queue ALREADY made (a dead track auto-advanced into the
            # prebuffered next). Refusing it — pacing, caps, staleness —
            # cannot un-advance the phone; it can only freeze this cursor
            # behind reality, after which every natural end the phone sends is
            # "non-current" (a no-op), no window ever refills, and the station
            # dies silently when the phone's ~5 local tracks drain. Pacing
            # throttles the POP, never the bookkeeping of a done deed.
            if target_video_id and _reconcile_cursor_to_target(
                mgr, sess, target_video_id,
            ):
                sess.auto_advance_ts.append(now)
                await _resolve_upcoming_variants(sess)
                _win = _upcoming_tracks(sess)
                if _win:
                    await broadcast_to_user(user_id, {
                        "type": "radio_upcoming",
                        "channel": channel,
                        "upcoming": _win,
                        "resolved_mode": sess.display_mode,
                    })
                await broadcast_to_user(user_id, sess.to_broadcast_dict())
                return
            # No target (or an unknown one): the phone is ASKING us to advance
            # it. These are the requests the discipline below exists for.
            #
            # A request naming a track the station already advanced past is
            # the same death reported twice (an ended+skip pair, or two client
            # error lanes racing) — honoring it moves the cursor twice.
            if from_video_id and from_video_id != sess.current_track_id:
                print(
                    f"[radio] auto_skip stale from={from_video_id} "
                    f"current={sess.current_track_id} — no-op",
                    flush=True,
                )
                await broadcast_to_user(user_id, sess.to_broadcast_dict())
                return
            if sess.auto_advance_ts and now - sess.auto_advance_ts[-1] < _AUTO_SKIP_MIN_INTERVAL_SEC:
                print(
                    f"[radio] auto_skip paced — last machine advance "
                    f"{now - sess.auto_advance_ts[-1]:.1f}s ago — re-anchoring",
                    flush=True,
                )
                # Never a SILENT refusal: the client defers-and-retries off
                # this state, and a bare return left it guessing (review
                # finding: a swallowed second death froze the station).
                await broadcast_to_user(user_id, sess.to_broadcast_dict())
                return
            if len(sess.auto_advance_ts) >= _AUTO_SKIP_MAX_PER_WINDOW:
                print(
                    f"[radio] auto_skip CAPPED — {len(sess.auto_advance_ts)} machine "
                    f"advances in {_AUTO_SKIP_WINDOW_SEC:.0f}s — station holds",
                    flush=True,
                )
                if now - sess.last_trouble_notice_ts >= 60.0:
                    sess.last_trouble_notice_ts = now
                    await broadcast_to_user(user_id, {
                        "type": "radio_notice",
                        "channel": channel,
                        "message": "Playback keeps failing — check your connection, or tap ⏭ to try the next track.",
                    })
                return
            sess.auto_advance_ts.append(now)
        await _advance_and_broadcast_next(
            user_id, channel, sess, trigger="skip_next",
            target_video_id=target_video_id,
        )


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


def _length_to_seconds(length: str | int | None) -> int:
    """StationTrack.length is YT Music's display string ("4:18", "1:02:07").
    0 on anything unparseable — the frame simply omits the field then."""
    if isinstance(length, int):
        return max(0, length)
    if not length or not isinstance(length, str):
        return 0
    parts = length.strip().split(":")
    try:
        secs = 0
        for p in parts:
            secs = secs * 60 + int(p)
        return max(0, secs)
    except (ValueError, TypeError):
        return 0


# Serializes concurrent display-mode flips per user. The handler is dispatched
# as a task (it must not block the WS receive loop — its variant resolves can
# take seconds), but two rapid pill taps interleaving through the awaits would
# corrupt the session's current-track/window state, so flips queue here.
_display_mode_locks: dict[str, asyncio.Lock] = {}


async def _handle_radio_display_mode(user_id: str, msg: dict) -> None:
    lock = _display_mode_locks.setdefault(user_id, asyncio.Lock())
    async with lock:
        await _handle_radio_display_mode_locked(user_id, msg)


async def _handle_radio_display_mode_locked(user_id: str, msg: dict) -> None:
    from app.agent.radio import get_radio_manager, RadioSessionManager
    from app.agent.radio.playlist import find_topic_version, find_music_video, StationTrack
    from app.agent.radio.player import broadcast_radio_track, warm_audio_cache

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

    # ACK FIRST. The variant resolves below run for up to ~15s
    # (find_music_video ≤6s + the 9s window budget), and the founder's P2
    # recording shows what holding the ack hostage to them costs: the client
    # flipped optimistically, heard nothing from the server, and sat in
    # silence with no authoritative mode. The state snapshot is cheap and
    # idempotent — a second one ships at the end for paths that mutate the
    # current track.
    await broadcast_to_user(user_id, sess.to_broadcast_dict())

    # Video→Song on the app channel means the phone is about to stream native
    # audio for the CURRENT id — audio that has never been through the proxy
    # (it was playing inside the WebKit iframe, which streams from YouTube
    # directly). Start the platform's spool/remux NOW, before the window
    # resolve spends its budget: by the time the phone's /audio_stream lands,
    # extraction is in flight or done and the spool is filling. This was the
    # structural half of the founder's 30s Video→Song silence (2026-08-09).
    # now_playing=True: on a pre-spool platform this downgrades to extract —
    # a build there would race the very stream the flip is about to start.
    if channel == "app" and sess.enabled and mode == "song" and sess.current_track_id:
        warm_audio_cache([sess.current_track_id], mode="build", now_playing=True)

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

    if channel == "app" and sess.enabled and mode == "song":
        # SONG direction only. The app plays song mode NATIVELY: an ATV swap
        # re-broadcasts the same song under a different video_id, which the
        # phone can only honor as a cold reload + mid-file seek — audible dead
        # air stacked on whatever flip (manual pill or the background
        # auto-flip) just happened, for audio that is the same song either
        # way. Skip it; FUTURE advances pick the right variant via the
        # upcoming pre-resolver, and the re-resolved window ships now so the
        # phone's prefetch queue matches what the pops will play.
        #
        # The VIDEO direction deliberately falls through to the normal swap
        # below: an explicit Video tap on a Topic/ATV track wants the actual
        # music video, the surface changing is the whole point, and the
        # iframe's loadVideoById + mode-flip position carry absorb the reload.
        #
        # Same budget as the Video direction: a Song tap is off the audio path
        # too (this branch deliberately does NO mid-track swap on the app), and
        # the window it produces decides whether every later advance starts
        # instantly or cold-builds. The 2s advance-path budget was left here
        # when the Video direction got its own, and under it this window
        # truncated to a single track.
        await _resolve_upcoming_variants(sess, budget=_VARIANT_RESOLVE_FLIP_BUDGET)
        upcoming = _upcoming_tracks(sess)
        if upcoming:
            await broadcast_to_user(user_id, {
                "type": "radio_upcoming",
                "channel": channel,
                "upcoming": upcoming,
                "resolved_mode": sess.display_mode,
            })
        print(
            f"[radio] mode_toggle app-channel song: no mid-track swap; "
            f"reshipped upcoming n={len(upcoming)}",
            flush=True,
        )
    elif sess.enabled and current_track is not None:
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

        # REVALIDATE the capture before mutating. The handler is a task now and
        # find_music_video/find_topic_version yield the loop for up to 6s —
        # an ungated skip or a natural media_ended can advance the station in
        # that window (skips serialize on a different lock). Executing the swap
        # against the stale capture would overwrite the NEW track's tape entry
        # and yank the card back to the track the user just skipped away from.
        # The mode flip itself stands; only the stale swap is dropped.
        if alt is not None and (
            not sess.enabled or sess.current_track_id != current_track.video_id
        ):
            print(
                f"[radio] mode_toggle swap SKIPPED — station moved during lookup "
                f"(captured={current_track.video_id} now={sess.current_track_id})",
                flush=True,
            )
            alt = None

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
            # reason: the frame used to default to "auto_advance", which the
            # client can only read as "new track, start from zero" — the P3
            # reset-to-0:00 card. "mv_swap"/"topic_swap" says "same song, new
            # surface, carry your position". duration: the backend holds
            # alt.length in hand; without it the client shows '--:--' (or a
            # STALE previous-track duration) until the player reports one.
            await broadcast_radio_track(
                user_id=user_id,
                video_id=alt.video_id,
                title=alt.display_title(),
                channel=channel,
                artist=alt.artist,
                thumbnail_url=alt.thumbnail_url,
                video_type=alt.video_type,
                reason=swap_log,
                duration=_length_to_seconds(alt.length),
            )
        elif swap_log:
            # Lookup was attempted but yielded nothing.
            print(
                f"[radio] {fail_log} fallback={current_track.video_id} "
                f"(staying on current track, {mode} UI still applies)",
                flush=True,
            )

        # The window is resolved PER MODE, and only the song direction above
        # re-ships it. So a Video tap swapped the CURRENT track to its music
        # video and left the phone holding the SONG-side (ATV) window for the
        # rest of the station — and the phone's optimistic skip reads that
        # window directly. Every ⏭ after a Video tap therefore jumped the card
        # to an ATV id the station would never play, which YouTube renders as
        # album art with no chrome: a song-looking card under a lit Video pill,
        # corrected only when the pop's own resolved media_play landed. That is
        # the "stale song card on Next, but only after toggling" report; it
        # needed a toggle precisely because a toggle is the only thing that
        # leaves the two sides disagreeing about the window.
        if channel == "app" and sess.enabled and mode == "video":
            # Bigger budget than the default 2s. This runs on an explicit user
            # tap, not on the audio path — the current track keeps playing
            # throughout — and `_upcoming_tracks` now refuses to advertise a
            # slot the pop would still swap. So a 2s budget that resolves
            # NOTHING does not ship a wrong window any more, it ships no window
            # at all, which silently costs the phone its instant skip and its
            # prefetch for the rest of the station. Measured 2026-08-07 against
            # a tenant on this image: a Video tap produced zero radio_upcoming.
            # Spend the time here instead; the reply is a frame, not audio.
            await _resolve_upcoming_variants(sess, budget=_VARIANT_RESOLVE_FLIP_BUDGET)
            _win = _upcoming_tracks(sess)
            if _win:
                await broadcast_to_user(user_id, {
                    "type": "radio_upcoming",
                    "channel": channel,
                    "upcoming": _win,
                    "resolved_mode": sess.display_mode,
                })
            print(
                f"[radio] mode_toggle app-channel video: reshipped upcoming n={len(_win)}",
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
        # Last turn id THIS connection registered — the connection-level
        # backstop for the in-flight registry (see the finally below). A turn
        # that loses its client keeps running inside this handler, so by the
        # time the finally runs the work is genuinely over.
        _conn_turn_mission: Optional[str] = None

        # ── Resume: announce a turn that is still running ──
        # The phone reconnects on every foreground (ensureChatConnected).
        # If this user's previous turn is still in flight — it kept running
        # headless while they were away — say so immediately, so the app can
        # put the working orb back up and keep narrating the agent's actions
        # instead of showing a dead thread until the answer lands.
        try:
            _resume = _get_active_turn(user_id)
            if _resume:
                await websocket.send_json(_turn_frame("turn_active", _resume, resumed=True))
                logger.info(
                    "[WS] announced in-flight turn %s to reconnecting client %s",
                    _resume.get("mission_id"), user_id[:8],
                )
        except Exception:  # noqa: BLE001 — never block a connect on the announce
            pass

        # ── Resume: re-anchor radio state ──
        # The contract had NO media resync on reconnect: radio_state is
        # broadcast only on radio EVENTS, so a phone returning from a long
        # background kept whatever frame it last saw — a dead card over an
        # expired stream, prev/next pills in the wrong state. Announce each
        # enabled session to THIS socket so the client can re-anchor and
        # decide to re-warm. Additive and idempotent: radio_state carries no
        # play command, and clients already receive it on every advance.
        try:
            from app.agent.radio import get_radio_manager as _grm
            for _rs in list(getattr(_grm(), "_sessions", {}).values()):
                # Recency-bounded: sessions never self-expire, so an
                # unbounded announce re-lit a days-old station's pills and
                # lock transport on every app open, forever (review finding).
                # A station idle >2h is history, not state worth resurrecting.
                if (
                    _rs.user_id == user_id
                    and _rs.enabled
                    and time.time() - _rs.last_activity_ts < 2 * 3600
                ):
                    await websocket.send_json(_rs.to_broadcast_dict())
        except Exception:  # noqa: BLE001 — never block a connect on the announce
            pass

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

                # ── "Is a turn still running for me?" ──
                # The connect-time announce only helps a socket that just
                # opened. A client whose chat screen REMOUNTS on a live socket
                # (every deep link does) asks here instead, and gets the same
                # answer: the running turn, or an explicit end.
                if msg_type == "turn_probe":
                    _probe = _get_active_turn(user_id)
                    try:
                        if _probe:
                            await websocket.send_json(
                                _turn_frame("turn_active", _probe, resumed=True)
                            )
                        else:
                            await websocket.send_json({
                                "type": "turn_ended",
                                "mission_id": msg.get("mission_id"),
                            })
                    except Exception:  # noqa: BLE001
                        pass
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
                    # Task, not await: the handler's variant resolves can run
                    # ~15s (6s MV lookup + 9s window budget), and awaiting it
                    # inline blocked every subsequent client frame behind a
                    # pill tap — including the media_ended/skip frames that
                    # keep the station moving. Rapid flips serialize on
                    # _display_mode_locks, so tasking this is order-safe.
                    asyncio.create_task(_handle_radio_display_mode(user_id, msg))
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
                # IANA timezone, e.g. "America/Toronto". Web/mobile send "tz";
                # the Chrome extension sidepanel sends "client_tz" (audit
                # A2-7 — it silently fell back to DB User.timezone before).
                client_tz = msg.get("tz") or msg.get("client_tz")
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

                # ── Instant server ack (2026-07-16 blank-response fix) ──
                # Every accepted message — including the FIRST message of a
                # session (session_id null, so user_message_persisted below
                # is skipped) — gets a sub-100ms status frame. Without it
                # the wire is silent through the whole pre-LLM pipeline +
                # the model's reasoning, and the app shows dead air for
                # ~10s on tool-first turns.
                #
                # The turn's mission id is minted HERE — before the first
                # status frame — so every status frame of the turn carries
                # it (additive field): the app uses it to correlate frames
                # with the turn's Live Activity card (chatturn:<hex> is
                # also the ActivityAttributes name in the start payload).
                _turn_mission_id = f"chatturn:{uuid.uuid4().hex[:12]}"
                # Per-task card title (Claude parity): the card says WHAT
                # is being worked on — the user's ask, one line — not a
                # generic "Working on your answer". Rides every status
                # frame so the app can name a locally-started card, and
                # every platform push for the turn. No new exposure: the
                # started push already carries a 180-char text preview.
                _turn_title = (
                    " ".join((_original_user_text or text or "").split())[:60].strip()
                    or "Working on your answer"
                )
                # Registered before the first frame: a client that reconnects
                # one second into the turn already gets `turn_active`.
                _set_active_turn(
                    user_id,
                    mission_id=_turn_mission_id,
                    title=_turn_title,
                    stage="thinking",
                    tool=None,
                    started_at=time.time(),
                )
                _conn_turn_mission = _turn_mission_id
                try:
                    await websocket.send_json({
                        "type": "status", "stage": "received",
                        "mission_id": _turn_mission_id,
                        "title": _turn_title,
                    })
                except Exception:
                    pass

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
                                    # No work will run under this turn's id —
                                    # retire it so a reconnecting client is
                                    # never told to wait on a phantom turn.
                                    _clear_active_turn(user_id, _turn_mission_id)
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

                # Presence-aware delivery (founder ask 2026-07-16): if the
                # user force-quits mid-turn, the turn keeps running; the
                # phone gets a Live Activity while we work and the final
                # answer as a push when no client is connected to receive
                # the done frame. One mission id ties start→complete.
                # _turn_mission_id was minted up at the 'received' status
                # frame — before the callbacks — so the status frames and
                # the interim-progress emitter close over the same id.
                _turn_flags = {"client_gone": False, "response_persisted": False,
                               "finished": False}

                async def _mirror_turn(stage: str, tool: Optional[str] = None) -> None:
                    """Keep the in-flight registry current and narrate the turn
                    to the user's OTHER live sockets (the reconnected phone).
                    Only real transitions go on the wire — a 400-chunk answer
                    mirrors 'writing' once, not 400 times."""
                    entry = _active_turns.get(user_id)
                    if entry is None or entry.get("mission_id") != _turn_mission_id:
                        return
                    if entry.get("stage") == stage and entry.get("tool") == tool:
                        return
                    entry["stage"] = stage
                    entry["tool"] = tool
                    try:
                        await broadcast_to_user(
                            user_id, _turn_frame("turn_status", entry),
                            exclude=broadcast_queue,
                        )
                    except Exception:  # noqa: BLE001 — narration is best-effort
                        pass

                # Interim Live Activity progress at tool boundaries.
                # Emits on EVERY turn (Claude parity: the backgrounded
                # card streams live status), but rows are update_only —
                # they refresh an existing card and can never START one,
                # so ordinary foreground turns still grow no card. The
                # card itself appears only via the app's local start on
                # backgrounding or the client-gone platform start.
                from app.agent.turn_progress import TurnProgressEmitter

                _turn_emitter = TurnProgressEmitter(
                    mission_id=_turn_mission_id,
                    mission_title=_turn_title,
                    base_progress=5,
                    ceiling=90,
                    route="chat",
                )

                # Stream callbacks
                async def on_text_chunk(chunk: str):
                    _streamed_chunks.append(chunk)
                    # Cheap inline guard: a long answer fires this callback
                    # thousands of times, and only the FIRST chunk after a
                    # stage change has anything to mirror.
                    _e = _active_turns.get(user_id)
                    if _e is not None and _e.get("stage") != "writing":
                        await _mirror_turn("writing")
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
                    await _mirror_turn("tool", tool_name)
                    try:
                        await websocket.send_json({"type": "tool_start", "tool": tool_name})
                    except Exception:
                        pass
                    # update_only rows: refresh the card if one exists.
                    try:
                        await _turn_emitter.on_tool_start(tool_name)
                    except Exception:
                        pass

                async def on_status(stage: str):
                    """Liveness signal during LLM dead-air (2026-07-16):
                    the runner emits 'thinking' before every LLM call so
                    the client can show a live indicator through
                    reasoning TTFT and post-tool iterations."""
                    if stage == "thinking":
                        # Between tools the agent is reasoning again — drop the
                        # finished tool so a resumed client stops narrating it.
                        await _mirror_turn("thinking")
                    try:
                        await websocket.send_json({
                            "type": "status", "stage": stage,
                            "mission_id": _turn_mission_id,
                            "title": _turn_title,
                        })
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
                # Persisted pointers → user Message.attachments so inbound images
                # survive reload / show in history / other devices, and are a
                # reusable source for the edit_image tool.
                _inbound_attachments: list = []
                _media_items = msg.get("media", [])
                if _media_items and isinstance(_media_items, list):
                    import tempfile, base64 as _b64, os as _os
                    from app.agent.doc_generators import _persist as _persist_att
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
                            _raw = _b64.b64decode(_mdata)
                            with open(_fpath, "wb") as _wf:
                                _wf.write(_raw)
                            _media_paths.append(_fpath)
                            # Persist to the storage backend + record a pointer.
                            # Best-effort: a storage hiccup must not drop the turn.
                            try:
                                _att = await _persist_att(_raw, _fname, _mtype, user_id)
                                _inbound_attachments.append(_att.to_dict())
                            except Exception as _pe:
                                logger.warning("[WS] Failed to persist inbound attachment: %s", _pe)
                        except Exception as _me:
                            logger.warning("[WS] Failed to process media attachment: %s", _me)

                # If the user message was already presaved (existing session),
                # back-fill its attachments now that inbound media is persisted.
                # The first-message path instead threads inbound_attachments into
                # run() (below), which saves the user row with attachments.
                if _inbound_attachments and _user_msg_presaved and _persisted_user_msg_id:
                    try:
                        from app.db.database import async_session_maker as _att_sm
                        async with _att_sm() as _att_db:
                            _urow = (await _att_db.execute(
                                __import__('sqlalchemy').select(DbMessage)
                                .where(DbMessage.id == _persisted_user_msg_id)
                            )).scalar_one_or_none()
                            if _urow is not None:
                                _urow.attachments = _inbound_attachments
                                await _att_db.commit()
                    except Exception as _ae:
                        logger.warning("[WS] Failed to back-fill presaved attachments: %s", _ae)

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
                            #
                            # EXCEPT on mobile, where an auto-enable is already on
                            # its way (a few lines below) and this frame would be a
                            # lie with a visible cost: record_user_seed turns the
                            # old station OFF for a new intent, so shipping that
                            # state drops the phone's card back to a bare "YOUTUBE"
                            # card with no prev/next for the ~3-6s the station takes
                            # to build, and again for every later play. The client
                            # renders the station optimistically; the _auto toggle's
                            # radio_state is the one that tells the truth.
                            _sess = _mgr.get(user_id, channel)
                            if _sess and channel != "mobile":
                                await broadcast_to_user(user_id, _sess.to_broadcast_dict())
                    except Exception as _re:
                        logger.warning("[radio] seed-record failed: %s", _re)

                # Mobile default: a "play me X" search starts RADIO mode, not a
                # one-off play. Mobile chat uses channel 'mobile', but its radio
                # session lives on the allow-listed 'app' channel the app
                # subscribes to. Fire-and-forget: the song already plays via the
                # fast-path; build the station + flip the toggle ON in the
                # background. `_auto` skips re-broadcasting the seed (already
                # playing → no cold-reload race). Web (channel 'web') is untouched.
                if _fast_result and channel == "mobile":
                    _ameta = _fast_result[1] or {}
                    if _ameta.get("video_id"):
                        # Audio-first: the phone starts this track NATIVELY
                        # within a second or two — warm the platform's remux
                        # cache for it immediately (deduped + capped server-
                        # side; never awaited).
                        # Deliberately NOT warming this id: the phone is about
                        # to stream these exact bytes and both requests share
                        # one residential proxy — warming here competes with the
                        # play it is supposed to speed up. The station's window
                        # is warmed instead, once it exists.
                        pass
                        asyncio.create_task(_handle_radio_toggle(user_id, {
                            "channel": "app",
                            "enabled": True,
                            "video_id": _ameta.get("video_id", ""),
                            "title": _ameta.get("title", "") or "Now Playing",
                            "seed_intent": text.strip(),
                            # Same surface the media_play frame announced, so
                            # the station's variant resolution matches what the
                            # phone actually put on screen.
                            "mode": _ameta.get("mode") or "song",
                            "_auto": True,
                        }))

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
                # (_turn_mission_id/_turn_flags minted above, before the
                # stream callbacks, so the progress emitter closes over them.)
                _user_text_preview = (_original_user_text or text or "")[:180]

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
                    on_status=on_status,
                    model_override=model,
                    save_user_message=not is_onboarding_msg and not _user_msg_presaved and not _is_system_action,
                    media_paths=_media_paths if _media_paths else None,
                    inbound_attachments=_inbound_attachments if _inbound_attachments else None,
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
                            elif m2.get("type") == "turn_probe":
                                # The chat screen remounted on this very socket
                                # (deep-link tap mid-turn) and lost its live
                                # state — answer with the turn it is standing on.
                                _p = _get_active_turn(user_id)
                                if _p:
                                    await websocket.send_json(
                                        _turn_frame("turn_active", _p, resumed=True)
                                    )
                                else:
                                    await websocket.send_json({"type": "turn_ended"})
                            elif m2.get("type") in _MID_TURN_PASSTHROUGH:
                                # Radio control frames must NOT be eaten here.
                                #
                                # This task owns the socket's single receive
                                # stream for the whole turn, so anything it
                                # doesn't handle is read and dropped — it never
                                # reaches the main dispatcher. For `media_ended`
                                # that is fatal rather than merely lossy: it is
                                # the ONLY thing that advances a station, so a
                                # track finishing while the agent happened to be
                                # talking ended the station outright. Nothing
                                # queued, no further media_ended possible (none
                                # can be produced without a track playing), and
                                # the radio pill still lit over a station that
                                # would never move again.
                                #
                                # Dispatched as tasks so a slow YT Music call
                                # inside a handler cannot stall the stop-watcher
                                # — a user pressing Stop must still be heard.
                                asyncio.create_task(_dispatch_radio_frame(user_id, m2))
                    except asyncio.CancelledError:
                        pass
                    except WebSocketDisconnect:
                        # Client left mid-turn (force-quit / lock). Keep
                        # the agent running — the answer persists and is
                        # push-delivered — and give the phone a live
                        # "working" card so the user sees progress from
                        # the lock screen / Dynamic Island.
                        _turn_flags["client_gone"] = True
                        logger.info(
                            "[WS] client left mid-turn (%s) — continuing headless",
                            user_id[:8],
                        )

                        async def _delayed_working_card():
                            # Grace window: quick turns (reminder arms,
                            # short answers) finish inside it and never
                            # spawn a working card — the user just gets
                            # the answer/reminder banner (founder rule
                            # 2026-07-20: no working-card noise; it
                            # also kept preempting live countdowns).
                            await asyncio.sleep(12)
                            if _turn_flags["finished"]:
                                return
                            try:
                                from app.services.agent_notify_client import notify
                                await notify(
                                    event_kind="mission_started",
                                    title="Working on your answer",
                                    body=_user_text_preview or None,
                                    data={
                                        "mission_id": _turn_mission_id,
                                        "mission_title": _turn_title,
                                        "subtitle": "Working on it…",
                                        "route": "chat",
                                        "kind": "chat_turn",
                                        "urgent": True,
                                        "timer_end_ms": int((time.time() + 120) * 1000),
                                    },
                                    priority="default",
                                    dedup_key=f"{_turn_mission_id}:started",
                                )
                            except Exception:  # noqa: BLE001 — best-effort card
                                return
                            # The card appears mid-turn after several
                            # tools already ran — let the next tool
                            # boundary emit progress immediately.
                            _turn_emitter.force_next()

                        asyncio.create_task(_delayed_working_card())

                stop_task = asyncio.create_task(_wait_for_stop())

                # Helper to safely send on WS (may be closed). Reports
                # delivery: starlette/uvicorn raise more than RuntimeError
                # on a dead socket (ConnectionClosed, ClientDisconnected…)
                # — an uncaught send used to fall into the generic agent
                # error handler and re-save the already-persisted answer
                # as a duplicate '*[Response interrupted]*' row.
                async def _safe_send(data: dict) -> bool:
                    try:
                        await websocket.send_json(data)
                        return True
                    except Exception:  # noqa: BLE001 — WS closed/broken
                        return False

                try:
                    response = await agent_task
                    # The runner has saved the assistant Message by now —
                    # nothing after this point may re-save a partial.
                    _turn_flags["response_persisted"] = True
                    # Cancels the delayed working card: a turn that
                    # finished inside the grace window spawns none.
                    _turn_flags["finished"] = True
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

                    # Alias the real model id to a neutral tier label before it
                    # crosses the WS boundary — the live `done` frame is the same
                    # user-facing model-id surface the REST serializers scrub
                    # (docs/security/audit-2026.md MI-2/MI-4). Flag-gated; the DB
                    # read path is already scrubbed in sessions.py.
                    _done_model = response.model
                    if settings.security_leak_filter and _done_model:
                        from app.services.model_alias import public_model_label
                        _done_model = public_model_label(_done_model)
                    _done_payload = {
                        "type": "done",
                        "text": response.text,
                        "session_id": response.session_id,
                        "tokens": {
                            "input": response.tokens_input,
                            "output": response.tokens_output,
                            "total": response.tokens_total,
                        },
                        "model": _done_model,
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
                    _done_delivered = await _safe_send(_done_payload)

                    # The asking socket is gone — but the user may already be
                    # back on a NEW one (the phone reconnects on every
                    # foreground). Hand that socket the persisted answer live;
                    # the app renders `message` frames straight into the open
                    # thread, so the reply appears the moment it exists instead
                    # of waiting for the next refetch.
                    if not _done_delivered and (response.text or "").strip():
                        try:
                            _late_frame = {
                                "type": "message",
                                "id": (
                                    getattr(response, "asst_message_id", None)
                                    or f"{_turn_mission_id}:answer"
                                ),
                                "role": "assistant",
                                "content": response.text,
                                "created_at": datetime.utcnow().isoformat() + "Z",
                            }
                            if getattr(response, "day_chat_id", None):
                                _late_frame["day_chat_id"] = response.day_chat_id
                            _late_sent = await broadcast_to_user(
                                user_id, _late_frame, exclude=broadcast_queue,
                            )
                            logger.info(
                                "[WS] late answer delivered to %d reconnected socket(s) user=%s",
                                _late_sent, user_id[:8],
                            )
                        except Exception:  # noqa: BLE001 — push + refetch remain
                            pass

                    # Answer delivery push — UNCONDITIONAL (founder
                    # decision 2026-07-17): every chat answer notifies
                    # the phone regardless of app state; the old gate
                    # (done frame failed AND no other WS client) meant a
                    # locked phone next to an open web tab never buzzed.
                    # Spam brakes: per-turn dedup key collapses retry
                    # replays, dismiss_after_s auto-clears the card,
                    # cap_exempt keeps the daily cap from eating answer
                    # #11, and the LA lane start-if-missing renders the
                    # banner even when no 'working' card was started.
                    # urgent bypasses quiet hours — they asked seconds
                    # ago.
                    try:
                        from app.services.agent_notify_client import notify
                        _answer_data = {
                            "mission_id": _turn_mission_id,
                            "mission_title": _turn_title,
                            "route": "chat",
                            "kind": "chat_turn",
                            "urgent": True,
                            "cap_exempt": True,
                            "progress": 100,
                            # 5 min, not 15 — a lingering finished card
                            # hogs the island and hides live countdowns.
                            "dismiss_after_s": 300,
                        }
                        if response.session_id:
                            _answer_data["session_id"] = response.session_id
                        if getattr(response, "day_chat_id", None):
                            _answer_data["day_chat_id"] = response.day_chat_id
                        if getattr(response, "asst_message_id", None):
                            _answer_data["message_id"] = response.asst_message_id
                        # REMINDER WINS at the source: when this turn
                        # created a reminder, its countdown card (armed
                        # by the routines API mid-turn) IS the
                        # confirmation — the user just read 'Done —
                        # I'll remind you at …' in chat. A loud
                        # 'Answer ready' card 20s later steals the
                        # Dynamic Island from the countdown for its
                        # 5-minute linger (founder repro 2026-07-22).
                        # silent: the LA lane skips the restart and
                        # ends any working card bannerlessly.
                        # routines__remind is the ONLY reminder-minting
                        # tool (routines__create's kind enum has no
                        # 'reminder'); reschedules via routines__update
                        # are not detectable here — the LA lane's
                        # yields_to_reminder guard covers them.
                        if any(
                            (tc.get("name") or "") == "routines__remind"
                            for tc in response.tool_calls
                        ):
                            _answer_data["silent"] = True
                        await notify(
                            event_kind="mission_completed",
                            title="Answer ready",
                            body=(response.text or "")[:180] or None,
                            data=_answer_data,
                            priority="high",
                            dedup_key=f"{_turn_mission_id}:completed",
                        )
                        logger.info(
                            "[WS] answer push queued user=%s ws_delivered=%s",
                            user_id[:8], _done_delivered,
                        )
                    except Exception as _pe:  # noqa: BLE001
                        logger.warning("[WS] answer push failed: %s", _pe)

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
                    _turn_flags["finished"] = True
                    # Tell a gone client's phone the turn died — the
                    # 'working' card must not sit there forever.
                    if _turn_flags["client_gone"]:
                        try:
                            from app.services.agent_notify_client import notify
                            await notify(
                                event_kind="mission_failed",
                                title="Couldn't finish your answer",
                                body="Something went wrong — open the app and ask again.",
                                data={
                                    "mission_id": _turn_mission_id,
                                    "mission_title": _turn_title,
                                    "route": "chat",
                                    "kind": "chat_turn",
                                    "urgent": True,
                                    "dismiss_after_s": 900,
                                },
                                priority="high",
                                dedup_key=f"{_turn_mission_id}:failed",
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    # Save partial response so it's not lost on refresh —
                    # but never after the runner already persisted the
                    # full answer (that's the duplicate-answer rider).
                    _partial = "" if _turn_flags["response_persisted"] else "".join(_streamed_chunks).strip()
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
                    # However this turn ended — answer, error, user stop,
                    # cancellation — it is no longer in flight. Retire the
                    # registry entry (so a client connecting a second later is
                    # not told to wait on finished work) and tell any resumed
                    # socket to drop its working orb.
                    _clear_active_turn(user_id, _turn_mission_id)
                    try:
                        await broadcast_to_user(user_id, {
                            "type": "turn_ended",
                            "mission_id": _turn_mission_id,
                        }, exclude=broadcast_queue)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            # Clean up broadcast queue and task
            broadcast_task.cancel()
            keepalive_task.cancel()
            _unregister_ws_queue(user_id, broadcast_queue)
            # Backstop for the in-flight registry: paths that never reach the
            # per-turn finally (a duplicate-replay skip, a pre-dispatch raise)
            # must not leave a phantom turn behind for the next connect to
            # announce. Guarded by mission id, so a NEWER turn is untouched.
            if _conn_turn_mission:
                _clear_active_turn(user_id, _conn_turn_mission)
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
