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
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from app.config import settings

logger = logging.getLogger(__name__)


def _friendly_error(exc: Exception) -> str:
    """Convert raw exceptions into user-friendly error messages."""
    name = type(exc).__name__
    msg = str(exc)

    # Anthropic API errors — check status code if available
    status = getattr(exc, "status_code", None)
    if status == 529 or "overloaded" in msg.lower() or "Overloaded" in msg:
        return "Claude is temporarily overloaded. Please try again in a few seconds."
    if status == 429 or "rate_limit" in msg.lower():
        return "Rate limit reached. Please wait a moment and try again."
    if status == 401 or "authentication" in msg.lower():
        return "Authentication error with the AI service. Please contact support."
    if status == 400 or "bad_request" in name.lower():
        return "There was an issue with the request. Please try rephrasing your message."
    if "timeout" in name.lower() or "timeout" in msg.lower():
        return "The request timed out. Please try again."
    if "connection" in name.lower() or "connection" in msg.lower():
        return "Connection error with the AI service. Please try again."

    # Generic fallback — don't expose internals
    logger.error(f"[WS] Unhandled error type for friendly message: {name}: {msg}")
    return "Something went wrong. Please try again in a moment."


# ── User WebSocket broadcast registry ────────────────────────────────
# Maps user_id → list of asyncio.Queues (one per active WS connection).
# Background tasks (e.g. app builder) push events here, and the WS
# handler forwards them to the client.
_user_ws_queues: Dict[str, List[asyncio.Queue]] = {}


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

import re as _re_mod

_PLAY_PATTERNS = [
    _re_mod.compile(r'^\s*play\s+(?:me\s+)?(?:a\s+)?(?:song\s+(?:of\s+|by\s+|called\s+)?|video\s+(?:of\s+|by\s+|called\s+)?|music\s+(?:of\s+|by\s+)?)?(.+?)(?:\s+(?:on|from|in)\s+(?:youtube|yt))?\s*$', _re_mod.I),
    _re_mod.compile(r'^\s*(?:put on|play me|play)\s+["\u201c]?(.+?)["\u201d]?\s*$', _re_mod.I),
]

_NETFLIX_KEYWORDS = _re_mod.compile(r'\b(?:netflix|disney|hulu|prime video|hbo)\b', _re_mod.I)


async def _fast_media_check(text: str, user_id: str, broadcast_queue: asyncio.Queue) -> Optional[str]:
    """If text looks like a play request, search YouTube and broadcast immediately.
    Returns a modified message text with a hint if handled, or None if not a media request."""
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

        # Broadcast media_play immediately — frontend opens YouTube embed
        event = {
            "type": "media_play",
            "provider": "youtube",
            "video_id": video_id,
            "title": video_title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
        broadcast_queue.put_nowait(event)
        logger.info("[FAST-MEDIA] Broadcast media_play in fast-path: %s - %s", video_id, video_title)

        # Return modified text so agent knows media is already playing
        return (
            f"{text}\n\n[SYSTEM: The video \"{video_title}\" (https://www.youtube.com/watch?v={video_id}) "
            f"is already playing in the user's browser. Do NOT call play_media. "
            f"Just respond conversationally about what's now playing.]"
        )
    except Exception as e:
        logger.warning("[FAST-MEDIA] Fast-path failed (agent will handle): %s", e)
        return None


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
            except asyncio.CancelledError:
                print(f"[BROADCAST_READER] Cancelled for user={user_id[:8]}", flush=True)

        broadcast_task = asyncio.create_task(_broadcast_reader())

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

                # If message comes from inside a built app, prepend context
                app_id_from_msg = msg.get("app_id")
                if channel == "app" and app_id_from_msg:
                    try:
                        from app.db.database import async_session_maker
                        from app.db.models import App, BuildJob
                        from sqlalchemy import select
                        import json as _json
                        async with async_session_maker() as _db:
                            _app = await _db.get(App, app_id_from_msg)
                            if _app:
                                _slug_safe = _app.slug.replace('-', '_')
                                _is_layer2 = msg.get("layer2") or False
                                _base_ctx = (
                                    f"[CONTEXT: The user is chatting from inside their '{_app.name}' app. "
                                    f"You are their in-app assistant.\n"
                                    f"- Be conversational and helpful. Greet naturally when they say hi.\n"
                                    f"- NEVER mention internal details (SQLite, bridges, connections, file paths, agent infrastructure).\n"
                                    f"- You have these app tools: app_{_slug_safe}__navigate (change screens), "
                                    f"app_{_slug_safe}__read_file / app_{_slug_safe}__write_file (edit the app), "
                                    f"app_{_slug_safe}__query_db (read/write app data).\n"
                                    f"- When the user asks to change something in the app (UI, content, settings), "
                                    f"use write_file/query_db to make it happen.\n"
                                    f"- Suggest helpful actions as [[Button Label]] chips.\n"
                                )
                                if _is_layer2:
                                    # ── Load Layer 1 context so agent knows what was already asked/built ──
                                    _layer1_context = ""
                                    try:
                                        _l1_result = await _db.execute(
                                            select(BuildJob)
                                            .where(BuildJob.app_id == app_id_from_msg, BuildJob.layer == 1)
                                            .order_by(BuildJob.created_at.desc())
                                            .limit(1)
                                        )
                                        _l1_job = _l1_result.scalar_one_or_none()
                                        if _l1_job:
                                            # Original user request
                                            _layer1_context += f"  LAYER 1 BUILD REQUEST: \"{_l1_job.prompt}\"\n"
                                            # Structured choices from checkpoint
                                            if _l1_job.checkpoint_json:
                                                try:
                                                    _ckpt = _json.loads(_l1_job.checkpoint_json)
                                                    _pc = _ckpt.get("plan_context") or {}
                                                    if _pc:
                                                        _parts = []
                                                        if _pc.get("screens"):
                                                            _parts.append(f"screens: {', '.join(_pc['screens'])}")
                                                        if _pc.get("features"):
                                                            _parts.append(f"features: {', '.join(_pc['features'])}")
                                                        if _pc.get("db_type"):
                                                            _parts.append(f"database: {_pc['db_type']}")
                                                        if _pc.get("design_notes"):
                                                            _parts.append(f"design: {_pc['design_notes']}")
                                                        if _parts:
                                                            _layer1_context += f"  LAYER 1 CHOICES: {'; '.join(_parts)}\n"
                                                    _ec = _ckpt.get("extra_context")
                                                    if _ec:
                                                        _layer1_context += f"  LAYER 1 EXTRA CONTEXT: {_ec[:500]}\n"
                                                except Exception:
                                                    pass
                                        # App plan (architecture decisions)
                                        if _app.plan_json:
                                            try:
                                                _plan = _json.loads(_app.plan_json)
                                                _plan_summary = _plan.get("summary", "")
                                                if _plan_summary:
                                                    _layer1_context += f"  LAYER 1 APP PLAN: {_plan_summary[:500]}\n"
                                            except Exception:
                                                pass
                                        # App description
                                        if _app.description:
                                            _layer1_context += f"  APP DESCRIPTION: {_app.description[:300]}\n"
                                    except Exception as _e:
                                        logger.debug(f"[WS] Could not load Layer 1 context: {_e}")

                                    # Layer 2 trigger — audit-first deep customization, building ON TOP of Layer 1
                                    _base_ctx += (
                                        f"\n- LAYER 2 CUSTOMIZATION MODE activated.\n"
                                    )
                                    if _layer1_context:
                                        _base_ctx += (
                                            f"\n  ── WHAT LAYER 1 ALREADY ESTABLISHED (DO NOT RE-ASK ANY OF THIS) ──\n"
                                            f"{_layer1_context}"
                                            f"  ── END LAYER 1 CONTEXT ──\n\n"
                                            f"  The above was ALREADY asked and answered during Layer 1. The app was ALREADY built with these parameters.\n"
                                            f"  You MUST NOT ask about any of the above topics again. They are settled.\n\n"
                                        )
                                    _base_ctx += (
                                        f"  CRITICAL: Layer 2 is an EDIT LAYER on top of Layer 1. You are NOT rebuilding the app.\n"
                                        f"  Layer 1 already created a functional app. Your job is to ENHANCE, FIX, and EXTEND it.\n"
                                        f"  NEVER propose replacing or rebuilding what Layer 1 created. Only improve it.\n\n"
                                        f"  STEP 1 (SILENT): Use app_{_slug_safe}__read_file to read the app's key files — "
                                        f"App.tsx, main screen components, database/seed data, config/constants. "
                                        f"Be EFFICIENT: read 3-5 key files, not every single file. "
                                        f"Do NOT tell the user you are reading files. Do NOT expose paths or technical details.\n"
                                        f"  As you read, identify ONLY things Layer 1 did poorly or left incomplete:\n"
                                        f"  - Placeholder/demo data that should be real content\n"
                                        f"  - Shallow features that need deeper implementation\n"
                                        f"  - Generic defaults that should be personalized\n"
                                        f"  - Missing functionality that would make the app truly useful\n"
                                        f"  - Hardcoded content that should be dynamic\n\n"
                                        f"  STEP 2: Ask 10+ questions that reference SPECIFIC things you found in the code.\n"
                                        f"  Each question MUST cite something concrete from the code (a number, a file, a feature).\n"
                                        f"  Example: 'I found 500 vocabulary words but they are all general English — should I focus them on academic passages for your field?'\n"
                                        f"  Example: 'The study plan is a fixed 90-day schedule — want me to make it adaptive based on your quiz performance?'\n"
                                        f"  Example: 'The reading section has 10 passages but they are placeholder text — should I generate real IELTS-style passages?'\n"
                                        f"  FORBIDDEN TOPICS (Layer 1 already handled these): target score, test date, study hours, color theme, "
                                        f"app name, which test type (academic/general), basic preferences, weekly availability.\n"
                                        f"  Every question MUST have [[option]] buttons on the NEXT LINE — buttons must be inline with their question, "
                                        f"NOT collected at the end.\n\n"
                                        f"  STEP 3 (AFTER user answers): IMMEDIATELY begin editing the app using write_file/query_db.\n"
                                        f"  Do NOT just acknowledge the answers or offer action buttons — you MUST apply actual code changes.\n"
                                        f"  Do NOT use memory_store to save preferences. Do NOT say 'let me store your preferences'.\n"
                                        f"  INSTEAD: Use write_file to rewrite app files with the user's choices applied.\n"
                                        f"  Replace placeholder data with real content, upgrade features, add algorithms.\n"
                                        f"  Be EFFICIENT — use write_file to write complete files, batch related changes together.\n"
                                        f"  Show brief progress after each edit.\n"
                                        f"  IMPORTANT: You have a limited number of tool calls. Be efficient — don't waste iterations "
                                        f"on unnecessary reads. Combine related edits into single write_file calls when possible.\n\n"
                                        f"  STEP 4 (COMPLETION): Give a BRIEF human-friendly summary of what you customized.\n"
                                        f"  NEVER expose internal operations (memory storage, file reading, database queries) to the user.\n"
                                        f"  NEVER say 'let me store' or 'let me save to memory' — the user does not care about internals.\n"
                                    )
                                text = f"{_base_ctx}]\n\n{text}"
                    except Exception as e:
                        logger.warning(f"[WS] Failed to load app context: {e}")

                # Terminal activity: show user message
                _tprint(f"\n{_CYAN_BOLD} user {_RESET} {text}")

                # Stream callbacks
                async def on_text_chunk(chunk: str):
                    try:
                        await websocket.send_json({"type": "text_chunk", "text": chunk})
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
                    import tempfile, base64 as _b64
                    for _mi in _media_items[:5]:  # Max 5 attachments
                        try:
                            _mtype = _mi.get("type", "image/png")
                            _mdata = _mi.get("data", "")
                            if not _mdata:
                                continue
                            _ext = {
                                "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
                                "image/webp": ".webp", "application/pdf": ".pdf",
                            }.get(_mtype, ".bin")
                            _tf = tempfile.NamedTemporaryFile(delete=False, suffix=_ext)
                            _tf.write(_b64.b64decode(_mdata))
                            _tf.close()
                            _media_paths.append(_tf.name)
                        except Exception as _me:
                            logger.warning("[WS] Failed to process media attachment: %s", _me)

                # ── Fast-path: detect play/music requests and fire media_play immediately ──
                # Returns modified text with system hint if media was found, so agent skips play_media
                _fast_text = await _fast_media_check(text, user_id, broadcast_queue)
                _agent_text = _fast_text or text

                # Run agent — use the modified text for LLM but save the original user text
                agent_task = asyncio.create_task(_agent_runner.run(
                    user_message=_agent_text,
                    display_user_message=text if _fast_text else None,
                    user_id=user_id,
                    session_id=session_id,
                    channel=channel,
                    on_text_chunk=on_text_chunk,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    model_override=model,
                    save_user_message=not is_onboarding_msg,
                    media_paths=_media_paths if _media_paths else None,
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
                                        _jdb.add(MsgModel(
                                            id=_jmid,
                                            conversation_id=response.session_id,
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
                    await _safe_send({"type": "stopped", "message": "Generation stopped"})
                except Exception as e:
                    stop_task.cancel()
                    try:
                        await stop_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    logger.exception(f"[WS] Agent error for {user_id}")
                    _tprint(f"\033[1;31m  ✗ Error: {e}{_RESET}")
                    # Show user-friendly error instead of raw exception
                    user_msg = _friendly_error(e)
                    await _safe_send({"type": "error", "message": user_msg})
        finally:
            # Clean up broadcast queue and task
            broadcast_task.cancel()
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
