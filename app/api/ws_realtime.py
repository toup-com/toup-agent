"""
OpenAI Realtime API WebSocket Proxy — ChatGPT-speed voice conversation.

Architecture:
  Browser ←WS→ This Proxy ←WS→ OpenAI Realtime API (GPT-4o native audio)

The proxy:
  1. Authenticates the user and loads their OpenAI API key
  2. Builds system instructions from Identity docs + memories
  3. Relays PCM16 audio bidirectionally
  4. Handles function calls (all agent tools) server-side
  5. Persists user/assistant messages to DB (same as regular chat)

Protocol (Browser ↔ Proxy):
  Client sends:
    { "type": "audio", "data": "<base64 PCM16>" }             — mic audio chunk
    { "type": "stop" }                                          — end session
    { "type": "config", "voice": "nova", "session_id": "..." } — session config

  Server sends:
    { "type": "audio_delta", "data": "<base64 PCM16>" }  — assistant audio chunk
    { "type": "transcript", "text": "..." }               — what user said
    { "type": "response_text", "text": "..." }            — what assistant said (partial)
    { "type": "response_done", "text": "..." }            — full assistant text
    { "type": "session_id", "session_id": "..." }         — DB session ID (for chat sync)
    { "type": "speech_started" }                           — user started speaking (barge-in)
    { "type": "state", "state": "listening|thinking|speaking" }
    { "type": "status", "stage": "authenticated|preparing|connecting_ai" }
                                                           — pre-ready progress beacons
    { "type": "ready" }                                    — session ready, start sending audio
    { "type": "error", "message": "..." }
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from app.config import settings
from app.agent.tool_definitions import get_agent_tools, get_extended_tools

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Realtime Voice"])

# ── Module refs (set from agent_main.py lifespan) ─────────────────────
_tool_executor = None
_agent_runner = None


def set_realtime_refs(tool_executor, agent_runner=None):
    """Set reference to tool executor and agent runner for handling function calls."""
    global _tool_executor, _agent_runner
    _tool_executor = tool_executor
    _agent_runner = agent_runner


# Warm-reopen context cache: user_id → (instructions, tools, monotonic_ts).
# Single-worker deployment (intentional — singleton reconcilers), so an
# in-process dict is correct. A reopen within the TTL puts the FULL personal
# context on the very first session.update — zero thin-personality window;
# _apply_full_context refreshes the entry on every session so staleness is
# bounded to one conversation's drift, not the TTL.
_instr_cache: dict = {}
_INSTR_CACHE_TTL = 300.0


def _base_voice_instructions() -> str:
    """Instant-start stub used until the personalized context lands (seconds).

    Explicitly self-aware about the loading window so the model never guesses
    at personal facts it doesn't have yet.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "You are the user's personal AI assistant in a LIVE VOICE conversation.\n"
        "- Respond naturally and conversationally; keep replies to 1-3 sentences "
        "unless the user asks for detail.\n"
        "- No markdown, lists, or formatting — spoken prose only.\n"
        "- Match the user's language.\n"
        "- Your personalized memory and tools are still loading for the first "
        "seconds of this call; if the user asks something personal before they "
        "arrive, say you're just waking up — never guess.\n"
        f"- The current date and time is {now_str}."
    )


# ── OpenAI Realtime tool definitions ──────────────────────────────────
# Tools that don't make sense in voice mode (voice already speaks, no need for TTS/voice chat)
# All other tools (including Telegram message, send_file, etc.) work via the tunnel
VOICE_INCOMPATIBLE_TOOLS = {
    "tts",         # Voice agent already speaks — no need for Telegram voice messages
    "talk_mode",   # Telegram voice chat mode — not applicable
    "spawn",       # Sub-agent spawning — Telegram-specific
}


def _build_realtime_tools():
    """Build Realtime API tool list from all agent tool definitions.

    Converts from agent format (input_schema) to OpenAI Realtime format (parameters),
    filters out Telegram-only tools, and appends the client-side navigate_to tool.
    """
    all_tools = get_agent_tools() + get_extended_tools()
    tools = []
    for t in all_tools:
        if t["name"] in VOICE_INCOMPATIBLE_TOOLS:
            continue
        tools.append({
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        })
    # Client-side navigation tool (handled in browser, not by ToolExecutor)
    tools.append({
        "type": "function",
        "name": "navigate_to",
        "description": (
            "Navigate the user's browser to a different page in the Toup platform. "
            "Use when the user asks to go somewhere, see a page, or when you want "
            "to show them something on a specific page. Available pages:\n"
            "- / — Hub (main landing page)\n"
            "- /chat — Chat (text conversation with you)\n"
            "- /brain/user — User Brain (view user's stored memories)\n"
            "- /brain/agent — Agent Brain (view agent's stored knowledge)\n"
            "- /workspace — Workspace (workflows and automations)\n"
            "- /dashboard — Dashboard (metrics, tasks, inbox, logs)\n"
            "- /agent — Agent Setup (configure agent settings)\n"
            "The voice conversation continues during navigation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The URL path to navigate to",
                    "enum": [
                        "/", "/chat", "/brain/user", "/brain/agent",
                        "/workspace", "/dashboard", "/agent",
                    ],
                },
            },
            "required": ["path"],
        },
    })
    # Reasoning tool — delegates questions to the best reasoning model
    # GPT-4o Realtime is for voice I/O only; actual reasoning goes through this tool
    tools.append({
        "type": "function",
        "name": "think",
        "description": (
            "Use a dedicated reasoning model to answer the user's question or solve their task. "
            "You MUST call this tool for ANY question, request, or task that requires knowledge, "
            "reasoning, analysis, coding, math, planning, research, explanations, or factual answers. "
            "Only skip this for simple greetings (hi, hello, bye), yes/no acknowledgments, and casual "
            "small talk that needs no real thought. Pass the user's full question as-is. "
            "Relay the result naturally in your own words."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The user's full question or task",
                },
            },
            "required": ["task"],
        },
    })
    return tools


REALTIME_TOOLS = _build_realtime_tools()


# ── Build system instructions from Identity + Memory ──────────────────
async def build_realtime_instructions(user_id: str, onboarding: bool = False) -> str:
    """Build system instructions for the Realtime API session.

    Reads ALL personal data (identities, memories, history) from the user's VPS
    via httpx API calls. The platform DB is NEVER used for personal data.
    """
    sections = []

    vps = await _get_vps_info(user_id)
    if vps:
        agent_url, agent_api_key = vps

        # Prefetch the four independent VPS reads concurrently. A cold or
        # stalled agent container used to burn their 15s timeouts one after
        # another (identity → agent brain → user brain → day-chats) while the
        # client heard nothing pre-`ready`; concurrent, the wall time is the
        # slowest single read. Processing below stays sequential so section
        # order is unchanged. (_vps_api never raises — it returns None — but
        # return_exceptions guards refactors.)
        _prefetched = await asyncio.gather(
            _vps_api(agent_url, agent_api_key, "GET", "/api/identity",
                     params={"active_only": "true"}),
            _vps_api(agent_url, agent_api_key, "GET", "/api/memories",
                     params={"brain_type": "agent", "limit": 10000}),
            _vps_api(agent_url, agent_api_key, "GET", "/api/memories",
                     params={"brain_type": "user", "limit": 10000}),
            _vps_api(agent_url, agent_api_key, "GET", "/api/day-chats",
                     params={"limit": 1}),
            return_exceptions=True,
        )
        identities_data, agent_mems_data, user_mems_data, dc_list_prefetch = (
            r if not isinstance(r, BaseException) else None for r in _prefetched
        )

        # 1. Load identity documents from VPS
        try:
            if identities_data:
                id_list = identities_data.get("identities", identities_data) if isinstance(identities_data, dict) else identities_data
                if isinstance(id_list, list):
                    # Sort by priority descending (highest first)
                    id_list.sort(key=lambda x: x.get("priority", 0), reverse=True)
                    for identity in id_list:
                        itype = identity.get("identity_type", "")
                        content = identity.get("content", "")
                        if itype == "soul":
                            sections.append(f"# Core Identity\n{content}")
                        elif itype == "agent_instructions":
                            sections.append(f"# Behavioral Guidelines\n{content}")
                        elif itype == "user_profile":
                            sections.append(f"# About the User\n{content}")
                        elif itype == "tools":
                            sections.append(f"# Tool Guidelines\n{content}")
        except Exception as e:
            logger.warning("[REALTIME] Failed to load VPS identities: %s", e)

        # 2. Load ALL agent brain memories from VPS
        try:
            if agent_mems_data:
                mems = agent_mems_data.get("memories", agent_mems_data) if isinstance(agent_mems_data, dict) else agent_mems_data
                if isinstance(mems, list) and mems:
                    lines = ["# Agent Brain (Permanent Knowledge)"]
                    for m in mems:
                        cat = m.get("category", "")
                        content = m.get("content", "")
                        lines.append(f"- [{cat}] {content}")
                    sections.append("\n".join(lines))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load VPS agent memories: %s", e)

        # 2b. Load ALL user brain memories from VPS
        try:
            if user_mems_data:
                mems = user_mems_data.get("memories", user_mems_data) if isinstance(user_mems_data, dict) else user_mems_data
                if isinstance(mems, list) and mems:
                    lines = ["# User Brain (What You Know About the User)"]
                    for m in mems:
                        cat = m.get("category", "")
                        content = m.get("content", "")
                        lines.append(f"- [{cat}] {content}")
                    sections.append("\n".join(lines))
                    logger.info("[REALTIME] Loaded %d user brain memories from VPS", len(mems))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load VPS user memories: %s", e)

        # 2c. Load today's chat history via Day-as-Chat (cross-channel context)
        # Uses /api/day-chats endpoint which loads ALL messages from today
        # across web, app, telegram, and voice channels.
        # Strategy: load all messages, keep user messages short (facts/requests),
        # keep recent assistant messages full for continuity.
        try:
            # Use the day-chats list endpoint to find today's actual date key
            # (handles timezone correctly — the agent resolves local_date)
            day_msgs = None
            try:
                dc_list = dc_list_prefetch
                if dc_list and isinstance(dc_list, list) and dc_list:
                    today_date = dc_list[0].get("local_date")
                    if today_date:
                        day_msgs = await _vps_api(
                            agent_url, agent_api_key, "GET",
                            f"/api/day-chats/{today_date}/messages",
                            params={"limit": 500},
                        )
            except Exception:
                pass

            # Fallback: try UTC date directly
            if not day_msgs:
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                day_msgs = await _vps_api(
                    agent_url, agent_api_key, "GET",
                    f"/api/day-chats/{today_str}/messages",
                    params={"limit": 500},
                )

            if day_msgs and isinstance(day_msgs, list):
                day_msgs.sort(key=lambda m: m.get("created_at", ""))
                total = len(day_msgs)

                # Load ALL messages — no truncation, no windowing.
                # The agent must remember everything from the entire day.
                lines = [f"# Today's Full Conversation History ({total} messages across all channels)"]
                for m in day_msgs:
                    role = m.get("role", "")
                    content = (m.get("content", "") or "").strip()
                    channel = m.get("channel", "")
                    if role in ("user", "assistant") and content:
                        speaker = "User" if role == "user" else "You"
                        ch = f" [{channel}]" if channel else ""
                        lines.append(f"{speaker}{ch}: {content}")

                if len(lines) > 1:
                    sections.append("\n".join(lines))
                    logger.info("[REALTIME] Loaded %d day-chat messages from VPS (Day-as-Chat, full)", total)
            else:
                logger.info("[REALTIME] No day-chat messages for today, falling back to sessions")
                # Fallback to session-based loading if day-chat endpoint not available
                sessions_data = await _vps_api(
                    agent_url, agent_api_key, "GET", "/api/sessions",
                    params={"limit": 5, "active_only": "false"},
                )
                if sessions_data:
                    sess_list = sessions_data.get("sessions", []) if isinstance(sessions_data, dict) else []
                    today_messages = []
                    for sess in sess_list[:5]:
                        sess_id = sess.get("id", "")
                        sess_updated = sess.get("updated_at", "")
                        if not sess_updated or today_str not in sess_updated:
                            continue
                        msgs_data = await _vps_api(
                            agent_url, agent_api_key, "GET",
                            f"/api/sessions/{sess_id}/messages",
                            params={"limit": 20},
                        )
                        if msgs_data and isinstance(msgs_data, list):
                            today_messages.extend(msgs_data)
                    if today_messages:
                        today_messages.sort(key=lambda m: m.get("created_at", ""))
                        recent = today_messages[-20:]
                        lines = ["# Today's Conversation History (most recent)"]
                        for m in recent:
                            role = m.get("role", "")
                            content = m.get("content", "")
                            if role in ("user", "assistant") and content:
                                speaker = "User" if role == "user" else "You"
                                truncated = content[:300] + "..." if len(content) > 300 else content
                                lines.append(f"{speaker}: {truncated}")
                        if len(lines) > 1:
                            sections.append("\n".join(lines))
                            logger.info("[REALTIME] Loaded %d today's messages from VPS (session fallback)", len(recent))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load VPS chat history: %s", e)

    if not sections:
        sections.append(
            "# Core Identity\n"
            "You are an intelligent AI assistant with persistent memory. "
            "You remember past conversations and learn about the user over time. "
            "If you don't know your name, ask the user what they'd like to call you."
        )

    # 3. Voice-specific instructions
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        "# Voice Conversation Mode\n"
        "You are in a LIVE VOICE conversation. Follow these rules:\n"
        "- Respond naturally and conversationally, as if speaking face-to-face.\n"
        "- Keep responses concise — aim for 1-3 sentences unless the user asks for detail.\n"
        "- Do NOT use markdown, code blocks, bullet points, or any text formatting.\n"
        "- Do NOT say 'here is a list' or read structured data verbatim.\n"
        "- Use natural speech patterns: contractions, casual phrasing.\n"
        "- Match the user's language — if they speak Farsi, respond in Farsi.\n"
        "- When you need to recall past information, use the memory_search tool.\n"
        "- When the user shares something worth remembering, use the memory_store tool.\n"
        "- You can navigate the user to different pages using the navigate_to tool. "
        "Offer to show them relevant pages when helpful.\n"
        "- You have FULL ACCESS to the user's computer terminal through a connected agent. "
        "You can run shell commands (exec), read files (read_file), write files (write_file), "
        "edit files (edit_file), search files (grep, find, ls), browse the web (web_search, browser), "
        "and more. Use these tools whenever the user asks you to do something on their computer.\n"
        "- When executing terminal commands, briefly tell the user what you're doing.\n"
        "- IMPORTANT: You have a 'think' tool. You MUST call it for ANY question, task, or request "
        "that requires knowledge, reasoning, analysis, coding, math, planning, research, explanations, "
        "factual answers, or problem-solving. The 'think' tool connects you to a powerful reasoning model. "
        "Only handle simple greetings (hi, hello, bye), yes/no acknowledgments, and casual small talk directly. "
        "For EVERYTHING ELSE, call think(task=<user's full question>). "
        "When you get the result, relay the answer naturally in your own words. "
        "NEVER mention the think tool, model switching, or reasoning models to the user.\n"
        "- The user may share their screen with you. When they do, you'll receive periodic "
        "[Screen context: ...] messages describing what's on their screen. Use this visual context "
        "to help them. Don't describe the screen unprompted every time — wait for the user to ask or reference it.\n"
        f"- The current date and time is {now_str}."
    )

    # 4. Onboarding mode — agent's first conversation to learn about the user
    if onboarding:
        sections.append(
            "# ONBOARDING MODE (ACTIVE — THIS IS YOUR FIRST CONVERSATION)\n"
            "You are meeting the user for the very first time. They just deployed you and "
            "you are coming alive! You are centered on their screen, looking at them.\n"
            "IMPORTANT: You do NOT have a name yet. The user will choose your name. "
            "Do NOT introduce yourself with any name — not 'Toup', not 'Agent', nothing.\n\n"

            "## CONVERSATION FLOW (FOLLOW THIS ORDER STRICTLY)\n\n"

            "### Phase 1: Names\n"
            "Your FIRST question MUST be to ask what the user wants to call you and what their name is.\n"
            "Wait for their answer. Then:\n"
            "- Store user's name: memory_store(brain_type='user', category='identity', "
            "content='User name: <name>')\n"
            "- Store your name: memory_store(brain_type='agent', category='agent_soul', "
            "content='My name is <name>')\n\n"

            "### Phase 2: Color\n"
            "After names are set, say something like: \"Now, what color would you like for me? "
            "Pick one from the options on your screen.\"\n"
            "Then IMMEDIATELY call: set_onboarding_phase(phase='color')\n"
            "This will show clickable color circles on the user's screen.\n"
            "WAIT for the user to pick. You will receive a message like "
            "'[COLOR_SELECTED: #hex]'. Acknowledge the color warmly.\n\n"

            "### Phase 3: Deep Profiling\n"
            "Continue naturally, ONE question at a time:\n"
            "- What they primarily need you for — goals, work domain. "
            "Store: brain_type='user', category='goals'\n"
            "- Their preferred language. "
            "Store: brain_type='user', category='preferences'\n"
            "- How they want you to communicate — formal/casual, concise/detailed, "
            "personality preferences. Store: brain_type='agent', category='agent_soul'\n"
            "- Any behavioral rules they want (things to always/never do). "
            "Store: brain_type='agent', category='agent_soul'\n"
            "- Anything else — hobbies, schedule, work style. "
            "Store: brain_type='user', category appropriate\n\n"

            "### Phase 4: Wrap Up\n"
            "After gathering core info (minimum: both names, color, goals, language, "
            "personality preference), summarize what you learned. Then call "
            "finalize_onboarding() to save the complete profiles and finish.\n\n"

            "RULES:\n"
            "- Be warm, enthusiastic, conversational. You're meeting your human!\n"
            "- Ask ONE question at a time. Never dump a list.\n"
            "- Use memory_store for EACH piece of info as you learn it.\n"
            "- Match the user's language automatically.\n"
            "- Do NOT call finalize_onboarding until you have gathered enough info."
        )

    return "\n\n".join(sections)


# ── Authentication (reuses ws_chat.py pattern) ────────────────────────
async def _authenticate_ws(token: str) -> Optional[str]:
    """Validate JWT token and return user_id, or None."""
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
        logger.warning("[REALTIME] Auth failed: %s", e)
        return None


async def _get_user_openai_key(user_id: str) -> Optional[str]:
    """Retrieve the user's stored OpenAI API key."""
    from app.db.database import async_session_maker
    from app.db import AgentConfig

    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
        )
        return result.scalar_one_or_none()


# ── VPS helpers — all personal data lives on user's VPS, never platform ──

async def _get_vps_info(user_id: str) -> Optional[tuple]:
    """Return (agent_url, agent_api_key) for the user's VPS agent."""
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(AgentConfig.user_id == user_id, AgentConfig.deploy_status == "active")
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs table may not exist on agent DBs
    return None


async def _vps_api(
    agent_url: str, agent_api_key: str, method: str, path: str,
    params: dict = None, json_body: dict = None,
):
    """Make an authenticated API call to the user's VPS agent.

    Uses X-Agent-Key header — VPS auth.py resolves to settings.user_id.
    """
    url = f"{agent_url}{path}"
    # TKT-LAT-007 (wave 3): shared agent_http client.
    from app.services.agent_http import get_agent_http_client
    try:
        client = get_agent_http_client()
        if method == "GET":
            resp = await client.get(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, timeout=15.0,
            )
        else:
            resp = await client.post(
                url, headers={"X-Agent-Key": agent_api_key},
                json=json_body or {}, timeout=15.0,
            )
        if resp.status_code == 200:
            return resp.json()
        logger.warning("[REALTIME] VPS API %s %s → %s", method, path, resp.status_code)
    except Exception as e:
        logger.warning("[REALTIME] VPS API %s %s failed: %s", method, path, e)
    return None


async def _ensure_vps_user(user_id: str):
    """Ensure the platform user exists in VPS users table (FK constraints).

    The VPS agent's agent_main.py already creates the owner user on startup
    (for all DB backends). This is a safety-net that verifies via health check.
    If the VPS is unreachable or user creation failed, voice still works —
    sessions and messages go through VPS HTTP API which handles auth internally.
    """
    # Just verify the VPS agent is healthy — it creates the user on startup
    vps = await _get_vps_info(user_id)
    if vps:
        agent_url, agent_api_key = vps
        # TKT-LAT-007 (wave 3): shared agent_http client.
        from app.services.agent_http import get_agent_http_client
        try:
            client = get_agent_http_client()
            resp = await client.get(
                f"{agent_url}/agent/health",
                headers={"X-Agent-Key": agent_api_key},
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info("[REALTIME] VPS agent healthy for user %s", user_id[:8])
                return
        except Exception as e:
            logger.warning("[REALTIME] VPS health check failed (non-fatal): %s", e)


# ── Execute function calls via Agent Tunnel or local ToolExecutor ─────
async def _execute_tool(user_id: str, func_name: str, arguments: dict) -> str:
    """Execute a Realtime API function call.

    On platform: routes through WebSocket tunnel to the user's terminal agent.
    On agent (local): uses the local ToolExecutor directly.
    """
    # Route through tunnel (platform mode → terminal agent)
    try:
        from app.api.ws_agent_tunnel import is_agent_connected, send_tool_call
        if is_agent_connected(user_id):
            logger.info("[REALTIME] Routing tool %s through tunnel for %s", func_name, user_id[:8])
            return await send_tool_call(user_id, func_name, arguments)
    except ImportError:
        pass

    # Local ToolExecutor (only available when running as agent_main.py, not platform)
    if _tool_executor:
        _tool_executor._current_user_id = user_id
        try:
            result = await _tool_executor.execute(func_name, arguments)
            return result
        except Exception as e:
            logger.exception("[REALTIME] Tool execution failed: %s", func_name)
            return f"ERROR: {e}"

    return "ERROR: Your terminal agent is not connected. Open your terminal and run the start command from Agent Settings to connect."


# ── DB persistence helpers ────────────────────────────────────────────
async def _get_or_create_voice_session(user_id: str, session_id: Optional[str]) -> str:
    """Get existing session or create a new one on VPS. Returns session_id.

    Uses VPS HTTP API (works for all DB backends: local postgres, Supabase, etc.)
    All conversation data lives on the user's VPS, never the platform DB.
    """
    import uuid as _uuid

    vps = await _get_vps_info(user_id)
    if vps:
        agent_url, agent_api_key = vps

        # Check if existing session is reusable (from today)
        if session_id:
            data = await _vps_api(
                agent_url, agent_api_key, "GET", f"/api/sessions/{session_id}",
                params={"include_messages": "false"},
            )
            if data and data.get("id"):
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                started = data.get("started_at", "") or data.get("updated_at", "")
                if today_str in str(started):
                    logger.info("[REALTIME] Reusing existing VPS session %s", session_id[:8])
                    return session_id

        # Create new session via VPS API
        new_data = await _vps_api(agent_url, agent_api_key, "POST", "/api/sessions", json_body={
            "title": "Voice Session",
            "channel": "voice",
        })
        if new_data and new_data.get("id"):
            logger.info("[REALTIME] Created VPS voice session via API %s", new_data["id"][:8])
            return new_data["id"]

    # Fallback: generate UUID locally (session still works for audio relay, just not persisted)
    fallback_id = session_id or str(_uuid.uuid4())
    logger.warning("[REALTIME] VPS unreachable, using local session ID %s (not persisted)", fallback_id[:8])
    return fallback_id


async def _save_voice_messages(
    user_id: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
    model: str = "gpt-4o-realtime",
) -> None:
    """Persist a user/assistant message pair to VPS via HTTP API.

    Uses POST /api/sessions/{id}/messages on the VPS agent.
    Works for all DB backends (local postgres, Supabase, etc.)
    """
    if not user_text and not assistant_text:
        return

    logger.info(
        "[REALTIME] Saving to VPS session %s: user=%d chars, assistant=%d chars",
        session_id[:8], len(user_text), len(assistant_text),
    )

    vps = await _get_vps_info(user_id)
    if not vps:
        logger.warning("[REALTIME] VPS info not available, cannot save voice messages")
        return

    agent_url, agent_api_key = vps
    count = 0

    if user_text:
        result = await _vps_api(
            agent_url, agent_api_key, "POST",
            f"/api/sessions/{session_id}/messages",
            json_body={"role": "user", "content": user_text},
        )
        if result:
            count += 1

    if assistant_text:
        result = await _vps_api(
            agent_url, agent_api_key, "POST",
            f"/api/sessions/{session_id}/messages",
            json_body={"role": "assistant", "content": assistant_text, "model_used": model},
        )
        if result:
            count += 1

    logger.info("[REALTIME] Saved %d message(s) to VPS session %s via API", count, session_id[:8])


# ── Background memory extraction (mirrors agent_runner._extract_memories) ──
async def _extract_voice_memories(user_id: str, user_text: str, assistant_text: str) -> None:
    """Extract and store memories from a voice conversation turn on VPS.

    Uses LLM extraction on platform, then pushes memories to VPS via tunnel
    memory_store tool. No personal data is written to the platform DB.
    """
    try:
        from app.services.memory_extractor import get_memory_extractor
        from app.api.ws_agent_tunnel import is_agent_connected, send_tool_call

        if not is_agent_connected(user_id):
            logger.info("[REALTIME] VPS not connected, skipping voice memory extraction")
            return

        # Fetch the user's own OpenAI API key for LLM extraction (operational data, not personal)
        user_api_key = None
        from app.db.database import async_session_maker
        from app.db import AgentConfig
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
            )
            user_api_key = result.scalar_one_or_none()

        extractor = get_memory_extractor()
        extracted = await extractor.extract_memories_with_llm(
            user_message=user_text,
            assistant_response=assistant_text,
            brain_type="user",
            max_memories=15,
            api_key=user_api_key,
        )

        if not extracted:
            return

        # Push each memory to VPS via tunnel memory_store (handles embedding + dedup on VPS)
        pushed = 0
        for mem in extracted:
            cat = mem.category.value if hasattr(mem.category, "value") else mem.category
            try:
                result = await send_tool_call(user_id, "memory_store", {
                    "content": mem.content,
                    "category": cat,
                    "brain_type": "user",
                    "importance": mem.importance,
                })
                if not result.startswith("ERROR"):
                    pushed += 1
                    logger.info("[REALTIME] Memory stored on VPS: %s", mem.content[:50])
                else:
                    logger.warning("[REALTIME] VPS memory_store failed: %s", result[:200])
            except Exception as e:
                logger.warning("[REALTIME] VPS memory_store exception: %s", e)

        logger.info(
            "[REALTIME] Voice memory extraction done: %d/%d pushed to VPS for %s",
            pushed, len(extracted), user_id[:8],
        )

    except Exception as e:
        logger.warning("[REALTIME] Voice memory extraction failed: %s", e)


# ── Deep Think — Claude Opus reasoning for complex voice tasks ────────
async def _think(user_id: str, task: str, session_id: Optional[str]) -> tuple:
    """
    Route reasoning to the best model using the model router.

    GPT-4o Realtime is voice I/O only. Actual reasoning goes through:
      - medium complexity → GPT-5.2 (settings.agent_model)
      - heavy complexity  → Claude Opus 4.6

    Returns (result_text, model_used).
    """
    from app.services.model_router import classify_request

    # Simple key-based routing — no classifier, just pick the best model
    decision = classify_request(task)
    model_override = decision.model

    logger.info("[REALTIME] think: model=%s", model_override)

    # Option A: Use agent_runner (preferred — full tool access + memory)
    if _agent_runner:
        try:
            # Voice has no client-side tz in the WebRTC payload; agent_runner
            # will fall back to User.timezone from DB (then UTC with warn log
            # if that's NULL). Channel is explicit so the unknown-channel
            # warning path isn't hit for legitimate voice traffic.
            response = await _agent_runner.run(
                user_message=task,
                user_id=user_id,
                session_id=session_id,
                channel="voice",
                model_override=model_override,
            )
            logger.info(
                "[REALTIME] think via agent_runner: %d chars, model=%s, %dms",
                len(response.text), response.model, response.processing_time_ms,
            )
            return response.text, response.model or model_override
        except Exception as e:
            logger.warning("[REALTIME] think via agent_runner failed: %s", e)

    # Option B: Direct API call (fallback — no tools but always works)
    try:
        # If it's a Claude model, use Anthropic API
        if model_override.startswith("claude-"):
            from app.services.anthropic_service import AnthropicService
            svc = AnthropicService()

            context_messages = []
            if session_id:
                try:
                    vps = await _get_vps_info(user_id)
                    if vps:
                        agent_url, agent_api_key = vps
                        msgs_data = await _vps_api(
                            agent_url, agent_api_key, "GET",
                            f"/api/sessions/{session_id}/messages",
                            params={"limit": 10},
                        )
                        if msgs_data and isinstance(msgs_data, list):
                            for m in msgs_data[-10:]:
                                role = m.get("role", "")
                                content = m.get("content", "")
                                if role in ("user", "assistant") and content:
                                    context_messages.append({"role": role, "content": content})
                except Exception:
                    pass

            context_messages.append({"role": "user", "content": task})

            chunks = []
            async for event in svc.create_message_stream(
                messages=context_messages,
                system="You are a powerful reasoning assistant. Answer thoroughly and accurately. Be concise enough for a voice response.",
                tools=[],
                model=model_override,
            ):
                if event.type == "text":
                    chunks.append(event.text)

            result = "".join(chunks)
            logger.info("[REALTIME] think via Anthropic direct: %d chars", len(result))
            return result or "I couldn't generate a response.", model_override

        else:
            # OpenAI model — direct chat completions call
            import httpx
            context_messages = [{"role": "user", "content": task}]
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    json={
                        "model": model_override,
                        "messages": [
                            {"role": "system", "content": "You are a reasoning assistant. Answer accurately and concisely for a voice response."},
                            {"role": "user", "content": task},
                        ],
                        "max_tokens": 2048,
                    },
                )
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]

            logger.info("[REALTIME] think via OpenAI direct: %d chars, model=%s", len(result), model_override)
            return result or "I couldn't generate a response.", model_override

    except Exception as e:
        logger.warning("[REALTIME] think fallback failed: %s", e)
        return "I'll do my best to answer directly.", "gpt-4o-realtime"


# ── Finalize onboarding — compile profiles + write .md files ─────────
async def _finalize_onboarding(user_id: str) -> str:
    """Compile agent and user memories into Identity records and .md files.

    ALL data is read from and written to the user's VPS. The platform DB
    is NEVER used for personal data (memories, identities, conversations).

    1. Read agent + user memories from VPS (they were stored via tunnel memory_store)
    2. Compile saul.md (agent soul) + identity.md (user profile)
    3. Write .md files to VPS via tunnel write_file
    4. Upsert Identity records in VPS DB via tunnel exec
    """
    from app.api.ws_agent_tunnel import is_agent_connected, send_tool_call

    try:
        if not is_agent_connected(user_id):
            return "Onboarding finalized but VPS agent is not connected. Memories were stored during conversation."

        # Ensure VPS user exists (for FK constraints)
        await _ensure_vps_user(user_id)

        # ── 1. Read memories from VPS ──
        vps = await _get_vps_info(user_id)
        agent_memories = []
        user_memories = []

        if vps:
            agent_url, agent_api_key = vps

            agent_mems_data = await _vps_api(
                agent_url, agent_api_key, "GET", "/api/memories",
                params={"brain_type": "agent", "limit": 10000},
            )
            if agent_mems_data:
                agent_memories = agent_mems_data.get("memories", agent_mems_data) if isinstance(agent_mems_data, dict) else agent_mems_data
                if not isinstance(agent_memories, list):
                    agent_memories = []

            user_mems_data = await _vps_api(
                agent_url, agent_api_key, "GET", "/api/memories",
                params={"brain_type": "user", "limit": 10000},
            )
            if user_mems_data:
                user_memories = user_mems_data.get("memories", user_mems_data) if isinstance(user_mems_data, dict) else user_mems_data
                if not isinstance(user_memories, list):
                    user_memories = []

        logger.info(
            "[REALTIME] Onboarding finalize: %d agent, %d user memories from VPS",
            len(agent_memories), len(user_memories),
        )

        # ── 2. Compile saul.md (agent soul) ──
        soul_sections = ["# Agent Soul\n"]
        agent_name = None
        for m in agent_memories:
            cat = m.get("category", "")
            content = m.get("content", "")
            if "my name is" in content.lower():
                agent_name = content.split("is")[-1].strip().rstrip(".")
            soul_sections.append(f"- [{cat}] {content}")
        saul_content = "\n".join(soul_sections)

        # ── 3. Compile identity.md (user profile) ──
        identity_sections = ["# User Profile\n"]
        for m in user_memories:
            cat = m.get("category", "")
            content = m.get("content", "")
            identity_sections.append(f"- [{cat}] {content}")
        identity_content = "\n".join(identity_sections)

        # ── 4. Write .md files to VPS ──
        # DEPRECATED: saul.md / identity.md no longer used — Identity table is source of truth.
        # Soul page now writes directly to Identity table via PUT /api/soul + soul sync.
        # await send_tool_call(user_id, "write_file", {"path": "/opt/toup-agent/saul.md", "content": saul_content})
        # await send_tool_call(user_id, "write_file", {"path": "/opt/toup-agent/identity.md", "content": identity_content})
        logger.info("[REALTIME] Skipped saul.md/identity.md write (deprecated) for user %s", user_id[:8])

        # ── 5. Upsert Identity records on VPS via HTTP API ──
        if vps:
            for id_type, id_name, id_content, id_priority in [
                ("soul", agent_name or "Agent Soul", saul_content, 100),
                ("user_profile", "User Profile", identity_content, 90),
            ]:
                try:
                    # Delete existing identity of this type first
                    existing = await _vps_api(
                        agent_url, agent_api_key, "GET", "/api/identity",
                        params={"identity_type": id_type},
                    )
                    if existing:
                        id_list = existing.get("identities", existing) if isinstance(existing, dict) else existing
                        if isinstance(id_list, list):
                            for old_id in id_list:
                                old_id_val = old_id.get("id", "")
                                if old_id_val:
                                    # TKT-LAT-007 (wave 3): shared agent_http client.
                                    from app.services.agent_http import get_agent_http_client
                                    _id_client = get_agent_http_client()
                                    await _id_client.delete(
                                        f"{agent_url}/api/identity/{old_id_val}",
                                        headers={"X-Agent-Key": agent_api_key},
                                        timeout=10.0,
                                    )

                    # Create new identity
                    await _vps_api(agent_url, agent_api_key, "POST", "/api/identity", json_body={
                        "identity_type": id_type,
                        "name": id_name,
                        "content": id_content,
                        "priority": id_priority,
                        "is_active": True,
                    })
                except Exception as e:
                    logger.warning("[REALTIME] Failed to push Identity %s to VPS: %s", id_type, e)

            logger.info("[REALTIME] Pushed Identity records to VPS for user %s", user_id[:8])

        return (
            f"Onboarding finalized! Agent soul profile ({len(agent_memories)} memories) "
            f"and user profile ({len(user_memories)} memories) saved to VPS. "
            "The user will be redirected to the Hub."
        )

    except Exception as e:
        logger.exception("[REALTIME] _finalize_onboarding failed: %s", e)
        return f"Onboarding finalized with some issues: {e}"


# ── WebSocket endpoint ────────────────────────────────────────────────
# GA Realtime API (the beta interface + `OpenAI-Beta: realtime=v1` header was
# shut down 2026-05-12 → "beta_api_shape_disabled"). GA uses the `gpt-realtime`
# model, no beta header, and a restructured session.update (audio nested under
# session.audio.input/output; renamed response.output_audio* events).
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"


@router.websocket("/ws/realtime")
async def realtime_voice_ws(
    websocket: WebSocket,
    token: str = Query(None),
    session_id: Optional[str] = Query(None),
    onboarding: bool = Query(False),
):
    """WebSocket proxy to OpenAI Realtime API for ChatGPT-speed voice conversation."""

    # ST-2: accept + subprotocol JWT extraction.
    from app.api._ws_auth_helpers import (
        accept_with_subprotocol_auth,
        log_deprecated_query_token,
        safe_send_close_ws,
    )
    subprotocol_token = await accept_with_subprotocol_auth(websocket)

    # ── 1. Authenticate ───────────────────────────────────────
    user_id = None
    if subprotocol_token:
        user_id = await _authenticate_ws(subprotocol_token)

    if not user_id and token:
        log_deprecated_query_token("/api/ws/realtime")
        user_id = await _authenticate_ws(token)

    client_disconnected = False
    if not user_id:
        # Try auth message
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") == "auth" and msg.get("token"):
                user_id = await _authenticate_ws(msg["token"])
        except (asyncio.TimeoutError, json.JSONDecodeError):
            pass
        except WebSocketDisconnect:
            client_disconnected = True
        except Exception:
            pass

    if not user_id:
        if client_disconnected:
            return
        await safe_send_close_ws(
            websocket, code=4401, message="Authentication failed",
        )
        return

    logger.info("[REALTIME] Session starting for user %s", user_id[:8])

    async def _status(stage: str) -> None:
        # Pre-ready progress beacons. The clients' connect watchdogs treat any
        # frame as "pipeline engaged"; total silence reads as a dead route and
        # fails their UI at ~20s. Best-effort — if the client is gone, the
        # relay loop below notices immediately anyway.
        try:
            await websocket.send_json({"type": "status", "stage": stage})
        except Exception:
            pass

    await _status("authenticated")

    # ── 2. Load OpenAI API key ────────────────────────────────
    # Fall back to the platform key (parity with /voice/transcribe + /voice/tts)
    # so managed/bundle users with no personal OpenAI key can still use live
    # voice. Realtime audio is billed to the platform key in that case.
    # Bounded: a stuck platform-DB pool acquisition here used to propagate as
    # a bare 1011 close with zero frames sent — the silent-forever class.
    try:
        openai_key = await asyncio.wait_for(_get_user_openai_key(user_id), timeout=8.0)
    except Exception:
        logger.exception("[REALTIME] OpenAI key lookup failed")
        openai_key = None
    openai_key = openai_key or settings.openai_api_key
    if not openai_key:
        await websocket.send_json({
            "type": "error",
            "message": "OpenAI API key not configured. Please set up your API key in Settings.",
        })
        await websocket.close(code=4400)
        return

    # ── 3+4+5 fan-out: `ready` waits ONLY on the OpenAI connect ──────────
    # The single hard prerequisite for the user to start talking is an OpenAI
    # socket with a VAD/format config written to it. Everything personal —
    # VPS session, instructions, tools — rides in behind and hot-swaps onto
    # the live session via a second session.update (responses created after
    # it use the new instructions). Warm reopens skip even that thin window
    # via _instr_cache. Previously all of this was serialized ahead of
    # `ready`: several seconds warm, up to 25s cold.
    await _status("preparing")

    async def _connect_openai():
        return await websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {openai_key}",
            },
            max_size=10 * 1024 * 1024,  # 10MB for audio chunks
        )

    async def _session_step() -> Optional[str]:
        try:
            return await _get_or_create_voice_session(user_id, session_id)
        except Exception:
            logger.exception("[REALTIME] Failed to create DB session")
            try:
                return await _get_or_create_voice_session(user_id, None)
            except Exception:
                logger.exception("[REALTIME] Failed to create fresh DB session")
                return None

    async def _instructions_step() -> Optional[str]:
        try:
            instr = await build_realtime_instructions(user_id, onboarding=onboarding)
            logger.info("[REALTIME] Built instructions (%d chars, onboarding=%s)", len(instr), onboarding)
            return instr
        except Exception:
            logger.exception("[REALTIME] Failed to build instructions")
            return None

    async def _tools_step() -> list:
        tools = REALTIME_TOOLS
        try:
            from app.db.database import async_session_maker
            from app.db.models import AgentConfig
            async with async_session_maker() as _db:
                _ac_res = await _db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == user_id)
                )
                _ac = _ac_res.scalars().first()
                if _ac and getattr(_ac, 'disabled_tools', None):
                    _user_disabled = set(json.loads(_ac.disabled_tools))
                    tools = [t for t in REALTIME_TOOLS if t["name"] not in _user_disabled]
                    logger.info("[REALTIME] Filtered %d disabled tools for user %s", len(_user_disabled), user_id[:8])
        except Exception as e:
            logger.warning("[REALTIME] Failed to load disabled tools: %s", e)
        if onboarding:
            tools = list(tools)  # copy
            tools.append({
                "type": "function",
                "name": "set_onboarding_phase",
                "description": "Set the onboarding UI phase. Call with phase='color' when asking the user to pick a color.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string", "enum": ["color", "profiling", "done"]},
                    },
                    "required": ["phase"],
                },
            })
            tools.append({
                "type": "function",
                "name": "finalize_onboarding",
                "description": "Call when onboarding conversation is complete to save all profiles and finish setup. Only call after gathering: both names, color, goals, language, and personality.",
                "parameters": {"type": "object", "properties": {}},
            })
        return tools

    async def _health_step() -> None:
        # FK safety-net health check — informational, must never block setup.
        try:
            await _ensure_vps_user(user_id)
        except Exception as e:
            logger.warning("[REALTIME] _ensure_vps_user failed (non-fatal): %s", e)

    _t0 = time.monotonic()
    _openai_t = asyncio.create_task(_connect_openai())
    _session_t = asyncio.create_task(_session_step())
    _instructions_t = asyncio.create_task(_instructions_step())
    _tools_t = asyncio.create_task(_tools_step())
    _health_t = asyncio.create_task(_health_step())
    _bg_tasks = [_session_t, _instructions_t, _tools_t, _health_t]

    def _cancel_bg() -> None:
        # Tap-and-close must not leave tasks poking a cold agent for 25s.
        for _t in _bg_tasks:
            _t.cancel()

    db_session_id: Optional[str] = None

    # ── 5. OpenAI connect — the critical path ─────────────────
    await _status("connecting_ai")
    # Realtime API voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
    voice = "coral"  # natural, expressive voice
    openai_ws = None

    try:
        openai_ws = await _openai_t
        logger.info("[REALTIME] Connected to OpenAI Realtime API (+%.0fms)", (time.monotonic() - _t0) * 1000)
    except Exception as e:
        _cancel_bg()
        logger.exception("[REALTIME] Failed to connect to OpenAI")
        err_str = str(e).lower()
        is_billing = any(kw in err_str for kw in ["quota", "billing", "rate_limit", "402", "429", "credit", "balance"])
        error_payload: dict = {
            "type": "error",
            "message": f"Failed to connect to OpenAI Realtime API: {e}",
        }
        if is_billing:
            error_payload["billing"] = True
            error_payload["billing_url"] = "https://platform.openai.com/settings/organization/billing/overview"
        await websocket.send_json(error_payload)
        await websocket.close(code=4502)
        return

    # ── 6. Configure session: cached-or-base config now, full context behind ──
    def _session_config(instr: str, tools: list) -> dict:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": instr,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "whisper-1"},
                        # Filters the input buffer before VAD — documented to
                        # reduce false speech_started triggers. far_field:
                        # the phone is a loudspeaker at arm's length, and the
                        # dominant false trigger is the agent's own speaker
                        # bleed (echo) reaching the mic.
                        "noise_reduction": {"type": "far_field"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.8,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 700,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": voice,
                    },
                },
                "tools": tools,
                "tool_choice": "auto",
            },
        }

    _cached = _instr_cache.get(user_id)
    _cache_hit = bool(
        _cached
        and (time.monotonic() - _cached[2]) < _INSTR_CACHE_TTL
        and not onboarding
    )
    if _cache_hit:
        _first_instructions, _first_tools = _cached[0], _cached[1]
    else:
        _first_instructions, _first_tools = _base_voice_instructions(), []

    try:
        await openai_ws.send(json.dumps(_session_config(_first_instructions, _first_tools)))
        logger.info("[REALTIME] Session configured (voice=%s, cache_hit=%s)", voice, _cache_hit)
    except Exception as e:
        _cancel_bg()
        logger.exception("[REALTIME] Failed to configure session")
        await websocket.send_json({"type": "error", "message": str(e)})
        await openai_ws.close()
        await websocket.close()
        return

    # The user can talk NOW. The VAD/format config is written to the OpenAI
    # socket ahead of any relayed audio (same-socket FIFO), so frames the
    # client sends immediately are processed under the right config. The
    # session_id follows on its own frame when the VPS session lands.
    try:
        await websocket.send_json({"type": "ready"})
    except Exception:
        _cancel_bg()
        try:
            await openai_ws.close()
        except Exception:
            pass
        return
    logger.info("[REALTIME] ready in %.0fms (cache_hit=%s)", (time.monotonic() - _t0) * 1000, _cache_hit)

    # Onboarding is the one flow where a first response on the base config
    # would be actively wrong (the whole flow lives in the instructions) —
    # its greet waits on this event (see audio_ready handler).
    full_context_applied = asyncio.Event()
    if _cache_hit:
        full_context_applied.set()

    async def _apply_full_context() -> None:
        # shield(): a timeout here abandons the wait, not the underlying
        # build — a late result still lands in the cache path below on the
        # next connect. _cancel_bg reaps everything at session end.
        instr: Optional[str] = None
        tools: list = list(_first_tools) if _first_tools else list(REALTIME_TOOLS)
        try:
            instr = await asyncio.wait_for(asyncio.shield(_instructions_t), timeout=40.0)
            tools = await asyncio.wait_for(asyncio.shield(_tools_t), timeout=10.0)
        except Exception:
            logger.warning("[REALTIME] Context build timed out — session continues on base config")
        try:
            final_instr = instr or _first_instructions
            await openai_ws.send(json.dumps(_session_config(final_instr, tools)))
            if instr:
                _instr_cache[user_id] = (instr, tools, time.monotonic())
            logger.info(
                "[REALTIME] Full context applied at +%.0fms (%d chars, %d tools)",
                (time.monotonic() - _t0) * 1000, len(final_instr), len(tools),
            )
        except Exception as e:
            logger.warning("[REALTIME] Full-context session.update failed: %s", e)
        finally:
            full_context_applied.set()

    _apply_t = asyncio.create_task(_apply_full_context())
    _bg_tasks.append(_apply_t)

    async def _announce_session() -> None:
        nonlocal db_session_id
        try:
            sid = await asyncio.shield(_session_t)
        except Exception:
            sid = None
        if sid and not db_session_id:
            db_session_id = sid
            logger.info("[REALTIME] DB session: %s", sid[:8])
            try:
                await websocket.send_json({"type": "session_id", "session_id": sid})
            except Exception:
                pass

    _announce_t = asyncio.create_task(_announce_session())
    _bg_tasks.append(_announce_t)

    # ── 6b. Auto-greet in onboarding mode ─────────────────────
    # Wait for client's "audio_ready" signal before greeting, so audio doesn't get dropped.
    # The client sends this after getUserMedia + AudioContext are fully set up.
    onboarding_greet_pending = onboarding

    # ── 7. Bidirectional relay ────────────────────────────────
    # Track state for transcript accumulation and persistence
    response_text_accum = ""
    last_user_text = ""  # Track last user message for memory extraction
    turn_model = "gpt-4o-realtime"  # Model used for current turn (changes if deep_think is called)

    screen_sharing_active = False
    first_frame_sent = False
    last_vision_call_time = 0.0
    vision_lock = asyncio.Lock()

    async def analyze_screen_frame(frame_data: str, is_first: bool):
        """Side-channel: call GPT-4o-mini vision to describe the screen, inject text into Realtime."""
        nonlocal last_vision_call_time
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                prompt = (
                    "Briefly describe what's on this screen in 1-2 sentences. "
                    "Focus on the main content, UI elements, and any text visible. "
                    "Be concise — this is context for a voice assistant."
                )
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": frame_data, "detail": "low"}},
                            ],
                        }],
                        "max_tokens": 200,
                    },
                )
                resp.raise_for_status()
                description = resp.json()["choices"][0]["message"]["content"]

            # Inject the screen description as a user message into the Realtime conversation
            await openai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"[Screen context: {description}]",
                    }],
                },
            }))

            # Only trigger a response on the first frame so agent acknowledges
            if is_first:
                await openai_ws.send(json.dumps({"type": "response.create"}))

            last_vision_call_time = time.monotonic()
            logger.info("[REALTIME] Screen vision analysis sent (%d chars)", len(description))

        except Exception as e:
            logger.warning("[REALTIME] Screen vision analysis failed: %s", e)

    async def client_to_openai():
        """Relay browser audio → OpenAI Realtime API."""
        nonlocal voice, db_session_id, screen_sharing_active, first_frame_sent, onboarding_greet_pending
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "audio":
                    # Relay PCM16 audio chunk
                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": msg["data"],
                    }))

                elif msg_type == "config":
                    # Update voice or other settings
                    VALID_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"}
                    if "voice" in msg:
                        requested = msg["voice"]
                        voice = requested if requested in VALID_VOICES else "alloy"
                        await openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "audio": {"output": {"voice": voice}},
                            },
                        }))
                    # Client can also pass session_id via config
                    if "session_id" in msg and msg["session_id"] and not db_session_id:
                        try:
                            db_session_id = await _get_or_create_voice_session(user_id, msg["session_id"])
                            await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                        except Exception as e:
                            logger.warning("[REALTIME] Failed to set session from config: %s", e)

                elif msg_type == "screen_share_start":
                    screen_sharing_active = True
                    first_frame_sent = False
                    logger.info("[REALTIME] Screen sharing started for user %s", user_id)

                elif msg_type == "screen_frame":
                    # Side-channel vision: analyze frame with GPT-4o-mini, inject text description
                    if not screen_sharing_active:
                        continue
                    frame_data = msg.get("data", "")
                    if not frame_data:
                        continue
                    is_first = not first_frame_sent
                    # Throttle: analyze first frame immediately, then every 5 seconds
                    elapsed = time.monotonic() - last_vision_call_time
                    if is_first or elapsed >= 5.0:
                        if not vision_lock.locked():
                            first_frame_sent = True
                            asyncio.create_task(analyze_screen_frame(frame_data, is_first))

                elif msg_type == "screen_share_stop":
                    screen_sharing_active = False
                    first_frame_sent = False
                    logger.info("[REALTIME] Screen sharing stopped for user %s", user_id)

                elif msg_type == "audio_ready":
                    # Client signals that AudioContext + mic are ready for playback
                    if onboarding_greet_pending:
                        onboarding_greet_pending = False

                        # The greet must not fire on the base config — the
                        # whole onboarding flow lives in the instructions, so
                        # wait for the full context (bounded; on timeout greet
                        # anyway rather than sit silent).
                        async def _greet_when_context_ready():
                            try:
                                await asyncio.wait_for(full_context_applied.wait(), timeout=45.0)
                            except asyncio.TimeoutError:
                                logger.warning("[REALTIME] Onboarding greet fired without full context (timeout)")
                            try:
                                await openai_ws.send(json.dumps({"type": "response.create"}))
                                logger.info("[REALTIME] Onboarding: client audio ready, triggered greeting")
                            except Exception:
                                pass

                        _bg_tasks.append(asyncio.create_task(_greet_when_context_ready()))

                elif msg_type == "inject_text":
                    # UI sent a text event (e.g., color selection) — inject into OpenAI conversation
                    inject_content = msg.get("text", "")
                    if inject_content:
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": inject_content}],
                            },
                        }))
                        await openai_ws.send(json.dumps({"type": "response.create"}))
                        logger.info("[REALTIME] Injected text: %s", inject_content[:60])

                elif msg_type == "stop":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("[REALTIME] client_to_openai error: %s", e)

    async def openai_to_client():
        """Relay OpenAI Realtime API events → browser."""
        nonlocal response_text_accum, db_session_id, last_user_text, turn_model
        try:
            async for raw_msg in openai_ws:
                event = json.loads(raw_msg)
                etype = event.get("type", "")

                # ── Audio response chunks → browser (GA: response.output_audio.delta) ──
                if etype == "response.output_audio.delta":
                    await websocket.send_json({
                        "type": "audio_delta",
                        "data": event.get("delta", ""),
                    })

                # ── Assistant text transcript (partial; GA: response.output_audio_transcript.delta) ──
                elif etype == "response.output_audio_transcript.delta":
                    delta = event.get("delta", "")
                    response_text_accum += delta
                    await websocket.send_json({
                        "type": "response_text",
                        "text": delta,
                        "partial": True,
                    })

                # ── Response complete ──
                elif etype == "response.done":
                    response = event.get("response", {})
                    # Extract final text from output items
                    full_text = response_text_accum
                    for item in response.get("output", []):
                        if item.get("type") == "message":
                            for content in item.get("content", []):
                                if content.get("type") in ("audio", "output_audio") and content.get("transcript"):
                                    full_text = content["transcript"]

                    if full_text:
                        await websocket.send_json({
                            "type": "response_done",
                            "text": full_text,
                            "model": turn_model,
                        })

                        # Persist assistant message to DB. Await the SHARED
                        # session task (still in flight on a cold agent) —
                        # creating a second session here would fork the
                        # conversation into two VPS threads.
                        if not db_session_id:
                            try:
                                db_session_id = await asyncio.wait_for(asyncio.shield(_session_t), timeout=10.0)
                                if db_session_id:
                                    logger.info("[REALTIME] Late-resolved DB session for assistant: %s", db_session_id[:8])
                                    await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                            except Exception:
                                logger.warning("[REALTIME] Session still unresolved at assistant persist — skipping this turn")
                        if db_session_id:
                            try:
                                await _save_voice_messages(user_id, db_session_id, "", full_text, model=turn_model)
                            except Exception as e:
                                logger.exception("[REALTIME] Failed to save assistant message")

                        # Auto-extract memories from this conversation turn (background)
                        if last_user_text and full_text and settings.auto_extract_memories:
                            asyncio.create_task(
                                _extract_voice_memories(user_id, last_user_text, full_text)
                            )
                            last_user_text = ""  # Reset so we don't re-extract

                    response_text_accum = ""
                    turn_model = "gpt-4o-realtime"  # Reset for next turn

                    await websocket.send_json({"type": "state", "state": "listening"})

                # ── User speech transcript ──
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript.strip():
                        user_text = transcript.strip()
                        await websocket.send_json({
                            "type": "transcript",
                            "text": user_text,
                        })
                        # Save user message to DB — same shared-session rule
                        # as the assistant persist above.
                        if not db_session_id:
                            try:
                                db_session_id = await asyncio.wait_for(asyncio.shield(_session_t), timeout=10.0)
                                if db_session_id:
                                    logger.info("[REALTIME] Late-resolved DB session: %s", db_session_id[:8])
                                    await websocket.send_json({"type": "session_id", "session_id": db_session_id})
                            except Exception:
                                logger.warning("[REALTIME] Session still unresolved at user persist — skipping this turn")
                        if db_session_id:
                            try:
                                await _save_voice_messages(user_id, db_session_id, user_text, "")
                            except Exception as e:
                                logger.exception("[REALTIME] Failed to save user transcript")
                        last_user_text = user_text

                # ── VAD: user started speaking (barge-in) ──
                elif etype == "input_audio_buffer.speech_started":
                    await websocket.send_json({"type": "speech_started"})
                    await websocket.send_json({"type": "state", "state": "listening"})

                # ── VAD: user stopped speaking → thinking ──
                elif etype == "input_audio_buffer.speech_stopped":
                    await websocket.send_json({"type": "state", "state": "thinking"})

                # ── Response started → speaking ──
                elif etype == "response.created":
                    await websocket.send_json({"type": "state", "state": "speaking"})

                # ── Function call completed → execute tool ──
                elif etype == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        func_name = item.get("name", "")
                        call_id = item.get("call_id", "")
                        try:
                            arguments = json.loads(item.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            arguments = {}

                        logger.info("[REALTIME] Function call: %s(%s)", func_name, arguments)
                        await websocket.send_json({"type": "state", "state": "tool_use"})

                        # ── Client-side tool: navigate_to ──
                        if func_name == "navigate_to":
                            path = arguments.get("path", "/")
                            _ALLOWED = {"/", "/chat", "/brain/user", "/brain/agent",
                                        "/workspace", "/dashboard", "/agent"}
                            if path not in _ALLOWED:
                                result = f"Invalid path '{path}'."
                            else:
                                await websocket.send_json({"type": "navigate", "path": path})
                                _NAMES = {"/": "Hub", "/chat": "Chat", "/brain/user": "User Brain",
                                          "/brain/agent": "Agent Brain", "/workspace": "Workspace",
                                          "/dashboard": "Dashboard", "/agent": "Agent Setup"}
                                result = f"Navigated to {_NAMES.get(path, path)}. Voice conversation continues."

                        # ── Think: delegate reasoning to best model ──
                        elif func_name == "think":
                            task = arguments.get("task", "")
                            result, turn_model_used = await _think(user_id, task, db_session_id)
                            turn_model = turn_model_used

                        # ── Onboarding: set UI phase ──
                        elif func_name == "set_onboarding_phase":
                            phase = arguments.get("phase", "")
                            await websocket.send_json({
                                "type": "onboarding_phase",
                                "phase": phase,
                            })
                            result = f"Onboarding UI phase set to '{phase}'. The user can now see the {phase} interface."

                        # ── Onboarding: finalize (compile profiles, mark complete) ──
                        elif func_name == "finalize_onboarding":
                            result = await _finalize_onboarding(user_id)
                            try:
                                from app.db.database import async_session_maker as _asm
                                from app.db.models import AgentConfig
                                async with _asm() as _db:
                                    _cfg = (await _db.execute(
                                        select(AgentConfig).where(AgentConfig.user_id == user_id)
                                    )).scalar_one_or_none()
                                    if _cfg:
                                        _cfg.onboarding_completed = True
                                        await _db.commit()
                                        logger.info("[REALTIME] Onboarding finalized for user %s", user_id[:8])
                            except Exception as oe:
                                logger.warning("[REALTIME] Failed to mark onboarding complete: %s", oe)
                            await websocket.send_json({"type": "onboarding_phase", "phase": "done"})

                            # Trigger personalized workspace generation on VPS
                            try:
                                _vps = await _get_vps_info(user_id)
                                if _vps:
                                    _agent_url, _agent_key = _vps
                                    await _vps_api(
                                        _agent_url, _agent_key, "POST",
                                        "/api/workflows/generate-from-onboarding",
                                    )
                                    logger.info("[REALTIME] Triggered workspace generation for %s", user_id[:8])
                            except Exception as _wg_err:
                                logger.warning("[REALTIME] Workspace generation trigger failed: %s", _wg_err)

                        else:
                            # ── Server-side tools ──
                            result = await _execute_tool(user_id, func_name, arguments)

                        # Check if onboarding just completed (legacy detection via memory_store)
                        if (onboarding and func_name == "memory_store"
                                and "onboarding complete" in arguments.get("content", "").lower()):
                            try:
                                from app.db.database import async_session_maker as _asm
                                from app.db.models import AgentConfig
                                async with _asm() as _db:
                                    _cfg = (await _db.execute(
                                        select(AgentConfig).where(AgentConfig.user_id == user_id)
                                    )).scalar_one_or_none()
                                    if _cfg:
                                        _cfg.onboarding_completed = True
                                        await _db.commit()
                                        logger.info("[REALTIME] Onboarding completed for user %s", user_id[:8])
                            except Exception as oe:
                                logger.warning("[REALTIME] Failed to mark onboarding complete: %s", oe)

                        # Send result back to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": result,
                            },
                        }))

                        # Tell OpenAI to continue
                        await openai_ws.send(json.dumps({
                            "type": "response.create",
                        }))

                # ── Errors from OpenAI ──
                elif etype == "error":
                    error_obj = event.get("error", {})
                    error_msg = error_obj.get("message", "Unknown OpenAI error")
                    error_code = error_obj.get("code", "")
                    logger.error("[REALTIME] OpenAI error: %s (code=%s)", error_msg, error_code)

                    # Detect billing / quota errors
                    billing_keywords = ["insufficient_quota", "billing", "exceeded", "rate_limit",
                                        "quota", "payment", "credit", "balance", "plan"]
                    is_billing = (
                        error_code in ("insufficient_quota", "billing_hard_limit_reached",
                                       "rate_limit_exceeded", "budget_exceeded")
                        or any(kw in error_msg.lower() for kw in billing_keywords)
                    )

                    if is_billing:
                        await websocket.send_json({
                            "type": "error",
                            "message": error_msg,
                            "billing": True,
                            "billing_url": "https://platform.openai.com/settings/organization/billing/overview",
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": error_msg,
                        })

                # ── Session events (log only) ──
                elif etype in ("session.created", "session.updated"):
                    logger.info("[REALTIME] %s", etype)

        except websockets.ConnectionClosed as e:
            logger.warning("[REALTIME] OpenAI WS closed: code=%s reason=%s", e.code, e.reason)
            reason = str(e.reason or e).lower()
            is_billing = any(kw in reason for kw in ["quota", "billing", "rate_limit", "credit", "balance", "exceeded"])
            error_payload: dict = {
                "type": "error",
                "message": f"OpenAI connection closed: {e.reason or e}",
            }
            if is_billing or e.code in (4002, 4003):
                error_payload["billing"] = True
                error_payload["billing_url"] = "https://platform.openai.com/settings/organization/billing/overview"
            try:
                await websocket.send_json(error_payload)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[REALTIME] openai_to_client error: %s", e)
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # ── Run both relay tasks ──────────────────────────────────
    try:
        await asyncio.gather(
            client_to_openai(),
            openai_to_client(),
            return_exceptions=True,
        )
    finally:
        logger.info("[REALTIME] Session ended for user %s", user_id[:8])
        _cancel_bg()
        try:
            await openai_ws.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
