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
    { "type": "played", "ms": 1234 }                            — V2: audio ms actually
                                                                  played before barge-in

  Server sends:
    { "type": "audio_delta", "data": "<base64 PCM16>" }  — assistant audio chunk
    { "type": "transcript", "text": "..." }               — what user said
    { "type": "response_text", "text": "..." }            — what assistant said (partial)
    { "type": "response_done", "text": "..." }            — full assistant text
    { "type": "session_id", "session_id": "..." }         — DB session ID (for chat sync)
    { "type": "speech_started" }                           — user started speaking (barge-in)
    { "type": "state", "state": "listening|thinking|speaking|tool_use" }
    { "type": "tool_call.started", "call_id","name","title","detail" }   — a tool began
    { "type": "tool_call.completed", "call_id","name","ok","result_preview" } — tool result
    { "type": "navigate", "path": "..." }                  — client-side navigate_to tool
    { "type": "onboarding_phase", "phase": "color|profiling|done" }
    { "type": "status", "stage": "authenticated|preparing|connecting_ai" }
                                                           — pre-ready progress beacons
    { "type": "ready" }                                    — session ready, start sending audio
    { "type": "session_expiring", "seconds_left": 300 }    — V2: OpenAI 60-min cap approaching
    { "type": "error", "message": "..." }
"""

import asyncio
import json
import contextvars
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import httpx
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from app.config import settings
from app.agent.tool_definitions import get_agent_tools, get_extended_tools
from app.api.voice import detect_script_language

from app.services.memory_log import describe_memory

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


# ── Per-connection V2 resolution ──────────────────────────────────────
# V2 is enabled when the global flag is on OR the connecting user is in the
# rollout allowlist. Resolved ONCE at the WS-handler entry and stored in this
# ContextVar; every module-level helper reads it via _v2_active(). When unset
# (unit tests calling these helpers directly, no live connection) it falls back
# to the global flag — so the existing tests keep pinning v1/v2 via settings.
_v2_ctx: contextvars.ContextVar = contextvars.ContextVar("realtime_v2", default=None)

# How long to wait before the single `think` retry against a transport failure.
# Sized for a container swap, not a network blip: an agent-image rollout replaces
# the container and the new one answers within a couple of seconds. Long enough
# to clear the swap, short enough that a genuinely dead agent still reaches the
# honest fallback well inside the voice turn budget.
_AGENT_RETRY_DELAY_S = 1.5

# The agent's GET /api/memories caps `limit` at 200 (422 above it). This asked
# for 10000 — so both brain reads had been failing on every voice session,
# silently, since that ceiling was introduced. Verified against the live agent
# 2026-07-31: limit=10000 → 422, limit=200 → 200 OK.
_MEMORIES_MAX_LIMIT = 200

# A play is one YouTube resolution (measured ~0.9-3.6s on prod) plus a websocket
# put. Well under `think`'s 60s budget on purpose: if it has not started by now
# something is wrong, and the user is standing there in silence waiting.
_PLAY_MEDIA_TIMEOUT_S = 20.0

# The only tools that actually EXECUTE on the relay for a hosted agent. Every
# other raw agent tool needs the terminal-agent tunnel or a local ToolExecutor,
# neither of which exists on platform-api — a direct call to one returns "your
# terminal agent is not connected" and the model narrates a connection failure
# mid-call. V2 filters the tool array to this set so the model is steered down
# paths that work. A tool missing from here is INVISIBLE, not broken, which is
# the failure mode worth remembering: the model then reasons from an empty
# toolbox to "I can't do that", which is how a play became a Spotify referral.
_REALTIME_NATIVE = {"think", "navigate_to", "play_media"}


def _resolve_v2_for_user(user_id: Optional[str]) -> bool:
    if settings.voice_realtime_v2:
        return True
    ids = {u.strip() for u in (settings.voice_realtime_v2_user_ids or "").split(",") if u.strip()}
    return bool(user_id and user_id in ids)


def _v2_active() -> bool:
    v = _v2_ctx.get()
    return settings.voice_realtime_v2 if v is None else v


def _agent_ctx_enabled_for(user_id: str) -> bool:
    """Serve the agent-built voice context: global flag OR per-user canary.

    The allowlist is the W-6 flip's canary path — `voice_context_from_agent`
    is a platform-process global, so without this a "canary flip" would be
    a fleet flip with a reassuring name.
    """
    if settings.voice_context_from_agent:
        return True
    raw = settings.voice_context_from_agent_user_ids or ""
    return user_id in {u.strip() for u in raw.split(",") if u.strip()}


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
        "- Match the user's language. The user may mix English product names "
        "(Grok, ChatGPT, Claude, Gemini…) into another language mid-sentence — "
        "hear those as English names, and if a name came through garbled, ask "
        "one short question instead of acting on a guess.\n"
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


# Tool descriptions that are WRONG in a voice call, overridden here rather than
# duplicated. The agent's own definitions are written for chat, where there is a
# browser and a card to look at.
#
# `play_media` matters most. It was in REALTIME_TOOLS all along — the V2 filter
# was what removed it, so the model was left with `think` as the only way to
# reach music: realtime model -> think -> a full 26k-token agent turn -> the
# same tool. Measured 13.0s to first sound on 2026-07-31. Now that it survives
# the filter (see _REALTIME_NATIVE) it needs a description that tells the model
# to prefer it over `think` and that the user is listening, not reading.
_VOICE_TOOL_DESCRIPTIONS = {
    "play_media": (
        "Start playing music on the user's device, immediately. Call this the moment they "
        "ask to play, put on, or hear something — a song, an artist, an album, a genre, or "
        "'something like X'. Do NOT use `think` for playback: this is far faster and it is "
        "the tool that actually starts the audio. Pass what they asked for in their own "
        "words. More music in the same style follows automatically after the track ends. "
        "For an open-ended ask — an artist, genre, or vibe rather than one specific song "
        "('play me some Drake', 'something chill') — also pass variety=true so a repeat "
        "request starts somewhere fresh instead of replaying the same track. "
        "Say ONE short line while it starts — they are listening to a speaker, not reading "
        "a screen — and never read out a list of alternatives.\n"
        "ONE exception to acting immediately: you must actually have heard WHAT to play. "
        "If the request came through garbled, or was cut off before the name, or you are "
        "guessing at an unfamiliar or non-English artist name, do NOT substitute an artist "
        "you happen to know and do NOT fall back to whatever was playing earlier in this "
        "call — ask one short question ('Who, sorry?' / 'Rihanna, right?') and call this "
        "tool as soon as they answer. Asking a three-word question is not refusing and is "
        "not a list of alternatives; playing the wrong artist is a worse failure than "
        "taking one extra second. If you did hear the name clearly, do not ask — just play."
    ),
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
            "description": _VOICE_TOOL_DESCRIPTIONS.get(t["name"], t["description"]),
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
        # Description stays capability-neutral: REALTIME_TOOLS is a module-level
        # constant shared by v1 and v2 sessions. What `think` actually reaches
        # (v1: a reasoning model; V2: the user's FULL agent with every tool,
        # skill, and connector) is set per-session in build_realtime_instructions.
        "description": (
            "Hand off the user's question or task to solve it. You MUST call this for ANY request "
            "that needs real knowledge, reasoning, research, coding, math, planning, up-to-date "
            "facts, or problem-solving — everything except greetings, acknowledgments, and small "
            "talk. Pass the user's full request verbatim, then relay the result naturally in your "
            "own words as your own work."
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


# ── Tool-activity labels (client tool-use UI) ─────────────────────────
# The realtime WS historically told the client only "a tool is running"
# (state:"tool_use") with no name/args/result. The tool UI needs a human label
# per call; map the common tools here, fall back to a humanized name.
_TOOL_TITLES = {
    "think": "Thinking",
    "navigate_to": "Opening a page",
    "web_search": "Searching the web",
    "browser": "Browsing the web",
    "read_file": "Reading a file",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "exec": "Running a command",
    "memory_search": "Recalling",
    "memory_store": "Remembering",
    "recall_day": "Recalling a past day",
    "play_media": "Starting the music",
}


def _tool_completed_frame(
    call_id: str, func_name: str, result_str: str, media: Optional[dict] = None,
) -> dict:
    """The tool_call.completed frame the voice client renders.

    For play_media the result string is the MODEL's sentence ("Now playing: …
    audible on the user's device…") — third-person English prose that used to
    render verbatim in the voice canvas, in every session language. The
    user-facing shape is the card: ship the structured media dict (same shape
    the thread persist writes) and keep the prose model-only — it still goes
    to OpenAI as the function_call_output, never to the phone.
    """
    frame = {
        "type": "tool_call.completed",
        "call_id": call_id,
        "name": func_name,
        "ok": not result_str.strip().upper().startswith("ERROR"),
        "result_preview": result_str[:600],
    }
    if func_name == "play_media":
        frame["result_preview"] = ""
        if media:
            frame["media"] = media
    return frame


def _tool_activity(func_name: str, arguments: dict) -> tuple:
    """(title, detail) for a client tool-activity row. `detail` is a short,
    human string pulled from the most salient argument (the search query, the
    task, the path…), trimmed so large blobs never hit the wire."""
    title = _TOOL_TITLES.get(func_name) or (func_name.replace("_", " ").strip().capitalize() or "Working")
    detail = ""
    if func_name == "think":
        detail = str(arguments.get("task", ""))[:200]
    elif func_name == "navigate_to":
        detail = str(arguments.get("path", ""))
    elif func_name in ("web_search", "browser"):
        detail = str(arguments.get("query") or arguments.get("url") or "")[:200]
    else:
        for v in arguments.values():
            if isinstance(v, str) and v.strip():
                detail = v.strip()[:200]
                break
    return title, detail


def _local_today_str(tz_name: Optional[str], now_utc: Optional[datetime] = None) -> str:
    """The user's LOCAL calendar date as ``YYYY-MM-DD``.

    The same day boundary DayChat buckets on (day_chat_resolver) and the
    same fallback rule as `_same_local_day` above: unknown or invalid tz
    degrades to UTC rather than raising. Reads this module's `datetime`
    so the frozen-clock test helper applies here too. `now_utc` lets the
    caller share one instant across the legacy and agent builders (W-6
    shadow: a date tick between the two builds must not read as a
    divergence).
    """
    now = now_utc or datetime.now(timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            return now.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
        except Exception:
            pass
    return now.strftime("%Y-%m-%d")


def _day_history_header(total: int, day_date: Optional[str], local_today: Optional[str]) -> str:
    """Header line for the day-history section of the voice instructions.

    `local_today` is None when `voice_day_context_date_guard` is off, or
    when the user's local date could not be resolved — in both cases the
    historical wording is returned byte-for-byte, so the flag-off path
    and the unresolvable-tz path are unchanged.

    With the guard on and `day_date` genuinely not today, the block is
    still included (nothing is dropped from the model's context) but it
    is labelled with the date it came from, because the Realtime model
    otherwise narrates a previous day as "earlier today".
    """
    _today = f"# Today's Full Conversation History ({total} messages across all channels)"
    if not local_today or not day_date or day_date == local_today:
        return _today
    return (
        f"# Conversation from {day_date} — the last day you and the user "
        f"spoke ({total} messages across all channels). This is NOT today: "
        f"today is {local_today} and nothing has been said today yet. Do "
        f"not describe any of it as having happened today."
    )


# ── The W-6 shadow's judgement ────────────────────────────────────────
# Module level on purpose: this decides whether the agent assembler is
# allowed to replace the legacy builder, and the offline parity harness
# (`scripts/w6_parity_harness.py`) imports it rather than reimplementing
# it — a comparator that exists in two copies eventually judges by two
# different rules.


def _voice_blocks(text: str) -> list:
    """`[(header, block)]` in document order, every block carrying its
    `# ` marker.

    `split("\\n\\n# ")` consumes the marker on every block except the
    first, so the original fingerprints measured section 0 over two more
    characters than every other section — which is the only reason a
    duplicated FIRST section ever showed up (as a phantom 2-char content
    difference). Normalising the marker back on removes that asymmetry.
    """
    out = []
    for i, block in enumerate((text or "").split("\n\n# ")):
        block = block.strip()
        if not block:
            continue
        if i:
            block = "# " + block
        header = block.split("\n", 1)[0].lstrip("# ").strip()[:40]
        out.append((header, block))
    return out


def voice_section_fingerprints(text: str) -> dict:
    """`{section header: (chars, 8-hex hash)}` — counts and digests only.

    Never the content: this goes in a log line, and the sections are the
    user's persona, brains and day transcript.

    A header seen more than once is suffixed `#2`, `#3`, … so a
    duplicated section cannot collapse into its twin and vanish from the
    comparison. `[A,B,B]` against `[A,B]` used to compare EQUAL across a
    whole missing section.
    """
    import hashlib

    out = {}
    for header, block in _voice_blocks(text):
        key, n = header, 1
        while key in out:
            n += 1
            key = f"{header}#{n}"
        out[key] = (
            len(block),
            hashlib.sha256(block.encode("utf-8")).hexdigest()[:8],
        )
    return out


def voice_section_order(text: str) -> list:
    """Section headers in the order they appear — template titles only,
    duplicates included."""
    return [header for header, _ in _voice_blocks(text)]


def compare_voice_contexts(agent_text: str, legacy_text: str) -> dict:
    """What the shadow reports, in one importable place.

    `match` is CONTENT equality: the same sections, duplicates included,
    with the same bytes in each. It deliberately does NOT include section
    ORDER — the agent assembler puts `identity_anchor` at index 1, where
    the text channel's runner puts it, rather than after the whole day
    transcript (`agent/voice_context.VOICE_SECTION_ORDER`, "Drift D2": a
    white-label guard the model reads 20k characters after the persona is
    a guard the model has already contradicted). That reorder is an
    intended improvement, so a comparator that failed on it would block
    the flip on a fix.

    Order must never be INVISIBLE either — an unintended reorder would
    otherwise read exactly like no change at all, which is how the
    intended one passed unremarked for a week. `order_match` and both
    sequences ride alongside the verdict so the flip is decided by
    someone who can see the difference.

    Everything returned is a header, a length or a digest. No section
    body ever leaves the process.
    """
    a_fp, l_fp = voice_section_fingerprints(agent_text), voice_section_fingerprints(legacy_text)
    a_order, l_order = voice_section_order(agent_text), voice_section_order(legacy_text)

    same = sorted(k for k in a_fp if k in l_fp and a_fp[k][1] == l_fp[k][1])
    differs = sorted(set(a_fp) ^ set(l_fp)) + sorted(
        k for k in a_fp if k in l_fp and a_fp[k][1] != l_fp[k][1]
    )
    return {
        "match": bool(agent_text and legacy_text) and not differs,
        "same": same,
        "differs": differs,
        "order_match": a_order == l_order,
        "agent_order": a_order,
        "legacy_order": l_order,
    }


# ── Build system instructions from Identity + Memory ──────────────────
async def build_realtime_instructions(
    user_id: str, onboarding: bool = False, now_utc: Optional[datetime] = None
) -> str:
    """Build system instructions for the Realtime API session.

    Reads ALL personal data (identities, memories, history) from the user's VPS
    via httpx API calls. The platform DB is NEVER used for personal data.

    V2 (VOICE_REALTIME_V2): the memory + day-history sections are trimmed to a
    character budget. Two reasons: (a) OpenAI caps instructions+tools at
    16,384 tokens and overflow behavior is undefined; (b) instructions are
    re-billed as input on every response — an unbounded blob makes each voice
    turn cost more for the same experience. Identity docs are never trimmed
    (they ARE the persona); memories keep their head (highest priority first),
    day history keeps its tail (newest messages).

    `now_utc` freezes the clock line; `_instructions_step` passes the same
    instant to the agent builder so the W-6 shadow never reads a minute
    tick between the two builds as a divergence.
    """
    sections = []
    _now_utc = now_utc or datetime.now(timezone.utc)
    _budget = settings.voice_realtime_instructions_budget_chars if _v2_active() else 0

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
            _load_identities_local(user_id),
            _vps_api(agent_url, agent_api_key, "GET", "/api/memories",
                     params={"brain_type": "agent", "limit": _MEMORIES_MAX_LIMIT}),
            _vps_api(agent_url, agent_api_key, "GET", "/api/memories",
                     params={"brain_type": "user", "limit": _MEMORIES_MAX_LIMIT}),
            _vps_api(agent_url, agent_api_key, "GET", "/api/day-chats",
                     params={"limit": 1}),
            return_exceptions=True,
        )
        identities_data, agent_mems_data, user_mems_data, dc_list_prefetch = (
            r if not isinstance(r, BaseException) else None for r in _prefetched
        )
        # SAY SO when a leg comes back empty. `_vps_api` turns every non-2xx into
        # None, which made a total context failure indistinguishable from a user
        # who simply has no memories — and on 2026-07-31 that hid the fact that
        # EVERY voice session was running with no persona and no brain at all:
        # /api/identity 404s (the router is not mounted on the agent) and
        # /api/memories was asked for limit=10000 against a ceiling of 200, so it
        # 422'd. Both had been silently failing, on every call, for as long as
        # those two contracts had been skewed. A prompt that lost its context
        # must never again look the same as a prompt that had none to lose.
        _missing = [
            _name for _name, _val in (
                ("identity", identities_data), ("agent_brain", agent_mems_data),
                ("user_brain", user_mems_data), ("day_chats", dc_list_prefetch),
            ) if not _val
        ]
        if _missing:
            logger.warning(
                "[REALTIME] voice context DEGRADED — no %s for user %s; the model "
                "is answering without it", ", ".join(_missing), user_id[:8],
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
                    text = "\n".join(lines)
                    if _budget:
                        text = _cap_chars(text, int(_budget * 0.2), keep="head")
                    sections.append(text)
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
                    text = "\n".join(lines)
                    if _budget:
                        text = _cap_chars(text, int(_budget * 0.3), keep="head")
                    sections.append(text)
                    logger.info("[REALTIME] Loaded %d user brain memories from VPS", len(mems))
        except Exception as e:
            logger.warning("[REALTIME] Failed to load VPS user memories: %s", e)

        # 2c. Load today's chat history via Day-as-Chat (cross-channel context)
        # Uses /api/day-chats endpoint which loads ALL messages from today
        # across web, app, telegram, and voice channels.
        # Strategy: load all messages, keep user messages short (facts/requests),
        # keep recent assistant messages full for continuity.
        try:
            # The user's LOCAL calendar date — the same boundary the text
            # runner buckets on. None keeps every line below on its exact
            # historical behaviour (guard off, or tz unresolvable).
            #
            # An UNKNOWN tz must leave the guard OFF, not fall back to UTC:
            # a Toronto user at 21:00 local is already on the next UTC date,
            # so a UTC "today" would tell the model that TODAY's own
            # conversation happened on a previous day — the exact error
            # this guard exists to prevent, inverted. Only a tz we actually
            # read justifies contradicting the day chat's own date.
            _tz_name = (
                await _get_user_tz_name(user_id)
                if settings.voice_day_context_date_guard else None
            )
            _local_today = _local_today_str(_tz_name, _now_utc) if _tz_name else None

            # Use the day-chats list endpoint to find today's actual date key
            # (handles timezone correctly — the agent resolves local_date)
            day_msgs = None
            # Which local_date the messages below actually came from. The
            # list endpoint is ordered local_date DESC (day_chats.py:205),
            # so `dc_list[0]` is the NEWEST day chat — today's only if the
            # user has already spoken today.
            _day_msgs_date = None
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
                        _day_msgs_date = today_date
            except Exception:
                pass

            # Fallback: try the date directly. Guard on → the user's LOCAL
            # date (what DayChat keys on); off → the server's UTC date,
            # which is a different day for most of the fleet's evening.
            if not day_msgs:
                today_str = _local_today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                day_msgs = await _vps_api(
                    agent_url, agent_api_key, "GET",
                    f"/api/day-chats/{today_str}/messages",
                    params={"limit": 500},
                )
                _day_msgs_date = today_str

            if day_msgs and isinstance(day_msgs, list):
                day_msgs.sort(key=lambda m: m.get("created_at", ""))
                total = len(day_msgs)

                # Load ALL messages — no truncation, no windowing.
                # The agent must remember everything from the entire day.
                if _local_today and _day_msgs_date and _day_msgs_date != _local_today:
                    logger.warning(
                        "[REALTIME] voice day-context is NOT today — loaded day_chat "
                        "local_date=%s but the user's local date is %s (user %s). "
                        "Labelling the block with its real date.",
                        _day_msgs_date, _local_today, user_id[:8],
                    )
                lines = [_day_history_header(total, _day_msgs_date, _local_today)]
                for m in day_msgs:
                    role = m.get("role", "")
                    content = (m.get("content", "") or "").strip()
                    channel = m.get("channel", "")
                    if role in ("user", "assistant") and content:
                        speaker = "User" if role == "user" else "You"
                        ch = f" [{channel}]" if channel else ""
                        lines.append(f"{speaker}{ch}: {content}")

                if len(lines) > 1:
                    text = "\n".join(lines)
                    if _budget:
                        text = _cap_chars(text, int(_budget * 0.5), keep="tail")
                    sections.append(text)
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
    now_str = _now_utc.strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        "# Voice Conversation Mode\n"
        "You are in a LIVE VOICE conversation. Follow these rules:\n"
        "- Respond naturally and conversationally, as if speaking face-to-face.\n"
        "- Keep responses concise — aim for 1-3 sentences unless the user asks for detail.\n"
        "- Do NOT use markdown, code blocks, bullet points, or any text formatting.\n"
        "- Do NOT say 'here is a list' or read structured data verbatim.\n"
        "- Use natural speech patterns: contractions, casual phrasing.\n"
        "- Match the user's language. Speak EVERY language with a natural, NATIVE "
        "accent and native pronunciation — never a foreign or English-accented one.\n"
        "- When the user speaks Persian/Farsi, reply in fluent, natural Farsi with a "
        "native Tehrani accent, pronouncing every Persian sound correctly (خ، غ، ق، ژ, "
        "and the tapped ر) exactly as a native speaker from Tehran would — NOT with an "
        "English accent. In Persian: «فارسی را کاملاً روان و طبیعی صحبت کن، با لهجهٔ "
        "بومیِ تهرانی و تلفّظِ درستِ فارسی، بدون هیچ لهجهٔ خارجی یا انگلیسی.»\n"
        "- The user may CODE-SWITCH mid-sentence: Persian speech with English product "
        "and brand names embedded (Grok, ChatGPT, Claude, Gemini, xAI, YouTube, "
        "Spotify…). When a word sounds foreign to the sentence's language, consider "
        "an English name FIRST, not a rare native word.\n"
        "- If you only half-caught a name or the request sounded garbled, ask ONE "
        "short clarifying question BEFORE acting on it — never research, explain, or "
        "answer at length from a low-confidence hearing. A user who said 'Grok bot' "
        "once got a lecture about rock music from exactly this mistake; asking "
        "'Grok — the xAI one?' costs two seconds and is always the better trade.\n"
        "- Everything you already know about the user and about yourself is "
        "provided ABOVE in this prompt — your identity, the user's profile, your "
        "memories, and today's conversation. Answer questions about the user's "
        "name, your OWN name, and any stored fact or preference DIRECTLY and "
        "instantly from it. NEVER stall or say you need to 'check what we have on "
        "record' for something already provided above.\n"
        "- If the user asks about something genuinely NOT in your provided context, "
        "hand it to the think tool to look it up — do not guess.\n"
        "- You can navigate the user to different pages using the navigate_to tool. "
        "Offer to show them relevant pages when helpful.\n"
        "- You have FULL ACCESS to the user's computer terminal through a connected agent. "
        "You can run shell commands (exec), read files (read_file), write files (write_file), "
        "edit files (edit_file), search files (grep, find, ls), browse the web (web_search, browser), "
        "and more. Use these tools whenever the user asks you to do something on their computer.\n"
        "- When executing terminal commands, briefly tell the user what you're doing.\n"
        + (
            # V2: `think` runs the user's FULL agent — every tool, skill, and connector.
            "- IMPORTANT: You have a 'think' tool that hands off to your FULL agent — the same brain, "
            "tools, skills, memory, and connected apps (email, calendar, drive, GitHub, and every "
            "connector) you have in text chat. You MUST call it for ANY question, task, action, or "
            "request that needs knowledge, reasoning, research, coding, math, planning, up-to-date facts, "
            "problem-solving, OR an action in the user's tools, accounts, or connected apps. "
            "Only handle simple greetings (hi, hello, bye), yes/no acknowledgments, and casual small talk directly. "
            "For EVERYTHING ELSE, call think(task=<user's full request>). "
            "When you get the result, relay it naturally in your own words as your own work. "
            "NEVER mention the think tool, model switching, reasoning models, or your internal setup to the user.\n"
            if _v2_active() else
            # v1: `think` reaches a reasoning model only (tool-less on platform) —
            # do not promise actions/connectors it cannot perform.
            "- IMPORTANT: You have a 'think' tool. You MUST call it for ANY question, task, or request "
            "that requires knowledge, reasoning, analysis, coding, math, planning, research, explanations, "
            "factual answers, or problem-solving. The 'think' tool connects you to a powerful reasoning model. "
            "Only handle simple greetings (hi, hello, bye), yes/no acknowledgments, and casual small talk directly. "
            "For EVERYTHING ELSE, call think(task=<user's full question>). "
            "When you get the result, relay the answer naturally in your own words. "
            "NEVER mention the think tool, model switching, or reasoning models to the user.\n"
        )
        + "- The user may share their screen with you. When they do, you'll receive periodic "
        "[Screen context: ...] messages describing what's on their screen. Use this visual context "
        "to help them. Don't describe the screen unprompted every time — wait for the user to ask or reference it.\n"
        f"- The current date and time is {now_str}."
    )

    # 3b. Identity anchor — voice parity with the text channel.
    # Without this the Realtime model (gpt-4o-realtime) falls back to its
    # baked-in self-identity and answers "I'm GPT-4o by OpenAI" when asked
    # what it is — breaking the white-label story on the one always-on
    # channel that had NO such guard (docs/security/audit-2026.md MI-1).
    # Mirrors agent_runner.py identity_anchor. Flag-gated (default on): it
    # only ADDS a guardrail, so nothing a user relies on changes.
    if settings.voice_identity_anchor:
        _agent_name = await _get_agent_name(user_id)
        # Back-port the text channel's POSITIVE name statement (agent_runner.py
        # identity_anchor). Voice previously carried only the negative guard, so
        # "what's your name?" answered "I don't have a name". State it when set.
        _name_line = (
            f"Your name is {_agent_name}. That is your name — use it when you "
            f"introduce or refer to yourself, and when the user asks your name, "
            f"answer {_agent_name} (never 'Toup', which is only the platform).\n"
            if _agent_name else ""
        )
        sections.append(
            "# Who you are (identity)\n"
            + _name_line
            + "You are the user's own personal agent on Toup. Toup is the "
            "platform you run on (toup.ai), not your name.\n"
            "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
            "specific provider model. If the user asks what model you are, "
            "who built you, what powers you, or what technology/stack Toup "
            "is built with, answer as their agent — never name the "
            "underlying LLM provider or version, and don't disclose the "
            "underlying tech stack (it's proprietary). The provider may "
            "change without notice; your identity to the user is stable.\n"
            "(If the user asks who FOUNDED or owns Toup — the company — that "
            "is a separate, allowed question; answer it if you know. This "
            "guard is only about the underlying model/technology.)"
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
    """Return the OpenAI key to use for this user's realtime voice session.

    Fully-hosted model: users no longer supply their own provider key — the
    platform provides the LLM. So we fall back to the platform's OpenAI key
    (per-tenant bundle key if provisioned, else the platform master). This key
    is used ONLY here on the PLATFORM side (the realtime WS proxy runs on the
    platform, not in the tenant container), so it is never exposed to a
    container. A legacy per-user key, if any, still wins.
    """
    from app.db.database import async_session_maker
    from app.db import AgentConfig

    key, _is_byok = await _get_user_openai_key_ex(user_id)
    return key


async def _get_user_openai_key_ex(user_id: str) -> tuple[Optional[str], bool]:
    """``(key, is_byok)`` — the key AND whether the USER owns it.

    These are two different questions and conflating them killed voice
    metering outright. The gate was ``using_platform_key = not openai_key``,
    but this resolver already falls back to platform-owned keys, so it is
    never None on a session that survives — and when it IS None the handler
    closes the socket before the relay loop. ``_maybe_meter_response`` was
    therefore unreachable: zero realtime rows in llm_proxy_events, ever, and
    no voice event type in credit_ledger.

    Only a LEGACY per-user ``openai_api_key`` is genuinely BYOK (that user
    pays OpenAI directly, so billing them credits too would double-bill —
    the rationale the old comment gave, correctly, for the wrong variable).
    ``bundle_openai_api_key`` is provisioned BY the platform, and both
    settings fallbacks are platform-owned: the platform pays for all three.
    """
    from app.db.database import async_session_maker
    from app.db import AgentConfig

    async with async_session_maker() as db:
        result = await db.execute(
            select(AgentConfig.openai_api_key, AgentConfig.bundle_openai_api_key)
            .where(AgentConfig.user_id == user_id)
        )
        row = result.first()
    user_key = (row[0] if row else None) or None
    bundle_key = (row[1] if row else None) or None
    key = (
        user_key or bundle_key
        or settings.platform_openai_api_key or settings.openai_api_key or None
    )
    return key, bool(user_key)


async def _get_agent_name(user_id: str) -> Optional[str]:
    """The agent's configured display name (AgentConfig.agent_name, platform DB)
    — the SAME source the text channel's identity anchor reads (agent_runner.py).
    Voice historically ported only the *negative* half of that anchor (never name
    the provider) and never stated the agent's own name, so it answered "I don't
    have a name". Read it so the voice anchor can state it positively."""
    from app.db.database import async_session_maker
    from app.db.models import AgentConfig
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentConfig.agent_name).where(AgentConfig.user_id == user_id)
            )
            row = result.first()
            if row and row[0]:
                return str(row[0]).strip() or None
    except Exception:
        pass  # agent_configs table may not exist on agent DBs
    return None


# ── VPS helpers — all personal data lives on user's VPS, never platform ──

async def _load_identities_local(user_id: str) -> Optional[dict]:
    """The user's identity documents (soul, instructions, user profile, tools).

    Read straight from the PLATFORM database, because that is where they live.
    This used to be `GET /api/identity` against the user's AGENT container, and
    it 404'd on every single voice session for as long as the call existed: the
    identity router is mounted only in platform_main (`app.include_router(
    identity_router, ...)`), the agent never had that table, and the relay
    already runs inside the platform process. So every voice call was built on
    a prompt with no soul, no behavioural guidelines and nothing about the user
    — invisibly, because `_vps_api` folds every non-2xx into None and an empty
    result is indistinguishable from a user who simply has none.

    Returned in the same {"identities": [...]} shape the HTTP route produced, so
    the consumer below is unchanged.
    """
    try:
        from app.db.database import async_session_maker
        from app.db.models import Identity

        async with async_session_maker() as db:
            rows = (await db.execute(
                select(Identity).where(
                    Identity.user_id == user_id,
                    Identity.is_active.is_(True),
                )
            )).scalars().all()
        return {
            "identities": [
                {
                    "identity_type": r.identity_type,
                    "content": r.content,
                    "priority": r.priority or 0,
                }
                for r in rows
            ]
        }
    except Exception as e:  # noqa: BLE001
        # Same failure shape as the old call so the DEGRADED warning still fires.
        logger.warning("[REALTIME] identity load failed for %s: %s", user_id[:8], e)
        return None


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
    params: dict = None, json_body: dict = None, timeout: float = 15.0,
):
    """Make an authenticated API call to the user's VPS agent.

    Uses X-Agent-Key header — VPS auth.py resolves to settings.user_id.
    `timeout` defaults to 15 s (fine for the short identity/memory reads); a
    full agent turn via /api/chat (think parity path) passes a larger value.
    """
    url = f"{agent_url}{path}"
    # TKT-LAT-007 (wave 3): shared agent_http client.
    from app.services.agent_http import get_agent_http_client
    try:
        client = get_agent_http_client()
        if method == "GET":
            resp = await client.get(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, timeout=timeout,
            )
        else:
            resp = await client.post(
                url, headers={"X-Agent-Key": agent_api_key},
                params=params or {}, json=json_body or {}, timeout=timeout,
            )
        # Any 2xx, not just 200. Every write route on the agent answers 201
        # Created (POST /api/sessions, POST /api/sessions/{id}/messages, POST
        # /api/identity), so a 200-only check discarded the response body of
        # each one. That is how voice transcripts were lost: session creation
        # really did succeed with 201, this returned None anyway, and
        # _get_or_create_voice_session fell through to a locally-invented UUID
        # that exists on no agent — so every later message POST 404'd.
        if 200 <= resp.status_code < 300:
            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                logger.warning(
                    "[REALTIME] VPS API %s %s → %s, non-JSON body",
                    method, path, resp.status_code,
                )
                return None
        logger.warning("[REALTIME] VPS API %s %s → %s", method, path, resp.status_code)
    except Exception as e:
        logger.warning("[REALTIME] VPS API %s %s failed: %s", method, path, e)
    return None


# ── Inner-tool relay (voice think) ────────────────────────────────────
# `think` hands the turn to the user's OWN agent, which runs the real tools
# (web search, browse, files, connectors). Those inner calls used to be
# invisible: the blocking endpoint returned only a COUNT, so voice could never
# show which tool ran, what it searched for, or where the answer came from.
# This relays the agent's live SSE frames onto the phone wire.
_INNER_MAX_ROWS    = 24
_INNER_DETAIL_MAX  = 200
_INNER_SOURCES_MAX = 5
_INNER_PREVIEW_MAX = 240
_INNER_FRAME_BYTES = 2048

# agent_url → monotonic ts of the last 404/405 on the streaming route. An agent
# image that predates the route is re-probed after the TTL, so a fleet roll
# heals itself without a platform-api deploy.
_stream_skew: dict = {}
_STREAM_SKEW_TTL = 600.0
_STREAM_IDLE_S   = 30.0     # per-chunk read timeout — NOT the turn budget


def _stream_ok(agent_url: str) -> bool:
    ts = _stream_skew.get(agent_url)
    return ts is None or (time.monotonic() - ts) > _STREAM_SKEW_TTL


class _InnerToolRelay:
    """Maps agent-side SSE frames onto the ALREADY-SHIPPED phone wire.

    Re-applies every cap: the relay does not trust the agent's caps, so a newer
    agent talking to an older platform can never flood the audio WS.
    """

    def __init__(self, websocket, outer_call_id: str):
        self._ws = websocket
        self._outer = outer_call_id
        self._open: dict = {}        # inner call_id → row call_id
        # A row opened from `tool.intent`, waiting for the `tool.start` that
        # names its arguments: (tool_name, row call_id). At most one.
        self._pending: Optional[tuple] = None
        self._rows = 0
        self.alive = True

    def _cid(self, inner: str) -> str:
        # Namespaced so inner ids can never collide with OpenAI's outer ids.
        return f"{self._outer}:{inner}"[:128]

    async def _send(self, frame: dict) -> bool:
        if not self.alive:
            return False
        try:
            if len(json.dumps(frame)) > _INNER_FRAME_BYTES:
                frame.pop("sources", None)
                frame["result_preview"] = str(frame.get("result_preview", ""))[:200]
            await self._ws.send_json(frame)
            return True
        except Exception:
            self.alive = False       # phone gone — stop emitting, keep draining
            return False

    async def on_event(self, ev: dict) -> bool:
        t = ev.get("type")
        if t == "tool.intent":
            # `tool.intent` fires at the model's tool_use_start — BEFORE the
            # arguments have finished streaming, so it carries a name but no
            # call_id. It used to set the coarse `tool_use` flag and open no
            # row, which meant every step appeared only once its arguments had
            # streamed: a second or two of "something is happening, we won't
            # say what" per step, on a surface whose whole job is to say what.
            #
            # So it opens a PROVISIONAL row now, named but detail-less, and the
            # matching `tool.start` adopts the same call_id — which the client
            # already handles, because re-arrival of a known call_id is an
            # UPDATE there, not a duplicate.
            name = str(ev.get("name", ""))[:64]
            if not name or self._pending or self._rows >= _INNER_MAX_ROWS:
                # One provisional row at a time. A model that opens two tool_use
                # blocks before either one's arguments land would otherwise have
                # its second step adopt the first's row. Falling back to the
                # coarse flag costs the head start, never correctness.
                return await self._send({"type": "state", "state": "tool_use"})
            cid = self._cid(f"intent{self._rows}")
            self._pending = (name, cid)
            self._rows += 1
            title, _ = _tool_activity(name, {})
            return await self._send({
                "type": "tool_call.started",
                "call_id": cid, "parent_call_id": self._outer,
                "name": name, "title": title, "detail": "",
            })
        if t == "tool.start":
            inner = str(ev.get("call_id", ""))
            name = str(ev.get("name", ""))[:64]
            args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
            title, detail = _tool_activity(name, args)
            # Adopt the provisional row only when it is unambiguously the same
            # step: same tool, and nothing else outstanding.
            pending = self._pending
            self._pending = None
            if pending and pending[0] == name:
                cid = pending[1]
            else:
                if pending:
                    # Claimed a row we are not going to fill — close it rather
                    # than leave the phone with a spinner that never resolves.
                    await self._send({"type": "tool_call.completed", "call_id": pending[1],
                                      "parent_call_id": self._outer, "name": pending[0],
                                      "ok": False, "result_preview": ""})
                if self._rows >= _INNER_MAX_ROWS:
                    return True
                self._rows += 1
                cid = self._cid(inner)
            self._open[inner] = cid
            return await self._send({
                "type": "tool_call.started",
                "call_id": cid,
                "parent_call_id": self._outer,
                "name": name, "title": title,
                "detail": str(detail)[:_INNER_DETAIL_MAX],
            })
        if t == "tool.end":
            inner = str(ev.get("call_id", ""))
            cid = self._open.pop(inner, None)
            if cid is None:
                return True          # start was capped/dropped — never orphan a completion
            srcs = [s for s in (ev.get("sources") or []) if isinstance(s, dict)][:_INNER_SOURCES_MAX]
            preview = str(ev.get("preview") or "")[:_INNER_PREVIEW_MAX]
            if not preview and srcs:
                doms = [str(s.get("domain") or "") for s in srcs if s.get("domain")]
                if doms:
                    preview = f"{len(srcs)} sources · " + " · ".join(doms[:3])
            return await self._send({
                "type": "tool_call.completed",
                "call_id": cid, "parent_call_id": self._outer,
                "name": str(ev.get("name", ""))[:64],
                "ok": bool(ev.get("ok", True)),
                "result_preview": preview,
                "sources": srcs,
                "elapsed_ms": int(ev.get("elapsed_ms") or 0),
            })
        if t == "status" and ev.get("stage") == "thinking":
            # Deliberately NOT {"type":"state","state":"thinking"} — the shipped
            # app clears usingTool on that, which would UN-DOCK the orb on every
            # post-tool iteration. A new type is inert on old builds.
            return await self._send({"type": "tool_call.progress",
                                     "call_id": self._outer, "stage": "thinking"})
        return True

    async def close_open(self) -> None:
        """Fail every still-running row so the phone never leaves a spinner."""
        if self._pending:
            # A turn that ends between tool_use_start and the arguments landing
            # leaves this row open; it is a spinner like any other.
            await self._send({"type": "tool_call.completed", "call_id": self._pending[1],
                              "parent_call_id": self._outer, "name": self._pending[0],
                              "ok": False, "result_preview": ""})
            self._pending = None
        for cid in list(self._open.values()):
            await self._send({"type": "tool_call.completed", "call_id": cid,
                              "parent_call_id": self._outer, "name": "",
                              "ok": False, "result_preview": ""})
        self._open.clear()


async def _vps_api_stream(agent_url: str, agent_api_key: str, path: str,
                          json_body: dict, relay: "_InnerToolRelay") -> tuple:
    """SSE sibling of _vps_api. NEVER raises.

    Returns (outcome, payload, frames_seen):
      ("stream", done_dict, n) — streamed and terminated with `done`
      ("json",   body_dict, 0) — agent answered with plain JSON (unexpected but fine)
      ("skew",   None,      0) — 404/405: agent build predates the route
      ("fail",   None,      n) — anything else. n>0 means the turn ALREADY RAN
                                 on the agent; the caller MUST NOT re-issue it.

    Timeout semantics: with stream=True httpx's `read` timeout is a PER-CHUNK
    idle timeout, so it cannot bound the turn — an agent heart-beating every
    10 s would hold the voice floor forever. The caller wraps this in
    asyncio.wait_for(voice_realtime_think_timeout_s), preserving the exact
    ceiling the buffered POST had.
    """
    from app.services.agent_http import get_agent_http_client
    client = get_agent_http_client()
    resp = None
    frames = 0
    try:
        req = client.build_request(
            "POST", f"{agent_url}{path}",
            headers={"X-Agent-Key": agent_api_key, "Accept": "text/event-stream"},
            json=json_body,
            timeout=httpx.Timeout(connect=5.0, read=_STREAM_IDLE_S, write=5.0, pool=5.0),
        )
        resp = await client.send(req, stream=True)
        if resp.status_code in (404, 405):
            await resp.aread()
            return ("skew", None, 0)
        if resp.status_code != 200:
            await resp.aread()
            logger.warning("[REALTIME] think stream → %s", resp.status_code)
            return ("fail", None, 0)
        if not resp.headers.get("content-type", "").startswith("text/event-stream"):
            body = await resp.aread()
            try:
                return ("json", json.loads(body), 0)
            except Exception:
                return ("fail", None, 0)
        async for line in resp.aiter_lines():
            if not line or line[0] == ":" or not line.startswith("data:"):
                continue                       # heartbeats / blank separators
            try:
                ev = json.loads(line[5:].strip())
            except Exception:
                continue
            frames += 1
            etype = ev.get("type")
            if etype == "done":
                return ("stream", ev, frames)
            if etype == "error":
                logger.warning("[REALTIME] think stream error frame: %s", ev.get("code"))
                return ("fail", None, frames)
            if relay is not None and relay.alive:
                await relay.on_event(ev)
        return ("fail", None, frames)          # stream ended with no terminal frame
    except Exception as e:
        logger.warning("[REALTIME] think stream failed: %s", e)
        return ("fail", None, frames)
    finally:
        if resp is not None:
            # MANDATORY. The shared client is a singleton with a bounded pool;
            # a stream=True response not closed here permanently burns a slot,
            # and enough leaks wedge ALL platform→agent HTTP (identity,
            # memories, day-chats).
            try:
                await resp.aclose()
            except Exception:
                pass


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
async def _get_user_tz_name(user_id: str) -> Optional[str]:
    """User.timezone from the local DB (platform copy on the relay,
    tenant row on the agent). None when unset/unreadable."""
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import User
        async with async_session_maker() as db:
            tz = (await db.execute(
                select(User.timezone).where(User.id == user_id)
            )).scalar_one_or_none()
            return tz or None
    except Exception:
        return None


async def _persist_user_tz(user_id: str, tz_name: str) -> None:
    """Fill a blank ``users.timezone`` with a device-detected zone.

    Restricted to rows where the column IS NULL: the account page's
    precise-location flow writes a zone the user chose deliberately, and
    a laptop that happens to be in another country must not overwrite it.
    """
    try:
        from sqlalchemy import update
        from app.db.database import async_session_maker
        from app.db.models import User
        async with async_session_maker() as db:
            await db.execute(
                update(User)
                .where(User.id == user_id, User.timezone.is_(None))
                .values(timezone=tz_name)
            )
            await db.commit()
    except Exception as e:
        logger.warning("[REALTIME] tz self-heal failed user=%s: %s", user_id[:8], e)


async def _apply_client_tz(user_id: str, raw) -> Optional[str]:
    """Validate a client-supplied IANA zone, persist it, return it.

    Voice sends no tz in the WebRTC payload, so it fell back to the
    PLATFORM copy of ``users.timezone`` — while chat and mobile persist
    the zone they collect to the TENANT copy. Same column name, different
    database: 36 of 43 platform rows were NULL on 2026-08-10, 23 of them
    belonging to users active in the last 30 days, and every one of those
    voice sessions resolved "today" in UTC.

    Validation is not optional here. ``resolve_local_date`` falls back to
    UTC on a zone it cannot parse, so an unvalidated string reintroduces
    exactly the bug this closes (#488) while looking like it fixed it.
    """
    if not raw or not isinstance(raw, str):
        return None
    tz_name = raw.strip()
    if not tz_name:
        return None
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz_name)
    except Exception:
        logger.warning("[REALTIME] ignoring unparseable client tz for user=%s", user_id[:8])
        return None
    await _persist_user_tz(user_id, tz_name)
    return tz_name


def _parse_utc_dt(raw) -> Optional[datetime]:
    """Parse an ISO timestamp from the VPS API. Naive values are UTC
    (Conversation.started_at is stored as naive utcnow)."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _same_local_day(started_utc: datetime, now_utc: datetime, tz_name: Optional[str]) -> bool:
    """Mirror of agent_runner.same_local_day (not imported — pulling
    agent_runner into the platform relay would drag the whole agent
    stack in). Unknown/invalid tz falls back to UTC, same as DayChat."""
    tz = None
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None
    if tz is not None:
        return started_utc.astimezone(tz).date() == now_utc.astimezone(tz).date()
    return started_utc.date() == now_utc.date()


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
                # Reuse only when the session started TODAY in the user's
                # LOCAL calendar day (same_local_day semantics) — the old
                # UTC-date comparison stranded post-local-midnight voice
                # transcripts in yesterday's session until UTC midnight.
                started_dt = _parse_utc_dt(
                    data.get("started_at") or data.get("updated_at")
                )
                if started_dt is not None and _same_local_day(
                    started_dt,
                    datetime.now(timezone.utc),
                    await _get_user_tz_name(user_id),
                ):
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


# Ceiling for the query-parameter compatibility shim below. Measured against a
# live agent rather than assumed: 64 KB of encoded content still returned 201,
# and the request only failed to complete somewhere before 100 KB. 16 KB keeps
# a 4x margin under the proven-good point, and still covers ~2,600 characters
# of Farsi (~6 encoded bytes each) — longer than any spoken turn.
_QUERY_SHIM_MAX = 16000


def _message_payload(
    role: str,
    content: str,
    model: Optional[str] = None,
    media: Optional[dict] = None,
):
    """Build (json_body, query_params) for POST /api/sessions/{id}/messages.

    Agent images that predate the body-aware route read role/content as QUERY
    parameters; newer ones prefer the JSON body. Sending both in ONE request
    satisfies either build — the old one reads the query and ignores the body,
    the new one lets the body win — so a platform deploy heals the whole fleet
    without waiting on an agent rollout.

    The shim is dropped when the encoded content would overrun a URL. A Farsi
    reply encodes to ~6 chars per character, so a long one blows past the
    request-line limit; body-only still works on any agent new enough to read
    it, and the caller logs the loss if that agent isn't.
    """
    body = {"role": role, "content": content}
    if model:
        body["model_used"] = model
    if media:
        # Body-only from here: a nested object cannot ride the query shim, and
        # an agent old enough to only read query params has no media column
        # handling anyway — it stores the text and drops this, which is exactly
        # today's behaviour rather than a regression.
        body["media"] = media
        return body, None
    if len(quote(content)) > _QUERY_SHIM_MAX:
        return body, None
    return body, dict(body)


async def _save_voice_messages(
    user_id: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
    model: str = "gpt-4o-realtime",
    media: Optional[dict] = None,
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
    saved = 0
    lost = 0

    for _role, _text, _model, _media in (
        ("user", user_text, None, None),
        # Media rides the ASSISTANT row, matching how a chat turn persists a
        # play — the card belongs to the reply that started it.
        ("assistant", assistant_text, model, media),
    ):
        if not _text:
            continue
        body, params = _message_payload(_role, _text, _model, _media)
        result = await _vps_api(
            agent_url, agent_api_key, "POST",
            f"/api/sessions/{session_id}/messages",
            params=params, json_body=body,
        )
        if result:
            saved += 1
        else:
            lost += 1

    if lost:
        # ERROR, not INFO. This is the only thing that persists a spoken turn —
        # `think` deliberately calls the agent with save=False to avoid a
        # duplicate day-chat entry, so a failure here loses the transcript
        # outright rather than degrading it.
        logger.error(
            "[REALTIME] LOST %d voice message(s) for session %s — transcript NOT persisted",
            lost, session_id[:8],
        )
    logger.info("[REALTIME] Saved %d message(s) to VPS session %s via API", saved, session_id[:8])


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

        # Key for LLM extraction (operational data, not personal). Under the
        # fully-hosted model the user has no key of their own, so reuse the
        # realtime key resolver which falls back to the platform key — else
        # voice memory extraction silently no-ops for every hosted user.
        user_api_key = await _get_user_openai_key(user_id)

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
                # ttl_days MUST cross the tunnel. The extractor computes it
                # for exactly the transient case ("play Ebi's music", "remind
                # me in 2 minutes"), and dropping it here is why the founder's
                # brain holds permanent rows like "The user requested to play
                # the song 'Setarehaye Sorbi'" with access_count=20 — media
                # requests are overwhelmingly a voice behaviour, and this is
                # the voice write path. Without a horizon they are immortal
                # and outrank real preferences in retrieval forever.
                result = await send_tool_call(user_id, "memory_store", {
                    "content": mem.content,
                    "category": cat,
                    "brain_type": "user",
                    "importance": mem.importance,
                    "ttl_days": getattr(mem, "ttl_days", None),
                    "memory_type": (
                        mem.memory_type.value
                        if hasattr(mem.memory_type, "value") else mem.memory_type
                    ),
                    "confidence": mem.confidence,
                })
                if not result.startswith("ERROR"):
                    pushed += 1
                    logger.info("[REALTIME] Memory stored on VPS: %s", describe_memory(mem.content, category=mem.category))
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


# System prompt for the tool-less Option B fallback in `_think`.
#
# Option B runs ONLY when the user's full agent could not be reached, and it has
# `tools=[]`. The old prompt said nothing about either fact, so a model asked to
# DO something — "play me asap rocky" — reasoned from the only evidence it had
# (no tools) to the only conclusion available (I cannot play music) and helpfully
# sent the user to a competitor. That is what the founder hit on 2026-07-31,
# eight minutes after the same feature had worked.
#
# The model, not a regex, decides whether the request was an action: it can read
# the request and it knows it has no tools. What it was missing is the context
# that its own inability is TEMPORARY and LOCAL to this one call.
_THINK_FALLBACK_SYSTEM = (
    "You are the user's own agent, answering over voice.\n"
    "\n"
    "IMPORTANT CONTEXT: your tools are momentarily unreachable — a transient "
    "connection problem on our side, already being retried. This is NOT a "
    "limitation of what you can do, and the user has done nothing wrong.\n"
    "\n"
    "If the request was a QUESTION, just answer it well.\n"
    "\n"
    "If the request was an ACTION (play music, send something, set a reminder, "
    "open something, change a setting), you must NOT attempt it and must NOT "
    "pretend it happened. Say plainly that you could not reach your tools just "
    "then and ask them to say it again in a moment. One short sentence.\n"
    "\n"
    "NEVER say you are unable to do something, that you lack a capability, or "
    "that a feature does not exist — you can do all of it, just not this "
    "second. NEVER suggest another app, service or website as a way to get it "
    "done (no Spotify, Apple Music, YouTube, or any other). Suggesting a "
    "competitor because your own tools blinked is the worst possible answer.\n"
    "\n"
    "Never state or hint at which underlying AI model or provider you are, or "
    "Toup's tech stack; refer to yourself only as the user's agent."
)


# ── Deep Think — Claude Opus reasoning for complex voice tasks ────────
async def _play_media_direct(user_id: str, query: str, variety: bool = False) -> str:
    """Start playback via the agent's tool-less /internal/play-media route.

    The whole point is that nothing here reasons. `think` exists for requests
    that need the agent's judgement; a play needs a search and a websocket
    frame, and putting a 26k-token agent turn in front of that is what made
    "play me asap rocky" take 13 seconds.

    Returns (text_for_the_model, media_dict_or_None). On failure the text is
    the REAL reason, prefixed ERROR so the client marks the step failed — never
    something the model could read as "this product cannot play music". The
    voice instructions forbid that reading explicitly; this makes sure the
    material it is reading from is true. `media` is None on every failure path,
    so a failed play can never persist a card for a song that isn't playing.
    """
    query = (query or "").strip()
    if not query:
        return "ERROR: no track was specified.", None

    vps = await _get_vps_info(user_id)
    if not vps:
        return ("ERROR: could not reach the user's agent to start playback. "
                "This is a temporary connection problem, not a missing feature."), None
    agent_url, agent_api_key = vps

    t0 = time.monotonic()
    try:
        data = await _vps_api(
            agent_url, agent_api_key, "POST", "/api/v1/internal/play-media",
            # Voice is always audio (a call has no screen for video); `variety`
            # rides through so an open-ended ask starts somewhere fresh.
            json_body={"query": query, "variety": variety},
            timeout=_PLAY_MEDIA_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[REALTIME] play_media failed for %s: %s", user_id[:8], e)
        return ("ERROR: could not start playback just now — a temporary problem "
                "reaching the user's agent. Ask them to try again in a moment."), None

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if not data or not data.get("ok"):
        logger.warning("[REALTIME] play_media no-result for %s in %dms", user_id[:8], elapsed_ms)
        return ("ERROR: could not start that track. It may not be available. "
                "Offer to try a different song — never suggest another app."), None

    title = (data.get("title") or "").strip()
    video_id = (data.get("video_id") or "").strip()
    logger.info("[REALTIME] play_media OK in %dms: %r", elapsed_ms, title[:60])
    # Hand the caller what it needs to persist a real media card, in the same
    # shape a chat turn writes (AgentRunner._save_messages → metadata_json
    # {"media": ...}). A voice play used to leave only plain text in the thread,
    # so a song the agent genuinely started looked, on reopening the app, like
    # nothing had happened.
    media = {
        "type": "youtube",
        "video_id": video_id,
        "title": title,
        "thumbnail_url": (data.get("thumbnail_url") or "")
        or (f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""),
    } if video_id else None
    # The title is the answer to "what's playing?" for the rest of the call —
    # it stays in the model's own conversation as this tool's result.
    text = (f"Now playing: {title}. It is already audible on the user's device, "
            f"and more in the same style will follow automatically.") if title else \
           "Playback started on the user's device."
    return text, media


async def _think(user_id: str, task: str, session_id: Optional[str],
                 relay: Optional["_InnerToolRelay"] = None) -> tuple:
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
                # Parity with the V2 relay path (it sends save=False, which
                # api_v1's agent-turn maps to exactly these three): `task` is
                # a string the REALTIME MODEL synthesised, not what the user
                # said — persisting it double-writes the day chat next to
                # _save_voice_messages' real transcripts, and mining it for
                # memories minted facts the user never stated (the 409A
                # incident api_v1.py documents). Voice memory extraction runs
                # from real transcripts via _extract_voice_memories instead.
                save_user_message=False,
                save_assistant_message=False,
                disable_post_processing=True,
            )
            logger.info(
                "[REALTIME] think via agent_runner: %d chars, model=%s, %dms",
                len(response.text), response.model, response.processing_time_ms,
            )
            return response.text, response.model or model_override
        except Exception as e:
            logger.warning("[REALTIME] think via agent_runner failed: %s", e)

    # Option A2 (platform-api, V2): the realtime relay runs on platform-api,
    # where the in-process agent_runner is absent. Route `think` to the user's
    # OWN agent over its internal full-agent endpoint — the SAME AgentRunner the
    # text chat runs — so voice gets the user's COMPLETE capability set: web +
    # browser + files + memory AND every skill and connected MCP connector
    # (Gmail, Calendar, Drive, GitHub, …). This is what makes voice a true peer
    # of chat rather than a stripped-down voice model; it also runs in the user's
    # agent session so the turn shares their chat history.
    #
    # NOTE: this MUST NOT be /api/chat — that route is a memory-augmented but
    # TOOL-LESS single LLM completion (app/api/chat.py) and it always persists,
    # which would both strip voice of tools and DOUBLE-write the day-chat. The
    # full agent lives behind /api/v1/internal/agent-turn (X-Agent-Key gated,
    # runs _agent_runner.run with save honored). V2-gated so v1 is unchanged
    # until the flag rolls out globally.
    if _v2_active():
        vps = await _get_vps_info(user_id)
        if vps:
            agent_url, agent_api_key = vps
            _body = {
                "message": task,
                "session_id": session_id,   # context (history) only
                "model": model_override,
                # Don't persist here — the voice handler already saves the
                # spoken user/assistant turn (avoids a duplicate day-chat
                # entry). Requires the agent image that exposes this
                # endpoint; enable VOICE_REALTIME_V2 for an account only
                # after its agent has that build (see deploy notes).
                "save": False,
            }
            data = None
            _no_retry = False
            try:
                # Preferred path: stream, so the phone sees each inner tool —
                # its name, its query, its sources — WHILE the turn runs.
                if (settings.voice_realtime_tool_events
                        and relay is not None and _stream_ok(agent_url)):
                    try:
                        outcome, payload, frames = await asyncio.wait_for(
                            _vps_api_stream(
                                agent_url, agent_api_key,
                                "/api/v1/internal/agent-turn/stream", _body, relay),
                            timeout=settings.voice_realtime_think_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        outcome, payload, frames = "fail", None, 1
                        logger.warning("[REALTIME] think stream exceeded %.0fs budget",
                                       settings.voice_realtime_think_timeout_s)
                    if outcome == "skew":
                        _stream_skew[agent_url] = time.monotonic()
                        logger.info("[REALTIME] agent predates think-stream route: %s", agent_url)
                    elif outcome in ("stream", "json") and payload and payload.get("text"):
                        data = payload
                    elif frames > 0:
                        # The turn ALREADY RAN on the agent. Re-issuing the
                        # blocking POST would double-charge credits and re-fire
                        # any mutating connector. Drop straight to Option B
                        # (tool-less, side-effect-free) instead.
                        #
                        # This is the branch the founder hit on 2026-07-31: the
                        # chip read "That one didn't work / Trying another way",
                        # which only a server tool_call.completed{ok:false} can
                        # produce — i.e. the agent WAS up and had already run a
                        # tool. So the answer came from Option B, and Option B is
                        # what has to be honest. See _THINK_FALLBACK_SYSTEM.
                        _no_retry = True
                    await relay.close_open()

                if data is None and not _no_retry:
                    # Retry ONCE on a transport-level failure. The agent
                    # container is replaced on every agent-image rollout, and a
                    # request that lands in that window gets a connect error
                    # against a container that is back seconds later. Without
                    # this, a routine deploy silently downgrades the user to the
                    # tool-less fallback below — which is what produced
                    # "your playback agent isn't connected right now, try
                    # Spotify" on 2026-07-31, eight minutes after music had
                    # played fine.
                    #
                    # Transport errors ONLY. An HTTP error from a container that
                    # answered, or a timeout (the agent may be mid-turn and
                    # about to charge credits), must not be replayed — same
                    # reasoning as `_no_retry` above.
                    for _attempt in (1, 2):
                        try:
                            data = await _vps_api(
                                agent_url, agent_api_key, "POST", "/api/v1/internal/agent-turn",
                                json_body=_body,
                                timeout=settings.voice_realtime_think_timeout_s,
                            )
                            break
                        except (ConnectionError, OSError) as _conn_err:
                            if _attempt == 2:
                                raise
                            logger.warning(
                                "[REALTIME] think agent unreachable (%s) — retrying once in %.1fs",
                                type(_conn_err).__name__, _AGENT_RETRY_DELAY_S,
                            )
                            await asyncio.sleep(_AGENT_RETRY_DELAY_S)
                if data and data.get("text"):
                    logger.info(
                        "[REALTIME] think via agent full-turn: %d chars, model=%s, tool_calls=%s",
                        len(data["text"]), data.get("model"), data.get("tool_calls"),
                    )
                    return data["text"], data.get("model") or model_override
                logger.warning("[REALTIME] think via agent full-turn: empty result, falling back")
            except Exception as e:
                logger.warning("[REALTIME] think via agent full-turn failed: %s", e)

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
                system=_THINK_FALLBACK_SYSTEM,
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
                            {"role": "system", "content": _THINK_FALLBACK_SYSTEM},
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
        # Both reads used limit=10000 against the agent's ceiling of 200 and so
        # 422'd every time — the same skew that was fixed in
        # build_realtime_instructions (see _MEMORIES_MAX_LIMIT). Here the cost
        # was worse than a thin prompt: this is the function that COMPILES
        # saul.md and identity.md at the end of onboarding, so every voice
        # onboarding finished by writing a soul and a profile built from zero
        # memories. Re-verified against the live agent 2026-08-01:
        # limit=200 → 200 OK, limit=201 → 422.
        vps = await _get_vps_info(user_id)
        agent_memories = []
        user_memories = []

        if vps:
            agent_url, agent_api_key = vps

            agent_mems_data = await _vps_api(
                agent_url, agent_api_key, "GET", "/api/memories",
                params={"brain_type": "agent", "limit": _MEMORIES_MAX_LIMIT},
            )
            if agent_mems_data:
                agent_memories = agent_mems_data.get("memories", agent_mems_data) if isinstance(agent_mems_data, dict) else agent_mems_data
                if not isinstance(agent_memories, list):
                    agent_memories = []

            user_mems_data = await _vps_api(
                agent_url, agent_api_key, "GET", "/api/memories",
                params={"brain_type": "user", "limit": _MEMORIES_MAX_LIMIT},
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
                    # Written to the PLATFORM database, not the agent.
                    #
                    # These three calls all targeted the agent's /api/identity,
                    # which does not exist there — the identity router is only
                    # ever mounted in platform_main, so every onboarding upsert
                    # 404'd and was swallowed by the warning below. Verified
                    # live 2026-08-01: GET {agent}/api/identity -> 404 while
                    # /api/soul and /api/memories both -> 200. Writing here puts
                    # the rows exactly where _load_identities_local now reads
                    # them, so a user's onboarding persona reaches their voice
                    # prompt instead of vanishing.
                    from app.db.database import async_session_maker
                    from app.db.models import Identity

                    async with async_session_maker() as _db:
                        old = (await _db.execute(
                            select(Identity).where(
                                Identity.user_id == user_id,
                                Identity.identity_type == id_type,
                            )
                        )).scalars().all()
                        for row in old:
                            await _db.delete(row)
                        _db.add(Identity(
                            user_id=user_id,
                            identity_type=id_type,
                            name=id_name,
                            content=id_content,
                            priority=id_priority,
                            is_active=True,
                        ))
                        await _db.commit()
                except Exception as e:
                    logger.warning("[REALTIME] Failed to save Identity %s: %s", id_type, e)

                # TENANT copy — the agent-side assembler (W-6) and text chat
                # read `identities` from the tenant DB; a persona written
                # only to the platform copy dies with the legacy builder.
                # Upsert via the agent's own identity router (mounted since
                # test_agent_serves_identity). Deactivate-then-create, not
                # delete: `_vps_api` speaks GET/POST only, and keeping the
                # old row inactive preserves history on a re-onboarding.
                try:
                    existing = await _vps_api(
                        agent_url, agent_api_key, "GET", "/api/identity",
                        params={"active_only": "true"},
                    )
                    ex_rows = (
                        existing.get("identities", [])
                        if isinstance(existing, dict) else (existing or [])
                    )
                    for row in ex_rows:
                        if isinstance(row, dict) and row.get("identity_type") == id_type and row.get("id"):
                            await _vps_api(
                                agent_url, agent_api_key, "POST",
                                f"/api/identity/{row['id']}/deactivate",
                            )
                    created = await _vps_api(
                        agent_url, agent_api_key, "POST", "/api/identity",
                        json_body={
                            "identity_type": id_type,
                            "name": id_name,
                            "content": id_content,
                            "priority": id_priority,
                            "is_active": True,
                        },
                    )
                    if created is None:
                        logger.warning(
                            "[REALTIME] tenant Identity %s write returned no row for %s",
                            id_type, user_id[:8],
                        )
                except Exception as e:
                    logger.warning(
                        "[REALTIME] Failed to save tenant Identity %s: %s", id_type, e
                    )

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


def realtime_model() -> str:
    """Model slug for this session. V2 tracks the current generation
    (gpt-realtime-2.1: 128k context, semantic VAD, better interruption
    behavior at the same audio price); V1 stays pinned to gen-1."""
    if _v2_active():
        return settings.voice_realtime_model
    return "gpt-realtime"


def realtime_url() -> str:
    return f"wss://api.openai.com/v1/realtime?model={realtime_model()}"


# ── Transcription language hint ───────────────────────────────────────
# Whisper re-detects the language of EVERY utterance independently, and on a
# short one carrying an English proper noun it reliably guesses wrong: measured
# 2026-07-31, Persian audio for "برام ASAP Rocky بذار" came back from whisper-1
# as "Badom asap rocky bizzare.", and production has seen the same speaker land
# in Devanagari. The realtime model hears the AUDIO and answers correctly, but
# the inner agent turn reads the TEXT — so a mis-detected utterance makes the
# agent answer a question the user never asked.
#
# Pinning `language` once per session fixes that (same measurement, same audio,
# language=fa → "برام ای سپ را کی بزار"). But a WRONG pin is far worse than no
# pin: English audio forced to fa returned "درست کنید که با کندرک لامار…",
# hallucinated Persian bearing no relation to the input. Hence: resolve from
# evidence, and return None whenever the evidence is thin.
#
# It must NOT be inferred from locale/timezone. There is no language column on
# User, AgentConfig or SoulConfig, and every non-UTC User.timezone in production
# is America/Toronto — including the Persian speaker's — so timezone resolves
# him to English, i.e. straight into the hallucination case above.
_LANG_HINT_ENV = "VOICE_TRANSCRIPTION_LANGUAGE_HINT"
# Days of history to scan, newest first. Only days the agent reports as having
# voice traffic are fetched, so this is a ceiling, not a cost.
_LANG_WINDOW_DAYS = 5
# A message counts as evidence only if one non-Latin script owns this share of
# its letters and appears at least this many times — one quoted foreign song
# title inside an English sentence must not be read as the speaker's language.
_LANG_MSG_MIN_SHARE = 0.30
_LANG_MSG_MIN_CHARS = 6
# Fire on PRESENCE, not majority. The bug destroys its own evidence: it turns
# spoken Persian into Latin gibberish, so on the account in the bug report only
# 2 of 122 recent user messages survive in Persian script (12.5% once narrowed
# to voice). A majority rule would never fire for the very user it is meant to
# fix. Two script-dominant messages is already decisive, because a speaker who
# does not use the script produces zero, never two.
_LANG_MIN_MESSAGES = 2
_LANG_MIN_SHARE = 0.10
# Which language someone speaks does not change within a call, or usually ever;
# the TTL exists so a genuine switch is picked up without a restart.
_LANG_CACHE_TTL = 30 * 60.0
_lang_cache: dict = {}


def _lang_hint_enabled() -> bool:
    """Kill switch. Defaults ON; set VOICE_TRANSCRIPTION_LANGUAGE_HINT=false to
    disable on a running deploy if the hint regresses somebody."""
    return os.getenv(_LANG_HINT_ENV, "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


async def _detect_voice_language(user_id: str) -> Optional[str]:
    """Language code for `user_id`'s voice, from their own transcript history.

    Scoped to the voice channel because that is the only place the evidence is
    honest: this user types English on mobile (IELTS practice, technical
    questions) and speaks Persian, so whole-history detection reports English.
    Assistant turns count too — they are the model's own text, never passed
    through Whisper, so they are the one part of a voice transcript the bug
    cannot corrupt.
    """
    vps = await _get_vps_info(user_id)
    if not vps:
        return None
    agent_url, agent_api_key = vps

    days = await _vps_api(
        agent_url, agent_api_key, "GET", "/api/day-chats", params={"limit": 30},
    )
    if not isinstance(days, list):
        return None
    # channels_active rides the list payload, so days with no voice traffic are
    # skipped without paying a request for their messages.
    voice_days = [
        d.get("local_date") for d in days
        if d.get("message_count") and "voice" in (d.get("channels_active") or [])
    ][:_LANG_WINDOW_DAYS]
    voice_days = [d for d in voice_days if d]
    if not voice_days:
        return None

    fetched = await asyncio.gather(*[
        _vps_api(
            agent_url, agent_api_key, "GET",
            f"/api/day-chats/{d}/messages", params={"limit": 500},
        )
        for d in voice_days
    ], return_exceptions=True)

    counts: dict = {}
    window = 0
    for result in fetched:
        if not isinstance(result, list):
            continue
        for m in result:
            if m.get("channel") != "voice":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            window += 1
            code = detect_script_language(
                content,
                min_share=_LANG_MSG_MIN_SHARE,
                min_chars=_LANG_MSG_MIN_CHARS,
            )
            if code:
                counts[code] = counts.get(code, 0) + 1

    if not counts:
        return None
    code, hits = max(counts.items(), key=lambda kv: kv[1])
    if hits < _LANG_MIN_MESSAGES or hits < _LANG_MIN_SHARE * window:
        logger.info(
            "[REALTIME] No language hint for %s — best=%s %d/%d below floor",
            user_id[:8], code, hits, window,
        )
        return None
    logger.info(
        "[REALTIME] Language hint %s for %s (%d/%d voice messages)",
        code, user_id[:8], hits, window,
    )
    return code


def _cached_voice_language(user_id: str) -> Optional[str]:
    """The already-resolved hint, or None. Never does I/O — this is what the
    first session.update reads, so a cold cache costs the connect path nothing."""
    entry = _lang_cache.get(user_id)
    if entry and (time.monotonic() - entry[1]) < _LANG_CACHE_TTL:
        return entry[0]
    return None


async def resolve_voice_language(user_id: str) -> Optional[str]:
    """Cached `_detect_voice_language`. None means "unknown" — leave Whisper on
    auto-detect, which is strictly today's behavior."""
    if not _lang_hint_enabled():
        return None
    # Reads the entry rather than calling _cached_voice_language, which cannot
    # tell "cached as None" from "never resolved". Most users ARE None, and
    # conflating the two would re-read their history on every single connect.
    entry = _lang_cache.get(user_id)
    if entry and (time.monotonic() - entry[1]) < _LANG_CACHE_TTL:
        return entry[0]
    code = await _detect_voice_language(user_id)
    _lang_cache[user_id] = (code, time.monotonic())
    return code


# ── Which realtime session currently OWNS a user's voice presence ────────
# user_id → session nonce, single-worker in-process (the deployment is
# deliberately one worker; same rule as the warm-reopen context cache above).
# A silent app reconnect opens the NEW session before (or moments after) the
# OLD one's finally runs, and without this the old teardown's Live Activity
# end killed the island card of the call the user was still on (review
# finding, 2026-08-16). The finally only ends the card if it is still the
# owner after a short grace — a clean client 'stop' skips the grace, because
# an ended call has no successor coming.
_voice_session_owner: dict = {}
_VOICE_LA_END_GRACE_S = 6.0
_deferred_la_tasks: set = set()


def _defer_voice_la_end(user_id: str, mission_id, nonce: str, immediate: bool) -> None:
    async def _run():
        try:
            if not immediate:
                await asyncio.sleep(_VOICE_LA_END_GRACE_S)
                if _voice_session_owner.get(user_id) != nonce:
                    return  # a reconnect superseded us — the card is theirs now
            from app.services.live_activity_service import end_voice_activities
            await asyncio.wait_for(end_voice_activities(user_id, mission_id), timeout=10.0)
        except Exception:  # noqa: BLE001
            logger.warning("[REALTIME] voice LA end failed for %s", user_id[:8])
        finally:
            if _voice_session_owner.get(user_id) == nonce:
                _voice_session_owner.pop(user_id, None)

    t = asyncio.create_task(_run())
    _deferred_la_tasks.add(t)
    t.add_done_callback(_deferred_la_tasks.discard)


class _ResponseGate:
    """ONE response at a time, by construction.

    The API rejects a `response.create` that lands while another response is
    active — "Conversation already has an active response in progress" — and
    this file used to have FOUR unguarded senders racing the VAD's own
    auto-created responses (tool continuation, inject_text, onboarding greet,
    screen-share first frame). The tool continuation is the reliable
    reproducer: a `think` turn holds the reader loop for seconds-to-tens of
    seconds, the user speaks meanwhile, semantic_vad (V1's server_vad too — it
    auto-creates by default) opens a response for the new utterance, and the
    continuation fires into it. On 2026-08-16 that error string reached a phone
    verbatim, the client treated it as terminal, and the session wedged.

    Semantics:
    - `create()` is the only sender. While a response is active it records the
      intent instead of sending — dropping it would eat the spoken half of a
      tool turn, because nothing else will ever ask for that reply.
    - `active` flips True on `response.created` (VAD- and relay-created alike)
      and optimistically inside `create()` (the send is a suspension point; the
      echo may not return before a second caller checks).
    - It flips False ONLY on `response.done`, which the API emits for every
      terminal status — completed, cancelled, failed, incomplete — so a
      deferred continuation can never strand.
    - `on_conflict()` is the self-heal for the one unwinnable ordering: our
      create raced a response whose `response.created` was still queued unread
      (the reader was blocked in a tool call). The API's error tells us it IS
      active; the intent stands and replays on the next done.

    Module-level (not a closure) so tests can pin this machine without a
    socket — same rule as `build_session_config`.
    """

    def __init__(self, send) -> None:
        self._send = send            # async fn taking the serialized frame
        self._lock = asyncio.Lock()
        self.active = False
        self.deferred = False

    async def create(self) -> None:
        async with self._lock:
            if self.active:
                self.deferred = True
                return
            self.active = True
            await self._send(json.dumps({"type": "response.create"}))

    def on_created(self) -> None:
        self.active = True

    async def on_done(self) -> bool:
        """Clear the active flag; report whether a deferred create wants to
        replay. The caller replays via `create()` AFTER finishing its own
        response.done bookkeeping, so the replayed response opens clean."""
        async with self._lock:
            self.active = False
            want, self.deferred = self.deferred, False
            return want

    def on_conflict(self) -> None:
        self.active = True
        self.deferred = True


# Benign edges of normal turn-taking: a cancel that lost the race to
# response.done, a VAD commit on an empty buffer, a truncate on an
# already-finished item. Nothing is wrong and nothing needs the user.
_BENIGN_ERROR_CODES = frozenset((
    "response_cancel_not_active",
    "input_audio_buffer_commit_empty",
    "item_truncate_audio_end_ms_too_large",
))

# The SESSION itself is dead — reconnecting is the only fix.
_FATAL_ERROR_STEMS = ("session_expired", "invalid_api_key", "auth",
                      "session_not_found", "invalid_session")


def classify_realtime_error(code: str, message: str) -> Optional[dict]:
    """Map an upstream error event to the frame the client receives — or None
    for the classes the user must never hear about.

    Raw upstream text NEVER crosses the WS boundary. On 2026-08-16
    "Conversation already has an active response in progress: resp_EDWtl…"
    reached a phone verbatim as the on-screen headline. The upstream message is
    for our logs; the client gets a `code` (to localize) and a `recoverable`
    bit (stay in the call vs tear down). Billing keeps its dedicated copy —
    the platform's OpenAI billing page is not the user's to visit, and must
    never be linked.

    Module-level so tests can pin every class without a socket.
    """
    code = code or ""
    if code in _BENIGN_ERROR_CODES:
        return None
    billing_keywords = ["insufficient_quota", "billing", "exceeded", "rate_limit",
                        "quota", "payment", "credit", "balance", "plan"]
    is_billing = (
        code in ("insufficient_quota", "billing_hard_limit_reached",
                 "rate_limit_exceeded", "budget_exceeded")
        or any(kw in message.lower() for kw in billing_keywords)
    )
    if is_billing:
        return {
            "type": "error",
            "code": "billing",
            "message": ("Voice is temporarily unavailable while we top "
                        "up capacity. Please try again shortly."),
            "billing": True,
        }
    fatal = any(stem in code for stem in _FATAL_ERROR_STEMS)
    return {
        "type": "error",
        "code": code or "voice_error",
        "recoverable": not fatal,
        "message": ("Voice is unavailable right now. Please try again."
                    if fatal else
                    "That didn't go through — I'm still listening."),
    }


# Product nouns a voice session actually says, in the Latin spellings the rest
# of the pipeline (the agent, the UI, memory extraction) needs back. Measured
# 2026-08-16 by scripts/eval_voice_transcription.py: WITHOUT this prompt,
# gpt-4o-transcribe Persianizes every one of them («گرواک‌بات», «کلاد»,
# «جمینای») — 5/9 keyword recovery; WITH it, 9/9, including the exact sentence
# the day's recording garbled into «گروه که ایلان ماسک».
_TRANSCRIPTION_BIAS_TERMS = (
    "Toup, Grok, ChatGPT, GPT, Claude, Gemini, xAI, OpenAI, Anthropic, "
    "YouTube, Spotify, WhatsApp, Telegram"
)


def transcription_prompt(language: Optional[str]) -> str:
    """The transcription side-channel's bias prompt.

    Written in the SESSION's language: a Farsi prompt tells the model the
    audio is colloquial Persian with English terms mixed in — which is what
    code-switching is, and what a hard `language` pin cannot express (a pin
    forces one language and came back with English utterances TRANSLATED into
    Farsi in the eval). Keep the term list in sync with
    scripts/eval_voice_transcription.py, which is the regression harness.
    """
    if language == "fa":
        return (
            "گفت‌وگوی کاربر با یک دستیار هوشمند. فارسی محاوره‌ای، همراه با "
            f"نام‌ها و اصطلاح‌های انگلیسی مانند {_TRANSCRIPTION_BIAS_TERMS}."
        )
    return f"A user talking to an AI assistant. May mention: {_TRANSCRIPTION_BIAS_TERMS}."


def build_session_config(
    instr: str, tools: list, voice: str, language: Optional[str] = None,
) -> dict:
    """The session.update payload. Module-level (not a closure) so tests can
    pin the exact V1/V2 shapes without a socket."""
    v2 = _v2_active()
    if v2:
        # semantic_vad scores the *words* for turn-completion probability —
        # a trailing "umm" extends the wait, a finished sentence ends it
        # quickly. This replaces the fixed 700ms silence gate, which is the
        # main thing that made v1 feel robotic (it interrupted thinking
        # pauses and always waited the full beat on clear completions).
        turn_detection: dict = {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": True,
        }
        # NO `language` pin on the V2 transcriber, deliberately. The pin was
        # whisper-1 medicine (it re-detects per utterance; the pin stopped
        # Persian audio coming back as Arabic script). On gpt-4o-transcribe
        # the same pin TRANSLATES: an English utterance in a fa-pinned
        # session came back rendered in Farsi (eval, en-only · pin=fa). The
        # session language rides the prompt instead, which biases without
        # forbidding.
        transcription = {
            "model": settings.voice_realtime_transcription_model,
            "prompt": transcription_prompt(language),
        }
    else:
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.8,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 700,
        }
        transcription = {"model": "whisper-1"}
        if language:
            transcription["language"] = language
    session: dict = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": instr,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": transcription,
                # Filters the input buffer before VAD — documented to
                # reduce false speech_started triggers. far_field:
                # the phone is a loudspeaker at arm's length, and the
                # dominant false trigger is the agent's own speaker
                # bleed (echo) reaching the mic.
                "noise_reduction": {"type": "far_field"},
                "turn_detection": turn_detection,
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
            },
        },
        "tools": tools,
        "tool_choice": "auto",
    }
    if v2:
        # Drop 20% of oldest history at once when the context ceiling hits,
        # instead of just-enough — just-enough truncation shifts the prompt
        # prefix every turn and busts the $0.40/M cache back to $32/M.
        session["truncation"] = {"type": "retention_ratio", "retention_ratio": 0.8}
    return {"type": "session.update", "session": session}


def _cap_chars(text: str, max_chars: int, keep: str = "head") -> str:
    """Trim a section to a budget on line boundaries. keep="head" preserves
    the start (memory lists: highest-priority entries come first);
    keep="tail" preserves the end (day history: newest messages last)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    lines = text.split("\n")
    header, body = lines[0], lines[1:]
    kept: list = []
    used = len(header)
    seq = body if keep == "head" else reversed(body)
    for ln in seq:
        if used + len(ln) + 1 > max_chars:
            break
        kept.append(ln)
        used += len(ln) + 1
    if keep == "tail":
        kept.reverse()
    marker = "- [context trimmed to budget]"
    return "\n".join([header, marker] + kept) if keep == "tail" else "\n".join([header] + kept + [marker])


def _usage_to_cost_cents(model: str, usage: dict) -> float:
    """True OpenAI cost (cents) of one response from its usage block.

    Audio and text price differently, and cached input tokens bill at ~1% of
    the uncached rate — the cached/uncached split is where realtime cost is
    won or lost, so it is modeled exactly, not averaged.
    """
    pricing = settings.voice_realtime_pricing_per_1m.get(model)
    if not pricing:
        pricing = settings.voice_realtime_pricing_per_1m.get("gpt-realtime-2.1", {})
    if not pricing:
        return 0.0
    in_d = usage.get("input_token_details", {}) or {}
    out_d = usage.get("output_token_details", {}) or {}
    cached_d = in_d.get("cached_tokens_details", {}) or {}
    cached_text = cached_d.get("text_tokens", 0) or 0
    cached_audio = cached_d.get("audio_tokens", 0) or 0
    text_in = max(0, (in_d.get("text_tokens", 0) or 0) - cached_text)
    audio_in = max(0, (in_d.get("audio_tokens", 0) or 0) - cached_audio)
    usd = (
        audio_in * pricing["audio_in"]
        + cached_audio * pricing["audio_in_cached"]
        + text_in * pricing["text_in"]
        + cached_text * pricing["text_in_cached"]
        + (out_d.get("audio_tokens", 0) or 0) * pricing["audio_out"]
        + (out_d.get("text_tokens", 0) or 0) * pricing["text_out"]
    ) / 1_000_000.0
    return usd * 100.0


def _voice_recorded_cost_cents(cost_cents: float):
    """The value written to llm_proxy_events.cost_cents for a voice turn.

    This was `int(round(cost_cents))`, which ROUNDS UP half the time —
    measured on production: 15 of 27 voice rows recorded HIGHER than the
    exact cost stored one line below in credit_ledger.underlying_cost_cents
    (worst +0.47¢). The R-3 rule is that recorded cost may only go DOWN or
    stay equal versus what the legacy expression produced, so the fix is
    `min(exact_4dp, legacy)`, the same bound llm_proxy._never_higher_cents
    applies — with THIS surface's legacy formula, not chat's 1¢ floor.

    The other half of int(round()) — a 0.35¢ turn recorded as 0 — is kept,
    deliberately: raising it to 0.35 would record MORE than legacy, which
    the authorization forbids. Documented residual, same as chat's
    truncation half.
    """
    from decimal import Decimal

    exact = Decimal(str(cost_cents)).quantize(Decimal("0.0001"))
    if exact <= 0:
        return Decimal("0")
    legacy = Decimal(int(round(cost_cents)))
    return min(exact, legacy)


async def _meter_voice_turn(user_id: str, model: str, usage: dict, response_id: str) -> None:
    """Charge one realtime response to the user's message credits.

    Mirrors llm_proxy._log_event: an LLMProxyEvent row for the cost
    dashboards + credit_service.try_charge keyed by the OpenAI response id,
    so relay retries / duplicate response.done events can't double-charge.
    Only called when the session runs on the PLATFORM key — BYOK users pay
    OpenAI directly and must not be charged twice.
    """
    try:
        from decimal import Decimal
        import uuid as _uuid
        from app.db.database import async_session_maker
        from app.db.models import LEDGER_CHAT_MESSAGE, LLMProxyEvent
        from app.services.credit_service import (
            credit_service, underlying_cost_to_credits, BUCKET_MESSAGE,
        )

        cost_cents = _usage_to_cost_cents(model, usage)
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        if input_tokens <= 0 and output_tokens <= 0:
            return
        credits = max(Decimal("0.1"), underlying_cost_to_credits(cost_cents))

        async with async_session_maker() as db:
            event = LLMProxyEvent(
                id=str(_uuid.uuid4()),
                user_id=user_id,
                provider="openai",
                model=model,
                endpoint="realtime_voice",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cents=_voice_recorded_cost_cents(cost_cents),
                was_fallback=False,
                latency_ms=0,
                status="ok",
            )
            db.add(event)
            result = await credit_service.try_charge(
                db, user_id, LEDGER_CHAT_MESSAGE, BUCKET_MESSAGE, credits,
                idempotency_key=f"rtv:{response_id}", event_id=event.id,
                model=model, provider="openai",
                input_tokens=input_tokens, output_tokens=output_tokens,
                underlying_cost_cents=Decimal(str(cost_cents)),
                metadata={"endpoint": "realtime_voice", "channel": "voice"},
                # This gate has never fired in production, so voice has always
                # been free. Turning it on and billing in the same deploy would
                # start charging for something users have never paid for, with
                # no idea what a real session costs. meter_only writes the row
                # with amount=0 and moves no balance — flip
                # `voice_metering_charge` once the series exists.
                meter_only=not getattr(settings, "voice_metering_charge", False),
                # The audio tokens are already spent at OpenAI by the time
                # response.done arrives; the daily cap must not zero them.
                already_incurred=True,
            )
            await db.commit()
        logger.info(
            "[REALTIME] metered response=%s user=%s model=%s tokens=%d/%d "
            "cost_cents=%.3f credits=%s balance_after=%s",
            response_id[:12], user_id[:8], model, input_tokens, output_tokens,
            cost_cents, credits, result.balance_after,
        )
    except Exception:
        # Metering must never kill a live call — but it must also never fail
        # silently into free voice; exception-level log so it pages in triage.
        logger.exception("[REALTIME] voice metering failed user=%s response=%s",
                         user_id[:8], response_id[:12])


def _maybe_meter_response(user_id: str, response: dict, using_platform_key: bool):
    """Metering gate for one response.done (v1 AND v2 — usage frames arrive
    identically on v1). Every response re-bills the accumulated context, so
    this — not the mic stream — is the billing unit. Platform-key sessions
    only (BYOK pays OpenAI). Returns the created task, or None if this
    response bills nothing."""
    if not using_platform_key:
        return None
    usage = response.get("usage") or {}
    rid = response.get("id") or ""
    if not usage or not rid:
        return None
    return asyncio.create_task(
        _meter_voice_turn(user_id, realtime_model(), usage, rid)
    )


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

    # Resolve V2 for THIS connection (global flag OR per-user allowlist) and pin
    # it for every helper this session calls (model, session config, think,
    # metering, …). One read, one source of truth for the whole connection.
    _v2_ctx.set(_resolve_v2_for_user(user_id))
    logger.info("[REALTIME] voice v2=%s for user %s", _v2_active(), user_id[:8])

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
        openai_key, is_byok = await asyncio.wait_for(
            _get_user_openai_key_ex(user_id), timeout=8.0,
        )
    except Exception:
        logger.exception("[REALTIME] OpenAI key lookup failed")
        openai_key, is_byok = None, False
    # Platform-key sessions are the ones we meter: a BYOK user pays OpenAI
    # directly, so charging credits too would double-bill. This USED to read
    # `not openai_key`, which is False for every session that survives (the
    # resolver falls back to platform keys) and True only where the socket is
    # closed below — so nothing was ever metered. See _get_user_openai_key_ex.
    using_platform_key = not is_byok
    openai_key = openai_key or settings.openai_api_key
    if not openai_key:
        await websocket.send_json({
            "type": "error",
            "message": "OpenAI API key not configured. Please set up your API key in Settings.",
        })
        await websocket.close(code=4400)
        return

    # V2 pre-flight: zero-balance gate, same semantics as the LLM proxy
    # (enforces only when credit_enforcement_enabled; shadow mode never
    # blocks). Points at TOUP billing — a platform-credit exhaustion must
    # never send users to platform.openai.com (the bundle-credit
    # misleading-copy incident).
    if _v2_active() and using_platform_key:
        try:
            from decimal import Decimal as _Dec
            from app.db.database import async_session_maker as _asm
            from app.services.credit_service import credit_service, BUCKET_MESSAGE
            if getattr(settings, "credit_enforcement_enabled", False):
                async with _asm() as _db:
                    preflight = await credit_service.check_balance(
                        _db, user_id, BUCKET_MESSAGE, _Dec("0.1"),
                    )
                if not preflight.success:
                    await websocket.send_json({
                        "type": "error",
                        "message": "You're out of Toup credits. Top up or upgrade your plan to keep talking.",
                        "billing": True,
                        "billing_url": "https://toup.ai/account?tab=billing",
                    })
                    await websocket.close(code=4402)
                    return
        except Exception:
            logger.warning("[REALTIME] credit pre-flight failed open — continuing", exc_info=True)

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
            realtime_url(),
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

    async def _agent_voice_context(now_utc: Optional[datetime] = None) -> Optional[str]:
        """Ask the tenant's own agent to assemble the instructions (G-19a).

        Returns None on ANY failure so the caller falls back to the legacy
        builder — a Realtime session must never open with no instructions,
        which is the 2026-07-31 shape.
        """
        vps = await _get_vps_info(user_id)
        if not vps:
            return None
        agent_url, agent_api_key = vps
        body = {
            "onboarding": onboarding,
            "budget_chars": settings.voice_realtime_instructions_budget_chars if _v2_active() else 0,
            "tz_name": await _get_user_tz_name(user_id),
            # One instant for both builders — the shadow must never read a
            # minute tick between the legacy build and this call as a
            # section divergence.
            "now": (now_utc or datetime.now(timezone.utc)).isoformat(),
        }
        data = await _vps_api(
            agent_url, agent_api_key, "POST",
            "/api/v1/internal/voice-context", json_body=body, timeout=20.0,
        )
        if not isinstance(data, dict):
            return None
        instr = data.get("instructions") or ""
        if not instr.strip():
            return None
        degraded = data.get("degraded") or []
        if degraded:
            logger.warning(
                "[REALTIME] agent voice-context DEGRADED user=%s legs=%s",
                str(user_id)[:8], ",".join(map(str, degraded)),
            )
        logger.info(
            "[REALTIME] agent voice-context ok user=%s chars=%d day=%s",
            str(user_id)[:8], len(instr), data.get("day_date"),
        )
        return instr


    async def _instructions_step() -> Optional[str]:
        _ctx_now = datetime.now(timezone.utc)
        legacy: Optional[str] = None
        try:
            legacy = await build_realtime_instructions(
                user_id, onboarding=onboarding, now_utc=_ctx_now
            )
            logger.info("[REALTIME] Built instructions (%d chars, onboarding=%s)", len(legacy), onboarding)
        except Exception:
            logger.exception("[REALTIME] Failed to build instructions")

        # Serve the agent's version only when explicitly enabled; otherwise
        # the agent call is a SHADOW — compared and logged, never served —
        # so the two can be shown to agree on real traffic first.
        want_agent = _agent_ctx_enabled_for(user_id)
        want_shadow = settings.voice_context_shadow and not want_agent
        if not (want_agent or want_shadow):
            return legacy

        agent_instr = None
        try:
            agent_instr = await _agent_voice_context(now_utc=_ctx_now)
        except Exception:
            logger.exception("[REALTIME] agent voice-context call failed")

        if want_shadow:
            if agent_instr and legacy:
                cmp_ = compare_voice_contexts(agent_instr, legacy)
                logger.info(
                    "[REALTIME] ctx_shadow match=%s order_match=%s "
                    "agent_chars=%d legacy_chars=%d same=%s differs=%s",
                    cmp_["match"], cmp_["order_match"],
                    len(agent_instr), len(legacy),
                    ",".join(cmp_["same"]) or "-",
                    ",".join(cmp_["differs"]) or "-",
                )
                if not cmp_["order_match"]:
                    # Sections carry the same bytes in a different sequence.
                    # Expected today (Drift D2 moves identity_anchor to the
                    # runner's position); logged every time so that an
                    # UNexpected reorder can never arrive silently.
                    logger.info(
                        "[REALTIME] ctx_shadow order agent=%s legacy=%s",
                        " > ".join(cmp_["agent_order"]),
                        " > ".join(cmp_["legacy_order"]),
                    )
            else:
                logger.info(
                    "[REALTIME] ctx_shadow unavailable agent=%s legacy=%s",
                    bool(agent_instr), bool(legacy),
                )
            return legacy

        # want_agent: serve it, but never at the cost of having nothing.
        if agent_instr:
            return agent_instr
        logger.warning(
            "[REALTIME] agent voice-context unavailable — falling back to the "
            "legacy builder for user=%s", str(user_id)[:8],
        )
        return legacy

    async def _lang_step() -> Optional[str]:
        try:
            return await resolve_voice_language(user_id)
        except Exception:
            logger.exception("[REALTIME] Failed to resolve transcription language")
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
        # V2 (hosted agent): the raw agent tools (web_search, browser, edit_file, memory_*,
        # …) execute only via the terminal-agent tunnel or a local ToolExecutor — neither of
        # which exists on platform-api for a hosted agent, so a DIRECT call to one returns
        # "your terminal agent is not connected" (_execute_tool) and the model narrates a
        # tool-connection failure mid-call. Offer only the tools that actually run on the
        # relay: `think` — which routes to the user's FULL hosted agent (every tool, skill,
        # and connector) via /api/v1/internal/agent-turn — plus the client-side `navigate_to`.
        # Onboarding tools are re-added below. This forces the model down the working path
        # instead of calling e.g. web_search directly.
        if _v2_active():
            tools = [t for t in tools if t["name"] in _REALTIME_NATIVE]
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
    _lang_t = asyncio.create_task(_lang_step())
    _bg_tasks = [_session_t, _instructions_t, _tools_t, _health_t, _lang_t]
    # Metering tasks are NOT in _bg_tasks. They used to be, and the session-end
    # `finally` cancels everything in that list — asyncio.CancelledError is a
    # BaseException, so _meter_voice_turn's `except Exception` could not catch
    # it and the final turn's cost vanished with no log at all. Every hang-up
    # (i.e. every call) silently dropped its last charge. These are drained,
    # never cancelled: the cost is already incurred at OpenAI.
    _meter_tasks: list[asyncio.Task] = []

    def _cancel_bg() -> None:
        # Tap-and-close must not leave tasks poking a cold agent for 25s.
        for _t in _bg_tasks:
            _t.cancel()

    async def _drain_meter_tasks() -> None:
        if not _meter_tasks:
            return
        try:
            # Bounded so a wedged DB cannot hold the socket open; each task
            # already logs its own failure, and gather never raises here.
            await asyncio.wait_for(
                asyncio.gather(*_meter_tasks, return_exceptions=True), timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[REALTIME] voice metering did not finish within 10s user=%s "
                "(%d task(s)) — cost may be unrecorded",
                user_id[:8], len(_meter_tasks),
            )

    db_session_id: Optional[str] = None

    # ── 5. OpenAI connect — the critical path ─────────────────
    await _status("connecting_ai")
    # Realtime API voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
    # V2 defaults to marin — the GA-new voice OpenAI documents as its most
    # natural (with cedar); clients can still request any valid voice.
    voice = "marin" if _v2_active() else "coral"
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
            "code": "connect_failed",
            "recoverable": False,
            "message": "Couldn't reach the voice engine. Tap to retry.",
        }
        if is_billing:
            # Platform-side capacity/credit problem. NEVER surface the
            # platform's own OpenAI billing page to a user — it is not their
            # account, and on iOS an external purchase link is a 3.1.1 hit.
            error_payload["billing"] = True
            error_payload["message"] = (
                "Voice is temporarily unavailable while we top up capacity. "
                "Please try again shortly."
            )
        await websocket.send_json(error_payload)
        await websocket.close(code=4502)
        return

    # ── 6. Configure session: cached-or-base config now, full context behind ──
    def _session_config(instr: str, tools: list, language: Optional[str] = None) -> dict:
        return build_session_config(instr, tools, voice, language)

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

    # Cache-only read: a first-ever connect goes out on auto-detect and the
    # hint lands with the full-context update below, rather than holding
    # `ready` behind history reads.
    _first_language = _cached_voice_language(user_id) if _lang_hint_enabled() else None

    try:
        await openai_ws.send(json.dumps(
            _session_config(_first_instructions, _first_tools, _first_language)
        ))
        logger.info(
            "[REALTIME] Session configured (voice=%s, cache_hit=%s, language=%s)",
            voice, _cache_hit, _first_language or "auto",
        )
    except Exception as e:
        _cancel_bg()
        logger.exception("[REALTIME] Failed to configure session")
        # The exception text is ours to read in the logs, never the user's.
        await websocket.send_json({
            "type": "error",
            "code": "session_setup_failed",
            "recoverable": False,
            "message": "Couldn't set up the voice session. Tap to retry.",
        })
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

    # V2: warn the client 5 minutes before OpenAI's hard 60-minute session cap
    # so the UI can offer a graceful reopen (the API has no resumption — the
    # alternative is a mid-sentence drop).
    if _v2_active():
        async def _expiry_warn() -> None:
            await asyncio.sleep(55 * 60)
            try:
                await websocket.send_json({"type": "session_expiring", "seconds_left": 300})
            except Exception:
                pass
        _bg_tasks.append(asyncio.create_task(_expiry_warn()))

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
        language: Optional[str] = _first_language
        try:
            instr = await asyncio.wait_for(asyncio.shield(_instructions_t), timeout=40.0)
            tools = await asyncio.wait_for(asyncio.shield(_tools_t), timeout=10.0)
        except Exception:
            logger.warning("[REALTIME] Context build timed out — session continues on base config")
        try:
            language = await asyncio.wait_for(asyncio.shield(_lang_t), timeout=10.0) or _first_language
        except Exception:
            logger.warning("[REALTIME] Language hint timed out — transcription stays on auto-detect")
        try:
            final_instr = instr or _first_instructions
            await openai_ws.send(json.dumps(_session_config(final_instr, tools, language)))
            if instr:
                _instr_cache[user_id] = (instr, tools, time.monotonic())
            logger.info(
                "[REALTIME] Full context applied at +%.0fms (%d chars, %d tools, language=%s)",
                (time.monotonic() - _t0) * 1000, len(final_instr), len(tools),
                language or "auto",
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
    # Media this turn started, attached to the assistant row when it persists so
    # the day thread renders a real Toup card instead of bare text. Survives the
    # function-call response (which carries no spoken text) and is consumed by
    # the spoken response that follows it.
    pending_media = None
    last_user_text = ""  # Track last user message for memory extraction
    # V1 kept the legacy "gpt-4o-realtime" label; V2 reports the real slug.
    default_turn_model = realtime_model() if _v2_active() else "gpt-4o-realtime"
    turn_model = default_turn_model  # Model used for current turn (changes if deep_think is called)

    # V2 shared state between the two relay loops: which assistant item is
    # currently voicing (for barge-in truncation) and end-of-user-speech
    # timestamps (for the ttfa_ms latency trail).
    last_audio_item: dict = {"id": None, "content_index": 0}
    speech_stopped_at: dict = {"t": 0.0}
    first_audio_of_response: dict = {"pending": False}

    # One response at a time, by construction — see _ResponseGate.
    _resp_gate = _ResponseGate(lambda payload: openai_ws.send(payload))

    # The app-reported Live Activity mission for THIS call (config frame).
    # A dict so both relay loops close over one slot.
    voice_activity_mission: dict = {"id": None}
    # This session now owns the user's voice presence; a reconnect that opens
    # a newer session takes the ownership with it, and the finally checks
    # before ending the island card (see _defer_voice_la_end).
    _session_nonce = str(uuid.uuid4())
    _voice_session_owner[user_id] = _session_nonce
    # A clean client 'stop' means the call ENDED — no successor is coming and
    # the card should die immediately, no grace.
    got_stop: dict = {"v": False}

    async def safe_response_create() -> None:
        await _resp_gate.create()

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
                await safe_response_create()

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
                    # Client's IANA zone. Voice has none in the WebRTC
                    # payload, so without this the relay falls back to a
                    # platform users row that is NULL for most users and
                    # resolves the user's day in UTC.
                    if msg.get("tz"):
                        await _apply_client_tz(user_id, msg.get("tz"))

                    # The call's Live Activity mission ("voice:<uuid>"),
                    # minted app-side. Held so the finally can END the
                    # island card when this socket dies — force-quit runs
                    # no app code, so the relay is the only party left who
                    # knows the call is over.
                    _va = msg.get("voice_activity")
                    if isinstance(_va, str) and _va.startswith("voice:"):
                        voice_activity_mission["id"] = _va[:64]

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
                                await safe_response_create()
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
                        await safe_response_create()
                        logger.info("[REALTIME] Injected text: %s", inject_content[:60])

                elif msg_type == "now_playing":
                    # The station moved on. Tell the model what is audible NOW.
                    #
                    # Radio advances are pushed over the CHAT websocket to the
                    # phone; this relay has no subscription to them, so the
                    # model's belief about what is playing was frozen at the
                    # last play_media result "for the rest of the call". The
                    # founder hit exactly that on 2026-07-31: the track had
                    # changed and the agent still named the previous song.
                    #
                    # Injected WITHOUT response.create on purpose. This is a
                    # context correction, not a turn — the model must quietly
                    # know the new title, not announce it. Announcing every
                    # station advance mid-call would be unbearable.
                    _np_title = str(msg.get("title") or "").strip()[:200]
                    if _np_title:
                        try:
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": (
                                            f"[System note, do not reply: the music moved on. "
                                            f"Now playing: {_np_title}. If asked what is playing, "
                                            f"say this.]"
                                        ),
                                    }],
                                },
                            }))
                            logger.info("[REALTIME] now_playing → %s", _np_title[:60])
                        except Exception as e:  # noqa: BLE001
                            logger.warning("[REALTIME] now_playing inject failed: %s", e)

                elif msg_type == "played":
                    # V2 barge-in truncation: the client reports how many ms of
                    # the interrupted reply it actually played. Truncating the
                    # conversation item to that point keeps the model's context
                    # matched to what the user heard — without it the model
                    # believes it said things the user never got to hear, and
                    # follow-ups go incoherent (documented OpenAI failure mode
                    # for WebSocket transports, where the client owns playback).
                    if _v2_active() and last_audio_item["id"] is not None:
                        try:
                            audio_end_ms = max(0, int(msg.get("ms", 0)))
                        except (TypeError, ValueError):
                            audio_end_ms = 0
                        if audio_end_ms > 0:
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.truncate",
                                "item_id": last_audio_item["id"],
                                "content_index": last_audio_item["content_index"],
                                "audio_end_ms": audio_end_ms,
                            }))
                            logger.info(
                                "[REALTIME] truncated %s at %dms (barge-in)",
                                last_audio_item["id"], audio_end_ms,
                            )

                elif msg_type == "interrupt":
                    # Explicit client barge-in: the user tapped the orb (or the
                    # client cut playback for its own reasons). The VAD can only
                    # cancel what it can hear, and a tap is silent — without
                    # this the model keeps generating a reply nobody is playing,
                    # and the next turn queues behind it. response.cancel on an
                    # already-finished response answers response_cancel_not_active,
                    # which the error branch swallows as benign.
                    if _resp_gate.active:
                        try:
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception:
                            pass

                elif msg_type == "stop":
                    got_stop["v"] = True
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("[REALTIME] client_to_openai error: %s", e)

    async def openai_to_client():
        """Relay OpenAI Realtime API events → browser."""
        nonlocal response_text_accum, db_session_id, last_user_text, turn_model
        nonlocal pending_media
        try:
            async for raw_msg in openai_ws:
                event = json.loads(raw_msg)
                etype = event.get("type", "")

                # ── Audio response chunks → browser (GA: response.output_audio.delta) ──
                if etype == "response.output_audio.delta":
                    if _v2_active():
                        if event.get("item_id"):
                            last_audio_item["id"] = event.get("item_id")
                            last_audio_item["content_index"] = event.get("content_index", 0) or 0
                        if first_audio_of_response["pending"]:
                            first_audio_of_response["pending"] = False
                            if speech_stopped_at["t"]:
                                # End-of-user-speech → first audio byte out: the
                                # number the latency acceptance criteria are
                                # measured against (target p50 ≤ 1000ms).
                                logger.info(
                                    "[REALTIME] ttfa_ms=%.0f",
                                    (time.monotonic() - speech_stopped_at["t"]) * 1000,
                                )
                    await websocket.send_json({
                        "type": "audio_delta",
                        "data": event.get("delta", ""),
                        # Which assistant item this audio belongs to. The
                        # client's played-ms clock is per-TURN; truncation is
                        # per-ITEM (a tool turn voices two items: pre-amble,
                        # then answer). Naming the item lets the client report
                        # a barge-in position the truncate can actually use.
                        "item": event.get("item_id"),
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
                    _replay = await _resp_gate.on_done()
                    response = event.get("response", {})

                    _meter_t = _maybe_meter_response(user_id, response, using_platform_key)
                    if _meter_t is not None:
                        _meter_tasks.append(_meter_t)
                    # Extract final text from output items
                    full_text = response_text_accum
                    for item in response.get("output", []):
                        if item.get("type") == "message":
                            for content in item.get("content", []):
                                if content.get("type") in ("audio", "output_audio") and content.get("transcript"):
                                    full_text = content["transcript"]

                    if full_text:
                        # Alias the model id before it crosses the WS boundary —
                        # voice is the always-on white-label channel; the raw
                        # turn_model (incl. cross-provider ids after a think turn)
                        # must not reach the client (docs/security/audit-2026.md
                        # MI-1/MI-2). Flag-gated; DB persist is scrubbed on read.
                        _rt_model = turn_model
                        if settings.security_leak_filter and _rt_model:
                            from app.services.model_alias import public_model_label
                            _rt_model = public_model_label(_rt_model)
                        await websocket.send_json({
                            "type": "response_done",
                            "text": full_text,
                            "model": _rt_model,
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
                                await _save_voice_messages(
                                    user_id, db_session_id, "", full_text,
                                    model=turn_model, media=pending_media,
                                )
                            except Exception as e:
                                logger.exception("[REALTIME] Failed to save assistant message")
                        # Consumed. Cleared HERE and not with the other per-turn
                        # resets below, because a tool turn fires response.done
                        # TWICE: once for the function-call response (no spoken
                        # text, so no persist) and again for the spoken reply
                        # that follows it. Clearing on the first would drop the
                        # card before the row that should carry it is written.
                        pending_media = None

                        # Auto-extract memories from this conversation turn (background)
                        if last_user_text and full_text and settings.auto_extract_memories:
                            asyncio.create_task(
                                _extract_voice_memories(user_id, last_user_text, full_text)
                            )
                            last_user_text = ""  # Reset so we don't re-extract

                    response_text_accum = ""
                    turn_model = default_turn_model  # Reset for next turn

                    await websocket.send_json({"type": "state", "state": "listening"})

                    # A continuation that was deferred because THIS response was
                    # active runs now. (If a VAD response was also queued behind
                    # us, the create below collides, the error handler re-queues
                    # it, and the next response.done replays it — self-healing
                    # by construction, invisible to the user.)
                    if _replay:
                        await safe_response_create()

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
                    if _v2_active():
                        speech_stopped_at["t"] = time.monotonic()
                        first_audio_of_response["pending"] = True
                    await websocket.send_json({"type": "state", "state": "thinking"})

                # ── Response started → speaking ──
                elif etype == "response.created":
                    _resp_gate.on_created()
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
                        # Structured card for THIS call's completed frame (play_media
                        # only). Deliberately separate from `pending_media`, which is
                        # the persist-side slot cleared on a later response.done — a
                        # failed play must not ship a previous play's card.
                        _frame_media = None
                        # Legacy coarse flag (kept for older app builds that only
                        # read state) + the discrete lifecycle event the tool UI uses.
                        await websocket.send_json({"type": "state", "state": "tool_use"})
                        _tc_title, _tc_detail = _tool_activity(func_name, arguments)
                        await websocket.send_json({
                            "type": "tool_call.started",
                            "call_id": call_id,
                            "name": func_name,
                            "title": _tc_title,
                            "detail": _tc_detail,
                        })

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

                        # ── Play media: straight to the agent's resolver ──
                        elif func_name == "play_media":
                            result, _played = await _play_media_direct(
                                user_id, str(arguments.get("query", "")),
                                variety=bool(arguments.get("variety")))
                            # Carried to this turn's assistant persist so the
                            # thread gets the same Toup media card a chat play
                            # produces. Cleared after the persist below.
                            if _played:
                                pending_media = _played
                                _frame_media = _played

                        # ── Think: delegate reasoning to best model ──
                        elif func_name == "think":
                            task = arguments.get("task", "")
                            _relay = _InnerToolRelay(websocket, call_id)
                            result, turn_model_used = await _think(
                                user_id, task, db_session_id, relay=_relay)
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

                        # Discrete completion event → client findings card.
                        _res_str = result if isinstance(result, str) else str(result)
                        await websocket.send_json(_tool_completed_frame(
                            call_id, func_name, _res_str, media=_frame_media))

                        # Send result back to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": result,
                            },
                        }))

                        # Ask for the spoken reply. By the time a tool result is
                        # ready, THIS response (the function-call one) is already
                        # done and its response.done is queued unread behind us —
                        # and if the user spoke during the tool run, the VAD has
                        # opened a response of its own. safe_response_create
                        # defers the continuation past whatever is active instead
                        # of colliding with it.
                        await safe_response_create()

                # ── Errors from OpenAI ──
                # THREE classes, and raw API text never crosses the WS boundary.
                # On 2026-08-16 "Conversation already has an active response in
                # progress: resp_EDWtl…" reached a phone verbatim, the client
                # treated it as terminal, and the session wedged with the mic
                # off. The message string is for our logs; the client gets a
                # `code` (to localize) and a `recoverable` bit (to decide
                # whether to stay in the call).
                elif etype == "error":
                    error_obj = event.get("error", {})
                    error_msg = error_obj.get("message", "Unknown OpenAI error")
                    error_code = error_obj.get("code", "") or ""
                    logger.error("[REALTIME] OpenAI error: %s (code=%s)", error_msg, error_code)

                    # Conversation-state race the relay itself resolves: our
                    # deferred continuation collided with a VAD-created
                    # response. Re-queue it for the next response.done and tell
                    # the user NOTHING — the session is healthy.
                    if error_code == "conversation_already_has_active_response":
                        _resp_gate.on_conflict()
                        continue

                    payload = classify_realtime_error(error_code, error_msg)
                    if payload is None:
                        continue
                    await websocket.send_json(payload)

                # ── Session events (log only) ──
                elif etype in ("session.created", "session.updated"):
                    logger.info("[REALTIME] %s", etype)

        except websockets.ConnectionClosed as e:
            # Same boundary rule as classify_realtime_error: the close reason
            # is upstream text, and it goes to our logs, never to the screen.
            logger.warning("[REALTIME] OpenAI WS closed: code=%s reason=%s", e.code, e.reason)
            reason = str(e.reason or e).lower()
            is_billing = any(kw in reason for kw in ["quota", "billing", "rate_limit", "credit", "balance", "exceeded"])
            error_payload: dict = {
                "type": "error",
                "code": "connection_closed",
                "recoverable": False,
                "message": "The voice connection dropped. Tap to reconnect.",
            }
            if is_billing or e.code in (4002, 4003):
                error_payload["billing"] = True
                error_payload["code"] = "billing"
                error_payload["message"] = (
                    "Voice is temporarily unavailable while we top up "
                    "capacity. Please try again shortly."
                )
            try:
                await websocket.send_json(error_payload)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[REALTIME] openai_to_client error: %s", e)
            try:
                await websocket.send_json({
                    "type": "error",
                    "code": "relay_error",
                    "recoverable": False,
                    "message": "Voice hit a snag. Tap to reconnect.",
                })
            except Exception:
                pass

    # ── Run both relay tasks ──────────────────────────────────
    # FIRST_COMPLETED, not gather: on a force-quit, client_to_openai returns
    # immediately (WebSocketDisconnect) but openai_to_client blocks on
    # `async for raw_msg in openai_ws` — and an idle Listening session has no
    # OpenAI event coming to unblock it, so a gather() held the finally (and
    # with it the mic-session close and the Live Activity end) until OpenAI's
    # own idle timeout. The recording's island claimed "Listening…" for as
    # long as anyone watched; teardown must start the moment either side dies.
    _t_client = asyncio.create_task(client_to_openai())
    _t_openai = asyncio.create_task(openai_to_client())
    try:
        await asyncio.wait({_t_client, _t_openai}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # The surviving sibling gets a short grace before the cancel: on a
        # clean 'stop' the openai side may be mid-way through persisting the
        # final assistant message, and an immediate cancel() lost that last
        # row from history (review finding). Bounded, so an idle blocked
        # reader still cannot hold teardown hostage.
        _pending = [t for t in (_t_client, _t_openai) if not t.done()]
        if _pending:
            await asyncio.wait(_pending, timeout=3.0)
        for _t in (_t_client, _t_openai):
            if not _t.done():
                _t.cancel()
                try:
                    await _t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        logger.info("[REALTIME] Session ended for user %s", user_id[:8])
        _cancel_bg()
        # Before the sockets go: these charges are for audio OpenAI has already
        # billed us for. Cancelling them (the old behaviour) lost the cost.
        await _drain_meter_tasks()
        try:
            await openai_ws.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        # The island/Lock-Screen card must not outlive the CALL — but a
        # silent reconnect is the same call on a new socket, so the end runs
        # detached, after a grace, and only if no newer session has taken
        # ownership of the user's voice presence. A clean 'stop' skips the
        # grace: the user ended the call, and no successor is coming.
        _defer_voice_la_end(
            user_id, voice_activity_mission["id"], _session_nonce, got_stop["v"],
        )
