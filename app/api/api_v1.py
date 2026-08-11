"""
Public API v1 — Programmatic access to Toup via API keys.

Endpoints:
  POST /api/v1/chat           — Send a message and get a response
  POST /api/v1/chat/stream    — Send a message and stream response (SSE)
  GET  /api/v1/sessions       — List sessions
  GET  /api/v1/sessions/{id}  — Get session messages
  POST /api/v1/memories/search — Search memories
  GET  /api/v1/skills         — List loaded skills

  POST /api/v1/keys           — Create a new API key
  GET  /api/v1/keys           — List your API keys
  DELETE /api/v1/keys/{id}    — Revoke an API key

Authentication:
  Header: Authorization: Bearer hx_...
  API keys are prefixed with "hx_" and hashed with SHA-256 for storage.

Rate limiting:
  Per-key configurable, default 60 requests/minute.
  Tracked in-memory with sliding window.
"""

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db, async_session_maker
from app.db.models import ApiKey, Conversation, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Public API v1"])

# References set at startup
_agent_runner = None
_skill_loader = None


def set_api_v1_refs(agent_runner, skill_loader=None):
    """Set references to the agent runner and skill loader (called from main.py lifespan)."""
    global _agent_runner, _skill_loader
    _agent_runner = agent_runner
    _skill_loader = skill_loader


# ======================================================================
# Rate limiter (in-memory sliding window)
# ======================================================================

_rate_windows: Dict[str, List[float]] = defaultdict(list)
_RATE_WINDOW_SECONDS = 60


def _check_rate_limit(key_id: str, limit: int) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    window = _rate_windows[key_id]

    # Remove timestamps outside the window
    cutoff = now - _RATE_WINDOW_SECONDS
    _rate_windows[key_id] = [t for t in window if t > cutoff]
    window = _rate_windows[key_id]

    if len(window) >= limit:
        return False

    window.append(now)
    return True


# ======================================================================
# Auth dependency
# ======================================================================

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def get_api_key_user(request: Request, db: AsyncSession = Depends(get_db)) -> str:
    """
    Dependency: Extract API key from Authorization header, validate, rate-limit.
    Returns user_id.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer hx_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Format: Authorization: Bearer hx_...",
        )

    raw_key = auth.removeprefix("Bearer ").strip()
    key_hash = _hash_key(raw_key)

    result = await db.execute(
        select(ApiKey).where(
            and_(
                ApiKey.key_hash == key_hash,
                ApiKey.is_active == True,
            )
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")

    # Rate limit
    if not _check_rate_limit(api_key.id, api_key.rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({api_key.rate_limit}/min)",
        )

    # Update last_used
    api_key.last_used_at = datetime.utcnow()
    await db.commit()

    return api_key.user_id


# ======================================================================
# Request/Response schemas
# ======================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: Optional[str] = None
    model: Optional[str] = None
    # When False, run the full agent (tools/skills/connectors + session context)
    # but do NOT persist this turn to the session. Used by the realtime-voice
    # `think` path: the voice handler already persists the spoken user/assistant
    # turn, so persisting here too would duplicate the day-chat. Default True
    # keeps existing API-chat behavior unchanged.
    save: bool = True


class ChatResponse(BaseModel):
    text: str
    session_id: str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    model: str = ""
    tool_calls: int = 0
    processing_time_ms: int = 0


class SessionSummary(BaseModel):
    id: str
    channel: str
    is_active: bool
    message_count: int
    total_tokens: int
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str
    model_used: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    brain_type: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(default=60, ge=1, le=1000)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class KeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    rate_limit: int
    is_active: bool
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str


class CreateKeyResponse(BaseModel):
    key: str  # Only returned on creation
    id: str
    name: str
    key_prefix: str


# ======================================================================
# Chat endpoints
# ======================================================================

@router.post("/chat", response_model=ChatResponse)
async def api_chat(
    req: ChatRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Send a message to the agent and get a response."""
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        response = await _agent_runner.run(
            user_message=req.message,
            user_id=user_id,
            session_id=req.session_id,
            channel="api",
            model_override=req.model,
            save_user_message=req.save,
            save_assistant_message=req.save,
        )

        # Alias the model id like the SSE sibling + messages endpoint
        # (docs/security/audit-2026.md MI-2, re-audit found this non-stream path).
        _resp_model = response.model
        if settings.security_leak_filter and _resp_model:
            from app.services.model_alias import public_model_label
            _resp_model = public_model_label(_resp_model)
        return ChatResponse(
            text=response.text,
            session_id=response.session_id,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tokens_total=response.tokens_total,
            model=_resp_model,
            tool_calls=len(response.tool_calls),
            processing_time_ms=response.processing_time_ms,
        )
    except Exception as e:
        logger.exception(f"API chat error for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


class PlayMediaRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    channel: str = Field(default="youtube", max_length=32)
    # Open-ended ask (artist/genre/vibe): pick a varied starting track instead
    # of the pinned top hit. See _tool_play_media's variety branch.
    variety: bool = Field(default=False)


@router.post("/internal/play-media", include_in_schema=False)
async def internal_play_media(req: PlayMediaRequest, request: Request):
    """Resolve a track and start it. NO LLM, NO agent turn.

    Why this exists. A voice "play me X" used to be routed through `think`,
    which runs a FULL agent turn on this container: load the session, the agent
    config and the day's context, embed a memory query, assemble ~26k tokens of
    prompt and the entire tool array, call the model so it emits one tool call
    with one string argument, run the tool, then call the model AGAIN to say one
    short sentence. Measured on prod for the founder on 2026-07-31: 13.0s to the
    media_play frame, of which 79% was inside this process and only 3.5s was
    inference — the two model calls produced 66 output tokens between them.

    The agent contributes exactly two things to a play: query → video_id, and
    pushing the frame to the phone. That is what this route does, and nothing
    else. Anything genuinely ambiguous ("something like that song from the
    film") still belongs on `think`, which keeps every skill and connector.

    Returns the resolved title so the caller can say what it started and answer
    "what's playing?" without another round trip.
    """
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")
    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    tools = getattr(_agent_runner, "tools", None)
    if tools is None:
        raise HTTPException(status_code=503, detail="Tool executor not available")

    # `_tool_play_media` reads BOTH of these off the executor: the user id to
    # broadcast to, and the channel to seed the radio station on. 'app' — not
    # 'voice' — because RADIO_ALLOWED_CHANNELS has no voice member, so seeding
    # on 'voice' would resolve the track and then silently decline to build a
    # station, and the music would stop after one song.
    # Per-call state is ContextVar-backed and per-asyncio-task (see
    # ToolExecutor's class docstring), and this handler owns its own task — so
    # setting it here cannot leak into a concurrent agent run, and there is
    # nothing to restore. Use the setters; `_current_*` are read-only properties.
    tools.set_user_id(user_id)
    tools.set_channel("app")
    result = await tools._tool_play_media({
        "query": req.query, "channel": req.channel, "variety": req.variety,
    })

    text = str(result or "")
    if text.upper().startswith("ERROR") or text.startswith("Could not find"):
        # Surface the real reason. The caller turns this into something the user
        # can act on; it must never become "I can't play music".
        raise HTTPException(status_code=502, detail=text[:300])

    last = getattr(tools, "_last_media", None) or {}
    # Consume it. `_last_media` is a one-slot mailbox that AgentRunner._save_messages
    # captures-and-clears when it persists a turn — so a value left here by a
    # voice play gets stapled onto the NEXT chat turn's assistant message, which
    # then renders a media card for a song nobody mentioned. This path never runs
    # an agent turn, so nothing else will ever drain it.
    try:
        tools._last_media = None
    except Exception:
        pass
    return {
        "ok": True,
        "title": last.get("title") or "",
        "video_id": last.get("video_id") or "",
        "thumbnail_url": (
            f"https://i.ytimg.com/vi/{last.get('video_id')}/hqdefault.jpg"
            if last.get("video_id") else ""
        ),
        "detail": text[:300],
    }


@router.post("/internal/agent-turn", response_model=ChatResponse, include_in_schema=False)
async def internal_agent_turn(req: ChatRequest, request: Request):
    """Internal-only: run a FULL agent turn (every tool/skill/connector) for the
    realtime-voice `think` path.

    The voice relay runs on platform-api, where the in-process agent_runner is
    absent — so voice reasoning has to hop to the user's OWN agent container to
    get the identical toolset chat has (web, browser, files, memory, and every
    connected MCP connector + skill). This endpoint is that hop.

    It is deliberately NOT part of the public API v1: it authenticates with the
    tenant's X-Agent-Key (the same primitive soul-sync / refresh-tools use, not
    a user `hx_` key), resolves to settings.user_id, and is invisible (404) on
    the platform process so only agent containers expose it.

    `save` is False when voice calls it: the realtime handler already persists the
    spoken user/assistant turn, so persisting here too would duplicate the
    day-chat. Session history (context) is still read from session_id regardless.
    """
    # Only meaningful on tenant agent containers. On the platform, 404 so the
    # endpoint is invisible to probers (mirrors agent.py:refresh-tools).
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    # X-Agent-Key auth — same primitive as agent.py:437.
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")

    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    try:
        response = await _agent_runner.run(
            user_message=req.message,
            user_id=user_id,
            session_id=req.session_id,
            model_override=req.model,
            channel="voice",
            save_user_message=req.save,
            save_assistant_message=req.save,
            # `save=False` means the realtime handler is persisting this turn
            # itself — so this run is not the turn of record, and must not mine
            # memories either. Persistence and post-processing are independent
            # gates in run(), and only the first was being set: a voice turn
            # therefore wrote memories with NO row in `messages`, from a
            # `user_message` that is not the user's utterance at all but the
            # `task` string the realtime model synthesised for its `think`
            # tool. That is exactly how five encyclopedia entries about 409A
            # valuations — restated from the agent's own spoken answer —
            # landed in the founder's brain on 2026-07-30.
            #
            # Voice memories still get extracted, once, on the platform side
            # (ws_realtime._extract_voice_memories) from the real transcript.
            disable_post_processing=not req.save,
        )
        _resp_model = response.model
        if settings.security_leak_filter and _resp_model:
            from app.services.model_alias import public_model_label
            _resp_model = public_model_label(_resp_model)
        return ChatResponse(
            text=response.text,
            session_id=response.session_id,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tokens_total=response.tokens_total,
            model=_resp_model,
            tool_calls=len(response.tool_calls),
            processing_time_ms=response.processing_time_ms,
        )
    except Exception as e:
        logger.exception(f"Internal agent-turn error for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


class VoiceContextRequest(BaseModel):
    onboarding: bool = Field(default=False)
    # 0 = no trimming. The relay passes its own
    # voice_realtime_instructions_budget_chars when V2 is on, so the
    # budget stays a caller decision — an agent container has no opinion
    # about what the Realtime API's instruction ceiling is today.
    budget_chars: int = Field(default=0, ge=0, le=1_000_000)
    # IANA zone from the client. None → the tenant's User.timezone, which
    # is what every other day-chat caller falls back to.
    tz_name: Optional[str] = Field(default=None, max_length=64)
    # The relay's clock instant. The W-6 shadow hashes sections on both
    # sides; without a shared instant a minute tick between the legacy
    # build and this call reads as a Voice Conversation Mode divergence.
    now: Optional[datetime] = None


class VoiceContextResponse(BaseModel):
    instructions: str
    day_date: Optional[str] = None
    sections: Dict[str, str] = Field(default_factory=dict)
    degraded: List[str] = Field(default_factory=list)


@router.post("/internal/voice-context", response_model=VoiceContextResponse,
             include_in_schema=False)
async def internal_voice_context(req: VoiceContextRequest, request: Request):
    """Internal-only: assemble the Realtime session's instructions HERE.

    Same hop, same authentication and same visibility rules as
    `/internal/agent-turn` above — voice reasoning already comes to this
    container for its tools; this brings the PROMPT here too, so the
    persona voice speaks from is the persona text chat speaks from, read
    from the tenant DB rather than from the platform's leftover copy of
    `identities` (see app/agent/voice_context.py for the full drift list
    and the #488 day-selection argument).

    PR-A ships it dark: nothing calls this yet. The relay swap is PR-B,
    behind a canary, and only then does ws_realtime's own builder go.
    """
    # Only meaningful on tenant agent containers. On the platform, 404 so the
    # endpoint is invisible to probers (mirrors agent.py:refresh-tools).
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    # X-Agent-Key auth — same primitive as agent.py:437.
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")

    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    try:
        from datetime import timezone as _tz

        from app.agent.voice_context import build_voice_context

        # Pydantic parses "now" from ISO; a bare timestamp is treated as
        # UTC so `.astimezone()` in the day-labelling leg cannot shift it
        # by the server's zone.
        _now = req.now
        if _now is not None and _now.tzinfo is None:
            _now = _now.replace(tzinfo=_tz.utc)

        async with async_session_maker() as db:
            ctx = await build_voice_context(
                db, user_id,
                onboarding=req.onboarding,
                budget_chars=req.budget_chars,
                tz_name=req.tz_name,
                now_utc=_now,
            )
            # Genuinely read-only since the day leg moved to the relay's
            # newest-day selection (W-6 parity): nothing here INSERTs any
            # more, so there is nothing to commit and nothing to roll back.
        return VoiceContextResponse(
            instructions=ctx.instructions,
            day_date=ctx.day_date,
            sections=ctx.sections,
            degraded=ctx.degraded,
        )
    except Exception as e:
        logger.exception(f"Internal voice-context error for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


# ── Voice inner-tool stream ───────────────────────────────────────────
# Live visibility for the realtime-voice `think` path: which tool is running,
# the exact query, and the sources the answer is grounded in — emitted WHILE
# the turn runs instead of being discarded (the blocking sibling above returns
# only len(tool_calls), an integer).
#
# Caps live here AND again in the relay (ws_realtime.py); neither side trusts
# the other's caps, so version skew in either direction can never flood the
# phone's audio socket.
_VS_QUEUE_MAX       = 512     # frames buffered between runner and generator
_VS_MAX_EVENTS      = 120     # tool.*/status frames per turn; `done` is exempt
_VS_HEARTBEAT_S     = 10.0
_VS_FRAME_BYTES_MAX = 4096
_VS_SRC_MAX         = 6
_VS_SRC_TITLE_MAX   = 120
_VS_SRC_URL_MAX     = 300
_VS_SRC_DOMAIN_MAX  = 64
_VS_ARG_VALUE_MAX   = 200
_VS_ARGS_BYTES_MAX  = 512
_VS_PREVIEW_MAX     = 240

# ALLOW-LIST, not a deny-list. A tool absent from this map ships with args={},
# which makes exec(command=…), write_file(content=…) and every connector body
# structurally unreachable rather than merely filtered.
_VS_ARG_ALLOW: Dict[str, tuple] = {
    "web_search":         ("query",),
    "extension_search":   ("query",),
    "extension_research": ("query",),
    "web_fetch":          ("url",),
    "extension_read":     ("url",),
    "browser":            ("url", "query"),
    "memory_search":      ("query",),
    "recall_day":         ("date",),
}
_VS_PREVIEW_ALLOW     = {"web_fetch", "extension_read"}
_VS_SOURCE_LIST_TOOLS = {"web_search", "extension_search", "extension_research"}
_VS_SOURCE_ONE_TOOLS  = {"web_fetch", "extension_read", "browser"}

_VS_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_VS_NUM_RE  = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")
_VS_URL_RE  = re.compile(r"^\s+(https?://\S+)\s*$")


def _vs_defence(s: str) -> str:
    """Strip the injection-fence envelope wrapped around every
    external-content tool result. Without this, `ok` is ALWAYS True (the
    string starts with '<external_content') and any preview is pure
    boilerplate rather than content."""
    if not s or not s.startswith("<external_content"):
        return s or ""
    i = s.find("\n---\n")
    j = s.rfind("\n---\n")
    return s[i + 5:j] if (i != -1 and j > i) else s


def _vs_clean(s: str) -> str:
    # Control chars stripped: search titles are attacker-influenced text (that
    # is exactly why the fence exists) and must not carry newlines or escapes
    # into a UI. Provider-name scrubbing is deliberately NOT applied to
    # external content — it would rewrite a legitimate result titled
    # "OpenAI ships X" into nonsense. The arg/preview allow-list is what closes
    # the stack-disclosure risk, structurally.
    return _VS_CTRL_RE.sub(" ", s or "").strip()


def _vs_source(title: str, url: str) -> dict:
    try:
        netloc = urlparse(url).netloc
    except Exception:
        netloc = ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return {
        "title":  _vs_clean(title)[:_VS_SRC_TITLE_MAX],
        "url":    url[:_VS_SRC_URL_MAX],
        "domain": netloc[:_VS_SRC_DOMAIN_MAX],
    }


def _vs_sources(name: str, tool_input: dict, result: str) -> list:
    """Structured sources from the FULL, de-fenced tool result.

    Every web_search backend emits the same block:
        N. Title
           https://url
           description…
    A parse miss degrades to [], never to an error."""
    body = _vs_defence(result)
    out: list = []
    if name in _VS_SOURCE_LIST_TOOLS:
        title = ""
        for ln in body.splitlines():
            m = _VS_NUM_RE.match(ln)
            if m:
                title = m.group(1)
                continue
            u = _VS_URL_RE.match(ln)
            if u and title:
                out.append(_vs_source(title, u.group(1)))
                title = ""
                if len(out) >= _VS_SRC_MAX:
                    break
    elif name in _VS_SOURCE_ONE_TOOLS:
        url = str((tool_input or {}).get("url", ""))
        head = ""
        for ln in body.splitlines()[:5]:
            if ln.startswith("# "):
                head = ln[2:].strip()
                break
        if url:
            out.append(_vs_source(head or url, url))
    return out


def _vs_args(name: str, tool_input: dict) -> dict:
    keys = _VS_ARG_ALLOW.get(name)
    if not keys or not isinstance(tool_input, dict):
        return {}
    out, budget = {}, _VS_ARGS_BYTES_MAX
    for k in keys:
        v = tool_input.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        v = _vs_clean(v)[:_VS_ARG_VALUE_MAX]
        if len(v) > budget:
            break
        out[k] = v
        budget -= len(v)
    return out


def _vs_sse(frame: dict) -> str:
    blob = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
    if len(blob) > _VS_FRAME_BYTES_MAX and frame.get("type") != "done":
        frame.pop("sources", None)
        frame.pop("preview", None)
        blob = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
    return f"data: {blob}\n\n"


@router.post("/internal/agent-turn/stream", include_in_schema=False)
async def internal_agent_turn_stream(req: ChatRequest, request: Request):
    """Streaming sibling of /internal/agent-turn.

    Same auth, same `save` semantics, same terminal payload — the only
    difference is that inner tool activity is emitted live and the
    ChatResponse body arrives as the final `done` event. The blocking
    endpoint above is left byte-identical, so rollback is "stop calling
    this route".

    NOTE: Starlette commits `http.response.start` BEFORE the generator runs,
    so nothing inside generate() can produce a non-200. Every fallible setup
    step therefore happens here, in the handler body.
    """
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")
    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    q: "asyncio.Queue" = asyncio.Queue(maxsize=_VS_QUEUE_MAX)
    budget = {"n": 0, "dropped": 0}
    cancelled = {"v": False}

    def _put(frame: dict) -> None:
        # NEVER awaits and NEVER raises: the producer is the agent loop, and a
        # slow or dead consumer must not be able to stall or kill a turn.
        if budget["n"] >= _VS_MAX_EVENTS:
            return
        budget["n"] += 1
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            budget["dropped"] += 1

    async def on_status(stage: str) -> None:
        try:
            if stage == "thinking":
                _put({"type": "status", "stage": "thinking"})
        except Exception:  # noqa: BLE001
            logger.debug("[VSTREAM] on_status sink failed", exc_info=True)

    async def on_tool_start(tool_name: str) -> None:
        # Earliest possible beat: fires at the LLM's tool_use_start, BEFORE the
        # arguments have finished streaming, so there is no call_id and no
        # input here. It can flip the orb to tool_use a second or two sooner;
        # it can NOT open a row.
        try:
            _put({"type": "tool.intent", "name": str(tool_name)[:64]})
        except Exception:  # noqa: BLE001
            logger.debug("[VSTREAM] on_tool_start sink failed", exc_info=True)

    async def on_tool_event(ev: Dict[str, Any]) -> None:
        try:
            name = str(ev.get("name", ""))[:64]
            cid = str(ev.get("call_id", ""))[:64]
            inp = ev.get("input") or {}
            if ev.get("phase") == "start":
                _put({"type": "tool.start", "call_id": cid, "name": name,
                      "args": _vs_args(name, inp),
                      "started_ms": int(ev.get("started_ms") or 0)})
            else:
                raw = ev.get("result") or ""
                body = _vs_defence(raw)
                frame = {
                    "type": "tool.end", "call_id": cid, "name": name,
                    # `ok` MUST come off the DE-FENCED string: post-fence every
                    # external result starts with '<external_content', so a
                    # naive startswith("ERROR") test reports a failed search ok.
                    "ok": not body.strip().upper().startswith("ERROR"),
                    "elapsed_ms": int(ev.get("elapsed_ms") or 0),
                    "sources": _vs_sources(name, inp, raw),
                }
                if name in _VS_PREVIEW_ALLOW:
                    frame["preview"] = _vs_clean(body)[:_VS_PREVIEW_MAX]
                _put(frame)
        except Exception:  # noqa: BLE001
            logger.debug("[VSTREAM] on_tool_event sink failed", exc_info=True)

    async def _run_wrapped():
        try:
            return await _agent_runner.run(
                user_message=req.message,
                user_id=user_id,
                session_id=req.session_id,
                model_override=req.model,
                channel="voice",
                save_user_message=req.save,
                save_assistant_message=req.save,
                # Same reasoning as the blocking sibling above — this is the
                # path voice actually takes when tool events are enabled, so
                # omitting it here would leave the defect fully live.
                disable_post_processing=not req.save,
                on_status=on_status,
                on_tool_start=on_tool_start,
                on_tool_event=on_tool_event,
                cancel_check=lambda: cancelled["v"],
                # on_text_chunk deliberately NOT passed: voice renders no token
                # deltas, and omitting it takes the frame count from thousands
                # per turn to 2-40, which makes backpressure a non-problem.
            )
        finally:
            try:
                q.put_nowait(None)          # terminal sentinel
            except asyncio.QueueFull:
                pass                        # drain loop's task.done() check covers it

    async def generate():
        task = asyncio.create_task(_run_wrapped())
        try:
            yield _vs_sse({"type": "ready"})
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=_VS_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    if task.done() and q.empty():
                        break
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield _vs_sse(item)

            try:
                response = await task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[VSTREAM] agent-turn stream failed for %s", user_id)
                yield _vs_sse({"type": "error", "code": type(e).__name__})
                return

            _m = response.model
            if settings.security_leak_filter and _m:
                from app.services.model_alias import public_model_label
                _m = public_model_label(_m)
            yield _vs_sse({
                "type": "done",
                "text": response.text,
                "session_id": response.session_id,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "tokens_total": response.tokens_total,
                "model": _m,
                "tool_calls": len(response.tool_calls),
                "processing_time_ms": response.processing_time_ms,
            })
            if budget["dropped"]:
                logger.warning("[VSTREAM] dropped %d frames (queue full)", budget["dropped"])
        finally:
            # Client gone (Starlette cancels the generator). Cooperative cancel
            # first — the runner polls cancel_check — then hard cancel.
            if not task.done():
                cancelled["v"] = True
                asyncio.get_running_loop().call_later(1.5, task.cancel)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/stream")
async def api_chat_stream(
    req: ChatRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Send a message and stream the response as Server-Sent Events."""
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    async def generate():
        try:
            async def on_text_chunk(chunk: str):
                data = json.dumps({"type": "text_chunk", "text": chunk})
                yield f"data: {data}\n\n"

            async def on_tool_start(tool_name: str):
                data = json.dumps({"type": "tool_start", "tool": tool_name})
                yield f"data: {data}\n\n"

            async def on_tool_end(tool_name: str, summary: str, tool_input: dict = None):
                data = json.dumps({"type": "tool_end", "tool": tool_name, "summary": summary})
                yield f"data: {data}\n\n"

            # We need to collect chunks for SSE because callbacks are coroutines
            chunks: list[str] = []
            tool_events: list[dict] = []

            async def collect_text(chunk: str):
                chunks.append(chunk)

            async def collect_tool_start(tool_name: str):
                tool_events.append({"type": "tool_start", "tool": tool_name})

            async def collect_tool_end(tool_name: str, summary: str):
                tool_events.append({"type": "tool_end", "tool": tool_name, "summary": summary})

            response = await _agent_runner.run(
                user_message=req.message,
                user_id=user_id,
                session_id=req.session_id,
                channel="api",
                model_override=req.model,
                on_text_chunk=collect_text,
                on_tool_start=collect_tool_start,
                on_tool_end=collect_tool_end,
            )

            # Emit collected events
            for event in tool_events:
                yield f"data: {json.dumps(event)}\n\n"

            # Emit final result — alias the model id like the message serializer
            # above (docs/security/audit-2026.md MI-2). Flag-gated.
            _sse_model = response.model
            if settings.security_leak_filter and _sse_model:
                from app.services.model_alias import public_model_label
                _sse_model = public_model_label(_sse_model)
            done = {
                "type": "done",
                "text": response.text,
                "session_id": response.session_id,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "model": _sse_model,
                "tool_calls": len(response.tool_calls),
                "processing_time_ms": response.processing_time_ms,
            }
            yield f"data: {json.dumps(done)}\n\n"

        except Exception as e:
            error = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ======================================================================
# Sessions
# ======================================================================

@router.get("/sessions", response_model=List[SessionSummary])
async def api_list_sessions(
    limit: int = 20,
    active_only: bool = False,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """List conversation sessions."""
    query = select(Conversation).where(Conversation.user_id == user_id)
    if active_only:
        query = query.where(Conversation.is_active == True)
    query = query.order_by(Conversation.updated_at.desc()).limit(limit)

    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        SessionSummary(
            id=s.id,
            channel=s.channel or "api",
            is_active=s.is_active,
            message_count=s.message_count,
            total_tokens=s.total_tokens,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
async def api_session_messages(
    session_id: str,
    limit: int = 50,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages from a specific session."""
    # Verify ownership
    result = await db.execute(
        select(Conversation).where(
            and_(Conversation.id == session_id, Conversation.user_id == user_id)
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))

    # Alias the real model id before it leaves the API (docs/security/
    # audit-2026.md MI-2). Flag-gated (default off).
    from app.config import settings as _settings
    _scrub = _settings.security_leak_filter
    if _scrub:
        from app.services.model_alias import public_model_label

    def _mu(v):
        return public_model_label(v) if (_scrub and v) else v

    return [
        MessageOut(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
            model_used=_mu(m.model_used),
        )
        for m in messages
    ]


# ======================================================================
# Memory search
# ======================================================================

@router.post("/memories/search")
async def api_memory_search(
    req: MemorySearchRequest,
    user_id: str = Depends(get_api_key_user),
):
    """Search memories via the API."""
    try:
        from app.services.embedding_service import get_embedding_service
        from app.services.memory_service import MemoryService

        emb = get_embedding_service()
        embedding = emb.embed(req.query)

        async with async_session_maker() as db:
            svc = MemoryService(db)
            results = await svc.search_memories_by_embedding(
                user_id=user_id,
                embedding=embedding,
                limit=req.limit,
                min_similarity=0.1,
                brain_types=[req.brain_type] if req.brain_type else None,
            )

        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


# ======================================================================
# Skills
# ======================================================================

@router.get("/skills")
async def api_list_skills(user_id: str = Depends(get_api_key_user)):
    """List all loaded skills and their tools."""
    if not _skill_loader:
        return {"skills": [], "count": 0}

    return {
        "skills": _skill_loader.get_summary(),
        "count": _skill_loader.loaded_count,
    }


# ======================================================================
# API Key management (uses JWT auth, not API key auth)
# ======================================================================

async def _get_jwt_user(request: Request, db: AsyncSession = Depends(get_db)) -> str:
    """Get user from JWT token (for key management endpoints)."""
    from app.api.auth import get_current_user
    user = await get_current_user(
        credentials=request.headers.get("Authorization", "").removeprefix("Bearer "),
        db=db,
    )
    return user.id


@router.post("/keys", response_model=CreateKeyResponse)
async def create_api_key(
    req: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_api_key_user),
):
    """Create a new API key. The raw key is only returned once."""
    # Generate key
    raw_key = f"hx_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:10]

    expires_at = None
    if req.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=req.expires_in_days)

    api_key = ApiKey(
        user_id=user_id,
        name=req.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        rate_limit=req.rate_limit,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return CreateKeyResponse(
        key=raw_key,
        id=api_key.id,
        name=api_key.name,
        key_prefix=key_prefix,
    )


@router.get("/keys", response_model=List[KeyOut])
async def list_api_keys(
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """List your API keys (without the actual key values)."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()

    return [
        KeyOut(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            rate_limit=k.rate_limit,
            is_active=k.is_active,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user_id: str = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke (deactivate) an API key."""
    result = await db.execute(
        select(ApiKey).where(
            and_(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await db.commit()

    return {"status": "revoked", "id": key_id}
