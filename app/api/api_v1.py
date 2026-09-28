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
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db, async_session_maker
from app.db.models import ApiKey, Conversation, Message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Public API v1"])


class VoiceDelegationStatusRequest(BaseModel):
    """A repaired Live socket's bounded, read-only reconciliation request."""

    session_id: str = Field(..., min_length=1, max_length=64)
    task_ids: List[str] = Field(..., min_length=1, max_length=4)


class VoiceDelegationSpeechClaimRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    task_id: str = Field(..., min_length=1, max_length=64)


class VoiceDelegationCancelRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    task_id: str = Field(..., min_length=1, max_length=64)
    job_id: str = Field(..., min_length=1, max_length=64)


def _voice_recovery_user(request: Request) -> str:
    """This internal seam is available only inside one authenticated tenant."""

    if settings.run_mode != "agent":
        raise HTTPException(status_code=404, detail="Not Found")
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid agent key")
    if not settings.user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")
    return settings.user_id


async def _owned_voice_session(db: AsyncSession, session_id: str, user_id: str) -> None:
    owned = (await db.execute(select(Conversation.id).where(
        Conversation.id == session_id,
        Conversation.user_id == user_id,
    ))).scalar_one_or_none()
    if not owned:
        raise HTTPException(status_code=404, detail="Session not found")


def _voice_delegation_row(message: Message, task_id: str) -> Optional[dict]:
    """Return only the persisted answer for this exact Live delegation."""

    try:
        meta = json.loads(message.metadata_json or "{}")
    except (TypeError, ValueError):
        return None
    voice = meta.get("voice") if isinstance(meta, dict) else None
    if not isinstance(voice, dict) or voice.get("delegation_id") != task_id:
        return None
    source = str(voice.get("source") or "")
    if source not in {"delegated", "delegated_record"}:
        return None
    accepted = source == "delegated" and not (
        voice.get("cancelled") or voice.get("superseded")
    )
    from app.api.day_chats import _serialize_tool_events

    return {
        "record_found": True,
        "status": (
            "completed" if accepted else
            "cancelled" if voice.get("cancelled") is True else "failed"
        ),
        "answer": str(message.content or "")[:6000] if accepted else "",
        "spoken": voice.get("spoken") is True,
        "recovery_speech_claimed": voice.get("recovery_speech_claimed") is True,
        "parent_user_turn_id": str(voice.get("parent_user_turn_id") or "")[:160],
        "title": str(voice.get("task_title") or "")[:160],
        # Same public projection as chat history; the raw metadata can hold
        # tool text the call surface is not allowed to show.
        "tool_events": (_serialize_tool_events(message) or [])[:16],
    }


@router.post("/internal/voice-delegations/status", include_in_schema=False)
async def internal_voice_delegations_status(
    req: VoiceDelegationStatusRequest, request: Request,
):
    """Resolve old Live cards from tenant rows after a socket repair.

    The original relay may have detached on another platform replica. The
    tenant's job and delegated answer rows are durable; this route only reads
    those rows and never re-runs a tool or an agent turn.
    """

    from app.db.models import BuildJob

    user_id = _voice_recovery_user(request)
    task_ids = list(dict.fromkeys(
        task_id for task_id in req.task_ids
        if isinstance(task_id, str) and 0 < len(task_id) <= 64
    ))[:4]
    if not task_ids:
        raise HTTPException(status_code=422, detail="Task id required")

    async with async_session_maker() as db:
        await _owned_voice_session(db, req.session_id, user_id)

        # Recent cards only; each candidate is checked against its exact
        # config_json identity. Old images lack this key, in which case the
        # answer row still settles the card by voice.delegation_id.
        jobs = (await db.execute(select(BuildJob).where(
            BuildJob.user_id == user_id,
            BuildJob.conversation_id == req.session_id,
            BuildJob.job_type == "agent_task",
        ).order_by(BuildJob.created_at.desc()).limit(100))).scalars().all()
        by_task: dict[str, dict] = {task_id: {"status": "unknown"} for task_id in task_ids}
        for job in jobs:
            config = job.config_json if isinstance(job.config_json, dict) else {}
            task_id = str(config.get("voice_delegation_id") or "")
            if task_id in by_task and "job_id" not in by_task[task_id]:
                by_task[task_id] = {
                    "status": str(job.status or "unknown")[:32],
                    "job_id": str(job.id),
                    "cancel_requested": job.stop_requested_at is not None,
                }

        # Read one bounded recent slice by indexed conversation identity,
        # then exact-match JSON in Python. One unindexed LIKE per task every
        # polling interval amplified long calls and let wildcard task IDs
        # broaden the DB scan even though the final JSON check was exact.
        candidates = (await db.execute(select(Message).where(
            Message.conversation_id == req.session_id,
            Message.role == "assistant",
        ).order_by(Message.created_at.desc()).limit(250))).scalars().all()
        for task_id in task_ids:
            for message in candidates:
                row = _voice_delegation_row(message, task_id)
                if row is not None:
                    by_task[task_id].update(row)
                    break

    return {"tasks": by_task}


@router.post("/internal/voice-delegations/cancel", include_in_schema=False)
async def internal_voice_delegation_cancel(
    req: VoiceDelegationCancelRequest, request: Request,
):
    """Request cooperative Stop for one exact running Live job.

    The row remains running until its original runner observes the marker and
    actually exits. A completed job wins the race and is never relabelled.
    """

    from app.db.models import BuildJob

    user_id = _voice_recovery_user(request)
    async with async_session_maker() as db:
        await _owned_voice_session(db, req.session_id, user_id)
        query = select(BuildJob).where(
            BuildJob.id == req.job_id,
            BuildJob.user_id == user_id,
            BuildJob.conversation_id == req.session_id,
            BuildJob.job_type == "agent_task",
        )
        if db.bind is not None and db.bind.dialect.name == "postgresql":
            query = query.with_for_update()
        job = (await db.execute(query)).scalar_one_or_none()
        config = job.config_json if job is not None and isinstance(job.config_json, dict) else {}
        if not job or config.get("voice_delegation_id") != req.task_id:
            raise HTTPException(status_code=404, detail="Voice task not found")
        if job.status != "running":
            return {"requested": False, "status": str(job.status or "unknown")}
        if job.stop_requested_at is None:
            job.stop_requested_at = datetime.utcnow()
            await db.commit()
        return {"requested": True, "status": "running", "job_id": job.id}


@router.post("/internal/voice-delegations/claim-speech", include_in_schema=False)
async def internal_voice_delegation_claim_speech(
    req: VoiceDelegationSpeechClaimRequest, request: Request,
):
    """Atomically allow one automatic recovered speech attempt per answer.

    The claim is deliberately made *before* the provider append. If that
    append or the phone's playback fails, the answer remains visible in chat
    and on the Live card; a later socket must wait for the user's explicit
    request to repeat rather than guessing whether audio was heard.
    """

    user_id = _voice_recovery_user(request)
    async with async_session_maker() as db:
        await _owned_voice_session(db, req.session_id, user_id)
        query = select(Message).where(
            Message.conversation_id == req.session_id,
            Message.role == "assistant",
        ).order_by(Message.created_at.desc()).limit(250)
        for candidate in (await db.execute(query)).scalars().all():
            if _voice_delegation_row(candidate, req.task_id) is None:
                continue
            locked = select(Message).where(Message.id == candidate.id).execution_options(
                populate_existing=True,
            )
            if db.bind is not None and db.bind.dialect.name == "postgresql":
                locked = locked.with_for_update()
            message = (await db.execute(locked)).scalar_one_or_none()
            if message is None:
                continue
            row = _voice_delegation_row(message, req.task_id)
            if row is None or row["status"] != "completed" or not row["answer"]:
                continue
            if row["spoken"] or row["recovery_speech_claimed"]:
                return {"claimed": False}
            meta = json.loads(message.metadata_json or "{}")
            voice = dict(meta["voice"])
            voice["recovery_speech_claimed"] = True
            meta["voice"] = voice
            message.metadata_json = json.dumps(meta)
            await db.commit()
            return {"claimed": True}
    return {"claimed": False}

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

#: How many context blocks the route will render, and how long each may be.
#: A prompt assembled from a request body is unbounded input; the model-facing
#: message is where an unbounded one costs money and context window. Declared
#: above `ChatRequest` so the SAME number bounds the body at parse time.
_CTX_BLOCKS_MAX = 8
_CTX_BLOCK_CHARS = 4000


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
    # ── R48: what the caller SAID, separately from what the model is given ──
    # On the Live path `message` is a constructed prompt: the utterance wrapped
    # in English framing plus the prior accepted results the model needs as
    # context. That blob is correct for the model and wrong for every surface
    # that names the work — the job card, the Live Activity, the push body were
    # all titled "Live-session context from earlier accepted delegations follo".
    # `display_request` is the clean utterance, used for display only.
    display_request: Optional[str] = Field(default=None, max_length=2000)
    # The language the CALLER just spoke, as a hint for the answer. The
    # delegation backend has never been told one, so a Persian question could
    # come back in English and be read out verbatim through `commentary.append`.
    # A short tag ("fa", "en", "fa-IR"), never a sentence.
    reply_language: Optional[str] = Field(default=None, max_length=32)
    # Prior accepted results, as structured blocks, for a caller that would
    # rather hand them over than pre-scaffold them into `message`. Exactly ONE
    # of the two mechanisms is in use per caller: a caller that already renders
    # its own context block (today's `delegated_agent_input`) sends nothing
    # here, and this field stays absent. Sending both feeds the context twice,
    # which `_compose_agent_message` refuses to do (it renders one and logs).
    context_blocks: Optional[List[Dict[str, Any]]] = Field(
        default=None, max_length=_CTX_BLOCKS_MAX)
    # WHICH delegation this turn is, in the relay's own vocabulary. The voice
    # card's cancelled close proves delivery from the thread, and on Live two
    # delegations can run at once — so without this the OTHER task's answer
    # closes this card `completed` (addendum C1).
    delegation_id: Optional[str] = Field(default=None, max_length=64)
    # Contract v0.3 §7 (addendum 6 R6-4 / R6-4b): the caller-turn ORDER of the
    # request that started this run, the provider session it belongs to and
    # that session's STAMP (the relay's hybrid logical clock, app-floored:
    # ordinals restart with each session), bound into the run so its
    # play_media is superseded only by a stop the caller asked for AFTER it
    # (pairwise: same scope by order, stamped scopes by stamp, otherwise by
    # arrival — `radio.control`). Voice-only and optional: absent
    # (every other caller, a legacy relay) keeps the arrival mark. Lenient on
    # purpose — a malformed value reads as absent rather than failing the turn.
    media_order: Optional[int] = None
    media_scope: Optional[str] = None
    media_scope_started_ms: Optional[int] = None

    @field_validator("media_order", mode="before")
    @classmethod
    def _check_media_order(cls, value):
        return _lenient_media_order(value)

    @field_validator("media_scope", mode="before")
    @classmethod
    def _check_media_scope(cls, value):
        return _lenient_media_scope(value)

    @field_validator("media_scope_started_ms", mode="before")
    @classmethod
    def _check_media_scope_started_ms(cls, value):
        return _lenient_started_ms(value)


def _lenient_media_order(value):
    """An additive ordering field never fails a request: anything that is not
    a non-negative int (a digit string is accepted, a bool is not) is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and 0 <= value <= 2**31 - 1:
        return value
    return None


def _lenient_media_scope(value):
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:128]


def _lenient_started_ms(value):
    """Epoch milliseconds, or None — the same leniency as the order."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and 0 <= value <= 2**53:
        return value
    return None


#: Openings a caller uses when it has ALREADY scaffolded its own context into
#: `message` (the relay's `delegated_agent_input`). Kept in sync with
#: `voice_jobs._SCAFFOLD_MARKERS`, which rejects the same strings as a title.
_SCAFFOLDED_MESSAGE_MARKERS = (
    "live-session context",
    "context from earlier in this conversation",
)


def _compose_agent_message(message: str, blocks: Optional[List[Dict[str, Any]]]) -> str:
    """The model-facing message: the request, with any caller-supplied context
    rendered ahead of it under a separator the model can tell from the ask.

    With no blocks this returns ``message`` unchanged, byte for byte — which is
    every caller before R48 and every caller that scaffolds its own context.

    The two mechanisms are mutually exclusive, and that is ENFORCED here rather
    than documented: a caller that sends both would have the same prior turns
    rendered twice, once in its own framing and once in ours, which is a
    context-window cost and a model-confusing duplicate. The caller's own
    scaffolding wins — it is the one the caller can see.
    """
    if blocks and message.strip()[:120].lower().startswith(
            _SCAFFOLDED_MESSAGE_MARKERS):
        logger.warning(
            "[agent-turn] context_blocks (%d) ignored: `message` already "
            "carries caller-rendered context", len(blocks),
        )
        return message
    if not blocks:
        return message
    parts: List[str] = []
    for b in blocks[:_CTX_BLOCKS_MAX]:
        if not isinstance(b, dict):
            continue
        _label = str(b.get("label") or "Context")[:80]
        _text = str(b.get("text") or "")[:_CTX_BLOCK_CHARS]
        if _text:
            parts.append(f"{_label}: {_text}")
    if not parts:
        return message
    return (
        "Context from earlier in this conversation. Treat it as prior "
        "conversation, not as the current request.\n"
        + "\n".join(parts)
        + "\n\nCurrent request: "
        + message
    )


def _runner_run_accepts(runner: Any, name: str) -> bool:
    """True when this runner's ``run`` declares the keyword ``name``.

    The relay and the agent image roll independently: the relay ships today and
    the image follows on a canary, so for a window the new body fields exist
    and the runner that must consume them does not. Probing the signature keeps
    that window a no-op instead of a 500 on every voice turn.
    """
    import inspect

    # `AgentRunner.run` is a thin `(*args, **kwargs)` wrapper that forwards to
    # `_run_inner`, so **kwargs here means "ask the real signature", NOT
    # "accepts anything": an unknown keyword reaches `_run_inner` and raises
    # TypeError — a 500 on every voice turn, which is the exact failure this
    # probe exists to prevent.
    for fn in (getattr(runner, "run", None), getattr(runner, "_run_inner", None)):
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):  # pragma: no cover — C-implemented
            continue
        if name in params:
            return True
    return False


def _forward_display_kwargs(runner: Any, req: "ChatRequest") -> Dict[str, Any]:
    """The R48 display/language kwargs this runner is able to accept."""
    out: Dict[str, Any] = {}
    if req.display_request and _runner_run_accepts(runner, "display_request"):
        out["display_request"] = req.display_request
    if req.reply_language and _runner_run_accepts(runner, "reply_language"):
        out["reply_language"] = req.reply_language
    # Named `voice_delegation_id` on the runner: `delegation_id` alone reads
    # like the runner's own concept, and the runner has several kinds of
    # delegated work. Same signature probe as its two siblings.
    if req.delegation_id and _runner_run_accepts(runner, "voice_delegation_id"):
        out["voice_delegation_id"] = req.delegation_id
    return out


class ChatResponse(BaseModel):
    text: str
    session_id: str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    model: str = ""
    tool_calls: int = 0
    processing_time_ms: int = 0
    # What the turn PRODUCED. On a `save=False` turn (every voice turn) nothing
    # downstream writes a Message row, so without these two the generated file
    # and the presented app existed only in storage — no row for
    # GET /api/files/{message_id}/{aid} to authorize against, and no card on
    # any surface. Optional and empty by default: an older relay ignores them,
    # an older agent image never sends them.
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    app_artifact: Optional[Dict[str, Any]] = None
    # …and the third thing a turn can produce. A `play_media` inside a voice
    # turn started a song and left NO card in the thread, because the caller
    # owns persistence on a `save=False` turn and was never told a card
    # existed. Same optional-and-absent contract as the two above.
    media: Optional[Dict[str, Any]] = None


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
    # Contract v0.3 §7: the caller-turn order of the request that asked for
    # this play, its provider session and that session's stamp (R6-4b). With
    # an order and a scope, a stop supersedes it only if the caller asked for
    # that stop AFTER this play (pairwise rule, `radio.control`).
    media_order: Optional[int] = None
    media_scope: Optional[str] = None
    media_scope_started_ms: Optional[int] = None

    @field_validator("media_order", mode="before")
    @classmethod
    def _check_media_order(cls, value):
        return _lenient_media_order(value)

    @field_validator("media_scope", mode="before")
    @classmethod
    def _check_media_scope(cls, value):
        return _lenient_media_scope(value)

    @field_validator("media_scope_started_ms", mode="before")
    @classmethod
    def _check_media_scope_started_ms(cls, value):
        return _lenient_started_ms(value)


class MediaControlRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    # stop/pause (contract v0.3 §H, `media_transport`) are confirmed by the
    # phone, not by the station; see `radio.control._execute_transport`.
    action: Literal["next", "previous", "stop", "pause"]
    channel: Optional[str] = Field(default="app", max_length=32)
    # Contract v0.3 §7, stop/pause only: the caller-turn order of this halt,
    # its provider session and that session's stamp (R6-4b). It then halts
    # exactly the ordered plays requested before it (pairwise across the
    # user's scopes, `radio.control`), and leaves a NEWER ordered item alone
    # (`reason: "newer_playing"`, ok false; `paused: true` when that item is
    # paused). Absent: exactly as before.
    before_order: Optional[int] = None
    media_scope: Optional[str] = None
    media_scope_started_ms: Optional[int] = None

    @field_validator("before_order", mode="before")
    @classmethod
    def _check_before_order(cls, value):
        return _lenient_media_order(value)

    @field_validator("media_scope", mode="before")
    @classmethod
    def _check_media_scope(cls, value):
        return _lenient_media_scope(value)

    @field_validator("media_scope_started_ms", mode="before")
    @classmethod
    def _check_media_scope_started_ms(cls, value):
        return _lenient_started_ms(value)


def _media_arrival_mark(user_id: str, req=None):
    """The media halt mark for a request arriving now (radio.control), or
    None when unavailable — the play tool then takes its own. With a request
    that carries the caller's order (`media_order` + `media_scope`, and
    `media_scope_started_ms`; contract v0.3 §7) the mark is ordered;
    otherwise it is the unchanged arrival mark."""
    try:
        from app.agent.radio.control import media_halt_mark
        return media_halt_mark(
            user_id,
            media_order=getattr(req, "media_order", None),
            media_scope=getattr(req, "media_scope", None),
            media_scope_started_ms=getattr(req, "media_scope_started_ms", None),
        )
    except Exception:  # noqa: BLE001 - the guard never blocks a turn
        return None


def _bind_media_run_mark(mark):
    try:
        from app.agent.radio.control import bind_run_halt_mark
        return bind_run_halt_mark(mark)
    except Exception:  # noqa: BLE001
        return None


def _reset_media_run_mark(token) -> None:
    if token is None:
        return
    try:
        from app.agent.radio.control import reset_run_halt_mark
        reset_run_halt_mark(token)
    except Exception:  # noqa: BLE001
        pass


def _vs_play_superseded(name: str, body) -> bool:
    """A play_media result that says its play was dropped for a newer stop."""
    if name != "play_media":
        return False
    try:
        from app.agent.radio.control import PLAY_SUPERSEDED_PREFIX
    except Exception:  # noqa: BLE001
        return False
    return str(body or "").lstrip().startswith(PLAY_SUPERSEDED_PREFIX)


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
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")
    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    # When this play was asked for. The search below takes seconds; a voice
    # stop/pause or a radio OFF that lands meanwhile is the caller's newer
    # intent, and the tool drops the play instead of restarting the music.
    # With the caller's order (contract v0.3 §7) an ordered stop supersedes
    # it only when the caller asked for that stop after this play.
    from app.agent.radio.control import PLAY_SUPERSEDED_PREFIX, media_halt_mark
    halt_mark = media_halt_mark(
        user_id,
        media_order=req.media_order,
        media_scope=req.media_scope,
        media_scope_started_ms=req.media_scope_started_ms,
    )

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
        # Voice is ALWAYS audio. Without this pin, infer_requested_mode can
        # stamp mode='video' on a spoken request ("play the music video for…"),
        # and because the frame rides the chat socket it may be surface-judged
        # AFTER the call ends — mounting an autoplaying WebView over the chat
        # the user just returned to. A voice session has no visible surface to
        # watch on, ever; the user can flip to Video from the card afterwards.
        "mode": "audio",
        "_halt_mark": halt_mark,
    })

    text = str(result or "")
    if text.startswith(PLAY_SUPERSEDED_PREFIX):
        # Nothing was sent to the phone. Not an error: the relay must neither
        # announce this play nor hand it to the full agent to play anyway.
        return {
            "ok": False,
            "reason": "superseded",
            "title": "",
            "video_id": "",
            "thumbnail_url": "",
            "detail": text[:300],
        }
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


@router.post("/internal/media-control", include_in_schema=False)
async def internal_media_control(req: MediaControlRequest, request: Request):
    """Move an active tenant radio session without running an agent turn.

    next/previous navigate the station. stop turns the station OFF and asks the
    phone to stop; pause asks the phone to pause. For those two `ok` is true
    only on the phone's ack, and `reason` says what happened
    (stopped | paused | nothing_playing | partially_confirmed | unacknowledged |
    delivery_failed | error). `partially_confirmed` (additive, ok false): every
    device that answered was idle but another socket that could be playing
    stayed silent; `acked_devices` / `silent_devices` count them.

    Contract v0.3 §7: a stop/pause may carry `before_order` + `media_scope`
    (+ `media_scope_started_ms`), the caller's order for it. It then
    supersedes only the ordered plays requested before it (pairwise across
    the user's scopes: same scope by order, stamped scopes by stamp, otherwise
    by arrival), and when the item last broadcast is a NEWER ordered one it is
    left alone: `{ok: false, reason: "newer_playing"}` (additive), with that
    item's `video_id` / `title` (+ `paused: true` when it is paused), and
    nothing is sent."""

    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")
    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")
    if not secrets.compare_digest(req.user_id, user_id):
        logger.error(
            "[media-control] body user=%s does not match tenant owner=%s",
            req.user_id[:8], user_id[:8],
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User mismatch")

    from app.agent.radio.control import (
        MediaControlOutcome,
        execute_media_control,
        media_control_budget_s,
    )

    # The stop/pause ack wait ends inside this same budget (minus a margin),
    # so an unanswered stop reports `unacknowledged` rather than `timeout`.
    timeout_s = media_control_budget_s()
    # Contract v0.3 §7: the halt's own order, passed only when the relay sent
    # one (and only for stop/pause), so a legacy body calls exactly as before.
    ordered = (
        {
            "before_order": req.before_order,
            "media_scope": req.media_scope,
            "media_scope_started_ms": req.media_scope_started_ms,
        }
        if req.action in ("stop", "pause")
        and req.before_order is not None and req.media_scope
        else {}
    )
    try:
        outcome = await asyncio.wait_for(
            execute_media_control(user_id, req.action, req.channel, **ordered),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        outcome = MediaControlOutcome(
            ok=False,
            user_id=user_id,
            action=req.action,
            channel=req.channel or "app",
            changed=False,
            # stop/pause answer from a closed reason set the relay words
            # from; a hung OFF or broadcast there is an error, not a state.
            reason="timeout" if req.action in ("next", "previous") else "error",
        )
    return outcome.as_dict()


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
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")

    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")

    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    # The run's media halt mark, taken on arrival: a stop/pause while the
    # agent is still thinking supersedes a play_media it calls later. With
    # the caller's order (`media_order` + `media_scope`, contract v0.3 §7) an
    # ordered stop supersedes it only when asked for after this request.
    _halt_token = _bind_media_run_mark(_media_arrival_mark(user_id, req))
    try:
        response = await _agent_runner.run(
            user_message=_compose_agent_message(req.message, req.context_blocks),
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
            # Only when this image's runner declares them — the relay ships
            # before the agent image rolls, so the fields can arrive at a
            # runner that has never heard of them.
            **_forward_display_kwargs(_agent_runner, req),
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
            # Read off `persisted` rather than out of the runner's tool state:
            # `persisted` is the documented echo of what this turn wrote — or,
            # when the caller owns persistence, of what it must write — and the
            # tool list has already been drained by the time we get here.
            attachments=list((response.persisted or {}).get("attachments") or []),
            app_artifact=(response.persisted or {}).get("app_artifact") or None,
            media=(response.persisted or {}).get("media") or None,
        )
    except Exception as e:
        logger.exception(f"Internal agent-turn error for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")
    finally:
        _reset_media_run_mark(_halt_token)


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
    # Which voice wire this prompt is for. The Realtime path has a `think`
    # tool, `navigate_to`, terminal access and screen share; the GPT-Live path
    # has NONE of them — its only lever over the backend is a delegation the
    # model decides to emit, and the prompt is the only channel that can tell
    # it when to. So the two paths need different channel documents, and
    # serving the Realtime one on Live instructed the model, imperatively, to
    # call tools it does not have. Default False: an older relay sends nothing
    # and gets exactly today's prompt.
    live: bool = Field(default=False)


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
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
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
                live=req.live,
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


# ── The voice turn's memory write (v3 §2.1.2) ─────────────────────────
#
# The realtime relay runs on platform-api; the curator runs agent-side,
# where the memory lives. Round 8 resolved that split the wrong way: it ran
# a full LLM EXTRACTION on the platform and pushed each result across the
# tunnel as a `memory_store` tool call with `explicit_save=True`, which
# disarmed three gate rules and is why the founder's brain held permanent
# rows about songs he asked to play once.
#
# The relay cannot simply reuse `/internal/agent-turn`: it deliberately
# calls that with save=False (which maps to disable_post_processing=True),
# because `think`'s `task` is a string the REALTIME MODEL synthesised, not
# what the user said. Mining that string is the 409A incident. So the honest
# seam is a second small internal route that takes the REAL transcript and
# runs the same writer the chat path runs — one hop, same auth, same
# visibility rules as /internal/voice-context above.


class CurateTurnRequest(BaseModel):
    # The user's own words, verbatim from transcription. NEVER the relay's
    # synthesised `task`, and never a tool result.
    user_text: str = Field(..., max_length=8000)
    assistant_text: str = Field(default="", max_length=8000)
    channel: str = Field(default="voice", max_length=32)


class CurateTurnResponse(BaseModel):
    applied: int = 0
    changed_files: List[str] = Field(default_factory=list)
    skipped: Optional[str] = None


@router.post("/internal/curate-turn", response_model=CurateTurnResponse,
             include_in_schema=False)
async def internal_curate_turn(req: CurateTurnRequest, request: Request):
    """Internal-only: write one spoken turn into the user's memory files."""
    # Only meaningful on tenant agent containers. On the platform, 404 so the
    # endpoint is invisible to probers (mirrors agent.py:refresh-tools).
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")

    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    try:
        from app.services import memory_curator

        async with async_session_maker() as db:
            result = await memory_curator.curate_turn(
                db, user_id,
                user_text=req.user_text,
                assistant_text=req.assistant_text,
                channel=req.channel or "voice",
            )
        # Current context's Today layer (v3 §6) — off the response path, in
        # its own session, and unable to fail this route. The relay is
        # waiting on this reply while the user is still in a live call.
        try:
            from app.services.current_context import spawn_refresh

            spawn_refresh(async_session_maker, user_id)
        except Exception as ctx_err:  # noqa: BLE001
            logger.warning("Current-context refresh not scheduled: %s", ctx_err)
        return CurateTurnResponse(
            applied=int(result.get("applied", 0)),
            changed_files=list(result.get("changed_files") or []),
            skipped=result.get("skipped"),
        )
    except Exception as e:
        logger.exception("Internal curate-turn error for user %s", user_id)
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
_VS_CANCEL_POLL_S   = 1.5
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
    # What the agent asked to play: a benign search string, the same class as
    # web_search's query (addendum 6 R6-6). Without it `tool.start` carried
    # args={} and the relay had no title for an agent play it had to stop, so
    # it quoted the caller's own sentence back as a track name (media-6).
    "play_media":         ("query",),
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

# ── needs_auth (A5-10) ────────────────────────────────────────────────────
# A connector tool that has lost its credential does not FAIL in any sense the
# user can act on by retrying — it is waiting for them to reconnect the
# account. The wire has an outcome for that (`needs_auth`, read by the relay's
# `_outcome_of`), and until now nothing produced it.
#
# The signal is STRUCTURAL, not model prose: `connector_mcp._serialize_result`
# turns each `ConnectorResult` variant into `{kind, message, …}` and
# `ToolExecutor._canonicalize_mcp_result` lifts `message`, which this server
# wrote as `[<kind>] …`. So the marker is the envelope's own `kind`, spelled
# by one function in this repo.
#
# `scope_missing` is here with `reauth_required` because the user's action is
# the same one — reconnect the account and grant it — and the client's
# needs_auth branch deep-links to exactly that screen. Both are a request for
# authorization; neither is a failure of the work.
_VS_NEEDS_AUTH_KINDS = ("reauth_required", "scope_missing")


def _vs_needs_auth(name: str, body: str) -> Optional[str]:
    """The connector slug when ``body`` is a connector-authorization result,
    else None.

    Both halves are structural. The KIND is the `[<kind>] …` head that
    `connector_mcp._serialize_result` writes from the `ConnectorResult`
    subclass; the SLUG is read ONLY out of the tool's own `<slug>__<action>`
    namespace (the same split `tool_executor` uses for `connector_id`), never
    out of the result string.

    The namespace gate is a security boundary, not tidiness. `body` is the
    DE-FENCED result, so for `web_fetch` / `web_search` / `browser` /
    `extension_*` it IS the fetched page — attacker-controlled by
    construction, which is the whole reason the fence exists
    (`tool_executor._EXTERNAL_CONTENT_TOOLS`, whose comment states the rule
    this function has to obey: a decision must not be derived from the result
    string). Without the gate, a page beginning with the marker forged a
    needs_auth prompt on the voice wire, and — because the slug used to be
    scraped from a `/agent/integrations/<slug>` URL inside that page — named
    an attacker-chosen connector for the user to "reconnect". None of those
    tools is namespaced, so requiring the namespace excludes every one of
    them, and the slug can only ever name the tool that actually ran.

    The slug is NOT put on the wire this round (ruling C10 — nothing reads it,
    and the client's reconnect CTA is a fixed `toup://connectors`); the caller
    only tests None-ness. It is still what this function returns, because the
    slug is the evidence that the verdict came from the tool THIS server ran
    and not from the body, and it is what a per-connector consumer would need.
    Returns "" only for the degenerate `__action` name. NEVER returns any part
    of the body.
    """
    if "__" not in name:
        return None
    head = (body or "").lstrip()[:24].lower()
    if not any(head.startswith(f"[{k}]") for k in _VS_NEEDS_AUTH_KINDS):
        return None
    return name.split("__", 1)[0][:32]


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


#: The step-attribution keys the runner stamps on every tool event
#: (``StepTracker.event_fields``). Flat scalars, all optional.
_VS_STEP_KEYS = ("job_id", "step_index", "step_name", "steps_total", "job_type")


def _vs_step_fields(ev: dict) -> dict:
    """Round 13: which declared step this action served.

    A chat turn writes these straight onto its persisted tool record, so the
    run view can bucket actions under steps. Voice's frames dropped them, and
    the relay therefore had nothing to persist — the same action, on the same
    job, rendered under "no step". Copied verbatim (a bounded set of flat
    scalars); ABSENT rather than null when the turn declared no job, so a
    jobless turn's frame is byte-identical to what shipped before.
    """
    out = {}
    for k in _VS_STEP_KEYS:
        v = ev.get(k)
        if v is None:
            continue
        out[k] = int(v) if k in ("step_index", "steps_total") else str(v)[:120]
    return out


def _vs_sources_from_domains(ev: dict) -> list:
    """Bare-domain sources from the runner's own `domains`, as a FALLBACK.

    `agent_runner` stamps an ordered, deduped host list on every web tool call
    (`WEB_DOMAIN_TOOLS` / `extract_web_refs`) and puts it on the very
    `on_tool_event` payload this module receives — and this module has never
    read it, re-deriving provenance from scratch by re-parsing the rendered
    text instead. That parse (`_vs_sources`) is anchored to the search
    gateway's `N. Title / url / description` block, so it returns [] for any
    tier that renders differently and for any result whose title line was
    reflowed. When that happens the phone shows a turn that searched the web
    and names nothing.

    Titles are structurally absent here — `extract_web_refs` is a URL regex —
    so this is strictly the weaker answer and is used only when the parse found
    nothing at all. The client already renders a source with a domain and no
    title as a one-line card."""
    doms = ev.get("domains")
    urls = ev.get("urls")
    if not isinstance(doms, list) or not doms:
        return []
    by_host: dict = {}
    if isinstance(urls, list):
        for u in urls:
            if not isinstance(u, str):
                continue
            try:
                h = (urlparse(u).hostname or "").lower()
            except Exception:  # noqa: BLE001
                continue
            if h.startswith("www."):
                h = h[4:]
            by_host.setdefault(h, u)
    out: list = []
    for d in doms[:_VS_SRC_MAX]:
        if not isinstance(d, str) or not d:
            continue
        out.append({
            "title": "",
            "url": str(by_host.get(d, ""))[:_VS_SRC_URL_MAX],
            "domain": d[:_VS_SRC_DOMAIN_MAX],
        })
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


def _vs_shrink(frame: dict) -> None:
    """Bring an oversized frame under `_VS_FRAME_BYTES_MAX` IN PLACE, giving up
    the least valuable thing first.

    This used to be `frame.pop("sources"); frame.pop("preview")`, i.e. the
    whole provenance for one byte over. At the caps here a source can be 120
    chars of title plus 300 of URL, so six of them are ~2.6 KB, and a turn
    whose results carry long tracking URLs shipped the phone nothing to show
    for a search it visibly ran. Same ladder as the relay's `_shrink_frame`
    (ws_realtime.py) — deliberately a second copy, because neither side trusts
    the other's caps and a version skew in either direction must not be able to
    flood the phone's audio socket."""
    def size() -> int:
        return len(json.dumps(frame, separators=(",", ":"), ensure_ascii=False))

    if size() > _VS_FRAME_BYTES_MAX and frame.get("sources"):
        frame.pop("preview", None)
    if size() <= _VS_FRAME_BYTES_MAX:
        return
    srcs = frame.get("sources")
    if not isinstance(srcs, list) or not srcs:
        frame.pop("preview", None)
        frame.pop("sources", None)
        return
    for src in srcs:
        u = src.get("url")
        if isinstance(u, str) and ("?" in u or "#" in u):
            src["url"] = u.split("#", 1)[0].split("?", 1)[0]
    for cap in (80, 48):
        if size() <= _VS_FRAME_BYTES_MAX:
            return
        for src in srcs:
            t = src.get("title")
            if isinstance(t, str) and len(t) > cap:
                src["title"] = t[:cap].rstrip()
    # Never all of them: one source is provenance, zero is none.
    while len(srcs) > 1 and size() > _VS_FRAME_BYTES_MAX:
        srcs.pop()


def _vs_sse(frame: dict) -> str:
    blob = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
    if len(blob) > _VS_FRAME_BYTES_MAX and frame.get("type") != "done":
        _vs_shrink(frame)
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
    if not settings.agent_api_key or not secrets.compare_digest(
        agent_key, settings.agent_api_key
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent key")
    if not _agent_runner:
        raise HTTPException(status_code=503, detail="Agent not available")
    user_id = settings.user_id
    if not user_id:
        raise HTTPException(status_code=503, detail="Agent user not configured")

    # When this request arrived, for media (addendum-2 item 9): bound for the
    # run below, so a stop/pause while the agent is still thinking supersedes
    # a play_media it calls later, not only one that lands during its search.
    # Carries the caller's order when the relay sent one (contract v0.3 §7):
    # then an ordered stop supersedes the run's play only when the caller
    # asked for it after this request — a run still thinking when an OLDER
    # stop fires plays.
    halt_mark = _media_arrival_mark(user_id, req)

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
            # Round 13: the step this action served. The runner stamps these
            # on every tool event (StepTracker.event_fields) and a chat turn
            # persists them straight onto its tool record — this frame dropped
            # them, so the identical action arriving over voice reached the
            # phone with no step to sit under. Absent keys, never nulls, so a
            # turn with no job is byte-identical to before.
            _attr = _vs_step_fields(ev)
            if ev.get("phase") == "start":
                _put({"type": "tool.start", "call_id": cid, "name": name,
                      "args": _vs_args(name, inp),
                      "started_ms": int(ev.get("started_ms") or 0), **_attr})
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
                    "sources": _vs_sources(name, inp, raw) or _vs_sources_from_domains(ev),
                    **_attr,
                }
                # `outcome` is ADDITIVE: `ok` keeps the value every shipped
                # client already reads, and a consumer that does not know the
                # key is byte-identical to today. Only named when this server
                # can prove what happened — everything else stays derived from
                # `ok` on the relay's side.
                #
                # The SLUG is deliberately not emitted (ruling C10). Nothing
                # downstream reads it: `_InnerToolRelay` builds
                # `tool_call.completed` key by key and has no `connector` key,
                # and the app's needs_auth CTA opens the fixed
                # `toup://connectors` with no per-connector route. A field
                # emitted and dropped at every hop is a false impression of
                # specificity; re-introduce it together with the consumer when
                # the web tool-activity UI lands.
                if _vs_needs_auth(name, body) is not None:
                    frame["outcome"] = "needs_auth"
                if _vs_play_superseded(name, body):
                    # The tool dropped its play because the user stopped or
                    # paused the music after asking (addendum-2 item 9). Not a
                    # started track: never ok, and named, so the relay and the
                    # app show a cancelled step, not "Starting the music".
                    frame["ok"] = False
                    frame["outcome"] = "cancelled"
                if name in _VS_PREVIEW_ALLOW:
                    frame["preview"] = _vs_clean(body)[:_VS_PREVIEW_MAX]
                _put(frame)
        except Exception:  # noqa: BLE001
            logger.debug("[VSTREAM] on_tool_event sink failed", exc_info=True)

    async def _run_wrapped():
        _halt_token = _bind_media_run_mark(halt_mark)
        try:
            return await _agent_runner.run(
                user_message=_compose_agent_message(req.message, req.context_blocks),
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
                # Same signature probe as the blocking sibling — the relay
                # ships before the agent image rolls.
                **_forward_display_kwargs(_agent_runner, req),
                # on_text_chunk deliberately NOT passed: voice renders no token
                # deltas, and omitting it takes the frame count from thousands
                # per turn to 2-40, which makes backpressure a non-problem.
            )
        finally:
            _reset_media_run_mark(_halt_token)
            try:
                q.put_nowait(None)          # terminal sentinel
            except asyncio.QueueFull:
                pass                        # drain loop's task.done() check covers it

    async def _watch_durable_voice_cancel(task: asyncio.Task) -> None:
        """The same tenant that executes tools observes the durable Stop bit.

        A repaired phone may be attached to another platform replica. This
        monitor is tied to the original agent run, not its socket, and the
        runner's existing cancel_check handles the next safe tool boundary.
        A bounded hard cancel follows for a tool that never yields back.
        """

        from app.db.models import BuildJob

        if not req.delegation_id or not req.session_id or req.save:
            return
        while not task.done():
            await asyncio.sleep(_VS_CANCEL_POLL_S)
            try:
                async with async_session_maker() as db:
                    jobs = (await db.execute(select(BuildJob).where(
                        BuildJob.user_id == user_id,
                        BuildJob.conversation_id == req.session_id,
                        BuildJob.job_type == "agent_task",
                    ).order_by(BuildJob.created_at.desc()).limit(100))).scalars().all()
                job = next((item for item in jobs
                            if isinstance(item.config_json, dict)
                            and item.config_json.get("voice_delegation_id") == req.delegation_id),
                           None)
            except Exception:  # noqa: BLE001
                logger.warning("[VSTREAM] voice cancel watch read failed")
                continue
            if job is None:
                continue  # voice job opens at its first planned tool round
            if job.status != "running":
                return  # terminal completion won the race
            if job.stop_requested_at is None:
                continue
            cancelled["v"] = True
            logger.info("[VSTREAM] voice cancel observed job=%s", job.id[:8])
            await asyncio.sleep(_VS_CANCEL_POLL_S)
            if not task.done():
                task.cancel()
            return

    async def generate():
        task = asyncio.create_task(_run_wrapped())
        cancel_watch = asyncio.create_task(_watch_durable_voice_cancel(task))
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
                if cancelled["v"]:
                    # The tenant observed an authenticated, exact-job Stop.
                    # Without a terminal event the relay sees EOF and may
                    # enter its tool-less fallback, answering a stopped turn.
                    yield _vs_sse({"type": "cancelled", "reason": "voice_stop"})
                    return
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
                # Same two fields the blocking sibling returns, for the same
                # reason: this is the path voice actually takes, so omitting
                # them here would leave the defect fully live. Keys are present
                # only when non-empty, so an older relay's parse is unchanged.
                **({"attachments": list((response.persisted or {}).get("attachments") or [])}
                   if (response.persisted or {}).get("attachments") else {}),
                **({"app_artifact": (response.persisted or {})["app_artifact"]}
                   if (response.persisted or {}).get("app_artifact") else {}),
                # The media card the turn started. `save=False` means this
                # process writes no row, and the relay — which does — was never
                # told a card existed, so a song started by voice reopened as
                # plain text with nothing to tap. Key present only when
                # non-empty, so an older relay's parse is unchanged.
                **({"media": (response.persisted or {})["media"]}
                   if (response.persisted or {}).get("media") else {}),
            })
            if budget["dropped"]:
                logger.warning("[VSTREAM] dropped %d frames (queue full)", budget["dropped"])
        finally:
            cancel_watch.cancel()
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
