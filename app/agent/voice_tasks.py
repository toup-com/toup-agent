"""Durable AgentRunner work started from a realtime voice conversation.

The websocket observes these jobs; it never owns them. ``build_jobs`` is the
execution ledger, a random claim token fences stale coroutines, and only queued
work is replayable. A running job whose lease expires is deliberately marked
unknown because an external operation may already have happened.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.agent.artifact_kinds import kind_for_mime, preview_policy
from app.agent.operation_identity import (
    mutation_invalidates_read,
    resolved_operation_is_read,
    tool_operation_cacheable_read,
    tool_operation_checkpoint,
    tool_operation_kind,
    tool_operation_key,
    tool_result_is_confirmation,
)
from app.db.models import (
    AgentNotifyOutbox, BuildJob, Conversation, DayChat, JobEvent, Message, User,
)

logger = logging.getLogger(__name__)

SOURCE_KIND = "voice_task"
RUNNING = "running"
WAITING = "waiting_on_user"
TERMINAL_DB = frozenset({"completed", "failed", "cancelled"})
OPEN_DB = frozenset({"queued", RUNNING, WAITING})
MAX_PROGRESS = 100
MAX_STEERING = 100
MAX_SOURCES = 40
MAX_ARTIFACTS = 30
MAX_TEXT = 100_000
MAX_RESOLVED_OPERATIONS = 100
MAX_RESOLVED_RESULT = 4_000
DEFAULT_LEASE_SECONDS = 90.0
DEFAULT_IDLE_POLL_SECONDS = 30.0
DEFAULT_WAITING_POLL_SECONDS = 5.0
APPROVAL_UNKNOWN_SECONDS = 300.0
ACTION_TERMINAL = frozenset({"executed", "failed", "rejected", "expired", "missing"})
_TASK_NAMESPACE = uuid.UUID("bed9ad67-1860-4f7a-95ed-f0d425ab8a72")

ActionReader = Callable[[str, str], Awaitable[Optional[dict[str, Any]]]]
ActionCanceller = Callable[[str, str], Awaitable[Optional[dict[str, Any]]]]
ClaimAllowed = Callable[[], bool]


def voice_task_supervisor_boot_allowed(
    *, enabled: bool, bound: bool, passive: bool,
) -> bool:
    """Only a serving tenant may perform durable-work discovery at boot."""
    return bool(enabled and bound and not passive)


def _agent_claims_allowed() -> bool:
    """Keep discovery alive while refusing new work in non-serving slots."""
    if settings.run_mode != "agent":
        return True
    from app.services import drain_state, runtime_identity

    if not runtime_identity.is_bound() or drain_state.is_draining():
        return False
    # A blue-green slot is already tenant-bound while the old slot still
    # serves. It becomes claim-eligible only after delayed promotion writes
    # the same marker used by agent_main.is_passive_boot().
    import os
    from pathlib import Path

    return not (
        os.environ.get("TOUP_BG_PASSIVE") == "1"
        and not Path("/app/workspace/.toup_bg_promoted").exists()
    )


class VoiceTaskError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class _UserCancelled(asyncio.CancelledError):
    pass


class _LeaseLost(asyncio.CancelledError):
    pass


class _DeadlineReached(asyncio.CancelledError):
    pass


@dataclass
class _Handle:
    token: str
    cancel_event: asyncio.Event
    lease_lost: asyncio.Event
    deadline_event: asyncio.Event
    task: Optional[asyncio.Task] = None
    deadline_task: Optional[asyncio.Task] = None

    def should_stop(self) -> bool:
        return (
            self.cancel_event.is_set()
            or self.lease_lost.is_set()
            or self.deadline_event.is_set()
        )


def task_id_for(user_id: str, session_id: str, request_id: str) -> str:
    """Exact request-id idempotency; similar text is intentionally unrelated."""
    return str(uuid.uuid5(_TASK_NAMESPACE, json.dumps([user_id, session_id, request_id])))


def _fingerprint(message: str) -> str:
    normalized = unicodedata.normalize("NFC", message).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _now() -> datetime:
    return datetime.utcnow()


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _now()).isoformat()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _state(job: BuildJob) -> dict[str, Any]:
    cfg = job.config_json if isinstance(job.config_json, dict) else {}
    state = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
    return _copy(state)


def _config(job: BuildJob, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _iso()
    cfg = dict(job.config_json or {})
    cfg["voice"] = state
    return cfg


def _public_status(job: BuildJob, state: dict[str, Any]) -> str:
    if job.outcome == "unknown":
        return "unknown"
    if state.get("cancel_requested_at") and job.status in OPEN_DB:
        return "cancelling"
    return job.status


def _wire_artifact(artifact: dict[str, Any], message_id: Optional[str]) -> dict[str, Any]:
    aid = str(artifact.get("id") or artifact.get("attachment_id") or "")
    mime = str(artifact.get("mime_type") or "")
    payload = {
        "id": aid, "attachment_id": aid, "message_id": message_id,
        "filename": str(artifact.get("filename") or ""), "mime_type": mime,
        "size_bytes": int(artifact.get("size_bytes") or 0),
        "created_at": artifact.get("created_at"),
    }
    if aid and message_id:
        payload["download_url"] = f"{settings.api_prefix}/files/{message_id}/{aid}"
        # ONE preview policy for every surface (L7's artifact_kinds). This
        # function used to carry its own third copy of the MIME set, and that
        # copy silently omitted PPTX — the same file was previewable in chat
        # and not previewable when a durable voice task produced it.
        if preview_policy(mime) != "none":
            payload["preview_url"] = (
                f"{settings.api_prefix}/files/{message_id}/{aid}/preview?format=html"
            )
        if artifact.get("has_thumb"):
            # Without it a task's image loaded the full-resolution original
            # into a ~370pt card; the live chat frame and the REST history path
            # both carry the thumb already.
            payload["thumb_url"] = (
                f"{settings.api_prefix}/files/{message_id}/{aid}?variant=thumb"
            )
    # Intrinsic to the file, not to the URL — so outside the guard above, or a
    # card with no message id yet lays out at a guessed ratio until it decodes.
    if artifact.get("width") and artifact.get("height"):
        payload["width"] = artifact["width"]
        payload["height"] = artifact["height"]
    payload["kind"] = artifact.get("kind") or kind_for_mime(mime, payload["filename"])
    if artifact.get("role"):
        payload["role"] = artifact["role"]
    return payload


def _snapshot(job: BuildJob) -> dict[str, Any]:
    state = _state(job)
    model = job.model or ""
    if model and settings.security_leak_filter:
        from app.services.model_alias import public_model_label
        model = public_model_label(model)
    metadata = dict(state.get("metadata") or {})
    actions = list(state.get("pending_actions") or [])
    if actions:
        metadata["pending_actions"] = actions
        metadata["pending_action"] = actions[-1]
    return {
        "task_id": job.id, "job_id": job.id,
        "session_id": job.conversation_id, "request_id": state.get("request_id"),
        # Additive: the identity a CLIENT may compare against. Falls back to the
        # conversation id for rows written before the day stamp existed, so an
        # old task keeps behaving exactly as it does today.
        "scope_key": state.get("day_chat_id") or job.conversation_id,
        "title": job.title, "status": _public_status(job, state),
        "user_message": state.get("user_message") or job.title,
        "generation": int(state.get("generation") or 0),
        "revision": int(job.state_revision or 0),
        "delivery_revision": int(job.delivery_revision or 0),
        "receipt_revision": int(job.receipt_revision or 0),
        "spoken_revision": int(job.spoken_revision or 0),
        "text": str(state.get("text") or ""), "model": model,
        "sources": list(state.get("sources") or []),
        "artifacts": [_wire_artifact(a, job.summary_message_id)
                      for a in list(state.get("artifacts") or [])],
        "progress": list(state.get("progress") or []),
        "steering": [{"request_id": s.get("request_id"), "status": s.get("status")}
                     for s in list(state.get("steering") or [])],
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": state.get("updated_at"), "message_id": job.summary_message_id,
        "error": job.user_message, "error_code": job.error_class,
        "metadata": metadata,
        "retry_safe": bool(job.status == "queued" and int(state.get("run_number") or 0) == 0),
    }


def _enqueue_lifecycle(db, job: BuildJob, state: dict[str, Any]) -> None:
    terminal = job.status in TERMINAL_DB
    if job.status == WAITING:
        kind = "needs_approval" if state.get("pending_actions") else "needs_input"
    elif job.status == "completed":
        kind = "mission_completed"
    elif terminal:
        kind = "mission_failed"
    else:
        kind = "mission_started"
    label = "Needs review" if job.outcome == "unknown" else job.status.capitalize()
    db.add(AgentNotifyOutbox(
        id=str(uuid.uuid4()), event_kind=kind,
        title=f"{label}: {job.title}"[:200],
        body=(job.user_message or state.get("text") or "Your voice task is running.")[:300],
        data_json={"route": "mission-control", "mission_id": job.id,
                   "mission_title": job.title[:80], "kind": "job",
                   "voice_task_id": job.id, "session_id": job.conversation_id,
                   "urgent": False, **({"dismiss_after_s": 900} if terminal else {})},
        priority="default",
        dedup_key=(f"voice:{job.id}:{job.status}:{job.outcome or ''}:"
                   f"{int(job.state_revision or 0)}"),
        created_at=_now(),
    ))


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _normalize_actions(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    cards = metadata.get("pending_actions")
    if not isinstance(cards, list):
        cards = [metadata.get("pending_action")] if metadata.get("pending_action") else []
    ids = [str(x) for x in metadata.get("pending_action_ids", []) if x]
    by_id: dict[str, dict[str, Any]] = {}
    for raw in cards:
        if not isinstance(raw, dict):
            continue
        aid = str(raw.get("action_id") or raw.get("id") or "")
        if aid:
            by_id[aid] = {**_copy(raw), "action_id": aid,
                          "status": raw.get("status") or "pending"}
    for aid in ids:
        by_id.setdefault(aid, {"action_id": aid, "status": "pending"})
    return list(by_id.values())


def _is_confirmation_result(value: str) -> bool:
    return tool_result_is_confirmation(value)


class VoiceTaskService:
    def __init__(
        self, runner, session_maker=None, *, max_seconds: Optional[float] = None,
        max_active: Optional[int] = None, lease_seconds: float = DEFAULT_LEASE_SECONDS,
        poll_seconds: Optional[float] = None, action_reader: Optional[ActionReader] = None,
        action_canceller: Optional[ActionCanceller] = None,
        idle_poll_seconds: Optional[float] = None,
        waiting_poll_seconds: Optional[float] = None,
        claim_allowed: Optional[ClaimAllowed] = None,
    ) -> None:
        if session_maker is None:
            from app.db.database import async_session_maker
            session_maker = async_session_maker
        self.runner = runner
        self.session_maker = session_maker
        self.max_seconds = float(max_seconds if max_seconds is not None
                                 else settings.voice_task_max_seconds)
        self.max_active = max(1, int(max_active if max_active is not None
                                     else settings.voice_task_max_active))
        self.lease_seconds = max(1.0, float(lease_seconds))
        configured_poll = max(
            0.05,
            float(poll_seconds if poll_seconds is not None
                  else settings.voice_task_poll_seconds),
        )
        # Three renewals per lease leaves room for a delayed event loop tick or
        # one transient DB failure even when an operator tunes polling upward.
        self.poll_seconds = min(configured_poll, self.lease_seconds / 3.0)
        self.idle_poll_seconds = max(
            self.poll_seconds,
            float(idle_poll_seconds if idle_poll_seconds is not None
                  else DEFAULT_IDLE_POLL_SECONDS),
        )
        self.waiting_poll_seconds = min(
            self.idle_poll_seconds,
            max(
                self.poll_seconds,
                float(waiting_poll_seconds if waiting_poll_seconds is not None
                      else DEFAULT_WAITING_POLL_SECONDS),
            ),
        )
        self.owner = uuid.uuid4().hex
        # Stable per-worker jitter prevents a fleet recreated together from
        # reconnecting to the transaction pooler in the same idle cadence.
        self._idle_jitter = 0.9 + (int(self.owner[:8], 16) / 0xFFFFFFFF) * 0.2
        self.handles: dict[str, _Handle] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._start_lock = asyncio.Lock()
        self._tick_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._supervisor: Optional[asyncio.Task] = None
        self._closing = False
        self._waiting_work = False
        self._action_reader = action_reader or self._read_action_from_platform
        self._action_canceller = action_canceller or self._cancel_action_on_platform
        self._claim_allowed = claim_allowed or _agent_claims_allowed

    @property
    def tasks(self) -> dict[str, asyncio.Task]:
        return {key: h.task for key, h in self.handles.items() if h.task is not None}

    def _filesystem_path_resolver(self):
        tools = getattr(self.runner, "tools", None)
        resolver = getattr(tools, "resolve_operation_path", None)
        return resolver if callable(resolver) else None

    async def start(self) -> None:
        async with self._start_lock:
            if self._supervisor is not None or self._closing:
                return
            initial = await self.tick_once()
            delay = self._next_poll_delay(
                self.poll_seconds, activity=bool(any(initial.values()))
            )
            self._supervisor = asyncio.create_task(
                self._supervise(delay), name=f"voice-task-supervisor:{self.owner[:8]}"
            )

    async def close(self) -> None:
        self._closing = True
        if self._supervisor is not None:
            self._supervisor.cancel()
            await asyncio.gather(self._supervisor, return_exceptions=True)
            self._supervisor = None
        handles = list(self.handles.items())
        for task_id, handle in handles:
            handle.lease_lost.set()
            await self._mark_unknown(
                task_id, handle.token,
                "Your agent restarted before this task's outcome was confirmed. Review existing results before starting it again.",
                "infra_interrupted",
            )
        for _, handle in handles:
            if handle.deadline_task:
                handle.deadline_task.cancel()
            if handle.task and not handle.task.done():
                handle.task.cancel()
        if handles:
            await asyncio.gather(*(h.task for _, h in handles if h.task is not None),
                                 return_exceptions=True)
        self.handles.clear()

    async def _lookup(self, db, user_id: str, task_id: str, *, lock: bool = False) -> BuildJob:
        query = select(BuildJob).where(
            BuildJob.id == task_id, BuildJob.user_id == user_id,
            BuildJob.source_kind == SOURCE_KIND,
        )
        if lock:
            query = query.with_for_update()
        job = (await db.execute(query)).scalar_one_or_none()
        if job is None:
            raise VoiceTaskError(404, "Voice task not found")
        return job

    async def _commit_state(
        self, db, job: BuildJob, state: dict[str, Any], *,
        values: Optional[dict[str, Any]] = None, token: Optional[str] = None,
        expected_status: Optional[str] = None, require_live_lease: bool = False,
    ) -> BuildJob:
        old_revision = int(job.state_revision or 0)
        stmt = update(BuildJob).where(BuildJob.id == job.id,
                                      BuildJob.state_revision == old_revision)
        if token is not None:
            stmt = stmt.where(BuildJob.claim_owner == self.owner,
                              BuildJob.claim_token == token)
        if expected_status is not None:
            stmt = stmt.where(BuildJob.status == expected_status)
        if require_live_lease:
            stmt = stmt.where(BuildJob.claim_expires_at > _now())
        result = await db.execute(stmt.values(
            config_json=_config(job, state), state_revision=old_revision + 1,
            **(values or {}),
        ).execution_options(synchronize_session=False))
        if (result.rowcount or 0) != 1:
            await db.rollback()
            raise _LeaseLost("Voice task execution claim changed")
        await db.flush()
        await db.refresh(job)
        return job

    async def submit(
        self, *, user_id: str, request_id: str, session_id: str, message: str,
        model: Optional[str] = None, user_message: Optional[str] = None,
        client_tz: Optional[str] = None,
    ) -> dict[str, Any]:
        if self._closing:
            raise VoiceTaskError(503, "Voice task worker is shutting down")
        await self.start()
        request_id, message = request_id.strip(), message.strip()
        task_id = task_id_for(user_id, session_id, request_id)
        fingerprint = _fingerprint(message)
        async with self.session_maker() as db:
            await db.execute(select(User.id).where(User.id == user_id).with_for_update())
            conv = (await db.execute(select(Conversation).where(
                Conversation.id == session_id, Conversation.user_id == user_id,
            ))).scalar_one_or_none()
            if conv is None:
                raise VoiceTaskError(404, "Voice conversation not found")
            existing = await db.get(BuildJob, task_id)
            if existing is not None:
                old = _state(existing)
                if (existing.user_id != user_id or existing.source_kind != SOURCE_KIND
                        or old.get("fingerprint") != fingerprint
                        or old.get("requested_model", "") != (model or "")):
                    raise VoiceTaskError(
                        409, "This request identifier was already used with different input"
                    )
                return _snapshot(existing)
            active_count = len((await db.execute(select(BuildJob.id).where(
                BuildJob.user_id == user_id, BuildJob.source_kind == SOURCE_KIND,
                BuildJob.status.in_(tuple(OPEN_DB)),
            ))).all())
            if active_count >= self.max_active:
                raise VoiceTaskError(
                    429, "There are already voice tasks running; check, correct, or cancel one first"
                )
            now = _now()
            state = {
                "request_id": request_id, "fingerprint": fingerprint,
                # The DAY this task belongs to, captured once at submit. It is
                # what scopes the task afterwards: keying on the conversation id
                # alone made a task unaddressable the moment the voice socket
                # reconnected (a new Conversation, a new id), so "cancel that"
                # seconds later answered "not available in this conversation".
                "day_chat_id": conv.day_chat_id,
                "requested_model": model or "", "original_prompt": message,
                "execution_prompt": message, "user_message": user_message,
                "client_tz": client_tz, "run_number": 0, "generation": 0,
                "text": "",
                "sources": [], "artifacts": [], "progress": [], "steering": [],
                "pending_actions": [], "completed_tools": [],
                "resolved_operations": [],
                "updated_at": _iso(now),
            }
            job = BuildJob(
                id=task_id, user_id=user_id, title=message[:200], prompt=message,
                job_type="agent_task", source_kind=SOURCE_KIND, source_id=session_id,
                conversation_id=session_id,
                idempotency_key=f"voice:{hashlib.sha256(request_id.encode()).hexdigest()}",
                status="queued", model=model or "", layer=0, steps_json="[]",
                config_json={"voice": state}, state_revision=1, created_at=now,
            )
            db.add(job)
            _enqueue_lifecycle(db, job, state)
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                existing = await self._lookup(db, user_id, task_id)
                old = _state(existing)
                if (old.get("fingerprint") != fingerprint
                        or old.get("requested_model", "") != (model or "")):
                    raise VoiceTaskError(409, "Request identifier conflict")
                return _snapshot(existing)
        # Claim synchronously so submission latency never depends on the idle
        # sweep. Then wake the supervisor: start() may have put it into a long
        # idle wait before this row existed, and the new owner needs its first
        # short-cadence heartbeat well inside the lease window.
        await self.tick_once()
        self._wake.set()
        await asyncio.sleep(0)
        return await self.get(user_id=user_id, task_id=task_id)

    async def get(self, *, user_id: str, task_id: str) -> dict[str, Any]:
        async with self.session_maker() as db:
            return _snapshot(await self._lookup(db, user_id, task_id))

    async def list(self, *, user_id: str, session_id: str, limit: int = 24) -> dict[str, Any]:
        await self.start()
        async with self.session_maker() as db:
            # Scope by the DAY, not by the socket. A task survives the call that
            # started it (the module docstring says so), but its list did not:
            # any reconnect or second call created a new Conversation, and the
            # work the user started minutes earlier vanished from the resume
            # snapshot while still running.
            scope_key = (await db.execute(select(Conversation.day_chat_id).where(
                Conversation.id == session_id, Conversation.user_id == user_id,
            ))).scalar_one_or_none()
            where = [BuildJob.user_id == user_id, BuildJob.source_kind == SOURCE_KIND]
            if scope_key:
                where.append(BuildJob.conversation_id.in_(
                    select(Conversation.id).where(
                        Conversation.user_id == user_id,
                        Conversation.day_chat_id == scope_key,
                    )
                ))
            else:
                where.append(BuildJob.conversation_id == session_id)
            jobs = (await db.execute(select(BuildJob).where(*where).order_by(
                BuildJob.created_at.desc()
            ).limit(min(100, max(1, limit))))).scalars().all()
            return {
                "tasks": [_snapshot(job) for job in reversed(jobs)],
                # What the caller should compare a task against. Absent on an
                # older agent image; the relay then falls back to session_id.
                "scope_key": scope_key or session_id,
            }

    async def ack(
        self, *, user_id: str, task_id: str, revision: int, kind: str,
    ) -> dict[str, Any]:
        column = {"delivery": "delivery_revision", "received": "receipt_revision",
                  "spoken": "spoken_revision"}.get(kind)
        if column is None:
            raise VoiceTaskError(422, "Unknown acknowledgement kind")
        async with self.session_maker() as db:
            job = await self._lookup(db, user_id, task_id, lock=True)
            revision = min(max(0, int(revision)), int(job.state_revision or 0))
            if revision > int(getattr(job, column) or 0):
                setattr(job, column, revision)
                await db.commit()
            return _snapshot(job)

    async def cancel(self, *, user_id: str, task_id: str) -> dict[str, Any]:
        action_ids: list[str] = []
        waiting = False
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id, lock=True)
                state = _state(job)
                if job.outcome == "unknown" or job.status in TERMINAL_DB:
                    return _snapshot(job)
                if job.status == "queued":
                    state["cancel_requested_at"] = _iso()
                    job = await self._commit_state(
                        db, job, state,
                        values={"status": "cancelled", "outcome": "cancelled",
                                "completed_at": _now(), "error_class": "user_cancelled",
                                "user_message": "Task cancelled before execution started.",
                                "claim_owner": None, "claim_token": None,
                                "claim_expires_at": None},
                        expected_status="queued",
                    )
                    _enqueue_lifecycle(db, job, state)
                    _result_frame = await self._persist_message(db, job, state)
                    await db.commit()
                    _cancelled, _uid = _snapshot(job), job.user_id
                    # Terminal, and it announced nothing at all: the
                    # cancellation row reached an open thread only on the next
                    # history fetch. Every terminal path announces its row.
                    await self._broadcast(_uid, _cancelled, message=_result_frame)
                    return _cancelled
                waiting = job.status == WAITING
                if not state.get("cancel_requested_at"):
                    state["cancel_requested_at"] = _iso()
                    job = await self._commit_state(db, job, state,
                                                   expected_status=job.status)
                    await db.commit()
                action_ids = [
                    str(a.get("action_id"))
                    for a in state.get("pending_actions", [])
                    if a.get("action_id") and a.get("status") in {"pending", "approved"}
                ]

        handle = self.handles.get(task_id)
        if handle:
            handle.cancel_event.set()
        # Pending-action cancellation is a separate guarded platform write. If
        # approval won the race, its authoritative state is reconciled instead.
        for action_id in action_ids:
            try:
                remote = await self._action_canceller(user_id, action_id)
                if remote:
                    await self._merge_action(task_id, {**remote, "action_id": action_id})
            except Exception:
                logger.warning(
                    "Could not cancel pending action %s; reconciliation will retry",
                    action_id,
                )
        if waiting and action_ids:
            await self._reconcile_job(task_id)
        elif waiting:
            await self._terminalize_waiting_cancel(task_id)
        self._wake.set()
        return await self.get(user_id=user_id, task_id=task_id)

    async def steer(
        self, *, user_id: str, task_id: str, request_id: str, message: str,
    ) -> dict[str, Any]:
        message = message.strip()
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id, lock=True)
                state = _state(job)
                steering = list(state.get("steering") or [])
                existing = next(
                    (s for s in steering if s.get("request_id") == request_id), None
                )
                if existing:
                    if existing.get("message") != message:
                        raise VoiceTaskError(
                            409, "This correction identifier was already used with different input"
                        )
                    return _snapshot(job)
                if job.outcome == "unknown":
                    raise VoiceTaskError(
                        409,
                        "This task has an unknown outcome. Review its existing results before starting new work.",
                    )
                if len(steering) >= MAX_STEERING:
                    raise VoiceTaskError(429, "This task has reached its correction limit")
                steering.append({"request_id": request_id, "message": message,
                                 "status": "pending", "created_at": _iso()})
                state["steering"] = steering
                # A correction is an explicit freshness boundary.  Read
                # checkpoints may describe the old scope; mutation outcomes
                # remain durable replay guards.
                state["resolved_operations"] = [
                    item for item in state.get("resolved_operations", [])
                    if isinstance(item, dict) and not resolved_operation_is_read(item)
                ]
                values: dict[str, Any] = {}
                if job.status in TERMINAL_DB:
                    state["generation"] = int(state.get("generation") or 0) + 1
                    state["execution_prompt"] = self._continuation_prompt(
                        job, state,
                        reason="The user supplied a correction after the previous result.",
                    )
                    state.pop("cancel_requested_at", None)
                    values = {"status": "queued", "outcome": None,
                              "completed_at": None, "error_class": None,
                              "user_message": None, "claim_owner": None,
                              "claim_token": None, "claim_expires_at": None}
                old_status = job.status
                job = await self._commit_state(
                    db, job, state, values=values, expected_status=old_status
                )
                await db.commit()
                snapshot = _snapshot(job)
        self._wake.set()
        return snapshot

    async def _drain_controls(
        self, task_id: str, user_id: str, token: str,
    ) -> list[str]:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id, lock=True)
                state = _state(job)
                if (job.status != RUNNING or job.claim_owner != self.owner
                        or job.claim_token != token or not job.claim_expires_at
                        or job.claim_expires_at <= _now()):
                    raise _LeaseLost("Voice task lease is no longer valid")
                if state.get("cancel_requested_at"):
                    raise _UserCancelled("Voice task cancellation requested")
                pending = [s for s in state.get("steering", [])
                           if s.get("status") == "pending"]
                if pending:
                    for correction in pending:
                        correction["status"] = "applied"
                        correction["applied_at"] = _iso()
                    job = await self._commit_state(
                        db, job, state, token=token, expected_status=RUNNING,
                        require_live_lease=True,
                    )
                    await db.commit()
                return [str(s.get("message") or "") for s in pending
                        if s.get("message")]

    async def _progress(
        self, task_id: str, user_id: str, token: str, event: dict[str, Any], *,
        sources: Optional[list[dict[str, Any]]] = None,
        artifact: Optional[dict[str, Any]] = None,
        resolved_operation: Optional[dict[str, Any]] = None,
    ) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id, lock=True)
                state = _state(job)
                if (job.status != RUNNING or job.claim_owner != self.owner
                        or job.claim_token != token or not job.claim_expires_at
                        or job.claim_expires_at <= _now()):
                    return
                event = {**_copy(event), "ts": _iso()}
                state["progress"] = [*state.get("progress", []), event][-MAX_PROGRESS:]
                known_sources = {
                    str(s.get("url")): s for s in state.get("sources", [])
                    if isinstance(s, dict) and s.get("url")
                }
                for source in sources or []:
                    if isinstance(source, dict) and source.get("url"):
                        known_sources[str(source["url"])] = _copy(source)
                state["sources"] = list(known_sources.values())[:MAX_SOURCES]
                if artifact:
                    artifacts = {
                        str(a.get("id") or a.get("filename")): a
                        for a in state.get("artifacts", []) if isinstance(a, dict)
                    }
                    artifacts[str(artifact.get("id") or artifact.get("filename"))] = _copy(artifact)
                    state["artifacts"] = list(artifacts.values())[:MAX_ARTIFACTS]
                if event.get("type") == "tool.end" and event.get("ok"):
                    name = str(event.get("name") or "")
                    completed = list(state.get("completed_tools") or [])
                    if name and name not in completed:
                        completed.append(name)
                    state["completed_tools"] = completed[-100:]
                if resolved_operation:
                    operation_name = str(resolved_operation.get("tool_name") or "")
                    operation_input = resolved_operation.get("tool_input") or {}
                    checkpoint = tool_operation_checkpoint(
                        operation_name,
                        operation_input,
                        filesystem_path_resolver=self._filesystem_path_resolver(),
                    )
                    operation_kind = tool_operation_kind(
                        operation_name, operation_input,
                    )
                    checkpointable = (
                        operation_kind == "mutation"
                        or tool_operation_cacheable_read(
                            operation_name, operation_input,
                        )
                    )
                    # Re-evaluate producer metadata here so a mixed-version
                    # runner cannot persist a volatile status/list/output read
                    # as a mutation replay guard.
                    if not checkpointable:
                        resolved_operation = None
                if resolved_operation:
                    operation_key = str(resolved_operation.get("operation_key") or "")
                    prior_operations = [
                        item for item in state.get("resolved_operations", [])
                        if isinstance(item, dict) and item.get("operation_key")
                    ]
                    if operation_kind == "mutation":
                        prior_operations = [
                            item for item in prior_operations
                            if not mutation_invalidates_read(
                                operation_name,
                                operation_input,
                                item,
                                filesystem_path_resolver=self._filesystem_path_resolver(),
                            )
                        ]
                    operations = {
                        str(item.get("operation_key")): item
                        for item in prior_operations
                    }
                    if operation_key:
                        durable_operation = {
                            **_copy(resolved_operation),
                            **checkpoint,
                        }
                        # Arguments are needed only while applying freshness;
                        # never retain them in the durable checkpoint.
                        durable_operation.pop("tool_input", None)
                        operations[operation_key] = durable_operation
                        state["resolved_operations"] = list(operations.values())[
                            -MAX_RESOLVED_OPERATIONS:
                        ]
                try:
                    job = await self._commit_state(
                        db, job, state, token=token, expected_status=RUNNING,
                        require_live_lease=True,
                    )
                except _LeaseLost:
                    return
                if artifact:
                    await self._persist_message(db, job, state)
                db.add(JobEvent(
                    job_id=job.id, user_id=user_id, kind="tool_call",
                    label=str(event.get("name") or event.get("stage") or "Working")[:200],
                    level="info", metadata_json=json.dumps(event, default=str),
                ))
                await db.commit()

    async def _run(self, task_id: str, user_id: str, handle: _Handle) -> None:
        try:
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id)
                state = _state(job)
                if (job.status != RUNNING or job.claim_owner != self.owner
                        or job.claim_token != handle.token or not job.claim_expires_at
                        or job.claim_expires_at <= _now()):
                    return
                prompt = str(state.get("execution_prompt") or job.prompt)
                session_id = str(job.conversation_id or "")
                model, client_tz = job.model, state.get("client_tz")
                resolved_operations = list(state.get("resolved_operations") or [])

            async def on_tool_event(ev: dict[str, Any]) -> None:
                from app.api.api_v1 import _vs_args, _vs_defence, _vs_sources
                name = str(ev.get("name") or "")[:128]
                raw = str(ev.get("result") or "")
                item = {"type": "tool.start" if ev.get("phase") == "start"
                        else "tool.end",
                        "call_id": str(ev.get("call_id") or "")[:64],
                        "name": name, "args": _vs_args(name, ev.get("input") or {})}
                found_sources: list[dict[str, Any]] = []
                resolved_operation = None
                if ev.get("phase") != "start":
                    ok = not _vs_defence(raw).strip().upper().startswith("ERROR")
                    item.update(ok=ok, elapsed_ms=int(ev.get("elapsed_ms") or 0))
                    found_sources = _vs_sources(name, ev.get("input") or {}, raw)
                    if (ok and name not in {"create_job", "update_job"}
                            and not _is_confirmation_result(raw)):
                        checkpoint = tool_operation_checkpoint(
                            name,
                            ev.get("input") or {},
                            filesystem_path_resolver=self._filesystem_path_resolver(),
                        )
                        checkpointable = (
                            checkpoint["operation_kind"] == "mutation"
                            or tool_operation_cacheable_read(
                                name, ev.get("input") or {},
                            )
                        )
                        if checkpointable:
                            resolved_operation = {
                                "operation_key": tool_operation_key(
                                    name, ev.get("input") or {},
                                ),
                                "operation_id": str(ev.get("call_id") or "")[:64],
                                "tool_name": name,
                                "status": "executed",
                                "ok": True,
                                "result": raw[:MAX_RESOLVED_RESULT],
                                "tool_input": _copy(ev.get("input") or {}),
                                **checkpoint,
                            }
                await self._progress(task_id, user_id, handle.token, item,
                                     sources=found_sources,
                                     resolved_operation=resolved_operation)

            async def on_attachment(_message_id, attachment) -> None:
                await self._progress(task_id, user_id, handle.token,
                                     {"type": "artifact"}, artifact=attachment)

            async def on_status(stage) -> None:
                await self._progress(task_id, user_id, handle.token,
                                     {"type": "status", "stage": str(stage)[:80]})

            result = await self.runner.run(
                user_message=prompt, user_id=user_id, session_id=session_id,
                model_override=model or None, channel="voice", managed_voice_task=True,
                save_user_message=False, save_assistant_message=False,
                disable_post_processing=True, current_job_id=task_id,
                client_tz=client_tz, on_tool_event=on_tool_event,
                on_attachment=on_attachment, on_status=on_status,
                cancel_check=handle.should_stop,
                steering_check=lambda: self._drain_controls(
                    task_id, user_id, handle.token
                ),
                managed_resolved_operations=resolved_operations,
            )
            await self._assert_final_barrier(task_id, user_id, handle)
            metadata = dict(getattr(result, "metadata", None) or {})
            if _normalize_actions(metadata):
                await self._finish_owned(task_id, user_id, handle.token, WAITING,
                                         result=result)
            elif getattr(result, "stopped_reason", None):
                await self._finish_owned(
                    task_id, user_id, handle.token, "failed", result=result,
                    error="The task stopped before finishing. Review its partial result.",
                    error_code=str(result.stopped_reason)[:40],
                )
            elif not str(getattr(result, "text", "") or "").strip():
                await self._finish_owned(
                    task_id, user_id, handle.token, "unknown", result=result,
                    error="The task returned no confirmed result. Review its progress before starting it again.",
                    error_code="empty_result",
                )
            else:
                await self._finish_owned(task_id, user_id, handle.token,
                                         "completed", result=result)
        except _UserCancelled:
            await self._finish_owned(
                task_id, user_id, handle.token, "cancelled",
                error="Task cancelled. An external action already in progress may still finish.",
                error_code="user_cancelled", allow_expired=True,
            )
        except _DeadlineReached:
            await self._mark_unknown(
                task_id, handle.token,
                "The task reached its time limit before completion was confirmed. Review progress before starting it again.",
                "timeout",
            )
        except _LeaseLost:
            await self._mark_unknown(
                task_id, handle.token,
                "Execution ownership was lost before the outcome was confirmed. Review existing results before starting it again.",
                "infra_interrupted",
            )
        except asyncio.CancelledError:
            if self._closing:
                raise
            if handle.cancel_event.is_set():
                await self._finish_owned(
                    task_id, user_id, handle.token, "cancelled",
                    error=(
                        "Task cancelled. An external action already in progress may still finish."
                    ),
                    error_code="user_cancelled", allow_expired=True,
                )
            elif handle.deadline_event.is_set():
                await self._mark_unknown(
                    task_id, handle.token,
                    "The task reached its time limit before completion was confirmed. Review progress before starting it again.",
                    "timeout",
                )
            else:
                await self._mark_unknown(
                    task_id, handle.token,
                    "Execution stopped before the task's outcome was confirmed. Review existing results before starting it again.",
                    "infra_interrupted",
                )
        except Exception:
            logger.exception("Voice task execution failed: %s", task_id)
            await self._mark_unknown(
                task_id, handle.token,
                "The task stopped before completion was confirmed. Review its progress and existing results.",
                "tool_failure",
            )

    async def _assert_final_barrier(
        self, task_id: str, user_id: str, handle: _Handle,
    ) -> None:
        if handle.deadline_event.is_set():
            raise _DeadlineReached()
        if handle.lease_lost.is_set():
            raise _LeaseLost()
        async with self.session_maker() as db:
            job = await self._lookup(db, user_id, task_id)
            state = _state(job)
            if (job.status != RUNNING or job.claim_owner != self.owner
                    or job.claim_token != handle.token or not job.claim_expires_at
                    or job.claim_expires_at <= _now()):
                raise _LeaseLost()
            if state.get("cancel_requested_at"):
                raise _UserCancelled()

    async def _persist_message(
        self, db, job: BuildJob, state: dict[str, Any], result=None,
    ) -> Optional[dict[str, Any]]:
        """Commit the task's result row. Returns the LIVE FRAME for it.

        The row was always written correctly and never announced: a task the
        user started by voice reached an already-open thread only on the next
        history fetch, so it looked like nothing had happened for as long as the
        user kept reading. The frame is built here, from the row we just wrote,
        because only this function knows the message id the artifact URLs are
        scoped to; it is SENT by `_broadcast`, after the commit.
        """
        conv = await db.get(Conversation, job.conversation_id)
        if not conv or conv.user_id != job.user_id:
            return None
        message_id = str(uuid.uuid5(_TASK_NAMESPACE, f"result:{job.id}"))
        msg = await db.get(Message, message_id)
        fresh = msg is None
        old_tokens = 0 if fresh else (msg.tokens_prompt or 0) + (msg.tokens_completion or 0)
        if fresh:
            from app.db.message_helpers import resolve_day_chat_id_for_now
            day_id = await resolve_day_chat_id_for_now(
                db, job.user_id, tz_override=state.get("client_tz")
            )
            msg = Message(
                id=message_id, conversation_id=conv.id, day_chat_id=day_id,
                role="assistant", channel="voice", source="voice_task", content="",
            )
            db.add(msg)
        msg.content = state.get("text") or job.user_message or f"Working on: {job.title}"
        msg.model_used = job.model
        # The file route authorizes against this field, committed before a task
        # snapshot publishes its download/preview URL.
        msg.attachments = list(state.get("artifacts") or []) or None
        actions = list(state.get("pending_actions") or [])
        metadata = {
            "voice_task_id": job.id, "job_id": job.id,
            "status": _public_status(job, state),
            "sources": list(state.get("sources") or []),
            **dict(state.get("metadata") or {}),
        }
        if actions:
            metadata["pending_actions"] = actions
            metadata["pending_action"] = actions[-1]
        msg.metadata_json = json.dumps(metadata, default=str)
        if result is not None:
            msg.tokens_prompt = int(getattr(result, "tokens_input", 0) or 0)
            msg.tokens_completion = int(getattr(result, "tokens_output", 0) or 0)
        delta = (msg.tokens_prompt or 0) + (msg.tokens_completion or 0) - old_tokens
        # Several managed tasks for the same user may finish together. Atomic
        # counter updates avoid lost increments and ORM StaleDataError across
        # those independent transactions/replicas.
        await db.execute(update(Conversation).where(
            Conversation.id == conv.id,
        ).values(
            message_count=func.coalesce(Conversation.message_count, 0) + int(fresh),
            total_tokens=func.coalesce(Conversation.total_tokens, 0) + delta,
        ).execution_options(synchronize_session=False))
        if msg.day_chat_id:
            await db.execute(update(DayChat).where(
                DayChat.id == msg.day_chat_id,
            ).values(
                message_count=func.coalesce(DayChat.message_count, 0) + int(fresh),
                total_tokens=func.coalesce(DayChat.total_tokens, 0) + delta,
                last_message_at=_now(),
            ).execution_options(synchronize_session=False))
        job.summary_message_id = message_id
        from app.api.message_frames import message_frame
        return message_frame(
            msg,
            channel="voice",
            day_chat_id=msg.day_chat_id,
            attachments=[
                _wire_artifact(a, message_id)
                for a in (state.get("artifacts") or [])
            ],
            # The row id is stable across generations, so a task corrected by
            # `steer` finishes a second time under the id the first result
            # already used. `state_revision` is monotonic per job, so "a higher
            # revision replaces what is on screen" is a rule the client can
            # apply without knowing anything about tasks.
            revision=int(getattr(job, "state_revision", 0) or 0),
        )

    async def _finish_owned(
        self, task_id: str, user_id: str, token: str, outcome: str, *,
        result=None, error: Optional[str] = None,
        error_code: Optional[str] = None, allow_expired: bool = False,
    ) -> Optional[dict[str, Any]]:
        snapshot: Optional[dict[str, Any]] = None
        _result_frame: Optional[dict[str, Any]] = None
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await self._lookup(db, user_id, task_id, lock=True)
                if (job.status != RUNNING or job.claim_owner != self.owner
                        or job.claim_token != token or (
                            not allow_expired and (
                                not job.claim_expires_at or job.claim_expires_at <= _now()
                            )
                        )):
                    return None
                state = _state(job)
                metadata = dict(getattr(result, "metadata", None) or {}) if result else {}
                result_values: dict[str, Any] = {}
                if result is not None:
                    state["text"] = str(getattr(result, "text", "") or "")[:MAX_TEXT]
                    state["metadata"] = {
                        key: value for key, value in metadata.items()
                        if key not in {"pending_action", "pending_actions",
                                      "pending_action_ids"}
                    }
                    # Keep result fields inside the same fenced UPDATE as the
                    # semantic state transition. Assigning them on the ORM row
                    # could autoflush before the claim predicate is checked.
                    result_values = {
                        "model": str(getattr(result, "model", "") or job.model)[:50],
                        "total_tokens": int(getattr(result, "tokens_total", 0) or 0),
                    }
                actions = _normalize_actions(metadata)
                if actions:
                    state["pending_actions"] = actions
                pending_corrections = any(
                    s.get("status") == "pending" for s in state.get("steering", [])
                )
                if state.get("cancel_requested_at"):
                    outcome = "cancelled"
                    error = error or (
                        "Task cancelled. An external action already in progress may still finish."
                    )
                    error_code = error_code or "user_cancelled"
                if outcome == WAITING:
                    values = {
                        "status": WAITING, "outcome": None, "completed_at": None,
                        "error_class": "confirmation_required",
                        "user_message": (
                            "This task needs your confirmation. Review each action card in the conversation."
                        ),
                        "claim_owner": None, "claim_token": None,
                        "claim_expires_at": None,
                        **result_values,
                    }
                elif outcome == "completed" and pending_corrections:
                    state["execution_prompt"] = self._continuation_prompt(
                        job, state,
                        reason="A correction arrived while the previous result was finishing.",
                    )
                    values = {
                        "status": "queued", "outcome": None, "completed_at": None,
                        "error_class": None, "user_message": None,
                        "claim_owner": None, "claim_token": None,
                        "claim_expires_at": None,
                        **result_values,
                    }
                else:
                    db_status = "failed" if outcome == "unknown" else outcome
                    values = {
                        "status": db_status, "outcome": outcome,
                        "completed_at": _now(), "error_class": error_code,
                        "user_message": error, "claim_owner": None,
                        "claim_token": None, "claim_expires_at": None,
                        **result_values,
                    }
                try:
                    job = await self._commit_state(
                        db, job, state, values=values, token=token,
                        expected_status=RUNNING, require_live_lease=not allow_expired,
                    )
                except _LeaseLost:
                    return None
                _result_frame = await self._persist_message(db, job, state, result=result)
                _enqueue_lifecycle(db, job, state)
                db.add(JobEvent(
                    job_id=job.id, user_id=job.user_id, kind="output_posted",
                    label=job.title, status=job.status, level="info",
                ))
                await db.commit()
                snapshot = _snapshot(job)
        if snapshot:
            await self._broadcast(user_id, snapshot, message=_result_frame)
        # _completed removes the old handle before waking the supervisor. A
        # direct wake here could reclaim this same queued task while the old
        # handle still occupies (and is about to mutate) the handle slot.
        return snapshot

    async def _mark_unknown(
        self, task_id: str, token: str, message: str, code: str,
    ) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = (await db.execute(select(BuildJob).where(
                    BuildJob.id == task_id, BuildJob.source_kind == SOURCE_KIND,
                ).with_for_update())).scalar_one_or_none()
                if (job is None or job.status != RUNNING
                        or job.claim_owner != self.owner or job.claim_token != token):
                    return
                state = _state(job)
                try:
                    job = await self._commit_state(
                        db, job, state,
                        values={"status": "failed", "outcome": "unknown",
                                "completed_at": _now(), "error_class": code,
                                "user_message": message, "claim_owner": None,
                                "claim_token": None, "claim_expires_at": None},
                        token=token, expected_status=RUNNING,
                    )
                except _LeaseLost:
                    return
                _result_frame = await self._persist_message(db, job, state)
                _enqueue_lifecycle(db, job, state)
                await db.commit()
                snapshot, uid = _snapshot(job), job.user_id
        await self._broadcast(uid, snapshot, message=_result_frame)

    async def _broadcast(
        self, user_id: str, snapshot: dict[str, Any],
        message: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from app.api.ws_chat import broadcast_to_user
            await broadcast_to_user(user_id, {
                "type": "job_update", "job_id": snapshot["task_id"],
                "status": snapshot["status"], "name": snapshot["title"],
            })
            await broadcast_to_user(
                user_id, {"type": "voice_task.updated", "task": snapshot}
            )
            if message:
                # The RESULT ROW itself, so an open thread paints the answer as
                # it lands instead of on the next history fetch. Safe to repeat:
                # the row id is the deterministic uuid5(result:<job id>) and the
                # client de-dupes by id, so a re-broadcast is a no-op.
                await broadcast_to_user(user_id, message)
        except Exception:
            logger.debug(
                "Voice task broadcast unavailable; canonical state remains persisted"
            )

    def _continuation_prompt(
        self, job: BuildJob, state: dict[str, Any], *, reason: str,
    ) -> str:
        actions = [{"action_id": a.get("action_id"),
                    "tool_name": a.get("tool_name"), "status": a.get("status"),
                    "result": a.get("result")}
                   for a in state.get("pending_actions", [])]
        operations = [{
            "operation_key": item.get("operation_key"),
            "operation_id": item.get("operation_id"),
            "tool_name": item.get("tool_name"),
            "status": item.get("status"),
            "result": item.get("result"),
            "staged_operation_key": item.get("staged_operation_key"),
            "operation_aliases": item.get("operation_aliases"),
        } for item in state.get("resolved_operations", []) if isinstance(item, dict)]
        return (
            "Continue the same managed task from its durable checkpoint. "
            f"{reason} Preserve completed research and files. Reuse the recorded result for an "
            "exact resolved operation; a distinct operation may use the same tool with different "
            "arguments. Produce a final written result that accurately "
            "describes rejected, failed, or uncertain operations.\n\n"
            f"Original request:\n{state.get('original_prompt') or job.prompt}\n\n"
            f"Previous checkpoint:\n{str(state.get('text') or '')[:20000]}\n\n"
            '<external_content untrusted="true" source="pending_action_ledger">\n'
            f"{json.dumps(actions, ensure_ascii=False, default=str)[:20000]}\n"
            "</external_content>\n\n"
            '<external_content untrusted="true" source="resolved_operation_ledger">\n'
            f"{json.dumps(operations, ensure_ascii=False, default=str)[:30000]}\n"
            "</external_content>"
        )

    async def _heartbeat_owned_in_session(
        self, db, *, now: datetime, jobs: Optional[list[BuildJob]] = None,
    ) -> None:
        expiry = now + timedelta(seconds=self.lease_seconds)
        live = {
            task_id: handle for task_id, handle in list(self.handles.items())
            if handle.task and not handle.task.done()
        }
        if not live:
            return
        if jobs is None:
            jobs = (await db.execute(select(BuildJob).where(
                BuildJob.id.in_(list(live)), BuildJob.source_kind == SOURCE_KIND,
            ).with_for_update())).scalars().all()
        by_id = {job.id: job for job in jobs}
        for task_id, handle in live.items():
            job = by_id.get(task_id)
            # A shared census uses SKIP LOCKED. Absence can therefore mean a
            # short control transaction owns the row, not that our lease died;
            # the next short-cadence heartbeat will decide authoritatively.
            if job is None:
                continue
            if (job.status != RUNNING or job.claim_owner != self.owner
                    or job.claim_token != handle.token or not job.claim_expires_at
                    or job.claim_expires_at <= now):
                handle.lease_lost.set()
                continue
            job.claim_expires_at = expiry
            if _state(job).get("cancel_requested_at"):
                handle.cancel_event.set()

    async def _heartbeat_owned(self) -> None:
        async with self.session_maker() as db:
            await self._heartbeat_owned_in_session(db, now=_now())
            await db.commit()

    async def _expire_running_in_session(
        self, db, *, now: datetime, jobs: Optional[list[BuildJob]] = None,
    ) -> list[tuple[str, dict[str, Any], Optional[dict[str, Any]]]]:
        if jobs is None:
            jobs = (await db.execute(select(BuildJob).where(
                BuildJob.source_kind == SOURCE_KIND, BuildJob.status == RUNNING,
                BuildJob.claim_expires_at.is_not(None),
                BuildJob.claim_expires_at <= now,
            ).with_for_update(skip_locked=True))).scalars().all()
        expired: list[tuple[str, dict[str, Any], Optional[dict[str, Any]]]] = []
        for job in jobs:
            if (job.status != RUNNING or not job.claim_expires_at
                    or job.claim_expires_at > now):
                continue
            handle = self.handles.get(job.id)
            if handle:
                handle.lease_lost.set()
            state = _state(job)
            job.status = "failed"
            job.outcome = "unknown"
            job.completed_at = now
            job.error_class = "lease_expired"
            job.user_message = (
                "Execution ownership expired before the outcome was confirmed. "
                "Review existing results before starting it again."
            )
            job.claim_owner = None
            job.claim_token = None
            job.claim_expires_at = None
            job.state_revision = int(job.state_revision or 0) + 1
            job.config_json = _config(job, state)
            # Terminal (failed / outcome unknown). The row was written and never
            # announced, so a lease expiry reached an open thread only on the
            # next history fetch — the same half of the defect the finish path
            # already closed.
            _result_frame = await self._persist_message(db, job, state)
            _enqueue_lifecycle(db, job, state)
            expired.append((job.user_id, _snapshot(job), _result_frame))
        return expired

    async def _expire_running(
        self, *, now: Optional[datetime] = None,
        task_ids: Optional[list[str]] = None,
    ) -> int:
        now = now or _now()
        if task_ids is None:
            async with self.session_maker() as db:
                task_ids = (await db.execute(select(BuildJob.id).where(
                    BuildJob.source_kind == SOURCE_KIND, BuildJob.status == RUNNING,
                    BuildJob.claim_expires_at.is_not(None),
                    BuildJob.claim_expires_at <= now,
                ))).scalars().all()
        expired = 0
        for task_id in task_ids:
            try:
                async with self.session_maker() as db:
                    job = (await db.execute(select(BuildJob).where(
                        BuildJob.id == task_id,
                        BuildJob.source_kind == SOURCE_KIND,
                    ).with_for_update())).scalar_one_or_none()
                    rows = await self._expire_running_in_session(
                        db, now=now, jobs=[job] if job is not None else []
                    )
                    await db.commit()
                for user_id, snapshot, result_frame in rows:
                    await self._broadcast(user_id, snapshot, message=result_frame)
                expired += len(rows)
            except Exception:
                # One malformed historical row must not roll back a healthy
                # task's already-committed heartbeat or block other expiries.
                logger.exception("Could not expire voice task %s", task_id)
        return expired

    async def _claim_queued_in_session(
        self, db, *, candidates: Optional[list[BuildJob]] = None,
    ) -> list[tuple[str, str, str]]:
        """Claim available rows, leaving task launch until after commit."""
        available = max(0, self.max_active - sum(
            1 for h in self.handles.values() if h.task and not h.task.done()
        ))
        try:
            claims_allowed = self._claim_allowed()
        except Exception:
            logger.exception("Voice task claim eligibility check failed")
            claims_allowed = False
        if not available or self._closing or not claims_allowed:
            return []
        if candidates is None:
            candidates = (await db.execute(select(BuildJob).where(
                BuildJob.source_kind == SOURCE_KIND, BuildJob.status == "queued",
            ).order_by(BuildJob.created_at.asc()).limit(available).with_for_update(
                skip_locked=True
            ))).scalars().all()
        claimed: list[tuple[str, str, str]] = []
        for candidate in candidates:
            if len(claimed) >= available:
                break
            if candidate.status != "queued":
                continue
            token, now = uuid.uuid4().hex, _now()
            candidate_state = _state(candidate)
            candidate_state["run_number"] = int(
                candidate_state.get("run_number") or 0
            ) + 1
            candidate.status = RUNNING
            candidate.claim_owner = self.owner
            candidate.claim_token = token
            candidate.claim_expires_at = now + timedelta(seconds=self.lease_seconds)
            candidate.config_json = _config(candidate, candidate_state)
            candidate.state_revision = int(candidate.state_revision or 0) + 1
            claimed.append((candidate.id, candidate.user_id, token))
        return claimed

    async def _claim_queued(self) -> int:
        async with self.session_maker() as db:
            claims = await self._claim_queued_in_session(db)
            await db.commit()
        self._launch_claimed(claims)
        return len(claims)

    def _launch_claimed(self, claims: list[tuple[str, str, str]]) -> None:
        for task_id, user_id, token in claims:
            handle = _Handle(token=token, cancel_event=asyncio.Event(),
                             lease_lost=asyncio.Event(), deadline_event=asyncio.Event())

            async def deadline(h: _Handle = handle, tid: str = task_id) -> None:
                await asyncio.sleep(max(1.0, self.max_seconds))
                h.deadline_event.set()
                await self._mark_unknown(
                    tid, h.token,
                    "The task reached its time limit before completion was confirmed. "
                    "Review progress before starting it again.",
                    "timeout",
                )
                if h.task and not h.task.done():
                    h.task.cancel()

            handle.deadline_task = asyncio.create_task(deadline())
            handle.task = asyncio.create_task(
                self._run(task_id, user_id, handle), name=f"voice-task:{task_id}"
            )
            self.handles[task_id] = handle
            handle.task.add_done_callback(
                lambda done, tid=task_id, tok=token: self._completed(tid, tok, done)
            )

    def _completed(self, task_id: str, token: str, task: asyncio.Task) -> None:
        handle = self.handles.get(task_id)
        if handle and handle.token == token:
            if handle.deadline_task:
                handle.deadline_task.cancel()
            self.handles.pop(task_id, None)
            lock = self._locks.get(task_id)
            if lock is not None and not lock.locked():
                self._locks.pop(task_id, None)
        if not task.cancelled():
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
            if error:
                logger.error("Voice task worker escaped: %s", type(error).__name__)
        self._wake.set()

    async def _read_action_from_platform(
        self, user_id: str, action_id: str,
    ) -> Optional[dict[str, Any]]:
        from app.services.agent_http import get_agent_http_client
        response = await get_agent_http_client().get(
            f"{settings.platform_api_url}/agent/pending-actions/{action_id}",
            headers={"X-Agent-Key": settings.agent_api_key,
                     "X-Agent-User-Id": user_id}, timeout=10.0,
        )
        if response.status_code == 404:
            return {"action_id": action_id, "status": "missing"}
        response.raise_for_status()
        return response.json()

    async def _cancel_action_on_platform(
        self, user_id: str, action_id: str,
    ) -> Optional[dict[str, Any]]:
        from app.services.agent_http import get_agent_http_client
        response = await get_agent_http_client().post(
            f"{settings.platform_api_url}/agent/pending-actions/{action_id}/cancel",
            headers={"X-Agent-Key": settings.agent_api_key,
                     "X-Agent-User-Id": user_id}, timeout=10.0,
        )
        if response.status_code == 404:
            return {"action_id": action_id, "status": "missing"}
        response.raise_for_status()
        return response.json()

    async def resolve_pending_action(
        self, *, action_id: str, outcome: str, detail: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
    ) -> int:
        """Best-effort callback accelerator; polling remains the durable path."""
        async with self.session_maker() as db:
            jobs = (await db.execute(select(BuildJob).where(
                BuildJob.source_kind == SOURCE_KIND, BuildJob.status == WAITING,
            ))).scalars().all()
        matched = 0
        for job in jobs:
            if action_id not in {
                str(a.get("action_id"))
                for a in _state(job).get("pending_actions", [])
            }:
                continue
            matched += 1
            # The action row is authoritative for edited arguments.  Read it
            # even on the callback path so callback-first and poll-first order
            # produce the same ledger.  Newer platform callbacks also carry
            # payload/result directly, which keeps this path complete during a
            # transient reconciliation-read failure.
            remote = None
            try:
                remote = await self._action_reader(job.user_id, action_id)
            except Exception:
                logger.debug(
                    "Authoritative action read unavailable during callback for %s",
                    action_id,
                )
            update_action = {
                **(remote if isinstance(remote, dict) else {}),
                "action_id": action_id,
                "status": outcome,
                "detail": detail,
            }
            if isinstance(payload, dict):
                update_action["payload"] = _copy(payload)
            if isinstance(result, dict):
                update_action["result"] = _copy(result)
            await self._merge_action(
                job.id, update_action,
            )
            await self._reconcile_job(job.id)
        return matched

    async def _merge_action(
        self, task_id: str, action: dict[str, Any],
    ) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await db.get(BuildJob, task_id, with_for_update=True)
                if not job or job.source_kind != SOURCE_KIND or job.status != WAITING:
                    return
                state = _state(job)
                aid = str(action.get("action_id") or action.get("id") or "")
                actions = list(state.get("pending_actions") or [])
                changed = False
                for index, current in enumerate(actions):
                    if str(current.get("action_id")) == aid:
                        current_status = str(current.get("status") or "pending")
                        incoming_status = str(action.get("status") or current_status)
                        # Callbacks and polls may cross. Never let a stale
                        # pending/approved observation resurrect a terminal
                        # operation or move an approved operation backwards.
                        if current_status in ACTION_TERMINAL:
                            if incoming_status != current_status:
                                return
                        if current_status == "approved" and incoming_status == "pending":
                            return
                        merged = {**current, **_copy(action), "action_id": aid}
                        current_payload = current.get("payload")
                        incoming_payload = action.get("payload")
                        if (isinstance(current_payload, dict)
                                and isinstance(incoming_payload, dict)
                                and current_payload != incoming_payload
                                and not isinstance(current.get("staged_payload"), dict)):
                            # payload is mutable while the card is reviewed;
                            # action_id is stable. Retain the proposal as an
                            # alias while making the approved payload current.
                            merged["staged_payload"] = _copy(current_payload)
                        if merged != current:
                            actions[index] = merged
                            changed = True
                        break
                if not changed:
                    return
                state["pending_actions"] = actions
                try:
                    job = await self._commit_state(
                        db, job, state, expected_status=WAITING
                    )
                except _LeaseLost:
                    return
                await self._persist_message(db, job, state)
                await db.commit()

    async def _reconcile_waiting(
        self, task_ids: Optional[list[str]] = None,
    ) -> int:
        if task_ids is None:
            async with self.session_maker() as db:
                task_ids = (await db.execute(select(BuildJob.id).where(
                    BuildJob.source_kind == SOURCE_KIND, BuildJob.status == WAITING,
                ))).scalars().all()
        count = 0
        for task_id in task_ids:
            if await self._reconcile_job(task_id):
                count += 1
        return count

    async def _reconcile_job(self, task_id: str) -> bool:
        async with self.session_maker() as db:
            job = await db.get(BuildJob, task_id)
            if not job or job.source_kind != SOURCE_KIND or job.status != WAITING:
                return False
            user_id = job.user_id
            state = _state(job)
            actions = list(state.get("pending_actions") or [])
            cancelling = bool(state.get("cancel_requested_at"))
        for action in actions:
            if action.get("status") not in {"pending", "approved"}:
                continue
            aid = str(action.get("action_id") or "")
            if not aid:
                continue
            remote = None
            if cancelling:
                try:
                    # The cancellation intent is durable. Reissue only the
                    # platform's guarded pending->rejected transition; if an
                    # approval already won, the authoritative row is returned
                    # unchanged. Read it once more because the approved action
                    # may already have advanced to executed while cancellation
                    # was crossing the service boundary.
                    remote = await self._action_canceller(user_id, aid)
                except Exception:
                    logger.debug(
                        "Pending action cancellation reconciliation unavailable for %s",
                        aid,
                    )
                if isinstance(remote, dict) and remote.get("status") == "approved":
                    try:
                        observed = await self._action_reader(user_id, aid)
                        if observed:
                            remote = observed
                    except Exception:
                        logger.debug(
                            "Approved action outcome unavailable during cancellation for %s",
                            aid,
                        )
            if remote is None:
                try:
                    remote = await self._action_reader(user_id, aid)
                except Exception:
                    logger.debug("Pending action reconciliation unavailable for %s", aid)
                    continue
            if remote:
                await self._merge_action(task_id, {**remote, "action_id": aid})

        async with self.session_maker() as db:
            job = await db.get(BuildJob, task_id)
            if not job or job.status != WAITING:
                return False
            state = _state(job)
            actions = list(state.get("pending_actions") or [])
            cancelling = bool(state.get("cancel_requested_at"))
        if any(a.get("status") == "pending" for a in actions):
            return False
        approved = [a for a in actions if a.get("status") == "approved"]
        if approved:
            now = _now()
            too_old = any(
                (now - (_parse_time(a.get("decided_at"))
                        or _parse_time(a.get("approval_seen_at")) or now)).total_seconds()
                >= APPROVAL_UNKNOWN_SECONDS
                for a in approved
            )
            if too_old:
                await self._terminalize_waiting_unknown(task_id)
                return True
            for action in approved:
                if not action.get("approval_seen_at"):
                    await self._merge_action(
                        task_id, {**action, "approval_seen_at": _iso()}
                    )
            return False
        if any(a.get("status") in {"missing", None, ""} for a in actions):
            await self._terminalize_waiting_unknown(task_id)
            return True
        if cancelling:
            await self._terminalize_waiting_cancel(task_id)
        else:
            await self._queue_after_actions(task_id)
        return True

    async def _queue_after_actions(self, task_id: str) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await db.get(BuildJob, task_id, with_for_update=True)
                if not job or job.status != WAITING:
                    return
                state = _state(job)
                operations = {
                    str(item.get("operation_key")): item
                    for item in state.get("resolved_operations", [])
                    if isinstance(item, dict) and item.get("operation_key")
                }
                action_operations: list[tuple[dict[str, Any], str, str, dict[str, Any], str]] = []
                for action in state.get("pending_actions", []):
                    status = str(action.get("status") or "")
                    name = str(action.get("tool_name") or "")
                    payload = action.get("payload")
                    if status not in {"executed", "failed", "rejected", "expired"}:
                        continue
                    if not name or not isinstance(payload, dict):
                        continue
                    operation_key = tool_operation_key(name, payload)
                    staged_payload = action.get("staged_payload")
                    staged_key = (
                        tool_operation_key(name, staged_payload)
                        if isinstance(staged_payload, dict) else operation_key
                    )
                    action_operations.append(
                        (action, status, name, payload, staged_key)
                    )

                # An independently authorised action whose executed arguments
                # equal another card's original draft is authoritative for that
                # hash.  Never let the earlier card's alias conflate the two.
                authoritative_keys = set(operations)
                authoritative_keys.update(
                    tool_operation_key(name, payload)
                    for _, _, name, payload, _ in action_operations
                )
                for action, status, name, payload, staged_key in action_operations:
                    operation_key = tool_operation_key(name, payload)
                    result = action.get("result")
                    checkpoint = tool_operation_checkpoint(
                        name,
                        payload,
                        filesystem_path_resolver=self._filesystem_path_resolver(),
                    )
                    if checkpoint["operation_kind"] == "mutation":
                        operations = {
                            key: item for key, item in operations.items()
                            if not mutation_invalidates_read(
                                name,
                                payload,
                                item,
                                filesystem_path_resolver=self._filesystem_path_resolver(),
                            )
                        }
                    record = {
                        "operation_key": operation_key,
                        "operation_id": str(action.get("action_id") or "")[:64],
                        "tool_name": name,
                        "status": status,
                        "ok": status == "executed",
                        "result": json.dumps(
                            result if result is not None else {
                                "status": status, "detail": action.get("detail"),
                            },
                            ensure_ascii=False, default=str,
                        )[:MAX_RESOLVED_RESULT],
                        **checkpoint,
                    }
                    if staged_key != operation_key:
                        record["staged_operation_key"] = staged_key
                        if staged_key not in authoritative_keys:
                            record["operation_aliases"] = [staged_key]
                    operations[operation_key] = record
                state["resolved_operations"] = list(operations.values())[
                    -MAX_RESOLVED_OPERATIONS:
                ]
                state["execution_prompt"] = self._continuation_prompt(
                    job, state,
                    reason="The platform has authoritatively resolved every approval card.",
                )
                try:
                    job = await self._commit_state(
                        db, job, state,
                        values={"status": "queued", "outcome": None,
                                "completed_at": None, "error_class": None,
                                "user_message": None, "claim_owner": None,
                                "claim_token": None, "claim_expires_at": None},
                        expected_status=WAITING,
                    )
                except _LeaseLost:
                    return
                # NOT broadcast: WAITING → queued is not a terminal transition,
                # and the result row it writes is still the placeholder
                # ("Working on: <title>", or the last checkpoint's text). The
                # row id is deterministic and ChatScreen de-dupes by id with no
                # update path, so an intermediate broadcast PINS the
                # placeholder and the real answer — sent later under the same id
                # — is discarded. Only terminal paths announce the row.
                await self._persist_message(db, job, state)
                await db.commit()
                snapshot, uid = _snapshot(job), job.user_id
        await self._broadcast(uid, snapshot)
        self._wake.set()

    async def _terminalize_waiting_cancel(self, task_id: str) -> None:
        await self._terminalize_waiting(
            task_id, status="cancelled", outcome="cancelled",
            code="user_cancelled",
            message=(
                "Task cancelled. Any action that already won the approval race "
                "is shown in its action card."
            ),
        )

    async def _terminalize_waiting_unknown(self, task_id: str) -> None:
        await self._terminalize_waiting(
            task_id, status="failed", outcome="unknown",
            code="approval_outcome_unknown",
            message=(
                "An approved external action did not reach a confirmed terminal record. "
                "Review the action card before doing anything again."
            ),
        )

    async def _terminalize_waiting(
        self, task_id: str, *, status: str, outcome: str,
        code: str, message: str,
    ) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            async with self.session_maker() as db:
                job = await db.get(BuildJob, task_id, with_for_update=True)
                if not job or job.status != WAITING:
                    return
                state = _state(job)
                try:
                    job = await self._commit_state(
                        db, job, state,
                        values={"status": status, "outcome": outcome,
                                "completed_at": _now(), "error_class": code,
                                "user_message": message},
                        expected_status=WAITING,
                    )
                except _LeaseLost:
                    return
                _result_frame = await self._persist_message(db, job, state)
                _enqueue_lifecycle(db, job, state)
                await db.commit()
                snapshot, uid = _snapshot(job), job.user_id
        await self._broadcast(uid, snapshot, message=_result_frame)

    async def _database_tick(
        self, *, now: datetime,
    ) -> tuple[
        list[str], list[str], list[tuple[str, str, str]],
    ]:
        """Perform one supervisor database pass on one connection.

        Pending-action reads cross a service boundary and are deliberately
        reconciled after this transaction. In the common no-task case this is
        the tick's only session and therefore its only NullPool connection.
        """
        async with self.session_maker() as db:
            # A single locked census replaces the former independent running,
            # waiting, and queued scans. Per-tenant admission bounds OPEN_DB,
            # and SKIP LOCKED keeps a request/control transaction off this hot
            # supervisor path without weakening the token fence.
            open_jobs = (await db.execute(select(BuildJob).where(
                BuildJob.source_kind == SOURCE_KIND,
                BuildJob.status.in_(tuple(OPEN_DB)),
            ).order_by(BuildJob.created_at.asc()).with_for_update(
                skip_locked=True
            ))).scalars().all()
            expired_ids = [
                job.id for job in open_jobs
                if (job.status == RUNNING and job.claim_expires_at is not None
                    and job.claim_expires_at <= now)
            ]
            waiting_ids = [job.id for job in open_jobs if job.status == WAITING]
            claims = await self._claim_queued_in_session(
                db, candidates=open_jobs
            )
            await db.commit()
        return expired_ids, waiting_ids, claims

    async def tick_once(self) -> dict[str, int]:
        async with self._tick_lock:
            now = _now()
            # Lease renewal is its own committed transaction.  A malformed
            # expiry row, claim failure, or census error must never roll back
            # the lease of a healthy in-flight task and cause its side effects
            # to become ambiguous.
            if self._has_live_handles():
                await self._heartbeat_owned()
            expired_ids, waiting_ids, claims = await self._database_tick(now=now)
            # Claims are durable before a coroutine can perform side effects.
            self._launch_claimed(claims)
            expired = await self._expire_running(now=now, task_ids=expired_ids)
            reconciled = await self._reconcile_waiting(waiting_ids)
            self._waiting_work = reconciled < len(waiting_ids)
            return {"expired": expired, "reconciled": reconciled,
                    "claimed": len(claims)}

    async def recover_expired(self, *, now: Optional[datetime] = None) -> int:
        # Kept as the focused test/operations entry point; production calls the
        # same atomic sweep from every supervisor tick.
        return await self._expire_running(now=now)

    def _has_live_handles(self) -> bool:
        return any(handle.task and not handle.task.done()
                   for handle in self.handles.values())

    def _next_poll_delay(self, current: float, *, activity: bool) -> float:
        if activity or self._has_live_handles():
            return self.poll_seconds
        if self._waiting_work:
            return self.waiting_poll_seconds
        return min(
            self.idle_poll_seconds,
            max(self.poll_seconds, current * 2.0),
        )

    def _poll_timeout(self, delay: float) -> float:
        if delay <= self.poll_seconds:
            return delay
        return delay * self._idle_jitter

    async def _supervise(self, initial_delay: Optional[float] = None) -> None:
        delay = max(self.poll_seconds, float(initial_delay or self.poll_seconds))
        while not self._closing:
            try:
                woke = False
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self._poll_timeout(delay)
                    )
                    woke = True
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                result = await self.tick_once()
                delay = self._next_poll_delay(
                    delay, activity=woke or bool(any(result.values()))
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Voice task supervisor tick failed")
                delay = self._next_poll_delay(delay, activity=False)


_services: dict[int, VoiceTaskService] = {}


def get_voice_task_service(runner) -> VoiceTaskService:
    key = id(runner)
    if key not in _services:
        _services[key] = VoiceTaskService(runner)
    return _services[key]


async def close_voice_task_services() -> None:
    services = list(_services.values())
    _services.clear()
    for service in services:
        await service.close()
