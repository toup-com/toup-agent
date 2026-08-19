"""Agent→platform notification ingest (Autopilot PR2).

``POST /agent/notify`` — the tenant agent's outbox (PR4) flushes
notification requests here; the platform dispatcher (PR3) later claims
queued rows and fans out (Expo push, agent-side Telegram/WhatsApp
fallback).

Design contract (docs/autopilot/PLAN.md D5):

- AUTH: X-Agent-Key bound to the claimed user_id via AgentConfig —
  the exact recipe of credits.agent_deduct (credits.py) — so one
  tenant's agent can never enqueue notifications for another user.
- IDEMPOTENT: UNIQUE(user_id, idempotency_key) on notification_queue.
  The outbox flushes at-least-once; replays return {"status":
  "duplicate"} and are not re-enqueued. This endpoint is the reason
  notification delivery is *never* fail-open-drop like the credit
  reporter: the agent keeps its outbox row until we've acknowledged
  one of queued/duplicate.
- DUMB INGEST: no prefs/quiet-hours/dedup evaluation here — prefs can
  change between enqueue and send, so suppression is the dispatcher's
  job at claim time. We only validate shape and persist.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.db.models import (
    AgentConfig,
    KNOWN_NOTIFY_KINDS,
    KNOWN_NQ_PRIORITIES,
    NOTIFY_KIND_ANNOUNCEMENT,
    NQ_PRIORITY_DEFAULT,
    NotificationQueue,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent Notify"])


# data_json rides inside the 4KB Expo push payload alongside
# title/body — cap it well below that so a chatty producer fails at
# ingest instead of at send time.
_MAX_DATA_JSON_BYTES = 2048


class AgentNotifyRequest(BaseModel):
    user_id: str = Field(min_length=8, max_length=64)
    # The agent outbox row id — replay guard. Required: an agent-side
    # producer without one cannot be flushed at-least-once safely.
    idempotency_key: str = Field(min_length=8, max_length=128)
    event_kind: str
    title: str = Field(min_length=1, max_length=200)
    body: Optional[str] = Field(default=None, max_length=2000)
    # Deep-link payload: route string + entity ids, scalars only.
    data: Optional[Dict[str, Any]] = None
    priority: str = NQ_PRIORITY_DEFAULT
    dedup_key: Optional[str] = Field(default=None, max_length=128)

    @field_validator("event_kind")
    @classmethod
    def _kind_known(cls, v: str) -> str:
        if v not in KNOWN_NOTIFY_KINDS:
            raise ValueError(
                f"unknown event_kind {v!r}; known: {sorted(KNOWN_NOTIFY_KINDS)}"
            )
        if v == NOTIFY_KIND_ANNOUNCEMENT:
            # Operator → user, and this route authenticates a TENANT
            # key — so anything arriving here is a user's own agent
            # asking to speak as Toup on the operator's card (brand orb,
            # 'from Toup' copy). The platform enqueues announcements
            # itself, in the admin-dispatch fan-out; ingest never does.
            raise ValueError(
                f"event_kind {v!r} is platform-authored; agents may not enqueue it"
            )
        return v

    @field_validator("priority")
    @classmethod
    def _priority_known(cls, v: str) -> str:
        if v not in KNOWN_NQ_PRIORITIES:
            raise ValueError(
                f"unknown priority {v!r}; known: {sorted(KNOWN_NQ_PRIORITIES)}"
            )
        return v

    @field_validator("data")
    @classmethod
    def _data_scalar_and_small(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        for key, val in v.items():
            if val is not None and not isinstance(val, (str, int, float, bool)):
                raise ValueError(f"data[{key!r}] must be a scalar")
        if len(json.dumps(v)) > _MAX_DATA_JSON_BYTES:
            raise ValueError(f"data exceeds {_MAX_DATA_JSON_BYTES} bytes")
        return v


class AgentNotifyResponse(BaseModel):
    status: str  # queued | duplicate
    id: Optional[str] = None


@router.post("/notify", response_model=AgentNotifyResponse)
async def agent_notify(
    body: AgentNotifyRequest,
    x_agent_key: Optional[str] = Header(None, alias="X-Agent-Key"),
    db: AsyncSession = Depends(get_db),
) -> AgentNotifyResponse:
    if not x_agent_key:
        raise HTTPException(401, "X-Agent-Key required")

    # Authenticate: agent_api_key must match the AgentConfig of the
    # claimed user_id (credits.agent_deduct recipe) — stops one
    # tenant's agent from notifying another tenant's devices.
    cfg_result = await db.execute(
        select(AgentConfig).where(
            AgentConfig.user_id == body.user_id,
            AgentConfig.agent_api_key == x_agent_key,
        )
    )
    if cfg_result.scalar_one_or_none() is None:
        raise HTTPException(403, "agent key mismatch")

    # Explicit SELECT-first so the common replay path (outbox re-flush
    # after a lost ack) is a cheap read, then INSERT with the unique
    # constraint as the race-proof backstop.
    existing = await db.execute(
        select(NotificationQueue.id).where(
            NotificationQueue.user_id == body.user_id,
            NotificationQueue.idempotency_key == body.idempotency_key,
        )
    )
    row_id = existing.scalar_one_or_none()
    if row_id is not None:
        return AgentNotifyResponse(status="duplicate", id=row_id)

    # Explicit id (the column default fires at flush, not construction)
    # so the response is built BEFORE commit — pgbouncer txn-mode rule.
    new_id = str(uuid.uuid4())
    row = NotificationQueue(
        id=new_id,
        user_id=body.user_id,
        source="agent",
        event_kind=body.event_kind,
        title=body.title,
        body=body.body,
        data_json=body.data,
        priority=body.priority,
        dedup_key=body.dedup_key,
        idempotency_key=body.idempotency_key,
    )
    db.add(row)
    try:
        await db.commit()
    except IntegrityError:
        # Two replicas ingesting the same replay concurrently — the
        # constraint wins; report duplicate.
        await db.rollback()
        existing = await db.execute(
            select(NotificationQueue.id).where(
                NotificationQueue.user_id == body.user_id,
                NotificationQueue.idempotency_key == body.idempotency_key,
            )
        )
        return AgentNotifyResponse(
            status="duplicate", id=existing.scalar_one_or_none()
        )

    # Progress fast lane (2026-07-16): interim Live Activity updates
    # are worthless 30s late — the whole point is bar motion at
    # tool-call rhythm. Best-effort inline dispatch; the status CAS is
    # the same primitive the dispatcher loop uses, so a concurrent
    # claim is impossible and any failure leaves the row queued for
    # the normal loop. Alert kinds stay on the loop on purpose — with
    # ONE exception (Round 3, 2026-08-18): a conversation job card's
    # start (`mission_started` + data.refresh_if_started). Its progress
    # rows ARE fast-laned and would otherwise reach the card up to 30s
    # before the start that is supposed to reset it for the new job
    # (the never-backwards clamp would then pin the new job's bar at the
    # previous job's 100%). Same kill switch, same CAS, same fallback.
    # Round 4 (item 5a): a REMINDER COUNTDOWN start rides it too. A
    # 60-second reminder's card used to wait for the 30 s tick and land
    # with seconds left (or after the fire) — the "no countdown" report.
    # `data.fast_lane` is the generic opt-in for a start row whose
    # value is in being on the phone NOW.
    _data = body.data or {}
    _fast = settings.notification_progress_fastlane_enabled and (
        body.event_kind == "progress"
        or (
            body.event_kind == "mission_started"
            and (
                bool(_data.get("refresh_if_started"))
                or bool(_data.get("fast_lane"))
                or str(_data.get("mission_id") or "").startswith("reminder:")
            )
        )
        # The reminder FIRE rides the fast lane too: the whole point of the
        # row is landing at the fire SECOND, and the dispatch loop's 30s tick
        # made the "now" banner + the countdown card's flip up to 30s late
        # (masked on iOS 26 by the local AlarmKit ring; naked everywhere
        # else). Reminder mission_ids never collide with conversation job
        # cards, so the never-backwards clamp concern above doesn't apply.
        or (
            body.event_kind == "mission_completed"
            and str(_data.get("mission_id") or "").startswith("reminder:")
        )
    )
    if _fast:
        try:
            from datetime import datetime as _dt

            from app.db.models import NotificationQueue as _NQ
            from app.services.notification_dispatcher import (
                NQ_QUEUED, NQ_SENDING, _cas_status, _dispatch_row,
            )

            _now = _dt.utcnow()
            claimed = await _cas_status(
                db, new_id, NQ_QUEUED, NQ_SENDING, _now,
                extra={"claimed_at": _now, "attempts": _NQ.attempts + 1},
            )
            if claimed:
                await _dispatch_row(db, new_id, _now)
        except Exception as _fl_err:  # noqa: BLE001 — loop picks it up
            logger.warning(
                "[agent_notify] progress fast lane failed (row stays "
                "queued for the dispatch loop): %s", _fl_err,
            )
    return AgentNotifyResponse(status="queued", id=new_id)
