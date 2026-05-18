"""Agent-side inbound dispatch endpoint for trigger events.

The platform-side webhook (POST /api/v1/webhooks/gmail) receives the
Pub/Sub push, verifies the JWT, decodes the envelope, resolves the
recipient via `connector_identities`, then pulls the new-message-id
delta from Gmail `history.list` using the connector's refresh token
(tokens stay on the platform — same isolation guarantee as Routines).

For each new message-id, the platform dispatches an envelope to this
endpoint on the user's per-tenant agent container. The contract is:

  POST /api/triggers/inbound
  X-Agent-Key: <tenant_agent_api_key>
  {
    "trigger_kind": "email_received",
    "user_id": "<must match X-Agent-Key tenant>",
    "events": [
      {
        "event_dedupe_id": "gmail_msg_id_abc",
        "external_payload": {              # provider-specific stub
          "gmail_message_id": "...",
          "history_id": "...",
          "internal_date_ms": 1715500000000
        }
      },
      ...
    ]
  }

This endpoint is hot-path. Pub/Sub wants 200 in <2s p50, hard cap 10s.
We MUST NOT do Gmail fetches, LLM calls, or Message writes here — the
runner (Gate T2) owns that. This endpoint's job is the idempotency
gate: INSERT…ON CONFLICT DO NOTHING into `trigger_events`. Pub/Sub
retries collapse on the UNIQUE (trigger_id, event_dedupe_id) constraint.

Auth contract:
  - The platform calls with `X-Agent-Key: <agent_configs.agent_api_key>`
    for the tenant. The agent container validates against
    `settings.agent_api_key` (the env-var copy of the same key — that's
    how the existing X-Agent-Key path in auth.py works).
  - Body `user_id` MUST match `settings.user_id` (the container owner).
    Defence-in-depth — if the platform routed a wrong envelope, we
    reject loudly instead of writing it to the wrong tenant's table.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Trigger, TriggerEvent, TRIGGER_KINDS


router = APIRouter(prefix="/triggers", tags=["triggers (inbound)"])
logger = logging.getLogger(__name__)


# Late-bound runner reference. agent_main injects this after the
# TriggerRunner is built so the inbound endpoint can fire-and-forget
# dispatch each newly-inserted row to the handler. None is fine — the
# drain loop picks up un-handed events on its next tick.
_runner_ref: Any = None


def set_runner_ref(runner: Any) -> None:
    """Called by agent_main lifespan after TriggerRunner is started."""
    global _runner_ref
    _runner_ref = runner


# ── Envelope ─────────────────────────────────────────────────────────


class TriggerInboundEvent(BaseModel):
    """One external event in the inbound batch.

    `event_dedupe_id` is the load-bearing field. For Gmail it's the
    message id; for future kinds it's whatever the provider gives us as
    the unique-per-event handle. `external_payload` is opaque to the
    inbound endpoint — the handler reads it in Gate T2.
    """

    event_dedupe_id: str = Field(..., min_length=1, max_length=255)
    external_payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_dedupe_id")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("event_dedupe_id must be non-empty after strip")
        return v


class TriggerInboundEnvelope(BaseModel):
    """One inbound dispatch from the platform webhook.

    A single dispatch may carry N events for ONE (user, trigger_kind)
    pair — the platform batches new-message deltas for the same user
    when Pub/Sub coalesces multiple inbox arrivals. The endpoint
    fan-outs the batch over the user's enabled triggers of that kind
    (filter evaluation happens in the runner, not here).
    """

    trigger_kind: str = Field(..., min_length=1, max_length=50)
    user_id: str = Field(..., min_length=1, max_length=36)
    events: list[TriggerInboundEvent] = Field(..., min_length=1, max_length=200)

    @field_validator("trigger_kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        if v not in TRIGGER_KINDS:
            raise ValueError(f"unsupported trigger_kind: {v!r}")
        return v


class TriggerInboundResult(BaseModel):
    """One row's outcome — `inserted` is the dedupe miss (new event),
    `dedupe_hit` is the conflict (Pub/Sub retry or sibling delivery),
    `no_trigger` is "no enabled trigger of this kind for this user."
    The platform doesn't act on these; they're for log correlation."""

    trigger_id: str
    event_dedupe_id: str
    outcome: str  # "inserted" | "dedupe_hit" | "no_trigger"


class TriggerInboundResponse(BaseModel):
    accepted_events: int
    inserted: int
    dedupe_hits: int
    no_trigger: int
    results: list[TriggerInboundResult]


# ── Auth ─────────────────────────────────────────────────────────────


def _require_platform_caller(
    x_agent_key: Optional[str], body_user_id: str,
) -> None:
    """Validate the X-Agent-Key + assert body.user_id matches the
    container owner. Two-layer defence:
      1) X-Agent-Key catches forged callers — platform is the only one
         with the per-tenant key.
      2) body.user_id mismatch catches a routing bug on the platform
         side where an envelope is dispatched to the wrong tenant
         container. Without this we'd silently insert another tenant's
         events into the wrong DB.

    Both failures return 401 — the platform should never see them in
    production, so granularity isn't useful, and 4xx is what Pub/Sub
    needs to NOT retry forever (a 5xx triggers retries).
    """
    expected_key = (getattr(settings, "agent_api_key", "") or "").strip()
    container_user_id = (getattr(settings, "user_id", "") or "").strip()

    if not expected_key or not container_user_id:
        # Agent is not configured for X-Agent-Key inbound (running in
        # legacy / dev mode). Fail closed — refuse to write rows on an
        # unconfigured agent so a stale dev container can't accidentally
        # poison another user's DB.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not configured for trigger inbound",
        )

    if not x_agent_key or x_agent_key.strip() != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid X-Agent-Key",
            headers={"WWW-Authenticate": 'X-Agent-Key realm="toup-agent"'},
        )

    if body_user_id != container_user_id:
        # The platform routed an envelope to the wrong tenant
        # container. This is a platform bug; log loudly so it gets
        # noticed.
        logger.error(
            "[trigger_inbound] envelope user_id=%s does not match container owner=%s — "
            "platform routing bug, refusing insert",
            body_user_id[:8] + "…", container_user_id[:8] + "…",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="envelope user_id does not match container owner",
        )


# ── Endpoint ─────────────────────────────────────────────────────────


@router.post(
    "/inbound",
    response_model=TriggerInboundResponse,
    summary="Inbound dispatch from the platform-side webhook",
)
async def trigger_inbound(
    envelope: TriggerInboundEnvelope,
    request: Request,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> TriggerInboundResponse:
    """Idempotently record N external events against all enabled
    triggers of `envelope.trigger_kind` for `envelope.user_id`.

    Returns 200 on every successful authenticated call, even if no
    enabled trigger matched — Pub/Sub treats 200 as "delivered, don't
    retry," which is exactly what we want once the call has been
    authenticated and the dedupe rows have been written. Pub/Sub will
    retry on a 5xx, which is appropriate for transient DB failures.
    """
    _require_platform_caller(x_agent_key, envelope.user_id)

    started_ns = _ns_now()
    results: list[TriggerInboundResult] = []
    inserted_count = 0
    dedupe_count = 0
    no_trigger_count = 0

    async with async_session_maker() as db:
        # Resolve enabled triggers for (user_id, kind). N is small
        # (per spec, one user typically has 0–3 triggers of a given kind),
        # so we materialise the list once and reuse across events.
        triggers = (await db.execute(
            select(Trigger).where(
                Trigger.user_id == envelope.user_id,
                Trigger.kind == envelope.trigger_kind,
                Trigger.enabled.is_(True),
            )
        )).scalars().all()

        if not triggers:
            # No enabled trigger of this kind — accept the dispatch
            # without inserting anything. The platform should ideally
            # filter at watch-provisioning time, but the agent might be
            # the source of truth that just learned about a disable.
            for ev in envelope.events:
                results.append(TriggerInboundResult(
                    trigger_id="",
                    event_dedupe_id=ev.event_dedupe_id,
                    outcome="no_trigger",
                ))
                no_trigger_count += 1
        else:
            # For each (trigger, event) pair, attempt INSERT with
            # ON CONFLICT DO NOTHING. We use one transaction per row so
            # a single conflict doesn't poison the rest of the batch.
            # The cost (N round-trips for a batch of N) is acceptable
            # at trigger volumes — typical Pub/Sub push is 1 event per
            # call. Larger coalesced batches still finish well under
            # the 2s p50 target.
            for trigger in triggers:
                for ev in envelope.events:
                    outcome = await _idempotent_insert(
                        db, trigger_id=trigger.id,
                        user_id=envelope.user_id,
                        event_dedupe_id=ev.event_dedupe_id,
                    )
                    results.append(TriggerInboundResult(
                        trigger_id=trigger.id,
                        event_dedupe_id=ev.event_dedupe_id,
                        outcome=outcome,
                    ))
                    if outcome == "inserted":
                        inserted_count += 1
                    elif outcome == "dedupe_hit":
                        dedupe_count += 1

    latency_ms = (_ns_now() - started_ns) // 1_000_000
    logger.info(
        "[trigger_inbound] kind=%s user_id=%s events=%d triggers=%d "
        "inserted=%d dedupe_hits=%d no_trigger=%d latency_ms=%d",
        envelope.trigger_kind, envelope.user_id[:8] + "…",
        len(envelope.events), len(triggers),
        inserted_count, dedupe_count, no_trigger_count, latency_ms,
    )

    return TriggerInboundResponse(
        accepted_events=len(envelope.events),
        inserted=inserted_count,
        dedupe_hits=dedupe_count,
        no_trigger=no_trigger_count,
        results=results,
    )


# ── Idempotent insert helper ─────────────────────────────────────────


async def _idempotent_insert(
    db, *, trigger_id: str, user_id: str, event_dedupe_id: str,
) -> str:
    """INSERT … ON CONFLICT DO NOTHING, returning "inserted" or
    "dedupe_hit" based on whether the row was actually created.

    Uses dialect-specific INSERT statements so Postgres gets ON CONFLICT
    (the real prod path) and SQLite gets OR IGNORE (the test path).
    Other dialects would need an explicit branch — failing loudly here
    is preferable to silently double-inserting.

    Fires the runner hand-off when (and only when) a new row was
    actually created — dedupe hits must NOT re-dispatch, that's the
    whole point of the UNIQUE gate.
    """
    new_id = str(uuid.uuid4())
    values = {
        "id": new_id,
        "trigger_id": trigger_id,
        "user_id": user_id,
        "event_dedupe_id": event_dedupe_id,
        "received_at": datetime.utcnow(),
        "status": "queued",
    }

    dialect_name = db.bind.dialect.name if db.bind else ""
    if dialect_name == "postgresql":
        stmt = pg_insert(TriggerEvent).values(**values)
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_trigger_events_dedupe",
        ).returning(TriggerEvent.id)
    elif dialect_name == "sqlite":
        stmt = sqlite_insert(TriggerEvent).values(**values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["trigger_id", "event_dedupe_id"],
        ).returning(TriggerEvent.id)
    else:
        # Unknown dialect — fail loud rather than silently re-insert.
        raise RuntimeError(
            f"trigger_inbound: unsupported DB dialect {dialect_name!r}. "
            "Add an ON CONFLICT branch before deploying."
        )

    result = await db.execute(stmt)
    inserted_id = result.scalar_one_or_none()
    await db.commit()

    if inserted_id is None:
        return "dedupe_hit"

    # ── Unified-jobs dual-write (PR 4a) ──────────────────────────
    # Mint a parallel BuildJob row so every TriggerEvent has a
    # mirrored Job that the activity feed in PR 6 can JOIN against.
    # Idempotency key = event_dedupe_id; source_id = trigger_id.
    # This mirrors the existing UNIQUE (trigger_id, event_dedupe_id)
    # on trigger_events: any racing call producing the same pair
    # gets a single Job back, never duplicates. Stamp
    # ``trigger_events.job_id`` so the runner's status-sync path
    # can update both rows when the handler terminates.
    #
    # Best-effort: a failed Job mint must NOT roll back the
    # TriggerEvent insert. The legacy fire path still works; we
    # just lose the Job linkage for this one event.
    try:
        from app.agent.job_runner import JobRunner, TaskSpec
        runner = JobRunner()
        spec = TaskSpec(
            user_id=user_id,
            channel="trigger",
            source_kind="trigger",
            source_id=trigger_id,
        )
        job = await runner.create_job(
            job_type="trigger_run",
            spec=spec,
            title=f"Trigger fire: {event_dedupe_id[:40]}",
            idempotency_key=event_dedupe_id,
        )
        await db.execute(
            sa_update(TriggerEvent)
            .where(TriggerEvent.id == str(inserted_id))
            .values(job_id=job.id)
        )
        await db.commit()
    except Exception as e:
        logger.warning(
            "[trigger_inbound] Job dual-write failed for event=%s: %s "
            "(legacy fire path unaffected)",
            inserted_id, e,
        )

    # Fire-and-forget runner hand-off. The event is in 'queued' state
    # in the DB; the runner picks it up, applies rate/coalesce/filter,
    # and dispatches to the handler. If the runner isn't wired yet
    # (boot race), its drain_loop picks the row up on its next tick.
    if _runner_ref is not None:
        try:
            _runner_ref.handle_event_background(str(inserted_id))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "[trigger_inbound] runner hand-off raised: %s (drain_loop will recover)",
                e,
            )

    return "inserted"


# ── Helpers ──────────────────────────────────────────────────────────


def _ns_now() -> int:
    """Monotonic nanosecond clock for latency measurement."""
    import time
    return time.perf_counter_ns()
