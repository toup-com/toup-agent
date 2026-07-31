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
gate: an explicit SELECT on ``build_jobs(source_id, idempotency_key)``
followed by ``JobRunner.create_job``. Pub/Sub retries collapse on the
composite UNIQUE in ``build_jobs`` (the same dedupe semantics the
legacy ``UNIQUE (trigger_id, event_dedupe_id)`` on ``trigger_events``
used to enforce — PR #49 of the cutover arc finished the move to
``build_jobs`` as the sole source of truth).

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
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Trigger, TRIGGER_KINDS


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
    """Mint a ``build_jobs`` row idempotently, returning "inserted" or
    "dedupe_hit" based on whether the Job was actually created.

    PR #49 of the unified-jobs cutover arc — ``build_jobs`` is the
    sole source of truth. The legacy ``trigger_events`` dual-write
    (PR #46 → PR #48) was removed once every reader was repointed
    at the Job. PR #50 drops the legacy table.

    ``JobRunner.create_job`` honours the composite UNIQUE on
    ``(source_id, idempotency_key)`` and returns the existing row on
    conflict, which preserves the dedupe semantics the legacy
    ``UNIQUE (trigger_id, event_dedupe_id)`` constraint used to
    enforce.

    Fires the runner hand-off with the BuildJob id when (and only
    when) a new Job row was actually created — dedupe hits must NOT
    re-dispatch, that's the whole point of the UNIQUE gate.
    """
    # ── 1. Explicit dedupe check ─────────────────────────────────
    # The partial UNIQUE on (source_id, idempotency_key) in mig 046
    # is the underlying gate, but we check explicitly here so the
    # fresh-vs-dedupe signal is unambiguous (no proximity heuristic
    # that breaks when two near-simultaneous calls land within
    # microseconds — e.g., in tests). One extra SELECT on an indexed
    # composite, negligible cost on the intake path.
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.db.models import BuildJob

    existing_job_id = (await db.execute(
        select(BuildJob.id).where(
            BuildJob.source_id == trigger_id,
            BuildJob.idempotency_key == event_dedupe_id,
        )
    )).scalar_one_or_none()
    if existing_job_id is not None:
        return "dedupe_hit"

    # ── 2. Fresh path: mint the Job ──────────────────────────────
    # ``JobRunner.create_job`` is idempotent under concurrent calls
    # (a race past the SELECT above collapses on the partial UNIQUE
    # INSERT), so a loser thread still gets a Job reference back —
    # we just return "inserted" twice in that very rare case, which
    # is benign (both threads dispatch the same Job; the runner's
    # claim step handles the race).
    spec = TaskSpec(
        user_id=user_id,
        channel="trigger",
        source_kind="trigger",
        source_id=trigger_id,
    )
    # Human name, never the dedupe id. `event_dedupe_id` is the raw
    # provider message id, so the old title put "Trigger fire:
    # 19e654e571e344a9" on the user's Activity board. One PK get on a
    # session we already hold — negligible next to the SELECT above.
    from app.db.models import Trigger

    _trig = await db.get(Trigger, trigger_id)
    _title = (getattr(_trig, "name", None) or "").strip() or (
        getattr(_trig, "kind", None) or "Automation"
    )

    job = await JobRunner().create_job(
        job_type="trigger_run",
        spec=spec,
        title=_title,
        idempotency_key=event_dedupe_id,
    )
    # JobRunner.create_job handles its own commit; we don't write
    # any further rows here.

    # ── 3. Runner hand-off ────────────────────────────────────────
    # The runner is BuildJob-native after the PR #49 cutover; the
    # hand-off carries the Job id straight through. If the runner
    # isn't wired yet (boot race), its drain_loop picks the row up
    # on its next tick.
    if _runner_ref is not None:
        try:
            _runner_ref.handle_event_background(job.id)
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
