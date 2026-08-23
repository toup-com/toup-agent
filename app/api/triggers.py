"""Triggers HTTP surface.

User-facing CRUD + event-history + test-fire endpoints. Same auth
contract as `routines.py` — single-user-per-container, `X-Agent-Key`
gate in `agent_main`. Validation is pinned to the canonical enum
constants in `app.db.models.trigger` so the API can't drift from the
handler registry.

Watch provisioning: the platform owns Gmail OAuth tokens, so the
`users.watch` call happens platform-side. v1 expects the operator to
arm the watch via the script documented in
`docs/runbooks/gmail-pubsub.md`. The trigger row exposes
`watch_provisioned` in its response so the UI can render a
"needs setup" badge until provisioning completes. Auto-provisioning
via an agent→platform RPC is on the post-T2 roadmap.

Endpoints:
  GET  /api/triggers                  — list user's triggers + recent events
  POST /api/triggers                  — create
  PATCH /api/triggers/{id}            — update filter/action/enabled/delivery
  DELETE /api/triggers/{id}           — delete + cascade events
  GET  /api/triggers/{id}/events      — paginated event history
  POST /api/triggers/{id}/test        — synthesize a test event (test:<n> prefix)
  GET  /api/triggers/_runner_status   — health + counters

Validation philosophy: reject malformed input loudly (4xx with a
specific message), accept extra unknown fields silently (forward
compatibility with future kinds). Pydantic's default extra=ignore
gives us the silent path; explicit `field_validator`s give us the
loud path for fields we do enforce.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.config import settings
from pydantic import BaseModel, Field, field_serializer, field_validator


def _utc_iso(v: Optional[datetime]) -> Optional[str]:
    """Render a datetime as RFC 3339 UTC with a trailing ``Z``.

    Pydantic's default `datetime` serialiser emits ISO 8601 without a
    timezone marker for naive datetimes — e.g. ``datetime.utcnow()``
    returns naive UTC, gets serialised as ``"2026-05-14T19:18:00"``,
    and the browser's ``new Date()`` parses that as LOCAL time. That
    was the 2026-05-14 dashboard bug: events stamped at 19:18 UTC
    rendered as "7:18 PM local" instead of "3:18 PM EDT". Force the
    ``Z`` suffix on every datetime field that crosses the wire to
    the frontend so the parse is unambiguous.
    """
    if v is None:
        return None
    if v.tzinfo is None:
        return v.isoformat() + "Z"
    return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
from sqlalchemy import desc, select

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/triggers", tags=["triggers"])


_runner = None  # set by agent_main.py via set_runner_ref()


def set_runner_ref(runner) -> None:
    """agent_main injects the TriggerRunner so the test-fire endpoint
    can hand-off newly synthesised events directly."""
    global _runner
    _runner = runner


# ── Helpers ──────────────────────────────────────────────────────────


def _user_id() -> str:
    """Single-user-per-container. Same pattern as routines."""
    from app.config import settings
    return settings.user_id


def _validate_kind(kind: str) -> None:
    """Reject kinds outside the canonical frozenset. Future kinds get
    added by extending `TRIGGER_KINDS` + registering a handler — no
    schema change."""
    from app.db.models import TRIGGER_KINDS
    if kind not in TRIGGER_KINDS:
        registered = sorted(TRIGGER_KINDS)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown trigger kind {kind!r}. Allowed: {registered}",
        )


def _validate_action(action: str) -> None:
    from app.db.models import TRIGGER_ACTIONS
    if action == "run_automation":
        # System-managed: only the automations compiler writes rows with
        # this action (same economy as kind='autopilot' being absent
        # from the routines API's creatable set). Listed in
        # TRIGGER_ACTIONS so the handler dispatch stays a closed enum.
        raise HTTPException(
            status_code=400,
            detail="Action 'run_automation' is system-managed and cannot "
                   "be set directly.",
        )
    if action not in TRIGGER_ACTIONS:
        allowed = sorted(TRIGGER_ACTIONS - {"run_automation"})
        raise HTTPException(
            status_code=400,
            detail=f"Unknown trigger action {action!r}. Allowed: {allowed}",
        )


def _kind_enabled_or_404(kind: str) -> None:
    """Feature flag gate. Mirrors routines' kind gate so a flag flip
    can mute the surface entirely."""
    from app.config import settings
    if kind == "email_received":
        if not getattr(settings, "triggers_email_enabled", False):
            raise HTTPException(status_code=404, detail="Feature not available")


_VALID_DELIVERY_CHANNELS = {"website", "telegram", "whatsapp"}


def _validate_delivery_channels(value: Any) -> Optional[list[str]]:
    """Normalise + validate the delivery_channels list. None / empty →
    `['website']`. Unknown values rejected loudly."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="delivery_channels must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        if not isinstance(v, str):
            raise HTTPException(
                status_code=400,
                detail=f"delivery_channels items must be strings; got {type(v).__name__}",
            )
        s = v.strip()
        if not s:
            continue
        if s not in _VALID_DELIVERY_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"delivery_channels item {s!r} not in {sorted(_VALID_DELIVERY_CHANNELS)}",
            )
        if s not in seen:
            out.append(s)
            seen.add(s)
    if not out:
        out = ["website"]
    elif "website" not in out:
        out.insert(0, "website")
    return out


def _merge_config(existing: Optional[dict], delivery_channels: Optional[list[str]],
                  raw_config: Optional[dict]) -> dict:
    """Combine the optional explicit delivery_channels with whatever
    config_json the caller passed. delivery_channels wins; the caller
    can also pre-stuff config.delivery_channels."""
    out = dict(existing or {})
    if raw_config:
        out.update(raw_config)
    if delivery_channels is not None:
        out["delivery_channels"] = delivery_channels
    elif "delivery_channels" in out:
        # Validate any pre-stuffed value too.
        out["delivery_channels"] = _validate_delivery_channels(out["delivery_channels"]) or ["website"]
    else:
        out["delivery_channels"] = ["website"]
    return out


# ── Pydantic models ──────────────────────────────────────────────────


class TriggerCreate(BaseModel):
    kind: str = Field(..., min_length=1, max_length=50)
    action: str = Field(..., min_length=1, max_length=50)
    name: Optional[str] = Field(default=None, max_length=100)
    enabled: bool = True
    filter_json: Optional[dict[str, Any]] = None
    config_json: Optional[dict[str, Any]] = None
    delivery_channels: Optional[list[str]] = None


class TriggerUpdate(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = Field(default=None, max_length=100)
    action: Optional[str] = Field(default=None, max_length=50)
    filter_json: Optional[dict[str, Any]] = None
    config_json: Optional[dict[str, Any]] = None
    delivery_channels: Optional[list[str]] = None


class TriggerEventResponse(BaseModel):
    id: str
    trigger_id: str
    event_dedupe_id: str
    received_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    status: str
    error_class: Optional[str] = None
    error_detail: Optional[str] = None
    summary_message_id: Optional[str] = None
    coalesced_into_event_id: Optional[str] = None

    # See `_utc_iso` — emit ``Z`` so the browser parses these as UTC.
    @field_serializer("received_at", "started_at", "finished_at", when_used="json")
    def _ser_dt(self, v: Optional[datetime]) -> Optional[str]:
        return _utc_iso(v)


class TriggerResponse(BaseModel):
    id: str
    kind: str
    action: str
    name: Optional[str]
    enabled: bool
    filter_json: Optional[dict[str, Any]]
    config_json: Optional[dict[str, Any]]
    provider_state_json: Optional[dict[str, Any]]
    last_fired_at: Optional[datetime]
    fire_count: int
    last_status: str
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    delivery_channels: list[str]
    watch_provisioned: bool
    # Derived health signal — the UI pill should read THIS, not
    # `last_status` directly. See `_derive_health` for the state machine.
    # Values:
    #   "delivering"            — at least one real event has fired successfully
    #   "test_passed"           — Test button worked, but no real event yet
    #   "awaiting_event"        — watch armed, no fires of any kind
    #   "needs_reauth"          — Gmail connector lost auth
    #   "setup_error"           — generic provisioning failure
    #   "setup_blocked_ops"     — GCP/Pub-Sub misconfig on the platform
    #   "setup_scope_error"     — Google rejected the watch (scope/IAM)
    #   "setup_transient"       — token mint, 5xx, network — retryable
    #   "setup_feature_disabled"— triggers_email_enabled=false somewhere
    #   "watch_expired"         — watch was armed but is past its 7d window
    #   "last_fire_failed"      — most recent REAL fire crashed; needs attention
    #
    # Defaulted to "setup_error" so callers / tests that build the model
    # without going through `_row_to_response` (which always derives the
    # real value) don't trip a `Field required`. Production code always
    # populates this via `_derive_health`.
    health: str = "setup_error"
    # Last successful REAL event delivery (not test fires, not empty batches).
    # Sourced from provider_state_json.last_real_fired_at.
    last_real_fired_at: Optional[str] = None
    # Last Test-button fire that wrote to Day-as-Chat.
    last_test_at: Optional[str] = None
    # Specific code from `_provision_email_watch` when the auto-arm
    # failed. Surfaced verbatim so the UI's expanded row can render
    # actionable detail (eg. "Gmail refresh token is rejected — reconnect").
    provision_error: Optional[str] = None
    provision_detail: Optional[str] = None
    # Latest delivery-path timestamps used by the diagnose probe + UI
    # banners. Strings (ISO-Z) because the JSONB blob already stores
    # them as strings; serializing as datetime would require a parse.
    last_webhook_dispatch_at: Optional[str] = None
    last_handler_completed_at: Optional[str] = None
    last_handler_status: Optional[str] = None
    recent_events: list[TriggerEventResponse] = Field(default_factory=list)

    # PR 7 of the unified-jobs arc: the last few mirrored BuildJob
    # rows for this trigger. Each fire (PR 4a wired the dual-write)
    # produces a parallel Job row in ``build_jobs`` with
    # ``source_id=trigger.id``. Surfacing them here lets the
    # Mission Control card render "Last 5 fires" as Job-detail-drawer
    # links — the activity-feed-from-the-trigger-side equivalent
    # of PR 6's cross-job feed.
    recent_jobs: list["TriggerRecentJob"] = Field(default_factory=list)

    @field_serializer("last_fired_at", "created_at", "updated_at", when_used="json")
    def _ser_dt(self, v: Optional[datetime]) -> Optional[str]:
        return _utc_iso(v)


class TriggerRecentJob(BaseModel):
    """Compact projection of a recent ``build_jobs`` row mirrored
    from a TriggerEvent (PR 4a). Only the fields the dashboard card
    needs to render a "Last fires" list item + link."""

    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    outcome: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def _derive_health(trigger, watch_provisioned: bool) -> str:
    """Compute the user-facing health signal from the trigger row.

    Decoupled from `last_status` because the runner's `last_status`
    field had to keep its existing semantics for back-compat with the
    skill + Mission Control read paths. The new `health` field is the
    single source of truth for the UI pill — same input row, more
    honest output.

    State machine (priority top to bottom):
      1. last_status == 'skipped_reauth'          → needs_reauth
      2. last_status == 'provisioning_failed'     → map provision_error
         to a specific setup_* code so the user sees what to do, not
         a generic "Setup error". Codes come from `_provision_email_watch`:
           - needs_reauth          → needs_reauth
           - ops_blocked           → setup_blocked_ops
           - scope_error           → setup_scope_error
           - transient / http_5xx
             / platform_unreachable→ setup_transient
           - feature_disabled      → setup_feature_disabled
           - anything else / unset → setup_error
      3. provider_state.watch_expires_at < now    → watch_expired
      4. provider_state.last_real_fired_at present → delivering
         (unless last_status=='failed' AFTER that → last_fire_failed)
      5. provider_state.last_test_at present      → test_passed
      6. watch_provisioned                        → awaiting_event
      7. default                                  → setup_error
    """
    ps = trigger.provider_state_json or {}
    ls = trigger.last_status or "never_fired"
    if ls == "skipped_reauth":
        return "needs_reauth"
    if ls == "provisioning_failed":
        err = str((ps.get("provision_error") or "")).strip()
        # Bucket the http_4xx / http_5xx variants. Google's 4xx for the
        # watch call is almost always a scope / IAM problem; 5xx is
        # transient.
        if err.startswith("http_"):
            try:
                code = int(err.split("_", 1)[1])
            except ValueError:
                code = 0
            if 500 <= code < 600:
                return "setup_transient"
            if 400 <= code < 500:
                return "setup_scope_error"
            return "setup_error"
        return {
            "needs_reauth": "needs_reauth",
            "ops_blocked": "setup_blocked_ops",
            "scope_error": "setup_scope_error",
            "transient": "setup_transient",
            "platform_unreachable": "setup_transient",
            "feature_disabled": "setup_feature_disabled",
        }.get(err, "setup_error")

    # Watch expired?
    expires_at_raw = ps.get("watch_expires_at") or ""
    if expires_at_raw:
        try:
            exp = datetime.fromisoformat(
                str(expires_at_raw).replace("Z", "+00:00")
            )
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= datetime.now(timezone.utc):
                return "watch_expired"
        except (ValueError, TypeError):
            pass

    last_real = ps.get("last_real_fired_at")
    last_test = ps.get("last_test_at")

    if last_real:
        # We have proof a real event delivered. If the most recent
        # batch failed AFTER that, surface that signal so the user
        # sees the regression — `last_fired_at` is post-real means
        # the runner ran after the last real success and failed.
        last_fired = trigger.last_fired_at
        try:
            real_dt = datetime.fromisoformat(
                str(last_real).replace("Z", "+00:00")
            )
            if real_dt.tzinfo is None:
                real_dt = real_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            real_dt = None
        if (
            ls == "failed"
            and last_fired is not None
            and real_dt is not None
        ):
            last_fired_aware = last_fired
            if last_fired_aware.tzinfo is None:
                last_fired_aware = last_fired_aware.replace(tzinfo=timezone.utc)
            if last_fired_aware > real_dt:
                return "last_fire_failed"
        return "delivering"

    if last_test:
        return "test_passed"

    if watch_provisioned:
        return "awaiting_event"

    return "setup_error"


# ── Response helpers ─────────────────────────────────────────────────


def _job_to_event_response(j) -> TriggerEventResponse:
    """Project a ``build_jobs`` row into the legacy ``TriggerEventResponse``
    shape so the events history endpoints can serve from BuildJob without
    a frontend contract change.

    PR #51 of the unified-jobs cutover arc: the legacy ``trigger_events``
    table is going away in PR #52 (mig 050). Every API consumer that used
    to SELECT TriggerEvent now reads BuildJob; this helper keeps the
    wire shape stable so dashboards/extensions don't notice the swap.

    Field mapping (mirrors the runner's terminal-write mapping in
    ``app/agent/triggers/runner.py``):

      BuildJob.id                 → id
      BuildJob.source_id          → trigger_id
      BuildJob.idempotency_key    → event_dedupe_id
      BuildJob.created_at         → received_at  (BuildJob has no
                                     received-vs-started split; the
                                     intake INSERT moment IS the
                                     received moment)
      None                        → started_at   (no equivalent on
                                     BuildJob; the legacy
                                     ``started_at`` column was set
                                     when the runner claimed the row.
                                     The new model doesn't track that
                                     boundary — the dashboard reads
                                     ``created_at`` for "when fired",
                                     ``completed_at`` for "when done")
      BuildJob.completed_at       → finished_at
      BuildJob.status + outcome   → status  (see _job_status_to_legacy)
      BuildJob.error_message      → error_detail  (single Text column
                                     on BuildJob; ``error_class`` has
                                     no separate column so we leave
                                     it None — the dashboard renders
                                     the detail string verbatim)
      BuildJob.summary_message_id → summary_message_id
      BuildJob.coalesced_into_job_id → coalesced_into_event_id
    """
    return TriggerEventResponse(
        id=j.id,
        trigger_id=j.source_id or "",
        event_dedupe_id=j.idempotency_key or "",
        received_at=j.created_at,
        started_at=None,
        finished_at=j.completed_at,
        status=_job_status_to_legacy(j.status, getattr(j, "outcome", None)),
        error_class=None,
        error_detail=j.error_message,
        summary_message_id=j.summary_message_id,
        coalesced_into_event_id=getattr(j, "coalesced_into_job_id", None),
    )


def _job_status_to_legacy(status: Optional[str], outcome: Optional[str]) -> str:
    """Collapse BuildJob's (status, outcome) two-column state back into
    the single legacy ``TriggerEvent.status`` string the frontend reads.

    Inverse of the runner's write mapping (see
    ``app/agent/triggers/runner.py:_dispatch_single``):

      completed + 'success'            → 'success'
      completed + 'skipped_rate_limit' → 'skipped_rate_limit'
      completed + 'skipped_filter'     → 'skipped_filter'
      completed + 'coalesced'          → 'coalesced'
      failed    + NULL                 → 'failed'
      queued    + NULL                 → 'queued'
      running   + NULL                 → 'running'

    Unknown combinations fall back to the raw status — better than
    returning an empty string and breaking the dashboard's icon
    switch."""
    if status == "completed":
        return outcome or "success"
    if status == "failed":
        return "failed"
    return status or "queued"


def _row_to_response(trigger, recent_events=(), recent_jobs=()) -> TriggerResponse:
    cfg = trigger.config_json or {}
    ps = trigger.provider_state_json or {}
    delivery = cfg.get("delivery_channels") or ["website"]
    watch_provisioned = bool(ps.get("gmail_history_id")) and not bool(
        ps.get("needs_refresh")
    )
    health = _derive_health(trigger, watch_provisioned)
    return TriggerResponse(
        id=trigger.id,
        kind=trigger.kind,
        action=trigger.action,
        name=trigger.name,
        enabled=trigger.enabled,
        filter_json=trigger.filter_json,
        config_json=trigger.config_json,
        provider_state_json=trigger.provider_state_json,
        last_fired_at=trigger.last_fired_at,
        fire_count=trigger.fire_count or 0,
        last_status=trigger.last_status,
        last_error=trigger.last_error,
        created_at=trigger.created_at,
        updated_at=trigger.updated_at,
        delivery_channels=delivery,
        watch_provisioned=watch_provisioned,
        health=health,
        last_real_fired_at=ps.get("last_real_fired_at"),
        last_test_at=ps.get("last_test_at"),
        provision_error=ps.get("provision_error"),
        provision_detail=ps.get("provision_detail"),
        last_webhook_dispatch_at=ps.get("last_webhook_dispatch_at"),
        last_handler_completed_at=ps.get("last_handler_completed_at"),
        last_handler_status=ps.get("last_handler_status"),
        # PR #51 cutover: ``recent_events`` is now a sequence of
        # BuildJob rows (source_kind='trigger', source_id=trigger.id).
        # ``_job_to_event_response`` projects each into the legacy
        # TriggerEventResponse shape so the wire contract is stable.
        recent_events=[_job_to_event_response(e) for e in recent_events],
        recent_jobs=[
            TriggerRecentJob(
                id=j.id,
                title=j.title,
                status=j.status,
                outcome=getattr(j, "outcome", None),
                created_at=j.created_at,
                completed_at=j.completed_at,
            )
            for j in recent_jobs
        ],
    )


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/_runner_status")
async def runner_status():
    """Lightweight health/introspection. Mirrors routines' endpoint."""
    if _runner is None:
        return {"running": False, "kinds_registered": []}
    return _runner.status_snapshot()


@router.get("", response_model=list[TriggerResponse])
async def list_triggers():
    """Return every trigger for the container's owner + the most
    recent 20 events per trigger. The frontend uses these for the
    Mission Control card grid."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Trigger).where(Trigger.user_id == _user_id())
            .order_by(desc(Trigger.created_at))
        )).scalars().all()
        out: list[TriggerResponse] = []
        for t in rows:
            # PR #51 cutover: source the recent-events list from
            # ``build_jobs`` instead of the legacy ``trigger_events``
            # table (which mig 050 drops in PR #52). Each fire mints
            # a BuildJob row with ``source_kind='trigger'`` +
            # ``source_id=trigger.id`` — same shape as ``recent_jobs``
            # below, just with a higher limit. We KEEP ``recent_events``
            # and ``recent_jobs`` as separate fields on the response so
            # the frontend's two-list rendering (legacy events table
            # vs. unified Jobs links) keeps working with no contract
            # change. The data is just one query now.
            recent = (await db.execute(
                select(BuildJob).where(
                    BuildJob.source_kind == "trigger",
                    BuildJob.source_id == t.id,
                ).order_by(desc(BuildJob.created_at)).limit(20)
            )).scalars().all()
            # PR 7 of the unified-jobs arc: surface the last 5
            # mirrored BuildJob rows for this trigger. Each fire
            # produces a parallel Job (PR 4a dual-write) with
            # ``source_id=trigger.id``. The dashboard card uses
            # this for a "Last 5 fires" list of detail-drawer
            # links — same shape the routine card will get in
            # this same PR. Best-effort: if the JOIN is empty
            # (no fires yet), ``recent_jobs`` is an empty list —
            # the card hides the section gracefully.
            recent_jobs = (await db.execute(
                select(BuildJob).where(
                    BuildJob.source_kind == "trigger",
                    BuildJob.source_id == t.id,
                ).order_by(desc(BuildJob.created_at)).limit(5)
            )).scalars().all()
            out.append(_row_to_response(t, recent, recent_jobs))
    return out


@router.post("", response_model=TriggerResponse, status_code=201)
async def create_trigger(req: TriggerCreate):
    """Create a trigger. Validation chain:
      1. kind in TRIGGER_KINDS + feature-flag gate (404 if disabled).
      2. action in TRIGGER_ACTIONS.
      3. delivery_channels normalisation.
      4. forward_to_telegram action implies telegram in delivery — we
         auto-add it instead of failing, since the action label is the
         user-facing source of truth.
    """
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    _kind_enabled_or_404(req.kind)
    _validate_kind(req.kind)
    _validate_action(req.action)
    delivery = _validate_delivery_channels(req.delivery_channels)
    config_json = _merge_config(None, delivery, req.config_json)

    # forward_to_telegram action auto-implies telegram in delivery.
    if req.action == "forward_to_telegram" and "telegram" not in config_json.get(
        "delivery_channels", []
    ):
        chs = list(config_json.get("delivery_channels") or ["website"])
        chs.append("telegram")
        config_json["delivery_channels"] = chs

    tid = str(uuid.uuid4())
    async with async_session_maker() as db:
        row = Trigger(
            id=tid,
            user_id=_user_id(),
            kind=req.kind,
            action=req.action,
            name=req.name,
            enabled=req.enabled,
            filter_json=req.filter_json,
            config_json=config_json,
            last_status="never_fired",
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

    # Auto-arm: for email_received, ask the platform to provision the
    # Gmail `users.watch`. Until the watch is armed, no event fires —
    # so we MUST attempt this synchronously and surface the outcome in
    # the response. Without it the user gets the false impression the
    # trigger is live (the bug we hit on 2026-05-14).
    if req.kind == "email_received":
        await _provision_email_watch(tid)
        async with async_session_maker() as db:
            row = await db.get(Trigger, tid)

    return _row_to_response(row)


async def _provision_email_watch(trigger_id: str) -> None:
    """Call the platform's `/v1/triggers/_provision_watch` RPC for this
    tenant and write the result into the trigger row's
    `provider_state_json` + `last_status`. Best-effort: every failure
    mode produces a structured last_status (`provisioning_failed`,
    `skipped_reauth`) the skill + UI surface verbatim, so the agent
    never lies about "live."

    Architecturally: the agent CAN'T call Gmail's `users.watch` directly
    because the refresh token lives platform-side. The platform RPC is
    the indirection.
    """
    import httpx
    from app.config import settings as _s

    base = (_s.platform_api_url or "").rstrip("/")
    key = (_s.agent_api_key or "").strip()
    uid = (_s.user_id or "").strip()
    if not base or not key or not uid:
        logger.warning(
            "[trigger_create] auto-arm skipped trigger=%s — agent missing "
            "platform_api_url/agent_api_key/user_id (run_mode=%s)",
            trigger_id, _s.run_mode,
        )
        await _stamp_trigger_state(
            trigger_id,
            last_status="provisioning_failed",
            provider_state_patch={"provision_error": "agent_misconfigured"},
        )
        return

    url = f"{base}/v1/triggers/_provision_watch"
    payload = {"user_id": uid, "connector_id": "gmail"}
    headers = {"X-Agent-Key": key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.RequestError as e:
        logger.warning(
            "[trigger_create] auto-arm transport failed trigger=%s err=%s",
            trigger_id, e,
        )
        await _stamp_trigger_state(
            trigger_id,
            last_status="provisioning_failed",
            provider_state_patch={
                "provision_error": "platform_unreachable",
                "provision_detail": str(e)[:200],
            },
        )
        return

    if r.status_code != 200:
        logger.warning(
            "[trigger_create] auto-arm non-200 trigger=%s status=%d body=%s",
            trigger_id, r.status_code, r.text[:200],
        )
        await _stamp_trigger_state(
            trigger_id,
            last_status="provisioning_failed",
            provider_state_patch={
                "provision_error": f"http_{r.status_code}",
                "provision_detail": r.text[:200],
            },
        )
        return

    try:
        data = r.json()
    except Exception:
        await _stamp_trigger_state(
            trigger_id,
            last_status="provisioning_failed",
            provider_state_patch={"provision_error": "malformed_response"},
        )
        return

    if data.get("provisioned"):
        await _stamp_trigger_state(
            trigger_id,
            last_status="never_fired",
            provider_state_patch={
                "gmail_history_id": data.get("history_id") or "",
                "watch_expires_at": data.get("expires_at") or "",
                "watch_provisioned_at": datetime.utcnow().isoformat(),
                # Clear any prior failure state.
                "provision_error": None,
                "provision_detail": None,
                "needs_refresh": None,
            },
        )
        return

    # Provisioned=false — map to the right status so the skill + UI
    # render the actionable message.
    err = (data.get("error") or "").strip()
    detail = (data.get("detail") or "")[:300]
    status_map = {
        "needs_reauth": "skipped_reauth",
        "ops_blocked": "provisioning_failed",
        "scope_error": "provisioning_failed",
        "transient": "provisioning_failed",
        "feature_disabled": "provisioning_failed",
        "unsupported_connector": "provisioning_failed",
    }
    new_status = status_map.get(err, "provisioning_failed")
    await _stamp_trigger_state(
        trigger_id,
        last_status=new_status,
        provider_state_patch={
            "provision_error": err or "unknown",
            "provision_detail": detail,
        },
    )


async def _stamp_trigger_state(
    trigger_id: str,
    *,
    last_status: str,
    provider_state_patch: dict[str, Any],
) -> None:
    """Merge `provider_state_patch` into the trigger's
    `provider_state_json` and set `last_status`. Drops keys whose value
    is None so callers can clear stale fields by passing them as None."""
    from sqlalchemy import update as _update
    from app.db.database import async_session_maker as _sm
    from app.db.models import Trigger as _Trigger

    async with _sm() as db:
        row = await db.get(_Trigger, trigger_id)
        if row is None:
            return
        existing = dict(row.provider_state_json or {})
        for k, v in provider_state_patch.items():
            if v is None:
                existing.pop(k, None)
            else:
                existing[k] = v
        await db.execute(
            _update(_Trigger)
            .where(_Trigger.id == trigger_id)
            .values(
                provider_state_json=existing,
                last_status=last_status,
                updated_at=datetime.utcnow(),
            )
        )
        await db.commit()


@router.patch("/{trigger_id}", response_model=TriggerResponse)
async def update_trigger(trigger_id: str, req: TriggerUpdate):
    """Partial update. Unset fields untouched. action change is
    permitted (with validation). Changing delivery_channels rewrites
    config_json.delivery_channels — partial channel changes are full
    list replacements (atomic from the user's PoV)."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    async with async_session_maker() as db:
        row = await db.get(Trigger, trigger_id)
        if row is None or row.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")

        if req.action is not None:
            _validate_action(req.action)
            row.action = req.action
        if req.name is not None:
            row.name = req.name
        if req.enabled is not None:
            row.enabled = req.enabled
        if req.filter_json is not None:
            row.filter_json = req.filter_json
        if req.config_json is not None or req.delivery_channels is not None:
            delivery = _validate_delivery_channels(req.delivery_channels)
            row.config_json = _merge_config(
                row.config_json, delivery, req.config_json,
            )
        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)
    return _row_to_response(row)


@router.delete("/{trigger_id}", status_code=204)
async def delete_trigger(trigger_id: str):
    """Delete a trigger. ON DELETE CASCADE drops associated events.
    Watch teardown (Gmail users.stop) is a platform-side concern —
    documented in the runbook; v1 expects manual teardown when the
    last trigger for a Gmail account is deleted."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    async with async_session_maker() as db:
        row = await db.get(Trigger, trigger_id)
        if row is None or row.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")
        await db.delete(row)
        await db.commit()
    return None


@router.get("/{trigger_id}/events", response_model=list[TriggerEventResponse])
async def list_events(
    trigger_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Paginated event history.

    PR #51 cutover: backed by ``build_jobs`` (PR #52 / mig 050 drops
    the legacy ``trigger_events`` table). The same composite index on
    ``(source_kind, source_id, created_at desc)`` keeps this cheap up
    to N=500 / offset=10k — see migration 046 for the index
    definition."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    async with async_session_maker() as db:
        # Auth: trigger must belong to the container owner.
        trig = await db.get(Trigger, trigger_id)
        if trig is None or trig.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")
        rows = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_kind == "trigger",
                BuildJob.source_id == trigger_id,
            )
            .order_by(desc(BuildJob.created_at))
            .limit(limit).offset(offset)
        )).scalars().all()
    return [_job_to_event_response(j) for j in rows]


@router.post("/{trigger_id}/test", response_model=TriggerEventResponse)
async def test_trigger(trigger_id: str):
    """Synthesise a test event. Mints a BuildJob row with
    ``idempotency_key`` prefix ``test:`` and a fresh uuid — guaranteed
    not to collide with any real Gmail message id. Hands off to the
    runner, returns the persisted row in the legacy
    ``TriggerEventResponse`` shape.

    For email_received: the handler detects the ``test:`` prefix in
    ``_fetch_all`` and substitutes a synthetic email envelope instead
    of calling MCP (which would 404 on the synthetic id). The
    summariser / notify / forward action then runs end-to-end against
    that payload, posting a real "Trigger wiring check" message into
    Day-as-Chat. The user sees an actual message in their chat
    (matching the dashboard banner) AND the trigger flips to
    ``last_status=active`` so the UI surfaces are consistent.

    PR #51 cutover: replaces the legacy ``INSERT TriggerEvent`` with
    ``JobRunner.create_job`` — same idempotency semantics
    (the composite UNIQUE on ``(source_id, idempotency_key)`` in
    mig 046 carries the legacy ``(trigger_id, event_dedupe_id)`` UNIQUE
    forward) and same runner hand-off, just one table to write to.
    Mirrors the inbound dispatch path in
    ``app/api/triggers_inbound.py:_idempotent_insert``."""
    from app.agent.job_runner import JobRunner, TaskSpec
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Trigger

    async with async_session_maker() as db:
        trig = await db.get(Trigger, trigger_id)
        if trig is None or trig.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")

    dedupe = f"test:{uuid.uuid4().hex}"
    spec = TaskSpec(
        user_id=_user_id(),
        channel="trigger",
        source_kind="trigger",
        source_id=trigger_id,
    )
    job = await JobRunner().create_job(
        job_type="trigger_run",
        spec=spec,
        # Human name, never the dedupe id. The old
        # `f"Trigger fire: {dedupe[:40]}"` put a raw uuid hex on the
        # user's Activity board ("Trigger fire: test:f2bdfda3add9…").
        # The "Test" suffix is honest: this IS a real production job,
        # minted by the user tapping Test — it is not test-suite leakage,
        # so it must be labelled rather than hidden.
        title=f"{(trig.name or '').strip() or trig.kind} (test)",
        idempotency_key=dedupe,
    )

    # Hand off to the runner — it'll claim the row + run the handler.
    if _runner is not None:
        try:
            _runner.handle_event_background(job.id)
        except Exception:
            pass

    # Re-read the fresh row so the response reflects whatever the
    # runner has stamped in the synchronous claim path (best-effort —
    # for inline test mode the runner's hand-off is fire-and-forget,
    # the row may still be queued).
    async with async_session_maker() as db:
        fresh = await db.get(BuildJob, job.id) or job
    return _job_to_event_response(fresh)


class TriggerProbeResponse(BaseModel):
    """Structured diagnosis returned by `/api/triggers/{id}/probe`.

    Frontend renders `diagnosis` verbatim as the headline + the four
    age timestamps in the modal body. The `diagnosis` enum is closed —
    the state machine that computes it lives in `_compute_diagnosis`.
    """

    watch_alive: bool
    watch_alive_reason: Optional[str] = None
    last_push_received_at: Optional[str] = None
    last_dispatch_received_at: Optional[str] = None
    last_handler_completed_at: Optional[str] = None
    last_handler_status: Optional[str] = None
    last_handler_error_class: Optional[str] = None
    last_push_age_seconds: Optional[int] = None
    last_dispatch_age_seconds: Optional[int] = None
    last_handler_age_seconds: Optional[int] = None
    # See `_compute_diagnosis` for the closed enum:
    #   pubsub_not_pushing | push_rejected_at_platform |
    #   dispatched_but_no_handler_response | handler_failing |
    #   healthy | needs_reauth | watch_expired
    diagnosis: str
    diagnosis_detail: Optional[str] = None


def _age_seconds(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - dt).total_seconds())


def _compute_diagnosis(
    *,
    watch_alive: bool,
    watch_alive_reason: Optional[str],
    push_age: Optional[int],
    dispatch_age: Optional[int],
    handler_age: Optional[int],
    handler_status: Optional[str],
) -> tuple[str, Optional[str]]:
    """Closed-enum diagnosis state machine.

    The "fresh push but stale handler" gap is the load-bearing
    signal: if push_age is recent but handler_age is None or much
    older, the chain is failing somewhere between platform dispatch
    and the agent's handler — exactly the kind of break that's
    invisible without these timestamps.
    """
    PUSH_STALE_SEC = 7 * 24 * 3600                # > 7d → assume Pub/Sub silent
    HANDLER_LAG_SEC = 5 * 60                       # handler runs within 5 min of push

    if not watch_alive:
        if watch_alive_reason in ("gmail_401", "token_mint_failed"):
            return ("needs_reauth",
                    "Gmail refused our token. Reconnect Gmail to resume.")
        if watch_alive_reason == "watch_expired":
            return ("watch_expired",
                    "Cached watch expiration is past. Click Verify to re-arm.")
        if watch_alive_reason == "no_connector_identity":
            return ("needs_reauth",
                    "No active Gmail identity for this user.")
        return ("pubsub_not_pushing",
                f"Watch not alive (reason: {watch_alive_reason or 'unknown'}).")

    if push_age is None or push_age > PUSH_STALE_SEC:
        return ("pubsub_not_pushing",
                "Watch is alive but no Pub/Sub push has reached the platform "
                "in 7+ days. Check the GCP Pub/Sub subscription's push URL "
                "and IAM bindings.")

    if dispatch_age is None:
        return ("push_rejected_at_platform",
                "Pub/Sub is pushing but the platform never dispatched. "
                "Common cause: JWT audience mismatch. Check "
                "PUBSUB_PUSH_AUDIENCE matches the subscription's audience.")

    if dispatch_age > push_age + HANDLER_LAG_SEC:
        # Dispatches are much older than the freshest push — the
        # platform stopped reaching the agent.
        return ("push_rejected_at_platform",
                "Recent pushes are not being dispatched. Platform-side "
                "history-fetch or agent reachability is broken.")

    if handler_age is None or handler_age > dispatch_age + HANDLER_LAG_SEC:
        return ("dispatched_but_no_handler_response",
                "Events reached the agent but the handler hasn't run. "
                "The TriggerRunner may be stuck or crashing.")

    if handler_status and handler_status not in ("success", "test_success",
                                                 "success_empty"):
        return ("handler_failing",
                f"Handler last terminated with status={handler_status}. "
                "Inspect Recent events for the specific error.")

    return ("healthy",
            "Watch alive, pushes received, dispatched, handler running clean.")


class RecordProvisionFailureReq(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=36)
    connector_id: str = Field(default="gmail", min_length=1, max_length=64)
    error: str = Field(..., min_length=1, max_length=100)
    detail: Optional[str] = Field(default=None, max_length=500)


class RecordProvisionFailureResponse(BaseModel):
    triggers_updated: int


@router.post(
    "/_record_provision_failure",
    response_model=RecordProvisionFailureResponse,
    summary="Platform-to-agent: record a watch arming failure on this user's triggers",
)
async def record_provision_failure(
    req: RecordProvisionFailureReq,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
):
    """Called by the platform's OAuth callback when
    `_arm_gmail_watch_post_connect` fails. Without this, an arm
    failure was invisible until the 6h refresh tick — the user
    reconnected Gmail, the trigger silently stayed dormant.

    Auth: same X-Agent-Key contract as the inbound dispatch endpoint
    (the env-var copy of `AgentConfig.agent_api_key`). Body's user_id
    must match the container owner.

    Idempotent at the row level — each call overwrites
    provision_error/detail/timestamp. Multiple Gmail triggers per
    user are all updated in one transaction.
    """
    expected_key = (getattr(settings, "agent_api_key", "") or "").strip()
    container_user_id = (getattr(settings, "user_id", "") or "").strip()
    if not expected_key or not container_user_id:
        raise HTTPException(
            status_code=503,
            detail="Agent not configured for provision-failure inbound",
        )
    if not x_agent_key or x_agent_key.strip() != expected_key:
        raise HTTPException(status_code=401, detail="invalid X-Agent-Key")
    if req.user_id != container_user_id:
        raise HTTPException(
            status_code=401,
            detail="envelope user_id does not match container owner",
        )

    # Only gmail today. We dispatch by connector_id explicitly so
    # future connectors (calendar, drive) get their own kind-of-trigger
    # update without accidentally hitting email_received rows.
    kind_for_connector = {
        "gmail": "email_received",
    }
    kind = kind_for_connector.get(req.connector_id)
    if kind is None:
        return RecordProvisionFailureResponse(triggers_updated=0)

    from app.db.database import async_session_maker
    from app.db.models import Trigger
    from sqlalchemy import update as _update

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    updated = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Trigger).where(
                Trigger.user_id == req.user_id,
                Trigger.kind == kind,
            )
        )).scalars().all()
        for row in rows:
            ps = dict(row.provider_state_json or {})
            ps["provision_error"] = req.error
            if req.detail:
                ps["provision_detail"] = req.detail[:300]
            ps["last_provision_attempt_at"] = now_iso
            row.provider_state_json = ps
            row.last_status = "provisioning_failed"
            row.last_error = (req.detail or req.error)[:1000]
            updated += 1
        await db.commit()

    logger.info(
        "[record_provision_failure] user=%s connector=%s error=%s updated=%d",
        req.user_id[:8], req.connector_id, req.error, updated,
    )
    return RecordProvisionFailureResponse(triggers_updated=updated)


@router.post("/{trigger_id}/probe", response_model=TriggerProbeResponse)
async def probe_trigger(trigger_id: str):
    """End-to-end delivery-chain diagnosis.

    Reads agent-side trigger timestamps + calls the platform's
    `_probe_watch` RPC for connector-side state. Returns a structured
    diagnosis the UI's Diagnose modal renders verbatim.

    Idempotent and side-effect-free apart from logging — safe to call
    on every dashboard refresh if the UI wants live status.
    """
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    async with async_session_maker() as db:
        row = await db.get(Trigger, trigger_id)
        if row is None or row.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")
        if row.kind != "email_received":
            raise HTTPException(
                status_code=400,
                detail=f"probe not supported for kind={row.kind!r}",
            )

    ps = row.provider_state_json or {}

    # Call platform's _probe_watch to get the connector-side bits.
    import httpx as _httpx
    from app.config import settings as _s

    base = (_s.platform_api_url or "").rstrip("/")
    key = (_s.agent_api_key or "").strip()
    uid = (_s.user_id or "").strip()
    watch_alive = False
    watch_reason: Optional[str] = "agent_misconfigured"
    last_push: Optional[str] = None
    if base and key and uid:
        try:
            async with _httpx.AsyncClient(timeout=12.0) as client:
                r = await client.post(
                    f"{base}/v1/triggers/_probe_watch",
                    json={"user_id": uid, "connector_id": "gmail"},
                    headers={
                        "X-Agent-Key": key,
                        "Content-Type": "application/json",
                    },
                )
            if r.status_code == 200:
                data = r.json()
                watch_alive = bool(data.get("watch_alive"))
                watch_reason = data.get("watch_alive_reason")
                last_push = data.get("last_push_received_at")
            else:
                watch_reason = f"platform_probe_http_{r.status_code}"
        except _httpx.RequestError as e:
            watch_reason = f"platform_probe_unreachable:{type(e).__name__}"

    last_dispatch = ps.get("last_webhook_dispatch_at")
    last_handler = ps.get("last_handler_completed_at")
    handler_status = ps.get("last_handler_status")
    handler_error_class = ps.get("last_handler_error_class")

    push_age = _age_seconds(last_push)
    dispatch_age = _age_seconds(last_dispatch)
    handler_age = _age_seconds(last_handler)

    diagnosis, detail = _compute_diagnosis(
        watch_alive=watch_alive,
        watch_alive_reason=watch_reason,
        push_age=push_age,
        dispatch_age=dispatch_age,
        handler_age=handler_age,
        handler_status=handler_status,
    )

    return TriggerProbeResponse(
        watch_alive=watch_alive,
        watch_alive_reason=watch_reason,
        last_push_received_at=last_push,
        last_dispatch_received_at=last_dispatch,
        last_handler_completed_at=last_handler,
        last_handler_status=handler_status,
        last_handler_error_class=handler_error_class,
        last_push_age_seconds=push_age,
        last_dispatch_age_seconds=dispatch_age,
        last_handler_age_seconds=handler_age,
        diagnosis=diagnosis,
        diagnosis_detail=detail,
    )


@router.post("/{trigger_id}/reverify", response_model=TriggerResponse)
async def reverify_trigger(trigger_id: str):
    """Re-run watch provisioning for this trigger and return the
    refreshed row.

    Why this exists: the auto-arm in `create_trigger` is best-effort.
    If it fails (platform RPC blip, transient Google 5xx, GCP
    misconfig at the time of creation), the trigger row sits forever
    with stale `provider_state_json`. The daily refresh job
    (`run_gmail_watch_refresh`) re-arms the *connector identity's*
    watch but doesn't touch this trigger row's metadata. Before this
    endpoint existed, the only recovery was delete + recreate, which
    loses the event history and any custom filters.

    The endpoint is idempotent at the Gmail end — a fresh `users.watch`
    over a live one is harmless and resets the expiration clock. So
    calling reverify on a healthy trigger is a no-op apart from
    updating `gmail_history_id` to whatever Gmail thinks is current.
    """
    from app.db.database import async_session_maker
    from app.db.models import Trigger

    async with async_session_maker() as db:
        row = await db.get(Trigger, trigger_id)
        if row is None or row.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")
        if row.kind != "email_received":
            raise HTTPException(
                status_code=400,
                detail=f"reverify not supported for kind={row.kind!r}",
            )

    await _provision_email_watch(trigger_id)

    async with async_session_maker() as db:
        row = await db.get(Trigger, trigger_id)
        # Eagerly load recent events for the response so the UI can
        # render the refreshed card without a separate GET round trip.
        # PR #51 cutover: same BuildJob-backed query as list_triggers.
        from app.db.models import BuildJob
        recent = (await db.execute(
            select(BuildJob).where(
                BuildJob.source_kind == "trigger",
                BuildJob.source_id == trigger_id,
            ).order_by(desc(BuildJob.created_at)).limit(20)
        )).scalars().all()
    return _row_to_response(row, recent)
