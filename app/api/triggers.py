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
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
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
    if action not in TRIGGER_ACTIONS:
        allowed = sorted(TRIGGER_ACTIONS)
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
    recent_events: list[TriggerEventResponse] = Field(default_factory=list)


# ── Response helpers ─────────────────────────────────────────────────


def _row_to_response(trigger, recent_events=()) -> TriggerResponse:
    cfg = trigger.config_json or {}
    ps = trigger.provider_state_json or {}
    delivery = cfg.get("delivery_channels") or ["website"]
    watch_provisioned = bool(ps.get("gmail_history_id")) and not bool(
        ps.get("needs_refresh")
    )
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
        recent_events=[
            TriggerEventResponse(
                id=e.id,
                trigger_id=e.trigger_id,
                event_dedupe_id=e.event_dedupe_id,
                received_at=e.received_at,
                started_at=e.started_at,
                finished_at=e.finished_at,
                status=e.status,
                error_class=e.error_class,
                error_detail=e.error_detail,
                summary_message_id=e.summary_message_id,
                coalesced_into_event_id=e.coalesced_into_event_id,
            )
            for e in recent_events
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
    from app.db.models import Trigger, TriggerEvent

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Trigger).where(Trigger.user_id == _user_id())
            .order_by(desc(Trigger.created_at))
        )).scalars().all()
        out: list[TriggerResponse] = []
        for t in rows:
            recent = (await db.execute(
                select(TriggerEvent).where(TriggerEvent.trigger_id == t.id)
                .order_by(desc(TriggerEvent.received_at)).limit(20)
            )).scalars().all()
            out.append(_row_to_response(t, recent))
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
    """Paginated event history. The hot path index
    `ix_trigger_events_trigger_received` makes this trivially cheap
    up to N=500 / offset=10k."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    async with async_session_maker() as db:
        # Auth: trigger must belong to the container owner.
        trig = await db.get(Trigger, trigger_id)
        if trig is None or trig.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")
        rows = (await db.execute(
            select(TriggerEvent).where(TriggerEvent.trigger_id == trigger_id)
            .order_by(desc(TriggerEvent.received_at))
            .limit(limit).offset(offset)
        )).scalars().all()
    return [
        TriggerEventResponse(
            id=e.id, trigger_id=e.trigger_id, event_dedupe_id=e.event_dedupe_id,
            received_at=e.received_at, started_at=e.started_at,
            finished_at=e.finished_at, status=e.status,
            error_class=e.error_class, error_detail=e.error_detail,
            summary_message_id=e.summary_message_id,
            coalesced_into_event_id=e.coalesced_into_event_id,
        )
        for e in rows
    ]


@router.post("/{trigger_id}/test", response_model=TriggerEventResponse)
async def test_trigger(trigger_id: str):
    """Synthesise a test event. Inserts a row with `event_dedupe_id`
    prefix `test:` and a fresh uuid — guaranteed not to collide with
    any real Gmail message id. Dispatches via the runner, returns the
    persisted row.

    For email_received: the handler will try to fetch the gmail_message_id
    `test:<uuid>` via MCP and fail. That's the expected outcome — the
    test verifies the wiring (auth, runner pickup, DB write), not a
    real summary. If a user wants to test against a real message they
    can copy a real id into the dedupe field manually."""
    from app.db.database import async_session_maker
    from app.db.models import Trigger, TriggerEvent

    async with async_session_maker() as db:
        trig = await db.get(Trigger, trigger_id)
        if trig is None or trig.user_id != _user_id():
            raise HTTPException(status_code=404, detail="Trigger not found")

        event_id = str(uuid.uuid4())
        dedupe = f"test:{uuid.uuid4().hex}"
        ev = TriggerEvent(
            id=event_id,
            trigger_id=trigger_id,
            user_id=_user_id(),
            event_dedupe_id=dedupe,
            received_at=datetime.utcnow(),
            status="queued",
        )
        db.add(ev)
        await db.commit()
        await db.refresh(ev)

    # Hand off to the runner — it'll claim the row + run the handler.
    if _runner is not None:
        try:
            _runner.handle_event_background(event_id)
        except Exception:
            pass

    return TriggerEventResponse(
        id=ev.id, trigger_id=ev.trigger_id, event_dedupe_id=ev.event_dedupe_id,
        received_at=ev.received_at, started_at=ev.started_at,
        finished_at=ev.finished_at, status=ev.status,
        error_class=ev.error_class, error_detail=ev.error_detail,
        summary_message_id=ev.summary_message_id,
        coalesced_into_event_id=ev.coalesced_into_event_id,
    )
