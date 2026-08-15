"""Confirmation cards for `elevation: true` connector tools.

When the agent calls a tool the user must approve first — sending mail,
posting to LinkedIn, writing a calendar event — the dispatcher stages
the arguments as a `ConnectorPendingAction` and returns without touching
the provider. This module is where the user says yes.

  GET   /connectors/pending-actions            — list (default: pending)
  GET   /connectors/pending-actions/{id}       — one, for card reload
  POST  /connectors/pending-actions/{id}/approve  — edit + run it
  POST  /connectors/pending-actions/{id}/reject   — discard it

THE THING TO PRESERVE IF YOU EDIT THIS FILE
───────────────────────────────────────────
Approval is a single guarded UPDATE:

    UPDATE connector_pending_actions
       SET status = 'approved', ...
     WHERE id = :id AND status = 'pending'

and we act only if it touched exactly one row. That is the whole
defence against sending the same email twice, and a double-tap on a
phone with a slow network is not a hypothetical — it is the default
user behaviour when a button does not visibly respond. Read-then-write,
ORM `row.status = "approved"`, or "check then execute then mark" all
reintroduce the race. Keep it one statement.

The second invariant: `connector_id` and `tool_name` come from the
DB row and NEVER from the request body. The client may edit the tool's
ARGUMENTS (that is the whole point of the card), but it must not be
able to turn an approved "send an email" into "delete the calendar".
`_coerce_payload` additionally whitelists the arguments down to the
properties the manifest declares, so nothing extra rides along.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import ConnectorPendingAction
from app.services import connector_dispatcher as dispatcher
from app.services.connector_registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connectors", tags=["Connectors"])

#: Short by design. This hop is a courtesy update to a progress surface —
#: the send already happened, and the user is looking at the card, not at
#: Mission Control. Blocking their approve response on a slow tenant agent
#: would trade the thing they care about for the thing they don't.
_RESUME_TIMEOUT_S = 5.0


async def _resume_job_for_action(
    db: AsyncSession, *, user_id: str, action_id: str, outcome: str,
    detail: Optional[str] = None,
) -> None:
    """Tell the tenant agent this card is decided, so the job it parked
    can terminalise.

    A job that parks on a confirmation card (`waiting_on_user`) has no
    other closer: the stalled-job reaper only sweeps queued/running and
    the turn-end finalizer only touches running. Without this call the
    honest "Waiting on you" becomes a permanent one, which is a quieter
    version of the same lie — the user approves, the mail goes out, and
    Mission Control still says they need to do something.

    Best-effort by contract. The card's own status is the record of what
    happened; a tenant agent that is asleep, mid-rollout or unreachable
    must never turn a successful send into a 5xx on the approve request.
    """
    from app.api.apps_proxy import _get_agent
    from app.services.agent_http import get_agent_http_client

    try:
        agent = await _get_agent(user_id, db)
        if not agent:
            return
        agent_url, agent_key = agent[0], agent[1]
        client = get_agent_http_client()
        resp = await client.post(
            f"{agent_url.rstrip('/')}/api/agent/jobs/resolve-pending-action",
            json={"action_id": action_id, "outcome": outcome, "detail": detail},
            headers={"X-Agent-Key": agent_key},
            timeout=_RESUME_TIMEOUT_S,
        )
        if resp.status_code >= 400:
            logger.warning(
                "[pending_actions] job resume returned %s for action=%s",
                resp.status_code, action_id,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[pending_actions] job resume failed for action=%s: %s", action_id, e,
        )


# ─── Schemas ─────────────────────────────────────────────────────────


class PendingActionOut(BaseModel):
    id: str
    connector_id: str
    tool_name: str
    summary: str
    payload: dict
    status: str
    channel: str
    conversation_id: Optional[str] = None
    created_at: str
    expires_at: str
    decided_at: Optional[str] = None
    decided_via: Optional[str] = None
    result: Optional[dict] = None
    # Which fields the card should render as editable text inputs, in
    # display order. Derived from the manifest so a new connector gets
    # a usable card without a frontend change.
    editable_fields: list["PendingActionField"] = Field(default_factory=list)


class PendingActionField(BaseModel):
    name: str
    label: str
    required: bool
    multiline: bool = False


class PendingActionListResponse(BaseModel):
    actions: list[PendingActionOut]


class ApproveRequest(BaseModel):
    """The (possibly edited) tool arguments, plus where the tap came from.

    `payload` omitted entirely = approve exactly what was staged. That
    is the common case — the user read the draft and hit Send.
    """
    payload: Optional[dict] = None
    decided_via: Optional[str] = None


class RejectRequest(BaseModel):
    decided_via: Optional[str] = None


PendingActionOut.model_rebuild()


# ─── Helpers ─────────────────────────────────────────────────────────


# Fields we render as a textarea rather than a single-line input.
_MULTILINE_FIELDS = frozenset({"body", "text", "message", "content", "description", "comment"})

# Human labels for the argument names that show up across connectors.
# Anything absent falls back to a title-cased field name, which reads
# fine ("video_id" → "Video id") and costs nothing to leave unmapped.
_FIELD_LABELS = {
    "to": "To",
    "cc": "Cc",
    "bcc": "Bcc",
    "subject": "Subject",
    "body": "Message",
    "text": "Text",
    "summary": "Title",
    "owner": "Owner",
    "repo": "Repository",
    "number": "Issue / PR number",
    "visibility": "Visibility",
}


def _tool_spec_or_404(tool_name: str):
    spec = get_registry().get_tool_spec(tool_name)
    if spec is None:
        # The connector was removed (or renamed) between staging and
        # approval. Refuse rather than guess — we have no schema to
        # validate the edited arguments against.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"The tool behind this request ({tool_name}) is no longer "
                f"available, so it cannot be run. Ask your agent again."
            ),
        )
    return spec


def _editable_fields(tool_name: str) -> list[PendingActionField]:
    spec = get_registry().get_tool_spec(tool_name)
    if spec is None:
        return []
    schema = spec.input_schema or {}
    props: dict = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: list[PendingActionField] = []
    for name, sub in props.items():
        # Only free-text-ish scalars are editable on the card. An
        # object/array argument is shown read-only in the JSON fallback
        # rather than given a text box that would round-trip badly.
        if (sub or {}).get("type") not in ("string", "integer", "number"):
            continue
        out.append(PendingActionField(
            name=name,
            label=_FIELD_LABELS.get(name, name.replace("_", " ").capitalize()),
            required=name in required,
            multiline=name in _MULTILINE_FIELDS,
        ))
    # Required fields first, then manifest order — so a card leads with
    # To / Subject / Message rather than Cc.
    out.sort(key=lambda f: (not f.required,))
    return out


def _coerce_payload(tool_name: str, staged: dict, edited: Optional[dict]) -> dict:
    """Merge the user's edits over the staged draft and validate.

    Whitelists to the properties the manifest declares — an argument the
    schema does not mention is DROPPED, not passed through. That is what
    stops a hand-rolled POST from smuggling extra arguments into a call
    the user approved on the strength of a card that never showed them.

    Deliberately hand-rolled rather than `jsonschema`: that package is
    not in any of backend/requirements*.txt, it is only present
    transitively via fastmcp, and a security boundary should not rest on
    a dependency nobody declared. Connector input schemas are flat
    objects of scalars, which is the entire surface handled here.
    """
    spec = _tool_spec_or_404(tool_name)
    schema = spec.input_schema or {}
    props: dict = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    merged = dict(staged)
    if edited:
        for k, v in edited.items():
            if k in props:
                merged[k] = v

    out: dict[str, Any] = {}
    for key, value in merged.items():
        if key not in props:
            continue  # not declared → dropped
        if value is None:
            continue
        sub = props.get(key) or {}
        want = sub.get("type")
        try:
            if want == "string":
                value = value if isinstance(value, str) else str(value)
            elif want == "integer":
                value = int(value)
            elif want == "number":
                value = float(value)
            elif want == "boolean":
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = bool(value)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{key!r} must be a {want}.",
            )
        allowed = sub.get("enum")
        if allowed and value not in allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{key!r} must be one of {allowed}.",
            )
        if want in ("integer", "number"):
            lo, hi = sub.get("minimum"), sub.get("maximum")
            if lo is not None and value < lo:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"{key!r} must be at least {lo}.",
                )
            if hi is not None and value > hi:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"{key!r} must be at most {hi}.",
                )
        out[key] = value

    missing = [
        r for r in required
        if r not in out or (isinstance(out[r], str) and not out[r].strip())
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Still missing: {', '.join(missing)}. Fill these in before "
                f"confirming."
            ),
        )
    return out


def _to_out(row: ConnectorPendingAction) -> PendingActionOut:
    try:
        payload = json.loads(row.payload_json) if row.payload_json else {}
    except (ValueError, TypeError):
        payload = {}
    result = None
    if row.result_json:
        try:
            result = json.loads(row.result_json)
        except (ValueError, TypeError):
            result = None
    return PendingActionOut(
        id=row.id,
        connector_id=row.connector_id,
        tool_name=row.tool_name,
        summary=dispatcher.summarize_pending_action(row.tool_name, payload),
        payload=payload,
        status=row.status,
        channel=row.channel,
        conversation_id=row.conversation_id,
        created_at=row.created_at.isoformat() + "Z",
        expires_at=row.expires_at.isoformat() + "Z",
        decided_at=(row.decided_at.isoformat() + "Z") if row.decided_at else None,
        decided_via=row.decided_via,
        result=result,
        editable_fields=_editable_fields(row.tool_name),
    )


async def _load_owned_or_404(
    db: AsyncSession, action_id: str, user_id: str,
) -> ConnectorPendingAction:
    """Fetch the row, 404 if it is not this user's.

    404 rather than 403 on an ownership miss: a 403 confirms the id
    exists, which is a (small) enumeration oracle over other users'
    pending sends. Not-found is the honest answer from this user's
    point of view anyway.
    """
    row = (await db.execute(
        select(ConnectorPendingAction)
        .where(ConnectorPendingAction.id == action_id)
        .where(ConnectorPendingAction.user_id == user_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such pending action.",
        )
    return row


# ─── List / read ─────────────────────────────────────────────────────


@router.get("/pending-actions", response_model=PendingActionListResponse)
async def list_pending_actions(
    action_status: str = Query(
        default="pending",
        alias="status",
        description="'pending' (default), any single status, or 'all'.",
    ),
    conversation_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PendingActionListResponse:
    """Cards awaiting this user, newest first.

    The clients call this on chat load so a card survives a refresh —
    an approval prompt that disappears when you reload is worse than no
    prompt, because the user assumes it went through.
    """
    q = (
        select(ConnectorPendingAction)
        .where(ConnectorPendingAction.user_id == str(current_user.id))
        .order_by(ConnectorPendingAction.created_at.desc())
        .limit(limit)
    )
    if action_status != "all":
        q = q.where(ConnectorPendingAction.status == action_status)
    if conversation_id:
        q = q.where(ConnectorPendingAction.conversation_id == conversation_id)

    rows = list((await db.execute(q)).scalars().all())

    # Expire lazily on read. A background sweep would be tidier, but a
    # card the user is looking at RIGHT NOW must never render as
    # actionable past its deadline, and this is the moment we know.
    now = datetime.utcnow()
    stale = [r for r in rows if r.status == "pending" and r.expires_at <= now]
    if stale:
        await db.execute(
            sa_update(ConnectorPendingAction)
            .where(ConnectorPendingAction.id.in_([r.id for r in stale]))
            .where(ConnectorPendingAction.status == "pending")
            .values(status="expired", decided_at=now)
        )
        await db.commit()
        for r in stale:
            r.status = "expired"

    return PendingActionListResponse(actions=[_to_out(r) for r in rows])


@router.get("/pending-actions/{action_id}", response_model=PendingActionOut)
async def get_pending_action(
    action_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PendingActionOut:
    row = await _load_owned_or_404(db, action_id, str(current_user.id))
    return _to_out(row)


# ─── Approve ─────────────────────────────────────────────────────────


@router.post("/pending-actions/{action_id}/approve", response_model=PendingActionOut)
async def approve_pending_action(
    action_id: str,
    req: ApproveRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PendingActionOut:
    """Run the staged call, optionally with the user's edits.

    Order is deliberate: validate BEFORE claiming. A 422 on a typo'd
    recipient must leave the card `pending` so the user can fix it —
    claiming first would burn the row and force them to ask the agent
    all over again.
    """
    user_id = str(current_user.id)
    row = await _load_owned_or_404(db, action_id, user_id)

    if row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This was already {row.status}."
                if row.status != "approved"
                else "This is already being sent."
            ),
        )

    now = datetime.utcnow()
    if row.expires_at <= now:
        await db.execute(
            sa_update(ConnectorPendingAction)
            .where(ConnectorPendingAction.id == row.id)
            .where(ConnectorPendingAction.status == "pending")
            .values(status="expired", decided_at=now)
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=(
                "This request expired before it was confirmed. Ask your "
                "agent to prepare it again."
            ),
        )

    try:
        staged = json.loads(row.payload_json) if row.payload_json else {}
    except (ValueError, TypeError):
        staged = {}
    final_payload = _coerce_payload(row.tool_name, staged, req.payload)

    # ── The claim. One statement, guarded on `status = 'pending'`. ──
    # Whoever's UPDATE lands first owns the send; everyone else sees
    # rowcount 0 and gets a 409. This is the double-tap defence — do
    # not refactor it into a read-then-write.
    claimed = await db.execute(
        sa_update(ConnectorPendingAction)
        .where(ConnectorPendingAction.id == row.id)
        .where(ConnectorPendingAction.status == "pending")
        .values(
            status="approved",
            decided_at=now,
            decided_via=(req.decided_via or row.channel or "web")[:32],
            payload_json=json.dumps(final_payload, sort_keys=True, default=str),
        )
    )
    await db.commit()
    if (claimed.rowcount or 0) != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This is already being sent.",
        )

    # ── Past this point the row is ours and can never be claimed again,
    # so every exit below records an outcome rather than re-opening it.
    logger.info(
        "[pending_actions] approved id=%s tool=%s user=%s",
        row.id, row.tool_name, user_id[:8],
    )

    outcome_status = "failed"
    result_payload: dict
    try:
        result = await dispatcher.execute(
            db=db,
            user_id=user_id,
            connector_id=row.connector_id,
            tool_name=row.tool_name,
            tool_input=final_payload,
            channel=row.channel or "web",
            agent_request_id=row.agent_request_id,
            approved_action_id=row.id,
        )
    except Exception as e:
        # The dispatcher is documented not to raise for normal error
        # paths, so this is a genuine defect somewhere below us. Record
        # it against the row — an approved action that left no trace is
        # the worst possible artifact of this feature.
        logger.exception(
            "[pending_actions] dispatcher raised for id=%s: %s", row.id, e,
        )
        result_payload = {
            "kind": "tool_error",
            "message": "Something broke while sending. Nothing was retried.",
        }
    else:
        from app.services.connector_mcp import _serialize_result
        result_payload = _serialize_result(result)
        if result_payload.get("kind") == "ok":
            outcome_status = "executed"

    await db.execute(
        sa_update(ConnectorPendingAction)
        .where(ConnectorPendingAction.id == row.id)
        .values(
            status=outcome_status,
            result_json=json.dumps(result_payload, default=str)[:8000],
        )
    )
    await db.commit()

    # AFTER the commit: the card's own row is the record of what happened,
    # and it must be durable before anything downstream is told about it.
    await _resume_job_for_action(
        db, user_id=user_id, action_id=row.id, outcome=outcome_status,
        detail=(
            None if outcome_status == "executed"
            else str(result_payload.get("message") or "")[:200] or None
        ),
    )

    refreshed = await _load_owned_or_404(db, action_id, user_id)
    return _to_out(refreshed)


# ─── Reject ──────────────────────────────────────────────────────────


@router.post("/pending-actions/{action_id}/reject", response_model=PendingActionOut)
async def reject_pending_action(
    action_id: str,
    req: RejectRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PendingActionOut:
    """Discard the draft. Idempotent for an already-rejected row so a
    double-tap on Cancel is a no-op rather than a 409 the user has to
    read and dismiss."""
    user_id = str(current_user.id)
    row = await _load_owned_or_404(db, action_id, user_id)

    if row.status == "rejected":
        return _to_out(row)
    if row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This was already {row.status}.",
        )

    now = datetime.utcnow()
    await db.execute(
        sa_update(ConnectorPendingAction)
        .where(ConnectorPendingAction.id == row.id)
        .where(ConnectorPendingAction.status == "pending")
        .values(
            status="rejected",
            decided_at=now,
            decided_via=(req.decided_via or row.channel or "web")[:32],
        )
    )
    await db.commit()
    logger.info(
        "[pending_actions] rejected id=%s tool=%s user=%s",
        row.id, row.tool_name, user_id[:8],
    )
    # A rejected card unblocks its job too — cancelled, not failed. The
    # user made a decision; nothing broke. Left out, saying "no" would
    # leave the job parked on a card that no longer exists.
    await _resume_job_for_action(
        db, user_id=user_id, action_id=row.id, outcome="rejected",
    )
    refreshed = await _load_owned_or_404(db, action_id, user_id)
    return _to_out(refreshed)
