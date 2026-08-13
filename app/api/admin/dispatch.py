"""Admin — Dispatch (operator → user announcements).

An admin composes one message and sends it to a single user or to everyone. It
lands in the user's chat as a card that is visibly NOT from their agent, and as
an ``announcement`` notification (Live Activity / lock screen). ``mode`` decides
its life: ``once`` is retracted the moment the user reads it, ``persistent``
stays and carries a reply, which lands in the Admin thread read here.

Routes (mounted at settings.api_prefix, e.g. /api):
  POST /admin/dispatch                    — compose + queue a dispatch
  GET  /admin/dispatch/preview            — authoritative pre-send headcount
  GET  /admin/dispatch                    — recent dispatches
  GET  /admin/dispatch/{id}               — one dispatch + per-target delivery
  POST /admin/dispatch/{id}/retry         — re-queue everything unfinished
  GET  /admin/dispatch/threads            — users with an Admin thread
  GET  /admin/dispatch/threads/{user_id}  — one thread (marks it admin-read)
  POST /admin/dispatch/threads/{user_id}  — reply into a thread

EVERY route here takes ``Depends(require_admin)``. The ``/api/admin/`` prefix is
a naming convention, not a guard — ``app/api/admin/system.py`` is mounted under
the same prefix with no auth on it at all.

Fan-out — resolving the audience, the tenant chat row, the notification row, the
agent hop — belongs to ``app/services/admin_dispatch_worker.py`` and runs off the
request path with its own session. ``audience='all'`` is every user on the
platform; the admin's HTTP request must not pay for a broadcast. This module
therefore writes the ``admin_dispatches`` row and nothing else: the target rows
are the worker's, which is also the only way the drip schedule and the CAS claim
have one owner.

``settings.admin_dispatch_enabled`` gates the two PRODUCING routes only
(``create_dispatch`` and ``retry_dispatch``). It is the kill switch for a bad
broadcast, and a kill switch that also blinds the operator to the replies that
broadcast caused is worse than useless: ``notices.py`` correctly stays open, so
users keep writing into a thread nobody could read.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import and_, case, func, or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.config import settings
from app.db import get_db, User
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadMessage,
    CHAT_DELIVERED,
    CHAT_NO_AGENT,
    CHAT_RETRACTED,
    DISPATCH_AUDIENCE_ALL,
    DISPATCH_AUDIENCE_USER,
    DISPATCH_QUEUED,
    TARGET_DONE,
    TARGET_FAILED,
    TARGET_PENDING,
    TARGET_SENDING,
    THREAD_IN,
    THREAD_OUT,
)
from app.services.admin_dispatch_worker import (
    build_reply_notification,
    count_recipients,
    preview_spec,
    spawn_dispatch_fanout,
    summarize_targets,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/dispatch", tags=["admin-dispatch"])

# An accidental broadcast is unrecallable — `once` retracts on read, but nothing
# un-notifies a phone. The admin has to type this word.
BROADCAST_CONFIRM = "BROADCAST"

# How long a CAS-claimed target may sit in `sending` before the retry route may
# take it back. A worker that was OOM-killed, redeployed or lost its replica
# mid-target leaves the row claimed forever: `_reconcile` counts it as
# unfinished, so the dispatch is pinned at `sending` with no route to
# completion and the retry button (which only reset `failed`) could not reach
# it.
#
# Reclaiming is safe because every write on the send path is idempotent BY
# CONSTRUCTION, not by convention: the tenant `Message` id is a uuid5 of
# (dispatch, user), the thread row's id is a uuid5 of the same pair, and the
# notification carries a deterministic `idempotency_key`. A second pass over a
# target the first pass had half-delivered re-hits those keys instead of writing
# anything twice — which is also why the same reset is safe for a `done` target
# whose chat card never landed.
#
# The window only has to exceed the longest honest delivery: one agent hop is
# capped at 15s (`admin_dispatch_worker._AGENT_HOP_TIMEOUT_S`).
_STUCK_CLAIM_MAX_AGE = timedelta(minutes=15)


def _utc_iso(v: Optional[datetime]) -> Optional[str]:
    """Emit RFC 3339 UTC with a trailing ``Z``. Mirrors ``app.api.routines``
    — the columns are naive UTC, and a bare isoformat is read as LOCAL time by
    both clients (the 2026-05-14 dashboard timezone bug)."""
    if v is None:
        return None
    if v.tzinfo is None:
        return v.isoformat() + "Z"
    return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_enabled() -> None:
    """Kill switch for the PRODUCING routes only — see the module docstring."""
    if not getattr(settings, "admin_dispatch_enabled", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin dispatch is disabled (set ADMIN_DISPATCH_ENABLED=true).",
        )


# ── Requests ─────────────────────────────────────────────────────────

class DispatchCreate(BaseModel):
    mode: str = Field(pattern="^(once|persistent)$")
    audience: str = Field(pattern="^(user|all)$")
    target_user_id: Optional[str] = Field(default=None, max_length=36)
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    urgent: bool = False
    sender_name: Optional[str] = Field(default=None, max_length=80)
    confirm: Optional[str] = Field(default=None, max_length=40)

    @field_validator("title", "body")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("sender_name")
    @classmethod
    def _sender_or_none(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        return v or None

    @model_validator(mode="after")
    def _audience_shape(self) -> "DispatchCreate":
        if self.audience == DISPATCH_AUDIENCE_USER and not (self.target_user_id or "").strip():
            raise ValueError("target_user_id is required when audience='user'")
        if self.audience == DISPATCH_AUDIENCE_ALL and self.target_user_id:
            raise ValueError("target_user_id is not allowed when audience='all'")
        return self


class ThreadReplyRequest(BaseModel):
    body: str = Field(min_length=1, max_length=4000)

    @field_validator("body")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


# ── Responses (wire shapes — the web client is built against these names) ──

class AdminDispatchOut(BaseModel):
    id: str
    mode: str
    audience: str
    target_user_id: Optional[str]
    target_email: Optional[str]
    sender_name: str
    title: str
    body: str
    urgent: bool
    status: str
    target_count: int
    # "Notified" — at least one surface landed on a target that did not fail.
    delivered_count: int
    # The two halves a single "delivered" used to hide: a recipient with no
    # running agent container gets the notification and the thread row but NO
    # chat card, and reporting that as delivered told the operator a broadcast
    # reached everyone when hundreds only saw a banner.
    chat_delivered_count: int
    no_agent_count: int
    read_count: int
    failed_count: int
    created_at: Optional[str]
    completed_at: Optional[str]
    created_by_email: Optional[str]


class AdminDispatchTargetOut(BaseModel):
    id: str
    user_id: str
    email: Optional[str]
    name: Optional[str]
    state: str
    chat_status: str
    chat_message_id: Optional[str]
    notification_id: Optional[str]
    read_at: Optional[str]
    attempts: int
    last_error: Optional[str]


class AdminThreadMessageOut(BaseModel):
    """One row of an Admin thread. Shared verbatim with the user-facing
    surface — ``app/api/notices.py`` imports this model rather than declaring a
    second copy that could drift from it."""
    id: str
    direction: str
    body: str
    created_at: Optional[str]
    dispatch_id: Optional[str]
    read_at: Optional[str]
    # Who the operator is, on this row. The chat card renders the dispatch's
    # `sender_name`; without this the thread had to invent one (mobile
    # hardcoded "Toup", web showed none) and the same operator appeared to the
    # user as two different parties.
    sender_name: Optional[str]


class AdminThreadSummaryOut(BaseModel):
    user_id: str
    email: Optional[str]
    name: Optional[str]
    last_message_at: Optional[str]
    last_body: Optional[str]
    unread_in: int
    total: int


class ThreadUserOut(BaseModel):
    id: str
    email: Optional[str]
    name: Optional[str]


class DispatchPreviewOut(BaseModel):
    """What ``GET /admin/dispatch/preview`` promises the compose form."""
    audience: str
    recipient_count: int
    with_agent_count: int


def thread_message_out(m: AdminThreadMessage) -> AdminThreadMessageOut:
    return AdminThreadMessageOut(
        id=m.id,
        direction=m.direction,
        body=m.body,
        created_at=_utc_iso(m.created_at),
        dispatch_id=m.dispatch_id,
        read_at=_utc_iso(m.read_at),
        # One operator identity. An `out` row with no stored name (written
        # before the column existed, or by a path that forgot it) would render
        # nameless, which is the exact split this field closes — so it falls
        # back to the configured default rather than to nothing. An `in` row is
        # the user's own words and carries no operator name at all.
        sender_name=(
            (m.sender_name or settings.admin_dispatch_sender_name)
            if m.direction == THREAD_OUT
            else None
        ),
    )


def _dispatch_out(
    d: AdminDispatch,
    *,
    target_email: Optional[str],
    created_by_email: Optional[str],
    ledger: Optional[Dict[str, int]] = None,
) -> AdminDispatchOut:
    """Serialise a dispatch, preferring the TARGET LEDGER over the parent's
    cached counters wherever the caller could afford to read it.

    The columns on ``admin_dispatches`` are a cache with two writers — the
    fan-out's reconcile and the read-receipt route, on two replicas — so they
    trail. The per-target rows are the truth, and ``read_count`` in particular
    is only ever correct there. ``ledger`` is a
    ``admin_dispatch_worker.summarize_targets`` dict (or the list route's
    single-query subset of it); missing keys fall back to the stored value,
    which is right for a dispatch that has no targets yet.
    """
    ledger = ledger or {}
    return AdminDispatchOut(
        id=d.id,
        mode=d.mode,
        audience=d.audience,
        target_user_id=d.target_user_id,
        target_email=target_email,
        sender_name=d.sender_name,
        title=d.title,
        body=d.body,
        urgent=bool(d.urgent),
        status=d.status,
        target_count=int(ledger.get("target_count", d.target_count or 0)),
        delivered_count=int(ledger.get("delivered_count", d.delivered_count or 0)),
        chat_delivered_count=int(ledger.get("chat_delivered_count", 0)),
        no_agent_count=int(ledger.get("no_agent_count", 0)),
        read_count=int(ledger.get("read_count", d.read_count or 0)),
        failed_count=int(ledger.get("failed_count", d.failed_count or 0)),
        created_at=_utc_iso(d.created_at),
        completed_at=_utc_iso(d.completed_at),
        created_by_email=created_by_email,
    )


async def _ledger_subset(db: AsyncSession, dispatch_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """``{dispatch_id: {chat_delivered_count, no_agent_count, read_count}}`` in
    ONE query, for the list view.

    ``summarize_targets`` is the definition of record and the detail route uses
    it directly, but it is seven scalar reads for ONE dispatch — 1400
    round-trips across a 200-row list. Only the three trivially-stable
    predicates are restated here; ``delivered_count`` and ``failed_count``
    deliberately are NOT (delivered is not a ``chat_status`` test) and keep
    coming from the reconciled columns.
    """
    ids = [i for i in dispatch_ids if i]
    if not ids:
        return {}

    def _sum(cond):
        return func.coalesce(func.sum(case((cond, 1), else_=0)), 0)

    rows = (await db.execute(
        select(
            AdminDispatchTarget.dispatch_id,
            # `retracted` counts as landed — it is what a `once` card BECOMES
            # once the user reads it, so excluding it would make "In chat" fall
            # as the dispatch succeeds. Mirrors summarize_targets.
            _sum(AdminDispatchTarget.chat_status.in_([CHAT_DELIVERED, CHAT_RETRACTED])),
            _sum(AdminDispatchTarget.chat_status == CHAT_NO_AGENT),
            _sum(AdminDispatchTarget.read_at.isnot(None)),
        )
        .where(AdminDispatchTarget.dispatch_id.in_(ids))
        .group_by(AdminDispatchTarget.dispatch_id)
    )).all()
    return {
        r[0]: {
            "chat_delivered_count": int(r[1] or 0),
            "no_agent_count": int(r[2] or 0),
            "read_count": int(r[3] or 0),
        }
        for r in rows
    }


async def _emails_for(db: AsyncSession, user_ids: set) -> dict:
    """{user_id: (email, name)} for the ids that still exist."""
    ids = [i for i in user_ids if i]
    if not ids:
        return {}
    rows = (await db.execute(
        select(User.id, User.email, User.name).where(User.id.in_(ids))
    )).all()
    return {r.id: (r.email, r.name) for r in rows}


# ── Compose + send ───────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_dispatch(
    body: DispatchCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Queue a dispatch and hand it to the fan-out worker."""
    _require_enabled()

    if body.audience == DISPATCH_AUDIENCE_ALL and body.confirm != BROADCAST_CONFIRM:
        raise HTTPException(
            status_code=400,
            detail=f"A broadcast is unrecallable — send confirm='{BROADCAST_CONFIRM}' to proceed.",
        )

    target_email: Optional[str] = None
    if body.audience == DISPATCH_AUDIENCE_USER:
        target = (await db.execute(
            select(User).where(User.id == body.target_user_id)
        )).scalar_one_or_none()
        if not target:
            raise HTTPException(404, "Target user not found")
        target_email = target.email

    dispatch = AdminDispatch(
        # Explicit — the column default only fires at flush, and the response is
        # built before the commit.
        id=str(uuid.uuid4()),
        created_by_user_id=admin.id,
        mode=body.mode,
        audience=body.audience,
        target_user_id=body.target_user_id if body.audience == DISPATCH_AUDIENCE_USER else None,
        sender_name=body.sender_name or settings.admin_dispatch_sender_name,
        title=body.title,
        body=body.body,
        urgent=bool(body.urgent),
        status=DISPATCH_QUEUED,
        created_at=datetime.utcnow(),
    )
    db.add(dispatch)

    # Build the response BEFORE commit — pgbouncer txn-mode rule. No `ledger`:
    # the worker has not created a target row yet, so the zeroed columns are
    # the honest answer.
    resp = {"dispatch": _dispatch_out(
        dispatch, target_email=target_email, created_by_email=admin.email,
    )}
    await db.commit()

    # AFTER the commit: the worker opens its own session and would not see an
    # uncommitted row. The await only creates the task — the fan-out itself
    # (N agent hops) does not run on this request.
    await spawn_dispatch_fanout(dispatch.id)
    logger.info("[admin-dispatch] %s queued: mode=%s audience=%s by %s",
                dispatch.id[:8], dispatch.mode, dispatch.audience, admin.id[:8])
    return resp


@router.get("/preview", response_model=DispatchPreviewOut)
async def preview_recipients(
    audience: str = Query(default=DISPATCH_AUDIENCE_ALL, pattern="^(user|all)$"),
    target_user_id: Optional[str] = Query(default=None, max_length=36),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> DispatchPreviewOut:
    """How many accounts this audience actually resolves to, right now.

    Declared ABOVE ``/{dispatch_id}`` so the literal path wins the match (same
    ordering rule as ``/threads``).

    The count comes from the worker's OWN enumeration (``count_recipients`` →
    ``_enumerate_recipients``), never from ``/admin/users`` or a parallel
    ``COUNT(*)``: the number an admin reads before pressing send on an
    unrecallable broadcast has to be the number the fan-out will walk. The
    previous compose form derived it from a separately-fetched user list, so a
    failed fetch rendered "Send to 0 accounts" on a button that then broadcast
    to everyone.

    Deliberately NOT behind ``_require_enabled``: it produces nothing, and the
    kill switch is for stopping sends, not for hiding the blast radius.
    """
    if audience == DISPATCH_AUDIENCE_USER and not (target_user_id or "").strip():
        raise HTTPException(422, "target_user_id is required when audience='user'")

    recipient_count, with_agent_count = await count_recipients(
        db, preview_spec(audience, target_user_id),
    )
    return DispatchPreviewOut(
        audience=audience,
        recipient_count=int(recipient_count),
        with_agent_count=int(with_agent_count),
    )


# ── Threads ──────────────────────────────────────────────────────────
# Declared ABOVE /{dispatch_id} so the literal path wins the match (same
# ordering rule as support.py's /corpus).

@router.get("/threads")
async def list_threads(
    limit: int = Query(default=200, ge=1, le=500),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every user with an Admin thread, newest activity first."""
    agg_rows = (await db.execute(
        select(
            AdminThreadMessage.user_id,
            func.count().label("total"),
            func.max(AdminThreadMessage.created_at).label("last_at"),
            func.coalesce(func.sum(case(
                (
                    (AdminThreadMessage.direction == THREAD_IN)
                    & (AdminThreadMessage.admin_read_at.is_(None)),
                    1,
                ),
                else_=0,
            )), 0).label("unread_in"),
        )
        .group_by(AdminThreadMessage.user_id)
        .order_by(func.max(AdminThreadMessage.created_at).desc())
        .limit(limit)
    )).all()

    if not agg_rows:
        return {"threads": []}

    # Last body per user in ONE query — a persistent broadcast gives every user
    # a thread, so a per-user follow-up read is an N+1 over the whole user base.
    latest = (
        select(
            AdminThreadMessage.user_id.label("user_id"),
            func.max(AdminThreadMessage.created_at).label("last_at"),
        )
        .group_by(AdminThreadMessage.user_id)
        .subquery()
    )
    body_rows = (await db.execute(
        select(AdminThreadMessage.user_id, AdminThreadMessage.body)
        .join(
            latest,
            (AdminThreadMessage.user_id == latest.c.user_id)
            & (AdminThreadMessage.created_at == latest.c.last_at),
        )
    )).all()
    last_body = {r.user_id: r.body for r in body_rows}

    users = await _emails_for(db, {r.user_id for r in agg_rows})
    return {"threads": [
        AdminThreadSummaryOut(
            user_id=r.user_id,
            email=users.get(r.user_id, (None, None))[0],
            name=users.get(r.user_id, (None, None))[1],
            last_message_at=_utc_iso(r.last_at),
            last_body=last_body.get(r.user_id),
            unread_in=int(r.unread_in or 0),
            total=int(r.total or 0),
        )
        for r in agg_rows
    ]}


@router.get("/threads/{user_id}")
async def get_thread(
    user_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """One user's Admin thread, oldest→newest. Marks the user's replies read."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    rows = (await db.execute(
        select(AdminThreadMessage)
        .where(AdminThreadMessage.user_id == user_id)
        .order_by(AdminThreadMessage.created_at.desc())
        .limit(limit)
    )).scalars().all()

    await db.execute(
        sa_update(AdminThreadMessage)
        .where(
            AdminThreadMessage.user_id == user_id,
            AdminThreadMessage.direction == THREAD_IN,
            AdminThreadMessage.admin_read_at.is_(None),
        )
        .values(admin_read_at=datetime.utcnow())
    )

    # Build the response BEFORE commit — pgbouncer txn-mode rule.
    resp = {
        "user": ThreadUserOut(id=user.id, email=user.email, name=user.name),
        "messages": [thread_message_out(m) for m in reversed(rows)],
    }
    await db.commit()
    return resp


@router.post("/threads/{user_id}", status_code=201)
async def reply_in_thread(
    user_id: str,
    body: ThreadReplyRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Write an admin→user row into the thread and alert the user.

    No agent hop: the thread is the operator's own channel on the platform
    (D2), and the agent must never see it. But a row on its own reaches NOBODY
    — neither client polls, so the mobile badge moves only if the user happens
    to open the drawer and the web badge does not move for the life of the tab,
    while both composers promise the answer gets through. So the reply rides
    the same `announcement` notification lane the dispatch itself does.

    ORDER: the row is committed FIRST. A notification announcing a message a
    rollback erased is a tap into an empty thread; the reverse leaves the reply
    readable in the thread, which is where the user was being sent anyway.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    now = datetime.utcnow()
    # One operator identity across both surfaces: the chat card renders a
    # dispatch's `sender_name`, so a follow-up in the thread must not arrive
    # unsigned or under a second name.
    sender_name = settings.admin_dispatch_sender_name
    msg = AdminThreadMessage(
        # id AND created_at explicit: both column defaults fire at flush, and
        # the response is built before the commit.
        id=str(uuid.uuid4()),
        user_id=user_id,
        dispatch_id=None,
        direction=THREAD_OUT,
        body=body.body,
        author_admin_id=admin.id,
        sender_name=sender_name,
        created_at=now,
    )
    db.add(msg)

    resp = {"message": thread_message_out(msg)}
    await db.commit()

    try:
        await build_reply_notification(
            db,
            user_id=user_id,
            message_id=msg.id,
            body=msg.body,
            sender_name=sender_name,
            now=now,
        )
    except Exception:
        # Re-raised, never swallowed: the operator has to learn their reply was
        # not announced. This log line is what tells the on-call that the reply
        # itself was NOT lost — it is committed and readable in the thread.
        logger.exception(
            "[admin-dispatch] reply %s to %s saved but not announced",
            msg.id[:8], user_id[:8],
        )
        raise

    return resp


# ── Read ─────────────────────────────────────────────────────────────

@router.get("")
async def list_dispatches(
    limit: int = Query(default=50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(AdminDispatch).order_by(AdminDispatch.created_at.desc()).limit(limit)
    )).scalars().all()

    users = await _emails_for(
        db,
        {d.target_user_id for d in rows} | {d.created_by_user_id for d in rows},
    )
    ledgers = await _ledger_subset(db, [d.id for d in rows])
    return {"dispatches": [
        _dispatch_out(
            d,
            target_email=users.get(d.target_user_id, (None, None))[0],
            created_by_email=users.get(d.created_by_user_id, (None, None))[0],
            ledger=ledgers.get(d.id),
        )
        for d in rows
    ]}


@router.get("/{dispatch_id}")
async def get_dispatch(
    dispatch_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """One dispatch plus its per-recipient delivery detail.

    Every count here is RECOMPUTED from the target ledger rather than read off
    the parent row: `read_count` is incremented by a different request on
    either replica, and the split of "delivered" into notified / in-chat /
    agent-down only exists per target.
    """
    dispatch = await db.get(AdminDispatch, dispatch_id)
    if not dispatch:
        raise HTTPException(404, "Dispatch not found")

    targets = (await db.execute(
        select(AdminDispatchTarget)
        .where(AdminDispatchTarget.dispatch_id == dispatch_id)
        .order_by(AdminDispatchTarget.created_at.asc())
    )).scalars().all()

    ledger = await summarize_targets(db, dispatch_id)
    users = await _emails_for(
        db,
        {t.user_id for t in targets}
        | {dispatch.target_user_id, dispatch.created_by_user_id},
    )
    return {
        "dispatch": _dispatch_out(
            dispatch,
            target_email=users.get(dispatch.target_user_id, (None, None))[0],
            created_by_email=users.get(dispatch.created_by_user_id, (None, None))[0],
            ledger=ledger,
        ),
        "targets": [
            AdminDispatchTargetOut(
                id=t.id,
                user_id=t.user_id,
                email=users.get(t.user_id, (None, None))[0],
                name=users.get(t.user_id, (None, None))[1],
                state=t.state,
                chat_status=t.chat_status,
                chat_message_id=t.chat_message_id,
                notification_id=t.notification_id,
                read_at=_utc_iso(t.read_at),
                attempts=int(t.attempts or 0),
                last_error=t.last_error,
            )
            for t in targets
        ],
    }


@router.post("/{dispatch_id}/retry")
async def retry_dispatch(
    dispatch_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Re-queue everything this dispatch has not finished, and re-spawn it.

    THREE classes are reset, not one — the button says it re-queues every
    unfinished target and it now does:

    1. `failed` — the obvious one.
    2. `done` + `chat_status='no_agent'` — the recipient had no live agent
       container when the fan-out reached them. They got the banner and (for a
       persistent dispatch) the thread row, but no chat card, and once the
       tenant recovers nothing was ever going to go back for it.
    3. `sending` older than `_STUCK_CLAIM_MAX_AGE` — a CAS claim whose worker
       died. Left alone it pins the dispatch at `sending` forever, because
       `_reconcile` counts it as unfinished.

    All three are safe for the same reason, spelled out at
    `_STUCK_CLAIM_MAX_AGE`: every write on the send path is keyed on something
    derived from (dispatch, user), so a second pass can only finish work.

    The worker is re-spawned unconditionally, not only when something was
    reset: a dispatch whose worker died mid-flight still has `pending` targets
    nobody is walking.
    """
    _require_enabled()
    dispatch = await db.get(AdminDispatch, dispatch_id)
    if not dispatch:
        raise HTTPException(404, "Dispatch not found")

    now = datetime.utcnow()
    res = await db.execute(
        sa_update(AdminDispatchTarget)
        .where(
            AdminDispatchTarget.dispatch_id == dispatch_id,
            or_(
                AdminDispatchTarget.state == TARGET_FAILED,
                and_(
                    AdminDispatchTarget.state == TARGET_DONE,
                    AdminDispatchTarget.chat_status == CHAT_NO_AGENT,
                ),
                and_(
                    AdminDispatchTarget.state == TARGET_SENDING,
                    AdminDispatchTarget.updated_at < now - _STUCK_CLAIM_MAX_AGE,
                ),
            ),
        )
        # `chat_status` is deliberately left as it stands: it is the last known
        # fact about the chat surface until the worker replaces it, and the
        # panel would otherwise show every reclaimed target as `pending` with
        # no record of why it was reclaimed.
        .values(state=TARGET_PENDING, last_error=None, updated_at=now)
    )
    retried = int(res.rowcount or 0)

    # The worker recomputes all four counters and the status when it drains
    # (_reconcile); these keep the panel honest for the seconds in between.
    # `failed_count` is set to 0 rather than decremented by `retried`: the
    # UPDATE above reset EVERY failed target of this dispatch, and `retried`
    # spans three classes so subtracting it would undercount.
    dispatch.failed_count = 0
    dispatch.status = DISPATCH_QUEUED
    dispatch.completed_at = None

    users = await _emails_for(db, {dispatch.target_user_id, dispatch.created_by_user_id})
    ledger = await summarize_targets(db, dispatch_id)
    resp = {
        "dispatch": _dispatch_out(
            dispatch,
            target_email=users.get(dispatch.target_user_id, (None, None))[0],
            created_by_email=users.get(dispatch.created_by_user_id, (None, None))[0],
            ledger=ledger,
        ),
        "retried": retried,
    }
    await db.commit()

    await spawn_dispatch_fanout(dispatch_id)
    return resp
