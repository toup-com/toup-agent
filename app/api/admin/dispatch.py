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
  GET  /admin/dispatch/threads/{user_id}  — one thread (read-only)
  POST /admin/dispatch/threads/{user_id}  — reply into a thread
  POST /admin/dispatch/threads/{id}/read  — mark the user's replies admin-read

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
from app.api.tenant_proxy import agent_proxy_info, proxy_to_agent
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
    PE_DISPATCH_CREATED,
    PE_ENTITY_DISPATCH,
    TARGET_DONE,
    TARGET_FAILED,
    TARGET_PENDING,
    TARGET_SENDING,
    THREAD_IN,
    THREAD_OUT,
)
from app.services import dispatch_tones
from app.services.admin_dispatch_worker import (
    build_reply_notification,
    count_recipients,
    preview_spec,
    spawn_dispatch_fanout,
    summarize_targets,
)
from app.services.product_events import emit_product_event

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


def broadcast_allowed() -> bool:
    """G-DISPATCH-BROADCAST. Default OFF — see ``config.py``."""
    return bool(getattr(settings, "dispatch_broadcast_enabled", False))


def _require_broadcast_allowed() -> None:
    """Refuse ``audience='all'`` unless the flag is deliberately on.

    Deliberately raised BEFORE the confirmation-word check, and deliberately
    without naming that word. The two messages are read by an operator who has
    just been stopped, and "send confirm='BROADCAST' to proceed" reads as an
    instruction for getting past the thing that stopped them — which is exactly
    wrong when the answer is that this deployment may not broadcast at all.
    """
    if not broadcast_allowed():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Broadcast to every account is disabled on this deployment. "
                "Send to a single user, or see docs/DISPATCH_BROADCAST.md."
            ),
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
    # A catalogue id from `app/services/dispatch_tones.py`. Omitted = the
    # system default.
    tone: Optional[str] = Field(default=None, max_length=32)

    @field_validator("tone")
    @classmethod
    def _tone_in_catalogue(cls, v: Optional[str]) -> Optional[str]:
        """422 rather than a silent fallback to the default.

        The read path collapses an unknown id (`dispatch_tones.normalize`),
        which is right for a row whose tone was later retired — but on the
        WRITE path a quiet collapse means an operator picks Knock, sees it
        accepted, and the recipient gets the default. Every failure in this
        area is already inaudible; this one at least can be loud.
        """
        v = (v or "").strip() or None
        if not dispatch_tones.is_valid(v):
            raise ValueError(f"unknown tone (expected one of: {', '.join(sorted(dispatch_tones.TONE_IDS))})")
        return v

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
    # Always a live catalogue id — never NULL, never a retired one (see
    # `_dispatch_out`), so the panel can render it without a fallback.
    tone: str
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
    # Why the dispatch-level status is `failed`. The per-target `last_error`
    # cannot answer it for the commonest failure — a worker that died before
    # materialising a single target row.
    last_error: Optional[str]
    created_at: Optional[str]
    completed_at: Optional[str]
    created_by_email: Optional[str]
    # Recall audit (migration 091). `revoked_at` set is what makes the panel
    # render this dispatch as recalled; the email is resolved like the
    # creator's so the record names a person, not a uuid.
    revoked_at: Optional[str] = None
    revoked_by_email: Optional[str] = None


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
    # Deletion state (migration 092). Both null on a live message. Optional so
    # a replica predating the migration still validates.
    hidden_from_user_at: Optional[str] = None
    deleted_at: Optional[str] = None


class AdminThreadSummaryOut(BaseModel):
    user_id: str
    email: Optional[str]
    name: Optional[str]
    last_message_at: Optional[str]
    last_body: Optional[str]
    unread_in: int
    total: int
    # Who spoke last. The sidebar prefixes the operator's own words with "You: "
    # — without this the list reads as if the user said everything, which is the
    # opposite of the one thing an operator scans this column for: who is
    # waiting on whom.
    last_direction: Optional[str] = None


class ThreadUserOut(BaseModel):
    id: str
    email: Optional[str]
    name: Optional[str]


class DispatchPreviewOut(BaseModel):
    """What ``GET /admin/dispatch/preview`` promises the compose form."""
    audience: str
    recipient_count: int
    with_agent_count: int
    # G-DISPATCH-BROADCAST, reported rather than assumed. The compose form must
    # not carry its own copy of this: a client-side constant says "disabled"
    # while the API happily accepts the send, and the first person to find out
    # is every user on the platform.
    broadcast_enabled: bool = False


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
        # The OPERATOR's view keeps deleted rows and marks them. Hiding them
        # here would mean an operator who deletes a message loses the record
        # that they sent it — which is the moment that record matters most.
        # The user's view filters them out entirely (`notices.py`).
        hidden_from_user_at=_utc_iso(getattr(m, "hidden_from_user_at", None)),
        deleted_at=_utc_iso(getattr(m, "deleted_at", None)),
    )


def _dispatch_out(
    d: AdminDispatch,
    *,
    target_email: Optional[str],
    created_by_email: Optional[str],
    ledger: Optional[Dict[str, int]] = None,
    # Defaulted, unlike the two above: only the routes that already resolve
    # emails pass it, and a recall is rare enough that the list route should
    # not grow a third id into its `_emails_for` batch for every row.
    revoked_by_email: Optional[str] = None,
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
        # NULL is every row written before 088; a retired id is one written
        # before the catalogue was trimmed. Both mean "the system default"
        # to the recipient, so both say so here rather than making each
        # reader re-derive it.
        tone=dispatch_tones.normalize(d.tone),
        status=d.status,
        target_count=int(ledger.get("target_count", d.target_count or 0)),
        delivered_count=int(ledger.get("delivered_count", d.delivered_count or 0)),
        chat_delivered_count=int(ledger.get("chat_delivered_count", 0)),
        no_agent_count=int(ledger.get("no_agent_count", 0)),
        read_count=int(ledger.get("read_count", d.read_count or 0)),
        failed_count=int(ledger.get("failed_count", d.failed_count or 0)),
        last_error=d.last_error,
        created_at=_utc_iso(d.created_at),
        completed_at=_utc_iso(d.completed_at),
        created_by_email=created_by_email,
        revoked_at=_utc_iso(getattr(d, "revoked_at", None)),
        revoked_by_email=revoked_by_email,
    )


async def _ledger_subset(db: AsyncSession, dispatch_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """Every count ``summarize_targets`` defines, off the target ledger, in ONE
    query — for the list view.

    ``summarize_targets`` is the definition of record and the detail route uses
    it directly, but it is seven scalar reads for ONE dispatch — 1400
    round-trips across a 200-row list. So the predicates are restated here, and
    they mirror it EXACTLY (a FAILED target is never also delivered; a
    `retracted` card still landed).

    Every counter or none: this used to recompute only the three
    ``chat_status``-shaped ones and leave ``target_count`` /
    ``delivered_count`` / ``failed_count`` on the parent's cached columns,
    which only the fan-out's ``_reconcile`` writes. One row then carried two
    vintages — a live ``read_count`` beside a stale ``0/0`` for a dispatch that
    had already delivered — and the operator reads ``0/0`` as "nothing was
    sent".
    """
    ids = [i for i in dispatch_ids if i]
    if not ids:
        return {}

    def _sum(cond):
        return func.coalesce(func.sum(case((cond, 1), else_=0)), 0)

    rows = (await db.execute(
        select(
            AdminDispatchTarget.dispatch_id,
            func.count(),
            # NOT a plain `chat_status` test: `notification_id` is non-null
            # whether or not the agent hop after it succeeded, so without the
            # state gate a broadcast that reached hundreds of banners and no
            # chats reports itself fully delivered.
            _sum(and_(
                AdminDispatchTarget.state != TARGET_FAILED,
                or_(
                    AdminDispatchTarget.chat_status == CHAT_DELIVERED,
                    AdminDispatchTarget.notification_id.isnot(None),
                ),
            )),
            # `retracted` counts as landed — it is what a `once` card BECOMES
            # once the user reads it, so excluding it would make "In chat" fall
            # as the dispatch succeeds. Mirrors summarize_targets.
            _sum(AdminDispatchTarget.chat_status.in_([CHAT_DELIVERED, CHAT_RETRACTED])),
            _sum(AdminDispatchTarget.chat_status == CHAT_NO_AGENT),
            _sum(AdminDispatchTarget.read_at.isnot(None)),
            _sum(AdminDispatchTarget.state == TARGET_FAILED),
        )
        .where(AdminDispatchTarget.dispatch_id.in_(ids))
        .group_by(AdminDispatchTarget.dispatch_id)
    )).all()
    return {
        r[0]: {
            "target_count": int(r[1] or 0),
            "delivered_count": int(r[2] or 0),
            "chat_delivered_count": int(r[3] or 0),
            "no_agent_count": int(r[4] or 0),
            "read_count": int(r[5] or 0),
            "failed_count": int(r[6] or 0),
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

    if body.audience == DISPATCH_AUDIENCE_ALL:
        # Flag first, word second. The flag is the rail; the word is only a
        # deliberation prompt for a deployment that is already allowed to
        # broadcast. Reversing these two leaks the word to a caller who may
        # not use it either way.
        _require_broadcast_allowed()
        if body.confirm != BROADCAST_CONFIRM:
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
        tone=body.tone,
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

    # AFTER the commit (the fact is "a dispatch row exists", and before the
    # commit it might not) and BEFORE the spawn: `emit_product_event` awaits
    # IO, so a fan-out task created first could deliver — and record its
    # `dispatch_delivered` rows — while this one is still in flight, putting
    # the funnel's first step after its second.
    #
    # `user_id` is None for a broadcast: there is no single subject. The
    # operator is the ACTOR either way.
    await emit_product_event(
        PE_DISPATCH_CREATED,
        user_id=dispatch.target_user_id,
        actor_user_id=admin.id,
        entity_type=PE_ENTITY_DISPATCH,
        entity_id=dispatch.id,
        payload={
            "mode": dispatch.mode,
            "audience": dispatch.audience,
            "urgent": bool(dispatch.urgent),
        },
        dedupe_key=f"{PE_DISPATCH_CREATED}:{dispatch.id}",
        occurred_at=dispatch.created_at,
    )

    # AFTER the commit: the worker opens its own session and would not see an
    # uncommitted row. The await only creates the task — the fan-out itself
    # (N agent hops) does not run on this request.
    await spawn_dispatch_fanout(dispatch.id)
    logger.info("[admin-dispatch] %s queued: mode=%s audience=%s by %s",
                dispatch.id[:8], dispatch.mode, dispatch.audience, admin.id[:8])
    return resp


@router.get("/tones")
async def list_tones(_admin: User = Depends(require_admin)) -> Dict[str, object]:
    """The tone catalogue the picker renders.

    Served rather than duplicated in the panel so the list an operator can
    choose from and the list `create_dispatch` validates against are the same
    object. A second literal in TypeScript is how a panel comes to offer a
    tone the app does not bundle — and the symptom on device is the system
    default playing, which is indistinguishable from a correct default.

    Declared ABOVE ``/{dispatch_id}`` so the literal path wins the match (same
    ordering rule as ``/threads`` and ``/preview``). Not gated by
    ``_require_enabled``: it produces nothing, and a picker that 503s while
    the kill switch is on would make the composer unreadable for no gain.
    """
    return {"tones": dispatch_tones.catalogue(), "default": dispatch_tones.TONE_DEFAULT}


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
        broadcast_enabled=broadcast_allowed(),
    )


# ── Threads ──────────────────────────────────────────────────────────
# Declared ABOVE /{dispatch_id} so the literal path wins the match (same
# ordering rule as support.py's /corpus).

@router.get("/threads")
async def list_threads(
    limit: int = Query(default=200, ge=1, le=500),
    query: Optional[str] = Query(default=None, max_length=200),
    cursor: Optional[str] = Query(default=None, max_length=40),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every user with an Admin thread, newest activity first.

    ``query`` searches the person AND what was said — email, name, and message
    bodies — because an operator looking for a conversation remembers one or the
    other and cannot know which field will hit.

    ``cursor`` is the ``last_message_at`` of the last row you were given, and the
    page is strictly older than it. Keyset, not OFFSET: this is a GROUP BY over
    every message in the system, so an offset scan re-aggregates the whole table
    to skip rows it already computed.
    """
    agg = (
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
    )

    q = (query or "").strip()
    if q:
        like = f"%{q}%"
        # Two ways in, unioned as a user-id set. The body arm is a separate
        # subquery rather than a WHERE on the aggregate: filtering the rows being
        # grouped would make `total` count only the MATCHING messages, so a
        # search for one word would rewrite every conversation's length.
        by_body = select(AdminThreadMessage.user_id).where(
            AdminThreadMessage.body.ilike(like)
        )
        by_person = select(User.id).where(
            or_(User.email.ilike(like), User.name.ilike(like))
        )
        agg = agg.having(
            or_(
                AdminThreadMessage.user_id.in_(by_body),
                AdminThreadMessage.user_id.in_(by_person),
            )
        )

    if cursor:
        # A bad cursor is the client's bug, and silently serving page 1 forever
        # is an infinite scroll that never advances. Say so.
        try:
            cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(422, "cursor must be an ISO-8601 timestamp")
        if cursor_dt.tzinfo is not None:
            cursor_dt = cursor_dt.astimezone(timezone.utc).replace(tzinfo=None)
        agg = agg.having(func.max(AdminThreadMessage.created_at) < cursor_dt)

    agg_rows = (await db.execute(agg)).all()

    if not agg_rows:
        return {"threads": [], "next_cursor": None}

    # Last body per user in ONE query — a persistent broadcast gives every user
    # a thread, so a per-user follow-up read is an N+1 over the whole user base.
    # Scoped to the page's ids: unscoped it re-aggregated every message in the
    # system to decorate at most `limit` rows.
    page_ids = [r.user_id for r in agg_rows]
    latest = (
        select(
            AdminThreadMessage.user_id.label("user_id"),
            func.max(AdminThreadMessage.created_at).label("last_at"),
        )
        .where(AdminThreadMessage.user_id.in_(page_ids))
        .group_by(AdminThreadMessage.user_id)
        .subquery()
    )
    body_rows = (await db.execute(
        select(
            AdminThreadMessage.user_id,
            AdminThreadMessage.body,
            AdminThreadMessage.direction,
        )
        .join(
            latest,
            (AdminThreadMessage.user_id == latest.c.user_id)
            & (AdminThreadMessage.created_at == latest.c.last_at),
        )
    )).all()
    last_row = {r.user_id: (r.body, r.direction) for r in body_rows}

    users = await _emails_for(db, set(page_ids))
    threads = [
        AdminThreadSummaryOut(
            user_id=r.user_id,
            email=users.get(r.user_id, (None, None))[0],
            name=users.get(r.user_id, (None, None))[1],
            last_message_at=_utc_iso(r.last_at),
            last_body=last_row.get(r.user_id, (None, None))[0],
            last_direction=last_row.get(r.user_id, (None, None))[1],
            unread_in=int(r.unread_in or 0),
            total=int(r.total or 0),
        )
        for r in agg_rows
    ]
    # Only when the page was FULL. A short page is the end of the list, and
    # handing back a cursor there makes the client fetch one empty page per
    # scroll forever.
    next_cursor = threads[-1].last_message_at if len(agg_rows) >= limit else None
    return {"threads": threads, "next_cursor": next_cursor}


@router.get("/threads/{user_id}")
async def get_thread(
    user_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """One user's Admin thread, oldest→newest. READ-ONLY.

    This used to stamp `admin_read_at` on every unread reply as a side effect,
    which was survivable only while a human pressing a row was the sole caller.
    The console now polls the open thread every few seconds, so the side effect
    would clear the operator's reply queue while the window was blurred and
    nobody had read anything — and `admin_read_at` is the only record that a
    reply was ever seen, so there is nothing to recover it from.

    Marking read is `POST /threads/{user_id}/read`, which the client calls only
    when the thread is open AND the document has focus.
    """
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    rows = (await db.execute(
        select(AdminThreadMessage)
        .where(AdminThreadMessage.user_id == user_id)
        .order_by(AdminThreadMessage.created_at.desc())
        .limit(limit)
    )).scalars().all()

    return {
        "user": ThreadUserOut(id=user.id, email=user.email, name=user.name),
        "messages": [thread_message_out(m) for m in reversed(rows)],
        # The count the sidebar badge and the "New messages" divider both key
        # off. Derived here rather than recomputed client-side from `messages`,
        # which is one page and would under-report a long thread.
        "unread_in": sum(
            1 for m in rows
            if m.direction == THREAD_IN and m.admin_read_at is None
        ),
    }


@router.post("/threads/{user_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_thread_admin_read(
    user_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Record that an operator has actually looked at this user's replies.

    Split out of the GET above (see its docstring). Blind UPDATE with no
    prior SELECT: two admins opening the same thread on two replicas race, and
    stamping a timestamp onto rows that already carry one is a no-op by the
    `IS NULL` predicate rather than a conflict to resolve.
    """
    if not (await db.execute(
        select(User.id).where(User.id == user_id)
    )).scalar_one_or_none():
        raise HTTPException(404, "User not found")

    await db.execute(
        sa_update(AdminThreadMessage)
        .where(
            AdminThreadMessage.user_id == user_id,
            AdminThreadMessage.direction == THREAD_IN,
            AdminThreadMessage.admin_read_at.is_(None),
        )
        .values(admin_read_at=datetime.utcnow())
    )
    await db.commit()


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
    return {
        "dispatches": [
            _dispatch_out(
                d,
                target_email=users.get(d.target_user_id, (None, None))[0],
                created_by_email=users.get(d.created_by_user_id, (None, None))[0],
                ledger=ledgers.get(d.id),
            )
            for d in rows
        ],
        # Reported here as well as on /preview because the compose form loads
        # this on mount and /preview only when 'Everyone' is already selected.
        # An option that is going to be refused has to read as unavailable
        # BEFORE it is clicked, not after a round-trip.
        "broadcast_enabled": broadcast_allowed(),
    }


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
    # The re-spawned fan-out clears this too, but not before this response is
    # built — and a row that reads `Queued` under "the worker is gone" is the
    # panel contradicting itself at the exact moment the operator acted on it.
    dispatch.last_error = None

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


# The tombstone body. A `both` delete clears the text — the point is that the
# words are gone — but the ROW stays, so the thread does not silently lose a
# turn and the reply above it stops making sense.
DELETED_BODY = "This message was deleted."


@router.delete("/threads/{user_id}/messages/{message_id}", status_code=200)
async def delete_thread_message(
    user_id: str,
    message_id: str,
    scope: str = Query(default="theirs", pattern="^(theirs|both)$"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Take a thread message back — from THEM, or from both sides.

    Two scopes, because "delete" means two different things and an operator has
    to be able to say which:

      theirs  the message stops being visible to the person it was sent to.
              The operator's thread keeps it, marked. This is almost always
              what is wanted: unsend the words, keep the record.
      both    nobody sees the words. The body is replaced with a tombstone and
              the row survives, so the conversation does not lose a turn and
              whatever was said next still has something to follow.

    NEITHER hard-deletes. A thread is a conversation between two people and one
    of them cannot un-remember it, so pretending the row never existed is a
    story the database tells and reality does not. Keeping the row is also what
    preserves the operator's own audit trail — needed most precisely when a
    message was sent by mistake.

    An operator may delete EITHER direction. Removing a user's own reply is a
    real need (someone pastes a password into a support thread) and the
    asymmetry would be arbitrary.

    NOT gated on ``_require_enabled``: that switch stops new dispatches being
    composed. A kill switch that also prevented taking back what already went
    out would be exactly backwards.
    """
    row = (await db.execute(
        select(AdminThreadMessage).where(
            AdminThreadMessage.id == message_id,
            # Scoped to the user whose thread this is, not just the id: a
            # message id is a uuid, but the route reads as "this user's
            # thread" and a mismatch should 404 rather than silently reach
            # into someone else's conversation.
            AdminThreadMessage.user_id == user_id,
        )
    )).scalars().first()
    if row is None:
        raise HTTPException(404, "Message not found in this thread")

    now = datetime.utcnow()
    if scope == "both":
        # Idempotent: deleting an already-deleted message must not overwrite
        # the tombstone with a second one, nor re-stamp who did it.
        if row.deleted_at is None:
            row.deleted_at = now
            row.deleted_by_user_id = admin.id
            row.body = DELETED_BODY
    else:
        if row.hidden_from_user_at is None:
            row.hidden_from_user_at = now
            row.deleted_by_user_id = admin.id

    resp = {"message": thread_message_out(row), "scope": scope}
    await db.commit()
    logger.info("[admin-dispatch] thread message %s deleted (%s) by %s",
                message_id[:8], scope, admin.id[:8])
    return resp


@router.post("/{dispatch_id}/revoke")
async def revoke_dispatch(
    dispatch_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Recall the CARD from every recipient's chat. The notification stands.

    This is the honest half of "unrecallable". Two surfaces were always
    delivered and they are recallable to different degrees:

      • the in-chat card — a row in the tenant's DB and a live WS frame, both
        of which we own and can take back at any time, for either mode;
      • the notification — already on a lock screen. Nothing un-sends it, and
        this route does not pretend to.

    The compose form used to flatten both into "Cannot be recalled", which is
    the wrong half to believe: an operator who has just sent the wrong body to
    everyone assumes there is nothing to be done and does nothing, when the
    thing most people will actually re-read — the card sitting in their chat —
    could have been gone in seconds.

    Mechanically this is the SAME tenant hop a `once` notice already makes when
    its reader acks it (`notices.py::mark_notice_read` →
    `POST /internal/admin-notice/retract`), applied to every target and driven
    by the operator instead of the reader. No new delivery path — R4, and the
    v2 brief's "no second pipeline".

    Best-effort PER TARGET, and deliberately not transactional across them: a
    thousand-recipient broadcast must not lose 999 recalls because one tenant
    container is down. Failures are recorded on the target and the count comes
    back, so the operator learns "930 of 1000" rather than "done".
    """
    _require_enabled()
    dispatch = await db.get(AdminDispatch, dispatch_id)
    if not dispatch:
        raise HTTPException(404, "Dispatch not found")

    # Only the targets that HAVE a card. `no_agent` never got one written, and
    # `retracted` already gave theirs up — re-hopping either is a round trip
    # per recipient for a row we know is not there.
    rows = (await db.execute(
        select(AdminDispatchTarget.id, AdminDispatchTarget.user_id)
        .where(
            AdminDispatchTarget.dispatch_id == dispatch_id,
            AdminDispatchTarget.chat_status == CHAT_DELIVERED,
        )
    )).all()

    # Stamp the audit BEFORE the hops. They take a round trip each and the
    # operator's intent is a fact the moment they press the button — a crash
    # halfway through must leave "this was recalled at 02:41 by X", not a
    # dispatch that looks untouched while half its cards are gone.
    now = datetime.utcnow()
    dispatch.revoked_at = now
    dispatch.revoked_by_user_id = _admin.id
    await db.commit()

    revoked = 0
    failed: List[str] = []
    for target_id, user_id in rows:
        info = await agent_proxy_info(user_id, db)
        if info is None:
            # No reachable agent. The card is still in their tenant DB, so
            # this is NOT a recall — record it and let a later press retry.
            failed.append(user_id)
            await db.execute(
                sa_update(AdminDispatchTarget)
                .where(AdminDispatchTarget.id == target_id)
                .values(last_error="revoke: agent unreachable", updated_at=now)
            )
            continue
        agent_url, agent_api_key = info
        try:
            await proxy_to_agent(
                agent_url, agent_api_key, "internal/admin-notice/retract", "POST",
                json_body={"user_id": user_id, "dispatch_id": dispatch_id},
            )
        except Exception as e:  # noqa: BLE001 — recorded on the target
            failed.append(user_id)
            await db.execute(
                sa_update(AdminDispatchTarget)
                .where(AdminDispatchTarget.id == target_id)
                .values(last_error=f"revoke failed: {type(e).__name__}: {e}"[:500],
                        updated_at=now)
            )
            continue
        revoked += 1
        await db.execute(
            sa_update(AdminDispatchTarget)
            .where(AdminDispatchTarget.id == target_id)
            .values(chat_status=CHAT_RETRACTED, last_error=None, updated_at=now)
        )

    await db.commit()
    logger.info("[admin-dispatch] %s revoked by %s: %d/%d cards, %d failed",
                dispatch_id[:8], _admin.id[:8], revoked, len(rows), len(failed))

    users = await _emails_for(db, {
        dispatch.target_user_id, dispatch.created_by_user_id, dispatch.revoked_by_user_id,
    })
    ledger = await summarize_targets(db, dispatch_id)
    return {
        "dispatch": _dispatch_out(
            dispatch,
            target_email=users.get(dispatch.target_user_id, (None, None))[0],
            created_by_email=users.get(dispatch.created_by_user_id, (None, None))[0],
            ledger=ledger,
            revoked_by_email=users.get(dispatch.revoked_by_user_id, (None, None))[0],
        ),
        "revoked": revoked,
        "eligible": len(rows),
        "failed": len(failed),
    }
