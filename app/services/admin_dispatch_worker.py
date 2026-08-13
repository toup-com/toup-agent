"""Admin dispatch fan-out — one operator message → N recipients.

Runs OFF the request path: a broadcast enumerates every user and does
three writes per user (one of them an HTTP hop into that user's tenant
agent), which is not a request budget. The two entry points mirror
``app/support/pipeline.py``: ``spawn_dispatch_fanout`` is
fire-and-forget with a module-level strong ref so the task is not GC'd
mid-flight, ``run_dispatch_fanout`` is the awaitable the retry route
and the tests drive directly.

REPLICA SAFETY — the load-bearing constraint: platform-api runs 2
Railway replicas with no leader election, so both can spawn a fan-out
for the same dispatch (and the retry route can re-enter one that is
still running). Every target is therefore claimed with a per-row
status CAS::

    UPDATE admin_dispatch_targets SET state='sending'
    WHERE id=:id AND state='pending'

and the loser sees rowcount=0 and skips. Same primitive as
``notification_dispatcher._cas_status``, and portable to the sqlite
test harness (no FOR UPDATE SKIP LOCKED — no precedent in this repo
and sqlite lacks it).

Two more entry points exist for the admin routes, and both are here
rather than there so there is exactly one definition of each:
``count_recipients`` answers the pre-send preview with the fan-out's OWN
audience query (a preview that can disagree with the send is how a
"Send to 0 accounts" button broadcast to everyone), and
``build_reply_notification`` enqueues the alert for an admin's thread
reply on this same ``announcement`` lane.

IDEMPOTENT BY CONSTRUCTION: every write is keyed on something derived
from (dispatch_id, user_id) — the thread row's uuid5 id, the queue
row's ``idempotency_key``, and the agent's own uuid5 message id
(``app/api/admin_notice.py``) — so re-running a half-finished dispatch
never double-sends. The parent's counters are RECOMPUTED from the
target rows rather than incremented, for the same reason.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadMessage,
    AgentConfig,
    NotificationQueue,
    User,
    CHAT_DELIVERED,
    CHAT_FAILED,
    CHAT_NO_AGENT,
    CHAT_PENDING,
    CHAT_RETRACTED,
    DISPATCH_AUDIENCE_ALL,
    DISPATCH_FAILED,
    DISPATCH_MODE_PERSISTENT,
    DISPATCH_SENDING,
    DISPATCH_SENT,
    NOTIFY_KIND_ANNOUNCEMENT,
    NQ_PRIORITY_DEFAULT,
    NQ_PRIORITY_HIGH,
    TARGET_DONE,
    TARGET_FAILED,
    TARGET_PENDING,
    TARGET_SENDING,
    THREAD_OUT,
)

logger = logging.getLogger(__name__)

_background_tasks: set = set()

# Drip (contract D8): a broadcast must never occupy more than a slice
# of the dispatcher's 20-row/30s claim batch, or reminders queue behind
# an announcement.
ADMIN_DISPATCH_BURST = 10
_DRIP_GAP_SEC = 30

# The agent hop writes one row and returns; it is not a turn. Well
# under the shared client's 30s default so a wedged tenant can't hold
# the whole fan-out.
_AGENT_HOP_TIMEOUT_S = 15.0


class _Recipient(NamedTuple):
    user_id: str
    agent_url: Optional[str]
    agent_api_key: Optional[str]

    @property
    def has_agent(self) -> bool:
        """The one definition of "this recipient can receive a chat card".

        Both the delivery path and the pre-send preview read it, so the
        `with_agent_count` an admin sees before an unrecallable broadcast is
        counted with the SAME predicate that decides `chat_status='no_agent'`
        an instant later. A join that matched but produced a row with a NULL
        key is not an agent we can hop to.
        """
        return bool(self.agent_url and self.agent_api_key)


class _DispatchSpec(NamedTuple):
    """An immutable snapshot of the dispatch row, taken once at the top.

    The fan-out commits after every target, and ``expire_on_commit`` is on
    for this project's sessionmaker — so a live ``AdminDispatch`` ORM object
    is EXPIRED from the first commit onward, and every later read of
    ``.mode`` / ``.title`` / ``.audience`` is an implicit lazy SELECT. In
    asyncio that refresh is IO outside the awaited call that established the
    greenlet context: it happened to survive a single sequential fan-out and
    raised ``MissingGreenlet`` the moment two ran concurrently — which is the
    production shape, because platform-api runs 2 replicas with no leader
    election and the retry route can re-enter a fan-out that is still running.

    Snapshotting also makes the values STABLE: every target of one broadcast
    gets byte-identical copy, notification payload and deep link, which is
    not something a re-read per target could promise.
    """
    id: str
    mode: str
    audience: str
    target_user_id: Optional[str]
    sender_name: str
    title: str
    body: str
    urgent: bool
    created_by_user_id: Optional[str]
    created_at: datetime


def _snapshot(dispatch: AdminDispatch) -> _DispatchSpec:
    """Read every field the fan-out needs while the row is still loaded."""
    return _DispatchSpec(
        id=dispatch.id,
        mode=dispatch.mode,
        audience=dispatch.audience,
        target_user_id=dispatch.target_user_id,
        sender_name=dispatch.sender_name or settings.admin_dispatch_sender_name,
        title=dispatch.title or "",
        body=dispatch.body or "",
        urgent=bool(dispatch.urgent),
        created_by_user_id=dispatch.created_by_user_id,
        created_at=dispatch.created_at or datetime.utcnow(),
    )


def preview_spec(audience: str, target_user_id: Optional[str] = None) -> _DispatchSpec:
    """A spec carrying only what `count_recipients` reads.

    The pre-send preview (DEC-4) runs BEFORE any dispatch row exists, so
    there is nothing to `_snapshot`. The rest of the fields are placeholders
    and must stay unread by the enumeration path — passing this anywhere near
    delivery would send an empty notice. Exists so the preview route cannot be
    tempted into writing its own recipient query, which is the entire defect.
    """
    return _DispatchSpec(
        id="", mode="", audience=audience, target_user_id=target_user_id,
        sender_name="", title="", body="", urgent=False,
        created_by_user_id=None, created_at=datetime.utcnow(),
    )


async def spawn_dispatch_fanout(dispatch_id: str) -> None:
    """Fire-and-forget the fan-out with a strong ref (dodges GC),
    mirroring support.pipeline.spawn.

    Split from ``run_dispatch_fanout`` so the fan-out is testable at
    all: a task nobody awaits can only be asserted on by sleeping.
    """
    task = asyncio.create_task(run_dispatch_fanout(dispatch_id))
    _background_tasks.add(task)
    task.add_done_callback(_fanout_done)


def _fanout_done(task: "asyncio.Task") -> None:
    _background_tasks.discard(task)
    # Nothing awaits this task, so without this the only trace of a
    # dead fan-out would be asyncio's "exception was never retrieved"
    # at GC time, with no dispatch id in it.
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "[admin_dispatch] fanout task died: %s",
            task.exception(), exc_info=task.exception(),
        )


async def run_dispatch_fanout(dispatch_id: str) -> Dict[str, Any]:
    """Enumerate, claim, deliver, reconcile. Its own session — the
    request that spawned this has long since returned its own."""
    async with async_session_maker() as db:
        row = await db.get(AdminDispatch, dispatch_id)
        if row is None:
            logger.warning("[admin_dispatch] dispatch %s not found", dispatch_id)
            return {"dispatch_id": dispatch_id, "status": "missing", "claimed": 0}
        # Snapshot BEFORE the status write below commits and expires the row.
        dispatch = _snapshot(row)

        await db.execute(
            update(AdminDispatch)
            .where(AdminDispatch.id == dispatch_id)
            .values(status=DISPATCH_SENDING, completed_at=None)
        )
        await db.commit()

        now = datetime.utcnow()
        claimed = 0
        try:
            recipients = await _enumerate_recipients(db, dispatch)
            await _ensure_targets(db, dispatch_id, [r.user_id for r in recipients], now)
            target_ids = await _target_ids(db, dispatch_id)

            for index, rec in enumerate(recipients):
                target_id = target_ids.get(rec.user_id)
                if target_id is None:
                    continue
                if not await _claim_target(db, target_id, datetime.utcnow()):
                    continue  # another replica owns this one
                claimed += 1
                try:
                    await _deliver_one(db, dispatch, rec, target_id, index)
                except Exception as e:  # noqa: BLE001 — recorded, not swallowed
                    # One target must never abort a broadcast to everyone
                    # behind it, and a claimed target left in `sending` is
                    # invisible to the retry sweep, which only resets
                    # `failed`.
                    logger.exception(
                        "[admin_dispatch] target %s (user %s) failed", target_id, rec.user_id,
                    )
                    await db.rollback()
                    await _finish_target(
                        db, target_id, state=TARGET_FAILED, chat_status=CHAT_FAILED,
                        chat_message_id=None, notification_id=None,
                        last_error=f"{type(e).__name__}: {e}"[:500],
                    )
        except Exception:
            # `admin_dispatches.status='failed'` means the FAN-OUT died —
            # per-target outcomes live in the target ledger and never
            # reach here. Record it and re-raise: nothing awaits this in
            # production, and a worker crash that leaves the row reading
            # `sending` forever is the same class of bug as an empty read
            # from a mirror.
            await db.rollback()
            await db.execute(
                update(AdminDispatch)
                .where(AdminDispatch.id == dispatch_id)
                .values(status=DISPATCH_FAILED, completed_at=datetime.utcnow())
            )
            await db.commit()
            raise

        summary = await _reconcile(db, dispatch_id)
        summary.update({"dispatch_id": dispatch_id, "claimed": claimed})
        # `no_agent` is logged beside `delivered` on purpose: it is the
        # half-delivery (banner, no chat card) that a single "delivered"
        # number used to hide, and the one thing a retry can still fix.
        logger.info(
            "[admin_dispatch] fanout id=%s audience=%s mode=%s targets=%d "
            "claimed=%d notified=%d in_chat=%d no_agent=%d failed=%d status=%s",
            dispatch_id, dispatch.audience, dispatch.mode,
            summary["target_count"], claimed, summary["delivered_count"],
            summary["chat_delivered_count"], summary["no_agent_count"],
            summary["failed_count"], summary["status"],
        )
        return summary


# ── Recipient enumeration ─────────────────────────────────────────


async def _enumerate_recipients(db: AsyncSession, dispatch: _DispatchSpec) -> List[_Recipient]:
    """Every user in the audience, each with its active agent's
    credentials when it has one.

    The `deploy_status` predicate lives in the JOIN condition, not in
    WHERE: in WHERE it degrades the outer join to an inner one and
    every user without a live agent silently drops out of the
    broadcast — they are still targets (they get the notification and
    the thread), they just get `chat_status='no_agent'`.

    Deliberately NOT built on admin/users.py's counts: that endpoint
    joins AGENT_ONLY tables and its numbers are legacy-monolith
    artefacts.
    """
    stmt = (
        select(User.id, AgentConfig.agent_url, AgentConfig.agent_api_key)
        .outerjoin(
            AgentConfig,
            and_(
                AgentConfig.user_id == User.id,
                AgentConfig.deploy_status == "active",
            ),
        )
        # Stable across re-runs so a target keeps its drip slot.
        .order_by(User.created_at.asc(), User.id.asc())
    )
    if dispatch.audience == DISPATCH_AUDIENCE_ALL:
        # A suspended account is not a recipient: it must not be pushed
        # to, written into, or counted in the operator's reach. Every
        # other platform-wide fan-out already gates on this
        # (scheduled_tasks.py, pool_service.py) — "everyone" has always
        # meant every ACTIVE user here.
        #
        # Only the broadcast branch. A single-user dispatch names its
        # recipient explicitly, and an operator messaging a suspended
        # account (about the suspension, typically) is the point.
        stmt = stmt.where(User.is_active.is_(True))
    else:
        if not dispatch.target_user_id:
            return []
        stmt = stmt.where(User.id == dispatch.target_user_id)

    rows = (await db.execute(stmt)).all()
    return [_Recipient(uid, url, key) for uid, url, key in rows]


async def count_recipients(db: AsyncSession, dispatch: _DispatchSpec) -> Tuple[int, int]:
    """(recipient_count, with_agent_count) for the admin's pre-send preview.

    Called by ``GET /api/admin/dispatch/preview``. It runs the enumerator
    ITSELF rather than a parallel ``COUNT(*)`` that happens to carry the same
    predicates today: the number an admin reads before pressing send on an
    unrecallable broadcast has to be the number the fan-out will use, and two
    queries that must agree eventually stop agreeing. The audience filter, the
    `is_active` gate and the outer join to a live agent are therefore defined
    in exactly one place (`_enumerate_recipients`).

    Cost is one row per user, which is what the fan-out pays anyway.
    """
    recipients = await _enumerate_recipients(db, dispatch)
    return len(recipients), sum(1 for r in recipients if r.has_agent)


async def _ensure_targets(
    db: AsyncSession, dispatch_id: str, user_ids: List[str], now: datetime,
) -> None:
    """Materialise one `pending` target per recipient. Safe to re-run:
    UNIQUE(dispatch_id, user_id) is the backstop behind the SELECT."""
    existing = await _target_user_ids(db, dispatch_id)
    missing = [u for u in user_ids if u not in existing]
    if not missing:
        return

    db.add_all([_new_target(dispatch_id, uid, now) for uid in missing])
    try:
        await db.commit()
        return
    except IntegrityError:
        # A second replica enumerated the same audience concurrently.
        # The rollback drops the whole batch, so re-read and insert
        # what is still genuinely missing one row at a time — a single
        # collision must not discard everyone behind it.
        await db.rollback()

    existing = await _target_user_ids(db, dispatch_id)
    for uid in [u for u in missing if u not in existing]:
        db.add(_new_target(dispatch_id, uid, now))
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()


def _new_target(dispatch_id: str, user_id: str, now: datetime) -> AdminDispatchTarget:
    return AdminDispatchTarget(
        id=str(uuid.uuid4()),
        dispatch_id=dispatch_id,
        user_id=user_id,
        state=TARGET_PENDING,
        chat_status=CHAT_PENDING,
        attempts=0,
        created_at=now,
        updated_at=now,
    )


async def _target_user_ids(db: AsyncSession, dispatch_id: str) -> set:
    rows = await db.execute(
        select(AdminDispatchTarget.user_id).where(
            AdminDispatchTarget.dispatch_id == dispatch_id
        )
    )
    return {r[0] for r in rows.all()}


async def _target_ids(db: AsyncSession, dispatch_id: str) -> Dict[str, str]:
    rows = await db.execute(
        select(AdminDispatchTarget.user_id, AdminDispatchTarget.id).where(
            AdminDispatchTarget.dispatch_id == dispatch_id
        )
    )
    return {uid: tid for uid, tid in rows.all()}


# ── Per-target delivery ───────────────────────────────────────────


async def _claim_target(db: AsyncSession, target_id: str, now: datetime) -> bool:
    """Single-row status CAS. False = another replica won the race."""
    result = await db.execute(
        update(AdminDispatchTarget)
        .where(
            AdminDispatchTarget.id == target_id,
            AdminDispatchTarget.state == TARGET_PENDING,
        )
        .values(
            state=TARGET_SENDING,
            attempts=AdminDispatchTarget.attempts + 1,
            updated_at=now,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def _deliver_one(
    db: AsyncSession,
    dispatch: _DispatchSpec,
    rec: _Recipient,
    target_id: str,
    index: int,
) -> None:
    """Thread row → notification row → agent hop, in that order, then
    record the outcome on the target.

    The agent hop's failure is CAUGHT rather than raised: it is
    recorded on the target (`state='failed'`, `last_error`) where the
    admin panel and the retry route both read it. Nothing here falls
    back to writing the chat row locally — `messages` is AGENT_ONLY.
    """
    now = datetime.utcnow()
    notification_id: Optional[str] = None
    chat_message_id: Optional[str] = None
    last_error: Optional[str] = None

    try:
        if dispatch.mode == DISPATCH_MODE_PERSISTENT:
            await _ensure_thread_row(db, dispatch, rec.user_id, now)
        notification_id = await _ensure_notification(db, dispatch, rec.user_id, index, now)
    except Exception as e:  # noqa: BLE001 — recorded on the target, not swallowed
        await db.rollback()
        logger.warning(
            "[admin_dispatch] platform writes failed dispatch=%s user=%s: %s: %s",
            dispatch.id, rec.user_id, type(e).__name__, e,
        )
        await _finish_target(
            db, target_id, state=TARGET_FAILED, chat_status=CHAT_FAILED,
            chat_message_id=None, notification_id=None,
            last_error=f"{type(e).__name__}: {e}"[:500],
        )
        return

    if not rec.has_agent:
        # No live agent to hold the chat row. The notification and the
        # thread still landed, so this is a completed target, not a
        # failed one. It IS a half delivery though — notified, never in
        # chat — so `_reconcile` counts it separately as `no_agent_count`
        # and the panel names it rather than letting one word ("Delivered")
        # cover two different outcomes.
        await _finish_target(
            db, target_id, state=TARGET_DONE, chat_status=CHAT_NO_AGENT,
            chat_message_id=None, notification_id=notification_id,
            last_error=None,
        )
        return

    chat_message_id, last_error = await _agent_hop(
        rec.agent_url, rec.agent_api_key, dispatch, rec.user_id,
    )
    await _finish_target(
        db, target_id,
        state=TARGET_DONE if last_error is None else TARGET_FAILED,
        chat_status=CHAT_DELIVERED if last_error is None else CHAT_FAILED,
        chat_message_id=chat_message_id,
        notification_id=notification_id,
        last_error=last_error,
    )


async def _ensure_thread_row(
    db: AsyncSession, dispatch: _DispatchSpec, user_id: str, now: datetime,
) -> None:
    """The `out` row a persistent dispatch opens the thread with.

    The id is a uuid5 of (dispatch, user) — the same trick the agent's
    message writer uses — so a retry that re-runs a target whose agent
    hop failed re-hits the PK instead of appending a second copy of the
    operator's message to the user's thread.
    """
    row_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"admin-thread:{dispatch.id}:{user_id}"))
    existing = await db.execute(
        select(AdminThreadMessage.id).where(AdminThreadMessage.id == row_id)
    )
    if existing.scalar_one_or_none() is not None:
        return

    db.add(AdminThreadMessage(
        id=row_id,
        user_id=user_id,
        dispatch_id=dispatch.id,
        direction=THREAD_OUT,
        body=dispatch.body,
        # The SAME name the chat card is headed with. This row opens the
        # thread, so leaving it to the column default made a dispatch sent
        # as e.g. "Toup Support" arrive as "Toup Support" in the chat and
        # "Toup" one tap later — one operator wearing two identities, which
        # is exactly what the sender_name column was added to stop.
        sender_name=dispatch.sender_name,
        author_admin_id=dispatch.created_by_user_id,
        created_at=now,
    ))
    try:
        await db.commit()
    except IntegrityError:
        # Two replicas delivering the same target — the PK wins.
        await db.rollback()


async def _ensure_notification(
    db: AsyncSession, dispatch: _DispatchSpec, user_id: str, index: int, now: datetime,
) -> str:
    """Enqueue the dispatch's announcement for one recipient.

    The key is deterministic per (dispatch, user), which is the entire
    replay protection: `idempotency_key` is nullable and Postgres allows
    multiple NULLs, so omitting it would mean none.
    """
    return await _enqueue_announcement(
        db,
        user_id=user_id,
        idem=f"admin-dispatch:{dispatch.id}:{user_id}",
        title=dispatch.title,
        body=dispatch.body,
        data={
            "mission_id": f"admin:{dispatch.id}",
            "dispatch_id": dispatch.id,
            "mode": dispatch.mode,
            # The tap target follows the MODE, because the two modes put the
            # message in different places. A `once` notice exists only as the
            # card in the chat — deep-linking it to the Admin thread would
            # land the user on an empty inbox (a `once` dispatch writes no
            # thread row) while the message they were alerted about sat
            # unread one screen away, and the "Got it" receipt the card
            # carries would never be pressed. A `persistent` notice's home
            # IS the thread. `?mission=` is a query VALUE, so its colon is
            # safe; the route segment must never contain one (App.tsx
            # rebuilds `toup://${route}` and WHATWG parses a colon suffix as
            # a port).
            "deep_link": (
                f"toup://notices?mission=admin:{dispatch.id}"
                if dispatch.mode == DISPATCH_MODE_PERSISTENT
                else f"toup://chat?mission=admin:{dispatch.id}"
            ),
            "urgent": dispatch.urgent,
        },
        priority=NQ_PRIORITY_HIGH if dispatch.urgent else NQ_PRIORITY_DEFAULT,
        scheduled_for=_drip_at(index, now),
        now=now,
    )


async def build_reply_notification(
    db: AsyncSession,
    *,
    user_id: str,
    message_id: str,
    body: str,
    sender_name: str,
    now: datetime,
) -> str:
    """Enqueue the alert for an admin's follow-up reply in the thread (DEC-9).

    Called by the admin reply route, not by the fan-out: a reply that only
    writes a row reaches nobody — neither client polls, so the badge moves
    only if the user happens to open the drawer, while the composer's hint
    promises the answer reaches them.

    ORDER: call this AFTER the thread row is committed. A notification that
    announces a message a rollback erased is a tap into an empty thread; the
    reverse (a row whose alert failed to enqueue) at least leaves the message
    readable and is recoverable by sending again — the deterministic
    `idempotency_key` makes that a no-op if the row did land.

    It commits, the way `_ensure_notification` does, and it raises rather than
    swallowing: the operator must learn that their reply was not announced.
    """
    return await _enqueue_announcement(
        db,
        user_id=user_id,
        # The thread row's id, so a re-send of the same reply is a no-op and
        # a genuinely new reply is genuinely new.
        idem=f"admin-thread:{message_id}",
        # There is no title on a thread reply — the operator's NAME is the
        # title, which is also the thing this whole feature exists to show:
        # the message is not from the agent.
        title=sender_name or settings.admin_dispatch_sender_name,
        body=body,
        data={
            "mission_id": f"admin-reply:{message_id}",
            "thread_message_id": message_id,
            # A reply only ever exists inside the ongoing thread, so it is
            # `persistent` by construction and its tap target is the thread.
            "mode": DISPATCH_MODE_PERSISTENT,
            "deep_link": f"toup://notices?mission=admin-reply:{message_id}",
            # Never urgent: `urgent` is a per-dispatch operator judgement that
            # bypasses quiet hours, and a reply in a conversation is not that.
            "urgent": False,
        },
        priority=NQ_PRIORITY_DEFAULT,
        scheduled_for=None,  # one user, no fan-out, nothing to drip
        now=now,
    )


async def _enqueue_announcement(
    db: AsyncSession,
    *,
    user_id: str,
    idem: str,
    title: str,
    body: str,
    data: Dict[str, Any],
    priority: str,
    scheduled_for: Optional[datetime],
    now: datetime,
) -> str:
    """Enqueue one `announcement` row, or return the row a previous run left.

    Explicit SELECT-first then INSERT with UNIQUE(user_id, idempotency_key)
    as the race-proof backstop — agent_notify.py's shape.

    ONE builder for both producers (the fan-out and the admin reply): the
    `announcement` lane's two escape hatches live in `data_json` — `kind`
    escapes the autopilot_push toggle and `cap_exempt` escapes cap slot #11 —
    and they are stamped HERE rather than by the caller, because a second
    builder is exactly how one producer silently loses one of them and its
    alerts are dropped by a suppression nobody thought applied.
    """
    existing = await db.execute(
        select(NotificationQueue.id).where(
            NotificationQueue.user_id == user_id,
            NotificationQueue.idempotency_key == idem,
        )
    )
    row_id = existing.scalar_one_or_none()
    if row_id is not None:
        return row_id

    new_id = str(uuid.uuid4())
    db.add(NotificationQueue(
        id=new_id,
        user_id=user_id,
        source="platform",
        event_kind=NOTIFY_KIND_ANNOUNCEMENT,
        title=(title or "")[:200],
        body=(body or "")[:400],
        # The two escape hatches are stamped LAST so no caller payload can
        # shadow them.
        data_json={**data, "kind": "announcement", "cap_exempt": True},
        priority=priority,
        idempotency_key=idem,
        scheduled_for=scheduled_for,
        # Explicit, not the column default: the alert and the row it
        # announces describe one instant, and the default would fire at
        # flush — after however long the caller's own writes took.
        created_at=now,
    ))
    try:
        await db.commit()
        return new_id
    except IntegrityError:
        await db.rollback()

    existing = await db.execute(
        select(NotificationQueue.id).where(
            NotificationQueue.user_id == user_id,
            NotificationQueue.idempotency_key == idem,
        )
    )
    row_id = existing.scalar_one_or_none()
    if row_id is None:
        raise RuntimeError(f"notification row for {idem} vanished after a unique violation")
    return row_id


def _drip_at(index: int, now: datetime) -> Optional[datetime]:
    """NULL (send now) for the first burst, then one 30s step per
    further burst — D8."""
    if index < ADMIN_DISPATCH_BURST:
        return None
    return now + timedelta(seconds=_DRIP_GAP_SEC * (index // ADMIN_DISPATCH_BURST))


async def _agent_hop(
    agent_url: str, agent_api_key: str, dispatch: _DispatchSpec, user_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    """POST the notice to the tenant agent. Returns (message_id, error)."""
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url.rstrip('/')}/api/internal/admin-notice"
    payload = {
        "user_id": user_id,
        "dispatch_id": dispatch.id,
        "mode": dispatch.mode,
        "title": dispatch.title,
        "body": dispatch.body,
        "sender_name": dispatch.sender_name,
        # The dispatch's own creation time, not utcnow(): a retry must
        # persist the same notice payload the first attempt would have.
        "sent_at": dispatch.created_at.isoformat(),
    }
    try:
        client = get_agent_http_client()
        resp = await client.post(
            url,
            headers={"X-Agent-Key": agent_api_key},
            json=payload,
            timeout=_AGENT_HOP_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 — recorded on the target
        return None, f"agent unreachable: {type(e).__name__}: {e}"[:500]

    # Any 2xx, never `== 200`: this route answers 201 on the write path
    # and 200 on the idempotent replay (ws_realtime.py:949 records what
    # an equality check costs).
    if not (200 <= resp.status_code < 300):
        return None, f"agent {resp.status_code}: {resp.text[:200]}"

    try:
        body = resp.json()
    except Exception:
        body = {}
    message_id = body.get("message_id") if isinstance(body, dict) else None
    return message_id, None


async def _finish_target(
    db: AsyncSession,
    target_id: str,
    *,
    state: str,
    chat_status: str,
    chat_message_id: Optional[str],
    notification_id: Optional[str],
    last_error: Optional[str],
) -> None:
    await db.execute(
        update(AdminDispatchTarget)
        .where(AdminDispatchTarget.id == target_id)
        .values(
            state=state,
            chat_status=chat_status,
            chat_message_id=chat_message_id,
            notification_id=notification_id,
            last_error=last_error,
            updated_at=datetime.utcnow(),
        )
    )
    await db.commit()


# ── Parent bookkeeping ────────────────────────────────────────────


async def summarize_targets(db: AsyncSession, dispatch_id: str) -> Dict[str, int]:
    """Every count the panel shows, straight off the target ledger.

    The ledger is the single source of truth for this dispatch's outcome, so
    the fan-out's reconcile and `GET /api/admin/dispatch/{id}` read it through
    this one function rather than through two sets of predicates that agree
    until they don't.

    `delivered_count` is deliberately narrower than "has any surface": a
    FAILED target is never also delivered. It used to be, because
    `notification_id` is non-null whether or not the agent hop that followed
    it succeeded — so a broadcast that reached hundreds of banners and no
    chats reported itself fully delivered with `failed_count = 0`. Hence the
    two counts beside it: `chat_delivered_count` and `no_agent_count` name the
    two halves, because one word cannot cover both.
    """
    def _count(*where):
        return (
            select(func.count())
            .select_from(AdminDispatchTarget)
            .where(AdminDispatchTarget.dispatch_id == dispatch_id, *where)
        )

    target_count = await db.scalar(_count()) or 0
    delivered = await db.scalar(_count(
        AdminDispatchTarget.state != TARGET_FAILED,
        or_(
            AdminDispatchTarget.chat_status == CHAT_DELIVERED,
            AdminDispatchTarget.notification_id.isnot(None),
        ),
    )) or 0
    # `retracted` counts as landed: it is what a `once` card becomes AFTER
    # the user read it, so excluding it would make "In chat" fall as the
    # dispatch succeeds — the panel would report chat cards that never
    # arrived, for the users who actually read them.
    chat_delivered = await db.scalar(_count(
        AdminDispatchTarget.chat_status.in_([CHAT_DELIVERED, CHAT_RETRACTED]),
    )) or 0
    no_agent = await db.scalar(
        _count(AdminDispatchTarget.chat_status == CHAT_NO_AGENT)
    ) or 0
    read = await db.scalar(_count(AdminDispatchTarget.read_at.isnot(None))) or 0
    failed = await db.scalar(_count(AdminDispatchTarget.state == TARGET_FAILED)) or 0
    unfinished = await db.scalar(
        _count(AdminDispatchTarget.state.in_([TARGET_PENDING, TARGET_SENDING]))
    ) or 0

    return {
        "target_count": target_count,
        "delivered_count": delivered,
        "chat_delivered_count": chat_delivered,
        "no_agent_count": no_agent,
        "read_count": read,
        "failed_count": failed,
        "pending_count": unfinished,
    }


async def _reconcile(db: AsyncSession, dispatch_id: str) -> Dict[str, Any]:
    """Recompute the parent's counters from the target rows and settle
    its status.

    Recomputed, never incremented: the other replica may have
    delivered half of these, `read_count` is written by the read
    receipt route from a different request, and a retry re-walks
    targets that already counted once.
    """
    counts = await summarize_targets(db, dispatch_id)
    target_count = counts["target_count"]
    unfinished = counts["pending_count"]

    now = datetime.utcnow()
    # Only the four persisted counters — `chat_delivered_count` and
    # `no_agent_count` have no columns and are computed per read, so they
    # cannot go stale between a fan-out and a retry.
    values: Dict[str, Any] = {
        "target_count": target_count,
        "delivered_count": counts["delivered_count"],
        "read_count": counts["read_count"],
        "failed_count": counts["failed_count"],
    }
    if unfinished:
        # The other replica still holds targets — it runs this same
        # reconcile when it drains and settles the status then.
        status = DISPATCH_SENDING
    else:
        # `sent` is "every target terminal", NOT "every target
        # succeeded": a broadcast where one tenant's container is down
        # has delivered, and the per-target failure is the ledger's to
        # report. `failed` is reserved for the fan-out itself dying —
        # of which "the audience resolved to nobody" is one case.
        status = DISPATCH_FAILED if target_count == 0 else DISPATCH_SENT
        values["completed_at"] = now
    values["status"] = status

    await db.execute(
        update(AdminDispatch).where(AdminDispatch.id == dispatch_id).values(**values)
    )
    await db.commit()

    return {**counts, "status": status}
