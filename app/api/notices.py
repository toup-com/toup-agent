"""User-facing half of Admin Dispatch — notices + the Admin thread.

The operator's side lives in ``app/api/admin/dispatch.py``; this is what the
phone and the web client call. Platform-only: ``admin_dispatch_targets`` and
``admin_thread_messages`` are PLATFORM_ONLY tables, and the thread is a
conversation between the operator and the user that the agent must never see
(D2).

Routes (mounted at settings.api_prefix, e.g. /api):
  GET  /notices/state              — badge counts for the rail / drawer row
  GET  /notices/thread             — the Admin thread
  POST /notices/thread             — reply to the operator
  POST /notices/thread/read        — mark the operator's rows read
  POST /notices/{dispatch_id}/read — read receipt; retracts a `once` notice

None of this is gated on ``settings.admin_dispatch_enabled``. That switch stops
new dispatches being composed; flipping it must not strand a card on a phone
with no way to read, retract or answer it.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.api.tenant_proxy import agent_proxy_info, proxy_to_agent
# One definition of the thread wire shape for both surfaces — see §7 of the
# dispatch contract; two copies would drift.
from app.api.admin.dispatch import thread_message_out
from app.db import get_db, User
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadMessage,
    CHAT_DELIVERED,
    CHAT_RETRACTED,
    DISPATCH_MODE_ONCE,
    TARGET_DONE,
    THREAD_IN,
    THREAD_OUT,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notices", tags=["Notices"])


class NoticeReplyRequest(BaseModel):
    body: str = Field(min_length=1, max_length=4000)

    @field_validator("body")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class NoticeStateOut(BaseModel):
    unread_notices: int
    thread_unread: int
    has_thread: bool


@router.get("/state", response_model=NoticeStateOut)
async def get_notice_state(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NoticeStateOut:
    """Counts behind the Admin row's badge.

    The two counters are DISJOINT and that is load-bearing: a persistent
    dispatch writes both a target row and a thread row, so an unscoped
    target count would announce one operator message as two. `once` notices
    live only in the chat (they have no thread and no sidebar destination),
    persistent ones live only in the thread — hence the mode filter here.

    What the clients do with them: the Admin row is the THREAD, so it is
    shown iff `has_thread` and badged with `thread_unread` alone.
    `unread_notices` is the un-acknowledged-in-chat count, reported for the
    operator-facing delivery view and for any future surface; it is
    deliberately not part of the sidebar badge on either client.
    """
    unread_notices = (await db.execute(
        select(func.count())
        .select_from(AdminDispatchTarget)
        .join(AdminDispatch, AdminDispatch.id == AdminDispatchTarget.dispatch_id)
        .where(
            AdminDispatchTarget.user_id == current_user.id,
            # `done` is the only state that has actually reached the user; a
            # pending or failed target is not something they can read yet.
            AdminDispatchTarget.state == TARGET_DONE,
            AdminDispatchTarget.read_at.is_(None),
            AdminDispatch.mode == DISPATCH_MODE_ONCE,
        )
    )).scalar_one()

    thread_total = (await db.execute(
        select(func.count()).select_from(AdminThreadMessage).where(
            AdminThreadMessage.user_id == current_user.id,
        )
    )).scalar_one()

    thread_unread = (await db.execute(
        select(func.count()).select_from(AdminThreadMessage).where(
            AdminThreadMessage.user_id == current_user.id,
            AdminThreadMessage.direction == THREAD_OUT,
            AdminThreadMessage.read_at.is_(None),
        )
    )).scalar_one()

    return NoticeStateOut(
        unread_notices=int(unread_notices or 0),
        thread_unread=int(thread_unread or 0),
        has_thread=bool(thread_total),
    )


# ── The thread ───────────────────────────────────────────────────────
# Declared ABOVE /{dispatch_id}/read: POST /notices/thread/read has the same
# shape as POST /notices/{dispatch_id}/read, and whichever is declared first
# wins — with the parametrised route first, "thread" arrives as a dispatch id.

@router.get("/thread")
async def get_thread(
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The Admin thread, oldest→newest. `unread` counts the WHOLE thread, not
    the page — the badge must not shrink because the page was short."""
    rows = (await db.execute(
        select(AdminThreadMessage)
        .where(AdminThreadMessage.user_id == current_user.id)
        .order_by(AdminThreadMessage.created_at.desc())
        .limit(limit)
    )).scalars().all()

    unread = (await db.execute(
        select(func.count()).select_from(AdminThreadMessage).where(
            AdminThreadMessage.user_id == current_user.id,
            AdminThreadMessage.direction == THREAD_OUT,
            AdminThreadMessage.read_at.is_(None),
        )
    )).scalar_one()

    return {
        "messages": [thread_message_out(m) for m in reversed(rows)],
        "unread": int(unread or 0),
    }


@router.post("/thread", status_code=201)
async def reply_to_admin(
    body: NoticeReplyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The user's reply. Platform-side only — the agent never sees it (D2)."""
    msg = AdminThreadMessage(
        # id AND created_at explicit: both column defaults fire at flush, and
        # the response is built before the commit.
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        dispatch_id=None,
        direction=THREAD_IN,
        body=body.body,
        author_admin_id=None,
        created_at=datetime.utcnow(),
    )
    db.add(msg)

    # Build the response BEFORE commit — pgbouncer txn-mode rule.
    resp = {"message": thread_message_out(msg)}
    await db.commit()
    return resp


@router.post("/thread/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_thread_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await db.execute(
        sa_update(AdminThreadMessage)
        .where(
            AdminThreadMessage.user_id == current_user.id,
            AdminThreadMessage.direction == THREAD_OUT,
            AdminThreadMessage.read_at.is_(None),
        )
        .values(read_at=datetime.utcnow())
    )
    await db.commit()


# ── Read receipt ─────────────────────────────────────────────────────

@router.post("/{dispatch_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notice_read(
    dispatch_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record that this user read the notice; retract it if it was a `once`.

    `admin_dispatch_targets.read_at` is the only read receipt in the system, so
    it is committed BEFORE the agent hop. The retract is best-effort on top: the
    reading device has already dropped the card, and an unreachable agent must
    not cost the user their receipt (which would re-serve the notice forever).
    A failure is recorded on the target and a later read retries it —
    `chat_status` only reaches 'retracted' when the agent confirms the delete.

    NOTHING here is loaded as an ORM entity, on purpose. A broadcast is read by
    thousands of phones within seconds, across two replicas with no leader
    election, so the read stamp is a CAS (`WHERE read_at IS NULL`) and the
    parent's counter is a SQL increment. The Python read-modify-write this
    replaces lost most of its increments and nothing ever recomputed the total.
    """
    row = (await db.execute(
        select(AdminDispatchTarget.id, AdminDispatchTarget.chat_status).where(
            AdminDispatchTarget.dispatch_id == dispatch_id,
            AdminDispatchTarget.user_id == current_user.id,
        )
    )).first()
    if row is None:
        raise HTTPException(404, "Notice not found")
    target_id, chat_status = row

    mode = (await db.execute(
        select(AdminDispatch.mode).where(AdminDispatch.id == dispatch_id)
    )).scalar_one_or_none()

    now = datetime.utcnow()
    claimed = int((await db.execute(
        sa_update(AdminDispatchTarget)
        .where(
            AdminDispatchTarget.id == target_id,
            AdminDispatchTarget.read_at.is_(None),
        )
        .values(read_at=now, updated_at=now)
    )).rowcount or 0)
    if claimed > 0:
        # Exactly-once per target: the CAS above is what licenses a blind
        # increment — a re-read of the same notice matches no row and never
        # reaches here. `admin_dispatches.read_count` stays a cache either way;
        # `GET /admin/dispatch/{id}` recomputes it from this ledger.
        await db.execute(
            sa_update(AdminDispatch)
            .where(AdminDispatch.id == dispatch_id)
            .values(read_count=AdminDispatch.read_count + 1)
        )

    needs_retract = mode == DISPATCH_MODE_ONCE and chat_status != CHAT_RETRACTED
    await db.commit()

    if not needs_retract:
        return

    info = await agent_proxy_info(current_user.id, db)
    if info is None:
        # "No agent" and "the agent is not reachable right now" are DIFFERENT
        # facts and only one of them means there is nothing to do.
        # `agent_proxy_info` returns None for BOTH: it requires
        # `deploy_status == 'active'`, and a container that is redeploying
        # ('deploying') or wedged ('error') fails that test while its
        # `messages` row sits there perfectly intact.
        #
        # The original comment here — "there is no tenant row to delete, so
        # there is nothing to retry either" — is true only in the first case.
        # In the second the card IS in the user's chat, the receipt has just
        # been committed above, and returning silently leaves a `once` notice
        # on screen permanently: exactly B6, reintroduced through the recovery
        # path rather than the delivery one.
        #
        # `chat_status` is what tells them apart, because it records whether a
        # row was ever written. Only 'delivered' means one exists.
        if chat_status == CHAT_DELIVERED:
            await db.execute(
                sa_update(AdminDispatchTarget)
                .where(AdminDispatchTarget.id == target_id)
                .values(last_error=(
                    "retract deferred: the agent was unreachable "
                    "(deploy_status is not 'active') while a delivered card "
                    "still needs retracting — press Retry once it is back"
                ))
            )
            await db.commit()
            logger.warning(
                "[notices] retract of %s for %s deferred — agent not active, "
                "but chat_status=delivered so the card is still on screen",
                dispatch_id[:8], current_user.id[:8],
            )
        return
    agent_url, agent_api_key = info
    try:
        await proxy_to_agent(
            agent_url,
            agent_api_key,
            "internal/admin-notice/retract",
            "POST",
            json_body={"user_id": current_user.id, "dispatch_id": dispatch_id},
        )
    except Exception as e:
        await db.execute(
            sa_update(AdminDispatchTarget)
            .where(AdminDispatchTarget.id == target_id)
            .values(last_error=f"retract failed: {type(e).__name__}: {e}"[:500])
        )
        await db.commit()
        logger.warning("[notices] retract of %s for %s failed: %s",
                       dispatch_id[:8], current_user.id[:8], e)
        return

    await db.execute(
        sa_update(AdminDispatchTarget)
        .where(AdminDispatchTarget.id == target_id)
        .values(chat_status=CHAT_RETRACTED, updated_at=datetime.utcnow())
    )
    await db.commit()
