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
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status,
)
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, func, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.api.tenant_proxy import agent_proxy_info, proxy_to_agent
# One definition of the thread wire shape for both surfaces — see §7 of the
# dispatch contract; two copies would drift.
from app.api.admin.dispatch import (
    attachment_response,
    load_thread_attachments,
    store_thread_attachment,
    thread_message_out,
)
from app.db import get_db, User
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadMessage,
    AdminThreadAttachment,
    CHAT_DELIVERED,
    CHAT_RETRACTED,
    DISPATCH_MODE_ONCE,
    PE_DISPATCH_READ,
    PE_DISPATCH_REPLIED,
    PE_ENTITY_DISPATCH,
    PE_ENTITY_THREAD_MESSAGE,
    TARGET_DONE,
    THREAD_IN,
    THREAD_OUT,
)
from app.services.product_events import emit_product_event

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


class ScreenshotReport(BaseModel):
    """What the client saw, not what it did about it.

    `surface` is where the screenshot happened — the Admin thread, or a chat
    card. `dispatch_id` is present only for a card.
    """
    surface: str = Field(pattern="^(thread|notice)$")
    dispatch_id: Optional[str] = Field(default=None, max_length=36)


class NoticeStateOut(BaseModel):
    unread_notices: int
    thread_unread: int
    has_thread: bool


def _visible_to_user():
    """Rows the USER may see. One definition, four call sites.

    A deleted message that still counted toward `thread_unread` would badge a
    thread whose new content the user cannot find — so the two state counters,
    the page and its unread total all read through this.

    `IS NULL` on both, never `!= <ts>`: in SQL a comparison against NULL is
    NULL, not TRUE, so a bare inequality would drop every row written before
    migration 092 — i.e. the entire history of every thread.
    """
    return and_(
        AdminThreadMessage.hidden_from_user_at.is_(None),
        AdminThreadMessage.deleted_at.is_(None),
    )


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
            # `has_thread` drives whether the Admin row EXISTS in the drawer.
            # An operator who deletes the only message in a thread has removed
            # the reason for the row, so it goes too.
            _visible_to_user(),
        )
    )).scalar_one()

    thread_unread = (await db.execute(
        select(func.count()).select_from(AdminThreadMessage).where(
            AdminThreadMessage.user_id == current_user.id,
            AdminThreadMessage.direction == THREAD_OUT,
            AdminThreadMessage.read_at.is_(None),
            _visible_to_user(),
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
        .where(AdminThreadMessage.user_id == current_user.id, _visible_to_user())
        .order_by(AdminThreadMessage.created_at.desc())
        .limit(limit)
    )).scalars().all()

    unread = (await db.execute(
        select(func.count()).select_from(AdminThreadMessage).where(
            AdminThreadMessage.user_id == current_user.id,
            AdminThreadMessage.direction == THREAD_OUT,
            AdminThreadMessage.read_at.is_(None),
            _visible_to_user(),
        )
    )).scalar_one()

    atts = await load_thread_attachments(db, [m.id for m in rows])
    return {
        "messages": [
            thread_message_out(m, attachments=atts.get(m.id, ()))
            for m in reversed(rows)
        ],
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

    # `dispatch_replied` is the RECIPIENT answering, and only that. The
    # operator's follow-up (`admin/dispatch.py::reply_in_thread`) writes an
    # `out` row and deliberately emits nothing: counting both directions
    # would make the reply rate — the one number that says whether a
    # persistent dispatch started a conversation — rise every time the
    # operator talked to themselves.
    #
    # The entity is the thread MESSAGE, not a dispatch: a user reply carries
    # `dispatch_id=None` (the thread outlives the dispatch that opened it),
    # so there is no dispatch id to point at.
    await emit_product_event(
        PE_DISPATCH_REPLIED,
        user_id=current_user.id,
            actor_user_id=current_user.id,
        entity_type=PE_ENTITY_THREAD_MESSAGE,
        entity_id=msg.id,
        # Length, never the words: this row is read by operator analytics and
        # the reply is the user's own private message to a human.
        payload={"direction": THREAD_IN, "body_chars": len(msg.body or "")},
        dedupe_key=f"{PE_DISPATCH_REPLIED}:{msg.id}",
        occurred_at=msg.created_at,
    )
    return resp


@router.post("/thread/attachment", status_code=201)
async def reply_to_admin_with_attachment(
    file: UploadFile = File(...),
    body: str = Form(default=""),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The user's reply, carrying a picture.

    A separate route from ``POST /thread`` for the same reason as the operator
    side: that one takes JSON and every existing client uses it for text.

    Message and attachment are written in ONE transaction. There is no
    "upload, then reference" step, so a user who abandons the compose leaves no
    orphaned blob for anyone to sweep up later.

    Still platform-side only — the agent never sees this row, and now never sees
    the picture either (D2).
    """
    msg = AdminThreadMessage(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        dispatch_id=None,
        direction=THREAD_IN,
        body=(body or "").strip(),
        author_admin_id=None,
        created_at=datetime.utcnow(),
    )
    db.add(msg)
    # Raises before the commit, so a rejected file leaves no message behind.
    att = await store_thread_attachment(
        db, message_id=msg.id, file=file, uploaded_by_user_id=current_user.id,
    )

    resp = {"message": thread_message_out(msg, attachments=[att])}
    await db.commit()

    await emit_product_event(
        PE_DISPATCH_REPLIED,
        user_id=current_user.id,
        actor_user_id=current_user.id,
        entity_type=PE_ENTITY_THREAD_MESSAGE,
        entity_id=msg.id,
        # Counts and sizes, never the words and never the bytes. Same rule as
        # the text reply: this is the user's private message to a human.
        payload={
            "direction": THREAD_IN,
            "body_chars": len(msg.body or ""),
            "attachments": 1,
            "attachment_bytes": int(att.size_bytes or 0),
        },
        dedupe_key=f"{PE_DISPATCH_REPLIED}:{msg.id}",
        occurred_at=msg.created_at,
    )
    return resp


@router.get("/thread/attachments/{att_id}")
async def get_thread_attachment(
    att_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve an attachment to the user who owns the thread it belongs to.

    The join to the parent message IS the authorisation: an id alone proves
    nothing, and these are uuids in a table shared by every user. The
    ``_visible_to_user()`` filter is applied for the same reason the thread list
    applies it — a picture on a message the operator removed for this user must
    not remain fetchable by its id after the words around it are gone.
    """
    att = (await db.execute(
        select(AdminThreadAttachment)
        .join(AdminThreadMessage, AdminThreadMessage.id == AdminThreadAttachment.message_id)
        .where(
            AdminThreadAttachment.id == att_id,
            AdminThreadMessage.user_id == current_user.id,
            _visible_to_user(),
        )
    )).scalar_one_or_none()
    if not att:
        raise HTTPException(404, "Attachment not found")
    return attachment_response(att)


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

    # ORDER: after the commit that persists `read_at` (a receipt a rollback
    # erased is not a read), and ABOVE the `needs_retract` early return —
    # below it, only `once` notices would ever be counted as read and the
    # whole persistent half of the funnel would silently report zero.
    #
    # Gated on the CAS: `claimed` is 0 for a second "Got it" on the same
    # notice, which is not a second read. The dedupe key says the same thing
    # structurally, so the metric survives someone removing the gate.
    if claimed > 0:
        await emit_product_event(
            PE_DISPATCH_READ,
            user_id=current_user.id,
                actor_user_id=current_user.id,
            entity_type=PE_ENTITY_DISPATCH,
            entity_id=dispatch_id,
            payload={"mode": mode},
            dedupe_key=f"{PE_DISPATCH_READ}:{dispatch_id}:{current_user.id}",
            occurred_at=now,
        )

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


# ── Screenshot signal ────────────────────────────────────────────────

@router.post("/screenshot", status_code=status.HTTP_204_NO_CONTENT)
async def report_screenshot(
    body: ScreenshotReport,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record that this user screenshotted an operator message.

    ⚠️ THIS IS DETECTION, NOT PREVENTION, AND THE DIFFERENCE IS THE WHOLE
    POINT. On iOS a screenshot cannot be blocked by any documented API — the
    system notifies the app AFTER the image is already in the camera roll.
    Android is genuinely different: `FLAG_SECURE` stops the capture, so on
    Android this route should almost never fire, and a report FROM Android is
    itself a signal worth reading (a rooted device, or the flag not applied).

    Nothing here is a security control. Someone can photograph the screen with
    a second phone, and WhatsApp — which does block screenshots, for View Once
    media only — says exactly that in its own help text. What this buys is an
    operator knowing a message left the app.

    Best-effort by construction: `emit_product_event` never raises, and this
    returns 204 regardless. A client whose report fails must not retry into a
    loop, and a user must never see an error because their own screenshot
    could not be logged.
    """
    from app.db.models.platform import PE_DISPATCH_SCREENSHOT_DETECTED
    from app.services.product_events import emit_product_event

    # try/except HERE rather than trusting the helper's own promise. The
    # docstring above says this route cannot fail the user; a claim that
    # depends on a distant module continuing to swallow is a claim this file
    # does not actually enforce. Proven by a test that makes the helper raise.
    try:
        await emit_product_event(
            PE_DISPATCH_SCREENSHOT_DETECTED,
            user_id=current_user.id,
        # The SUBJECT and the ACTOR are the same person here, and that is the
        # unusual case worth stating: every other dispatch event has an
        # operator acting on a user.
            actor_user_id=current_user.id,
            entity_type="dispatch" if body.dispatch_id else "thread",
            entity_id=body.dispatch_id or current_user.id,
            payload={"surface": body.surface},
        # NO dedupe key. A second screenshot is a second screenshot — that is
        # precisely the number an operator wants. Every other event in this
        # table dedupes on a fact that can only happen once; this one cannot.
            dedupe_key=None,
        )
    except Exception as e:  # noqa: BLE001 — telemetry must never reach the user
        logger.warning("[notices] screenshot report not recorded: %s", e)
        return
    logger.info("[notices] screenshot reported by %s on %s",
                current_user.id[:8], body.surface)
