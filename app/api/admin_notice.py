"""Agent-side write endpoint for operator notices (Admin Dispatch).

An admin composes a message on the platform and it arrives in the user's
chat as a card from the OPERATOR, not from their agent. The platform owns
the dispatch, the targets, the thread and the push lane — but the chat row
has to be written HERE, because `messages` is AGENT_ONLY and
`broadcast_to_user` is an in-process dict on this container. So the
platform's fan-out worker HTTP-hops once per recipient with the tenant's
X-Agent-Key, the same shape `notify_deliver` uses in the other direction.

Auth is three layers and each catches a different mistake:

  1. 404 unless ``run_mode == "agent"``. These routes are mounted only by
     agent_main, but the gate means a platform deploy can never answer for
     them either.
  2. Our OWN X-Agent-Key compare. ``AgentAPIKeyMiddleware`` FALLS OPEN when
     ``settings.agent_api_key`` is unset (agent_main.py:348) — a route that
     writes a third party's message into the user's chat cannot rely on a
     guard with an open state.
  3. ``body.user_id == settings.user_id``. The agent has exactly one user,
     and `get_current_user`'s agent branch auto-creates a stub User for a
     mis-routed envelope (auth.py:206-228), so a platform routing bug would
     otherwise land silently in the wrong tenant.

Mounted in agent_main (agent entrypoint — NOT platform_main).
"""

from __future__ import annotations

import hmac
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.agent.admin_message_writer import (
    broadcast_admin_notice,
    broadcast_admin_notice_retract,
    build_admin_notice_payload,
    delete_admin_notice,
    write_admin_notice,
)
from app.config import settings
from app.db.database import async_session_maker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/admin-notice", tags=["Admin Notice"])


class AdminNoticeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=36)
    dispatch_id: str = Field(min_length=1, max_length=36)
    mode: str = Field(pattern="^(once|persistent)$")
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    sender_name: str = Field(default="Toup", min_length=1, max_length=80)
    sent_at: Optional[str] = Field(default=None, max_length=40)


class AdminNoticeResponse(BaseModel):
    message_id: str
    # Always null — the row is deliberately outside the day chat so the
    # agent's context can't see it. Echoed so the platform's worker can log
    # that it got the shape it expected rather than a missing field.
    day_chat_id: Optional[str] = None
    ws_count: int


class AdminNoticeRetractRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=36)
    dispatch_id: str = Field(min_length=1, max_length=36)


class AdminNoticeRetractResponse(BaseModel):
    deleted: int
    ws_count: int


def _require_platform_caller(x_agent_key: Optional[str], body_user_id: str) -> None:
    """Run the three gates in the docstring's order. Raises, or returns."""
    if settings.run_mode != "agent":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    expected_key = (getattr(settings, "agent_api_key", "") or "").strip()
    container_user_id = (getattr(settings, "user_id", "") or "").strip()
    if not expected_key or not container_user_id:
        # Pool-lobby container with no bound tenant — there is no user to
        # write to. 401 rather than 5xx so the dispatch worker records the
        # target and moves on instead of retrying a lobby (notify_deliver's
        # reasoning).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="agent not bound",
        )

    provided = (x_agent_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid agent key",
            headers={"WWW-Authenticate": 'X-Agent-Key realm="toup-agent"'},
        )

    if body_user_id != container_user_id:
        logger.error(
            "[admin_notice] envelope user_id=%s does not match container owner=%s — "
            "platform routing bug, refusing write",
            body_user_id[:8], container_user_id[:8],
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="envelope user_id does not match container owner",
        )


@router.post(
    "",
    response_model=AdminNoticeResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def write_notice(
    body: AdminNoticeRequest,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> AdminNoticeResponse:
    """Write the operator's message into the user's chat and push it live.

    Idempotent by construction: the Message id is a uuid5 of
    (dispatch, user), so a worker retry returns the first row's id instead
    of giving the user a second copy. Still 201 on that path — the platform
    asked for the row to exist, and it does.
    """
    _require_platform_caller(x_agent_key, body.user_id)

    notice = build_admin_notice_payload(
        dispatch_id=body.dispatch_id,
        mode=body.mode,
        title=body.title,
        sender_name=body.sender_name,
        sent_at=body.sent_at,
    )
    async with async_session_maker() as db:
        message_id, day_chat_id = await write_admin_notice(
            db, user_id=body.user_id, content=body.body, notice=notice,
        )

    # Broadcast only after the session closed on a committed row: a client
    # that reacts to the frame by refetching history has to find it there.
    ws_count = await broadcast_admin_notice(
        body.user_id,
        message_id=message_id,
        created_at=datetime.utcnow().isoformat(),
        content=body.body,
        notice=notice,
    )
    logger.info(
        "[admin_notice] delivered dispatch=%s user=%s msg=%s mode=%s ws=%d",
        body.dispatch_id, body.user_id[:8], message_id, body.mode, ws_count,
    )
    return AdminNoticeResponse(
        message_id=message_id, day_chat_id=day_chat_id, ws_count=ws_count,
    )


@router.post(
    "/retract",
    response_model=AdminNoticeRetractResponse,
    include_in_schema=False,
)
async def retract_notice(
    body: AdminNoticeRetractRequest,
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> AdminNoticeRetractResponse:
    """End a `once` notice: delete the row, then tell live clients to drop
    the card. ``deleted == 0`` is success — the platform replays a read
    receipt whose retract already landed."""
    _require_platform_caller(x_agent_key, body.user_id)

    async with async_session_maker() as db:
        deleted = await delete_admin_notice(
            db, user_id=body.user_id, dispatch_id=body.dispatch_id,
        )

    # Broadcast even when nothing was deleted: another device may still be
    # showing a card it took off the live frame, and this is what takes it
    # down there.
    ws_count = await broadcast_admin_notice_retract(
        body.user_id, dispatch_id=body.dispatch_id,
    )
    logger.info(
        "[admin_notice] retracted dispatch=%s user=%s deleted=%d ws=%d",
        body.dispatch_id, body.user_id[:8], deleted, ws_count,
    )
    return AdminNoticeRetractResponse(deleted=deleted, ws_count=ws_count)
