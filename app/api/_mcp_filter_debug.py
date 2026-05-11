"""Temporary diagnostic — remove after the connector_filter mismatch
that was blocking Gmail tools from surfacing in chat is resolved.

Returns the same data the ConnectorToolFilterMiddleware uses to
decide whether a connector tool is shown to the agent: the user_id
resolved from X-Agent-Key on the platform, the set of
ConnectorIdentity rows that vault.list_active returns for that
user_id, and a flag for each so we can tell whether the row is
active or in some other state.

Lives outside of agent.py so it can be removed cleanly once the
investigation is done."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.database import async_session_maker
from app.db.models import ConnectorIdentity
from app.db.models.connectors import ConnectorEvent
from app.mcp_auth import _resolve_agent_key_to_user_id
from app.services import connector_vault as vault

router = APIRouter(prefix="/agent", tags=["Diagnostic"])


class IdentityRow(BaseModel):
    connector_id: str
    status: str
    has_token: bool


class EventRow(BaseModel):
    connector_id: str
    event_type: str
    occurred_at: str


class FilterDebugResp(BaseModel):
    resolved_user_id: str | None
    active_connector_ids: list[str]
    all_identities: list[IdentityRow]
    # Were any oauth events EVER recorded for this user_id? Tells us
    # whether the OAuth callback even ran. EVENT_CONNECTED rows would
    # show "the OAuth completed but vault.put failed silently"; EVENT
    # rows of any other type narrow it further.
    recent_events: list[EventRow]
    # Sanity check: total identities across all users in the platform
    # DB. If this is 0, no OAuth has ever persisted anywhere — points
    # to a global connector_identities write bug rather than a per-
    # user mismatch.
    total_identities_all_users: int


@router.get("/_mcp_filter_debug")
async def mcp_filter_debug(
    x_agent_key: str | None = Header(default=None, alias="X-Agent-Key"),
):
    # Wrap everything in try/except and return the traceback so we can
    # see what's happening without Railway log access. Diagnostic-only.
    import traceback as _tb
    try:
        return await _mcp_filter_debug_impl(x_agent_key)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": _tb.format_exc()}


async def _mcp_filter_debug_impl(
    x_agent_key: str | None,
) -> FilterDebugResp:
    if not x_agent_key:
        raise HTTPException(status_code=401, detail="X-Agent-Key required")

    user_id = await _resolve_agent_key_to_user_id(x_agent_key)
    if user_id is None:
        return FilterDebugResp(
            resolved_user_id=None,
            active_connector_ids=[],
            all_identities=[],
        )

    async with async_session_maker() as db:
        active_rows = await vault.list_active(db, user_id)
        all_rows = (
            await db.execute(
                select(ConnectorIdentity).where(
                    ConnectorIdentity.user_id == user_id
                )
            )
        ).scalars().all()
        recent_events = (
            await db.execute(
                select(ConnectorEvent)
                .where(ConnectorEvent.user_id == user_id)
                .order_by(ConnectorEvent.occurred_at.desc())
                .limit(20)
            )
        ).scalars().all()
        total_count = (
            await db.execute(select(func.count(ConnectorIdentity.user_id)))
        ).scalar_one()

    return FilterDebugResp(
        resolved_user_id=user_id,
        active_connector_ids=sorted({r.connector_id for r in active_rows}),
        all_identities=[
            IdentityRow(
                connector_id=r.connector_id,
                status=r.status,
                has_token=bool(r.access_token_enc),
            )
            for r in all_rows
        ],
        recent_events=[
            EventRow(
                connector_id=e.connector_id,
                event_type=e.event_type,
                occurred_at=e.occurred_at.isoformat() + "Z",
            )
            for e in recent_events
        ],
        total_identities_all_users=int(total_count),
    )
