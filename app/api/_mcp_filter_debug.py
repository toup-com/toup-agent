"""Diagnostic — re-added 2026-05-11 because the user's connector
identity disappeared after a container rollout and we need DB-level
visibility into why. Returns: resolved user_id from X-Agent-Key, the
set of active connector ids vault.list_active sees, every
ConnectorIdentity row regardless of status, and the last 20
ConnectorEvent rows for that user. The combo tells us whether the
OAuth row was deleted, flipped to reauth_required, or never
persisted.

Removable once the regression is understood."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.database import async_session_maker
from app.db.models import ConnectorIdentity, User
from app.db.models.connectors import ConnectorEvent
from app.mcp_auth import _resolve_agent_key_to_user_id
from app.services import connector_vault as vault

router = APIRouter(prefix="/agent", tags=["Diagnostic"])


class IdentityRow(BaseModel):
    connector_id: str
    status: str
    has_token: bool
    has_refresh: bool
    connected_at: str | None


class EventRow(BaseModel):
    connector_id: str
    event_type: str
    occurred_at: str


class GlobalRow(BaseModel):
    user_id_prefix: str
    connector_id: str
    status: str
    # Email of the user that owns this row — needed to disambiguate
    # the two-account-same-email case from two-different-accounts.
    user_email: str | None


class FilterDebugResp(BaseModel):
    resolved_user_id: str | None
    active_connector_ids: list[str]
    all_identities: list[IdentityRow]
    recent_events: list[EventRow]
    total_identities_all_users: int
    # Every row in connector_identities, anonymised. Only safe to expose
    # because v0 has at most a handful of rows during this debugging
    # window — rip this field out before re-publishing the diagnostic.
    all_rows_global: list[GlobalRow]
    # Email of the user the X-Agent-Key resolves to. If this differs
    # from the all_rows_global[*].user_email, we're in a
    # two-different-account scenario; if they match, two accounts
    # share an email which is the platform's bug.
    resolved_user_email: str | None


@router.get("/_mcp_filter_debug")
async def mcp_filter_debug(
    x_agent_key: str | None = Header(default=None, alias="X-Agent-Key"),
):
    import traceback as _tb
    try:
        return await _impl(x_agent_key)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": _tb.format_exc()}


async def _impl(x_agent_key: str | None) -> FilterDebugResp:
    if not x_agent_key:
        raise HTTPException(status_code=401, detail="X-Agent-Key required")
    user_id = await _resolve_agent_key_to_user_id(x_agent_key)
    if user_id is None:
        return FilterDebugResp(
            resolved_user_id=None,
            active_connector_ids=[],
            all_identities=[],
            recent_events=[],
            total_identities_all_users=0,
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
        global_rows = (
            await db.execute(select(ConnectorIdentity))
        ).scalars().all()
        # Bulk email lookup for every user_id we touch in this response.
        wanted_ids = {r.user_id for r in global_rows} | {user_id}
        users_by_id = {
            u.id: u.email
            for u in (
                await db.execute(
                    select(User).where(User.id.in_(wanted_ids))
                )
            ).scalars()
        }
    return FilterDebugResp(
        resolved_user_id=user_id,
        active_connector_ids=sorted({r.connector_id for r in active_rows}),
        all_identities=[
            IdentityRow(
                connector_id=r.connector_id,
                status=r.status,
                has_token=bool(r.access_token_enc),
                has_refresh=bool(r.refresh_token_enc),
                connected_at=(r.connected_at.isoformat() + "Z") if r.connected_at else None,
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
        all_rows_global=[
            GlobalRow(
                user_id_prefix=(r.user_id or "")[:8],
                connector_id=r.connector_id,
                status=r.status,
                user_email=users_by_id.get(r.user_id),
            )
            for r in global_rows
        ],
        resolved_user_email=users_by_id.get(user_id),
    )
