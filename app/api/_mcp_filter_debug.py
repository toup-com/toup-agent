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
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import ConnectorIdentity
from app.mcp_auth import _resolve_agent_key_to_user_id
from app.services import connector_vault as vault

router = APIRouter(prefix="/agent", tags=["Diagnostic"])


class IdentityRow(BaseModel):
    connector_id: str
    status: str
    has_token: bool


class FilterDebugResp(BaseModel):
    resolved_user_id: str | None
    active_connector_ids: list[str]
    all_identities: list[IdentityRow]


@router.get("/_mcp_filter_debug", response_model=FilterDebugResp)
async def mcp_filter_debug(
    x_agent_key: str | None = Header(default=None, alias="X-Agent-Key"),
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
        # Pull every row for the user too — including non-active — so
        # we can tell "no row at all" from "row exists but status !=
        # active".
        all_rows = (
            await db.execute(
                select(ConnectorIdentity).where(
                    ConnectorIdentity.user_id == user_id
                )
            )
        ).scalars().all()

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
    )
