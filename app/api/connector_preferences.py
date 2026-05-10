"""
T2c — Per-user connector preferences + event log.

Three endpoints, all under `/api/connectors/`:

  GET   /preferences/{connector_id}     — read current overrides
  PUT   /preferences/{connector_id}     — update enabled / channel
                                          overrides / trust_until
  GET   /events/{connector_id}          — last N events for the
                                          connector × user, paged.

Architecture §4.4 + §5.4 reference. Channel-deny resolution at the
dispatcher reads `ConnectorUserPreference` (T1e left a hook); when
the row is absent the manifest defaults apply. T2c populates the
row; the dispatcher needs no changes here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    ConnectorEvent,
    ConnectorUserPreference,
)
from app.services.connector_registry import get_registry

router = APIRouter(prefix="/connectors", tags=["Connectors"])


# ─── Schemas ───────────────────────────────────────────────────────────


class ChannelOverride(BaseModel):
    """Per-tool channel-deny map. Keys are channel ids
    (`web|mobile|voice|telegram|whatsapp`), values are explicit
    enable/disable. Missing keys mean "use the manifest default for
    this (channel, tool)" — NOT "deny all"."""

    web: Optional[bool] = None
    mobile: Optional[bool] = None
    voice: Optional[bool] = None
    telegram: Optional[bool] = None
    whatsapp: Optional[bool] = None


class PreferenceResponse(BaseModel):
    connector_id: str
    enabled: bool
    per_tool_overrides: dict[str, ChannelOverride] = Field(default_factory=dict)
    trust_until: Optional[str] = None
    updated_at: Optional[str] = None


class PreferenceUpdate(BaseModel):
    """Partial update — only fields provided are touched."""

    enabled: Optional[bool] = None
    per_tool_overrides: Optional[dict[str, ChannelOverride]] = None
    # Special values:
    #   - omitted → no change
    #   - None    → clear trust (i.e. require confirmation)
    #   - "+24h"  → set trust_until = now + 24 hours
    trust_until: Optional[str] = None


class EventEntry(BaseModel):
    id: str
    event_type: str
    occurred_at: str
    channel: Optional[str] = None
    tool_name: Optional[str] = None
    agent_request_id: Optional[str] = None
    metadata: Optional[dict] = None


class EventsResponse(BaseModel):
    connector_id: str
    events: list[EventEntry]
    total: int
    has_more: bool


# ─── Preferences GET / PUT ─────────────────────────────────────────────


def _serialize_pref(pref: Optional[ConnectorUserPreference], connector_id: str) -> PreferenceResponse:
    if pref is None:
        return PreferenceResponse(
            connector_id=connector_id,
            enabled=True,
            per_tool_overrides={},
            trust_until=None,
            updated_at=None,
        )
    raw_overrides = (
        json.loads(pref.per_tool_channel_overrides_json)
        if pref.per_tool_channel_overrides_json
        else {}
    )
    return PreferenceResponse(
        connector_id=connector_id,
        enabled=pref.enabled,
        per_tool_overrides={
            k: ChannelOverride(**v) for k, v in raw_overrides.items()
        },
        trust_until=(
            pref.trust_until.isoformat() + "Z" if pref.trust_until else None
        ),
        updated_at=pref.updated_at.isoformat() + "Z",
    )


def _validate_connector(connector_id: str) -> None:
    if get_registry().get(connector_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown connector {connector_id!r}",
        )


@router.get(
    "/preferences/{connector_id}", response_model=PreferenceResponse,
)
async def get_preferences(
    connector_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _validate_connector(connector_id)
    pref = (
        await db.execute(
            select(ConnectorUserPreference)
            .where(ConnectorUserPreference.user_id == str(current_user.id))
            .where(ConnectorUserPreference.connector_id == connector_id)
        )
    ).scalar_one_or_none()
    return _serialize_pref(pref, connector_id)


@router.put(
    "/preferences/{connector_id}", response_model=PreferenceResponse,
)
async def update_preferences(
    connector_id: str,
    update: PreferenceUpdate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _validate_connector(connector_id)
    user_id = str(current_user.id)

    pref = (
        await db.execute(
            select(ConnectorUserPreference)
            .where(ConnectorUserPreference.user_id == user_id)
            .where(ConnectorUserPreference.connector_id == connector_id)
        )
    ).scalar_one_or_none()
    if pref is None:
        pref = ConnectorUserPreference(
            user_id=user_id,
            connector_id=connector_id,
            enabled=True,
        )
        db.add(pref)

    if update.enabled is not None:
        pref.enabled = update.enabled

    if update.per_tool_overrides is not None:
        # Strip Nones so the on-disk JSON is minimal.
        compacted: dict[str, dict] = {}
        for tool_name, ov in update.per_tool_overrides.items():
            row = {k: v for k, v in ov.model_dump().items() if v is not None}
            if row:
                compacted[tool_name] = row
        pref.per_tool_channel_overrides_json = (
            json.dumps(compacted) if compacted else None
        )

    if update.trust_until is not None:
        if update.trust_until == "+24h":
            pref.trust_until = datetime.utcnow() + timedelta(hours=24)
        elif update.trust_until == "":
            pref.trust_until = None
        else:
            try:
                pref.trust_until = datetime.fromisoformat(
                    update.trust_until.rstrip("Z")
                )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid trust_until: {update.trust_until!r}",
                )

    pref.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(pref)
    return _serialize_pref(pref, connector_id)


# ─── Events ────────────────────────────────────────────────────────────


@router.get("/events/{connector_id}", response_model=EventsResponse)
async def get_events(
    connector_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    before: Optional[str] = Query(default=None, description="ISO timestamp (exclusive upper bound)"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent events for this connector, newest first.
    Cursor-based pagination via `before` (set to the oldest event's
    `occurred_at` to fetch the next page)."""
    _validate_connector(connector_id)
    user_id = str(current_user.id)

    stmt = (
        select(ConnectorEvent)
        .where(ConnectorEvent.user_id == user_id)
        .where(ConnectorEvent.connector_id == connector_id)
        .order_by(desc(ConnectorEvent.occurred_at))
    )
    if before:
        try:
            cutoff = datetime.fromisoformat(before.rstrip("Z"))
            stmt = stmt.where(ConnectorEvent.occurred_at < cutoff)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid before cursor: {before!r}",
            )

    # Fetch limit+1 to detect has_more without a separate count query.
    rows = (await db.execute(stmt.limit(limit + 1))).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]

    events = [
        EventEntry(
            id=r.id,
            event_type=r.event_type,
            occurred_at=r.occurred_at.isoformat() + "Z",
            channel=r.channel,
            tool_name=r.tool_name,
            agent_request_id=r.agent_request_id,
            metadata=(json.loads(r.metadata_json) if r.metadata_json else None),
        )
        for r in rows
    ]
    return EventsResponse(
        connector_id=connector_id,
        events=events,
        total=len(events),
        has_more=has_more,
    )
