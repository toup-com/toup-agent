"""
T5b — Connector force-quarantine admin endpoint.

Operator break-glass for "this connector is misbehaving across users
and we need to take it off the wire NOW." Two surfaces:

  POST /api/admin/connectors/quarantine/{connector_id}
       body: { reason: str }
       Marks every active identity for this connector as
       `reauth_required` so the next call surfaces a reconnect chip
       to every affected user. Logs an EVENT_FORCE_QUARANTINED row
       per identity for the audit trail.

  POST /api/admin/connectors/release/{connector_id}
       Lifts the quarantine — operator-set `reauth_required` rows
       remain (per-user reauth is the user's decision); only the
       global block is cleared.

Auth: admin role (per `current_user.role == 'admin'`). No
sensitive-action token required — the operator is already
authenticated to the admin surface; force-quarantine is a
defensive primitive, not a destructive one.

Implementation: rather than scrubbing every row, we keep an
in-memory "quarantined connector ids" set + a DB table
(`connector_quarantine`) for persistence across restarts. The
dispatcher consults the set BEFORE touching the vault — same shape
as the channel-policy check. Quarantine entry takes priority over
every other rule.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    EVENT_FORCE_QUARANTINED,
    EVENT_FORCE_RELEASED,
)
from app.services.connector_quarantine import (
    QuarantineEntry,
    add as quarantine_add,
    list_active as quarantine_list,
    remove as quarantine_remove,
)

router = APIRouter(prefix="/admin/connectors", tags=["Admin — Connectors"])


def _require_admin(current_user) -> None:
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )


class QuarantineRequest(BaseModel):
    reason: str


class QuarantineEntryOut(BaseModel):
    connector_id: str
    reason: str
    quarantined_by_user_id: Optional[str] = None
    quarantined_at: str


@router.get("/quarantine", response_model=list[QuarantineEntryOut])
async def list_quarantine(current_user=Depends(get_current_user)):
    _require_admin(current_user)
    return [
        QuarantineEntryOut(
            connector_id=e.connector_id,
            reason=e.reason,
            quarantined_by_user_id=e.actor_user_id,
            quarantined_at=e.quarantined_at.isoformat() + "Z",
        )
        for e in quarantine_list()
    ]


@router.post("/quarantine/{connector_id}", response_model=QuarantineEntryOut)
async def force_quarantine(
    connector_id: str,
    req: QuarantineRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    entry = QuarantineEntry(
        connector_id=connector_id,
        reason=req.reason,
        actor_user_id=str(current_user.id),
        quarantined_at=datetime.utcnow(),
    )
    quarantine_add(entry)

    # Mark every active identity as reauth_required so the next
    # tool call surfaces a reconnect chip. Audit row per identity.
    rows = (
        await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.connector_id == connector_id,
                ConnectorIdentity.status == "active",
            )
        )
    ).scalars().all()
    for ident in rows:
        ident.status = "reauth_required"
        db.add(
            ConnectorEvent(
                user_id=ident.user_id,
                connector_id=connector_id,
                event_type=EVENT_FORCE_QUARANTINED,
                metadata_json=f'{{"reason": "{req.reason[:200]}", "actor": "{str(current_user.id)[:8]}"}}',
            )
        )
    await db.commit()
    return QuarantineEntryOut(
        connector_id=entry.connector_id,
        reason=entry.reason,
        quarantined_by_user_id=entry.actor_user_id,
        quarantined_at=entry.quarantined_at.isoformat() + "Z",
    )


@router.post("/quarantine/{connector_id}/release")
async def release_quarantine(
    connector_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    quarantine_remove(connector_id)
    db.add(
        ConnectorEvent(
            user_id=str(current_user.id),
            connector_id=connector_id,
            event_type=EVENT_FORCE_RELEASED,
            metadata_json=f'{{"actor": "{str(current_user.id)[:8]}"}}',
        )
    )
    await db.commit()
    return {"status": "released", "connector_id": connector_id}
