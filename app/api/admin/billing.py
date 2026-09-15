"""Admin billing — the UNLIMITED override, grant and revoke.

Every route is ``Depends(require_admin)``, the same guard the rest of the
admin surface uses. There is deliberately no second auth mechanism here: an
entitlement that can be handed out is exactly the kind of thing that must not
have its own back door.

The routes are thin. All of the rules — who may sponsor whom, when a grant is
refused, what the grantee's balance and ``plan_source`` become, what happens
to the entitlement when a grant ends — live in
``app.services.entitlement``, which ``python -m app.scripts.unlimited_grant``
also calls. One implementation, two front ends; a second copy of the rules is
a second set of rules.

Nothing here deletes a grant. ``DELETE /admin/unlimited-grants/{id}`` REVOKES
— it stamps ``revoked_at``/``revoked_by``/``revoked_reason`` on the row and
leaves it in place forever. The listing returns revoked and lapsed grants by
default for the same reason: how a comp ended is the part of the record an
operator most often needs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_admin
from app.db.database import get_db
from app.db.models import User
from app.services import entitlement
from app.services.entitlement import GrantError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/unlimited-grants", tags=["admin-billing"])


class GrantCreate(BaseModel):
    # Either identifier for the grantee; email is what an operator has to hand.
    user_id: Optional[str] = None
    email: Optional[str] = None

    sponsor_kind: str = Field(description="apple | stripe")
    sponsor_original_transaction_id: Optional[str] = None
    sponsor_user_id: Optional[str] = None
    sponsor_stripe_subscription_id: Optional[str] = None

    reason: str
    expires_at: Optional[datetime] = None


class GrantRevoke(BaseModel):
    revoked_reason: str = ""


class GrantOut(BaseModel):
    id: str
    granted_to_user_id: str
    granted_to_email: Optional[str] = None
    sponsor_kind: str
    sponsor_user_id: Optional[str] = None
    sponsor_email: Optional[str] = None
    sponsor_original_txn_id: Optional[str] = None
    sponsor_stripe_sub_id: Optional[str] = None
    reason: str
    expires_at: Optional[datetime] = None
    granted_by_user_id: str
    granted_by_email: str
    granted_at: datetime
    revoked_at: Optional[datetime] = None
    revoked_by_email: Optional[str] = None
    revoked_reason: Optional[str] = None
    lapsed_at: Optional[datetime] = None
    lapsed_reason: Optional[str] = None
    live: bool
    sponsor_live: bool


async def _render(db: AsyncSession, grant, *, now: Optional[datetime] = None) -> GrantOut:
    now = now or datetime.utcnow()
    sponsor_live = await entitlement._sponsor_is_live(db, grant, now=now)
    live = (
        grant.revoked_at is None and grant.lapsed_at is None
        and (grant.expires_at is None or grant.expires_at > now)
        and sponsor_live
    )
    grantee = await db.get(User, grant.granted_to_user_id)
    sponsor = await db.get(User, grant.sponsor_user_id) if grant.sponsor_user_id else None
    return GrantOut(
        id=grant.id,
        granted_to_user_id=grant.granted_to_user_id,
        granted_to_email=getattr(grantee, "email", None),
        sponsor_kind=grant.sponsor_kind,
        sponsor_user_id=grant.sponsor_user_id,
        sponsor_email=getattr(sponsor, "email", None),
        sponsor_original_txn_id=grant.sponsor_original_txn_id,
        sponsor_stripe_sub_id=grant.sponsor_stripe_sub_id,
        reason=grant.reason,
        expires_at=grant.expires_at,
        granted_by_user_id=grant.granted_by_user_id,
        granted_by_email=grant.granted_by_email,
        granted_at=grant.granted_at,
        revoked_at=grant.revoked_at,
        revoked_by_email=grant.revoked_by_email,
        revoked_reason=grant.revoked_reason,
        lapsed_at=grant.lapsed_at,
        lapsed_reason=grant.lapsed_reason,
        live=live,
        sponsor_live=sponsor_live,
    )


async def _resolve_user_id(
    db: AsyncSession, user_id: Optional[str], email: Optional[str],
) -> str:
    if user_id:
        return user_id
    if not email:
        raise HTTPException(422, "one of user_id or email is required")
    # Case-insensitive: an operator types the address as the customer wrote it.
    row = (await db.execute(
        select(User).where(func.lower(User.email) == email.strip().lower())
    )).scalars().first()
    if row is None:
        raise HTTPException(404, f"no user with email {email}")
    return row.id


@router.post("", status_code=201, response_model=GrantOut)
async def create_unlimited_grant(
    body: GrantCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> GrantOut:
    """Comp an account onto UNLIMITED, linked to a sponsor who is paying.

    Refusals come straight from the service so this route and the CLI answer
    identically: 404 unknown grantee, 409 an open grant already exists, 422 a
    sponsor that is missing / not live / the grantee themselves, or a grantee
    already on a paid plan of their own.
    """
    grantee_id = await _resolve_user_id(db, body.user_id, body.email)
    try:
        grant = await entitlement.create_grant(
            db,
            granted_to_user_id=grantee_id,
            sponsor_kind=body.sponsor_kind,
            sponsor_user_id=body.sponsor_user_id,
            sponsor_original_txn_id=body.sponsor_original_transaction_id,
            sponsor_stripe_sub_id=body.sponsor_stripe_subscription_id,
            reason=body.reason,
            expires_at=body.expires_at,
            granted_by_user_id=admin.id,
            granted_by_email=admin.email,
        )
    except GrantError as e:
        # The service already rolled nothing back — the grant row may have been
        # flushed before the sponsor-liveness refusal. Roll back explicitly so a
        # refused grant leaves NO row behind.
        await db.rollback()
        raise HTTPException(e.status, e.detail)
    out = await _render(db, grant)
    await db.commit()
    return out


@router.get("")
async def list_unlimited_grants(
    live: Optional[bool] = Query(None, description="filter to live / dead grants"),
    user_id: Optional[str] = Query(None),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Grant history, newest first. UNFILTERED by default — revoked and lapsed
    grants are records, not clutter.
    """
    grants = await entitlement.list_grants(db, user_id=user_id, live=live)
    return {"grants": [(await _render(db, g)).model_dump() for g in grants]}


@router.delete("/{grant_id}", response_model=GrantOut)
async def revoke_unlimited_grant(
    grant_id: str,
    body: GrantRevoke,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> GrantOut:
    """REVOKE — never delete. The row stays, stamped with who ended it, when
    and why; the grantee returns to free unless they qualify some other way.

    Idempotent: revoking an already-dead grant returns it unchanged rather
    than rewriting how it died.
    """
    try:
        grant = await entitlement.revoke_grant(
            db, grant_id,
            revoked_by_user_id=admin.id,
            revoked_by_email=admin.email,
            revoked_reason=body.revoked_reason,
        )
    except GrantError as e:
        await db.rollback()
        raise HTTPException(e.status, e.detail)
    out = await _render(db, grant)
    await db.commit()
    return out
