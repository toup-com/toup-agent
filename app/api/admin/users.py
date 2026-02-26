"""
Admin — User & Invite Management (closed beta access control)

Moved from app/api/admin_users.py → app/api/admin/users.py
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, User, Invite
from app.api.auth import get_current_user
from app.api.admin.deps import require_admin
from app.services.auth_service import create_user, get_user_by_email

# ─── Schemas ───────────────────────────────────────────────────

class InviteCreate(BaseModel):
    email: Optional[str] = None
    role: str = Field(default="beta_user", pattern="^(admin|beta_user)$")
    note: Optional[str] = Field(None, max_length=500)
    expires_in_days: int = Field(default=7, ge=1, le=90)


class InviteResponse(BaseModel):
    id: str
    token: str
    email: Optional[str]
    role: str
    note: Optional[str]
    status: str
    created_by: str
    used_by: Optional[str] = None
    used_at: Optional[datetime] = None
    expires_at: datetime
    created_at: datetime
    invite_url: str

    class Config:
        from_attributes = True


class InviteListResponse(BaseModel):
    invites: List[InviteResponse]
    total: int


class UserAdminResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    memory_count: int = 0
    session_count: int = 0
    password_plain: Optional[str] = None

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    users: List[UserAdminResponse]
    total: int


class UserUpdateRequest(BaseModel):
    role: Optional[str] = Field(None, pattern="^(admin|beta_user)$")
    is_active: Optional[bool] = None
    name: Optional[str] = None


class InviteSignupRequest(BaseModel):
    token: str
    email: str
    password: str = Field(min_length=6)
    name: Optional[str] = None


class InviteValidateResponse(BaseModel):
    valid: bool
    email: Optional[str] = None
    role: Optional[str] = None
    expires_at: Optional[datetime] = None
    message: Optional[str] = None


# ─── Admin Router (protected) ─────────────────────────────────

router = APIRouter(prefix="/admin", tags=["Admin — Users & Invites"])

INVITE_BASE_URL = "https://brain.toup.ai/admin/invite"


@router.post("/invites", response_model=InviteResponse, status_code=201)
async def create_invite(
    body: InviteCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new invite token. Only admins."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)

    invite = Invite(
        token=token,
        created_by=admin.id,
        email=body.email,
        role=body.role,
        note=body.note,
        status="pending",
        expires_at=expires_at,
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)
    return _invite_to_response(invite)


@router.get("/invites", response_model=InviteListResponse)
async def list_invites(
    status_filter: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all invites. Optional filter by status."""
    query = select(Invite).order_by(Invite.created_at.desc())
    if status_filter:
        query = query.where(Invite.status == status_filter)

    result = await db.execute(query)
    invites = result.scalars().all()

    now = datetime.utcnow()
    for inv in invites:
        if inv.status == "pending" and inv.expires_at < now:
            inv.status = "expired"
    await db.commit()

    return InviteListResponse(
        invites=[_invite_to_response(i) for i in invites],
        total=len(invites),
    )


@router.delete("/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a pending invite."""
    result = await db.execute(select(Invite).where(Invite.id == invite_id))
    invite = result.scalar_one_or_none()
    if not invite:
        raise HTTPException(404, "Invite not found")
    if invite.status != "pending":
        raise HTTPException(400, f"Cannot revoke invite with status '{invite.status}'")
    invite.status = "revoked"
    await db.commit()
    return {"success": True, "message": "Invite revoked"}


@router.get("/users", response_model=UserListResponse)
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users with stats. Only admins."""
    from app.db.models import Memory, Conversation

    mem_sub = (
        select(Memory.user_id, func.count(Memory.id).label("mem_count"))
        .where(Memory.is_deleted == False)
        .group_by(Memory.user_id)
        .subquery()
    )
    sess_sub = (
        select(Conversation.user_id, func.count(Conversation.id).label("sess_count"))
        .group_by(Conversation.user_id)
        .subquery()
    )

    query = (
        select(
            User,
            func.coalesce(mem_sub.c.mem_count, 0).label("memory_count"),
            func.coalesce(sess_sub.c.sess_count, 0).label("session_count"),
        )
        .outerjoin(mem_sub, User.id == mem_sub.c.user_id)
        .outerjoin(sess_sub, User.id == sess_sub.c.user_id)
        .order_by(User.created_at.desc())
    )

    result = await db.execute(query)
    rows = result.all()

    users = []
    for user, mem_count, sess_count in rows:
        users.append(UserAdminResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=getattr(user, "role", "beta_user"),
            is_active=user.is_active,
            created_at=user.created_at,
            memory_count=mem_count,
            session_count=sess_count,
            password_plain=getattr(user, "password_plain", None),
        ))
    return UserListResponse(users=users, total=len(users))


@router.patch("/users/{user_id}", response_model=UserAdminResponse)
async def update_user(
    user_id: str,
    body: UserUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update a user's role, active status, or name. Only admins."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    if user.id == admin.id and body.is_active is False:
        raise HTTPException(400, "Cannot deactivate your own account")

    if user.id == admin.id and body.role and body.role != "admin":
        admin_count_result = await db.execute(
            select(func.count(User.id)).where(User.role == "admin", User.is_active == True)
        )
        if admin_count_result.scalar() <= 1:
            raise HTTPException(400, "Cannot remove the last admin")

    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.name is not None:
        user.name = body.name

    await db.commit()
    await db.refresh(user)

    return UserAdminResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=getattr(user, "role", "beta_user"),
        is_active=user.is_active,
        created_at=user.created_at,
        password_plain=getattr(user, "password_plain", None),
    )


# ─── Public Invite Endpoints (no auth) ────────────────────────

invite_router = APIRouter(prefix="/auth", tags=["Authentication"])


@invite_router.get("/invite/{token}", response_model=InviteValidateResponse)
async def validate_invite(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """Public endpoint: check if an invite token is valid."""
    result = await db.execute(select(Invite).where(Invite.token == token))
    invite = result.scalar_one_or_none()
    if not invite:
        return InviteValidateResponse(valid=False, message="Invalid invite link")

    now = datetime.utcnow()
    if invite.status != "pending":
        return InviteValidateResponse(valid=False, message=f"Invite has been {invite.status}")
    if invite.expires_at < now:
        invite.status = "expired"
        await db.commit()
        return InviteValidateResponse(valid=False, message="Invite has expired")

    return InviteValidateResponse(
        valid=True,
        email=invite.email,
        role=invite.role,
        expires_at=invite.expires_at,
    )


@invite_router.post("/register/invite")
async def register_with_invite(
    body: InviteSignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """Redeem an invite token and create a new account."""
    from app.services.auth_service import create_access_token

    result = await db.execute(select(Invite).where(Invite.token == body.token))
    invite = result.scalar_one_or_none()
    if not invite:
        raise HTTPException(400, "Invalid invite token")

    now = datetime.utcnow()
    if invite.status != "pending":
        raise HTTPException(400, f"Invite has already been {invite.status}")
    if invite.expires_at < now:
        invite.status = "expired"
        await db.commit()
        raise HTTPException(400, "Invite has expired")

    if invite.email and invite.email.lower() != body.email.lower():
        raise HTTPException(400, f"This invite is reserved for {invite.email}")

    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(400, "Email already registered")

    user = await create_user(db, body.email, body.password, body.name)
    user.role = invite.role
    await db.flush()

    invite.status = "used"
    invite.used_by = user.id
    invite.used_at = now

    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
    }


# ─── Helpers ───────────────────────────────────────────────────

def _invite_to_response(invite: Invite) -> InviteResponse:
    return InviteResponse(
        id=invite.id,
        token=invite.token,
        email=invite.email,
        role=invite.role,
        note=invite.note,
        status=invite.status,
        created_by=invite.created_by,
        used_by=invite.used_by,
        used_at=invite.used_at,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
        invite_url=f"{INVITE_BASE_URL}/{invite.token}",
    )
