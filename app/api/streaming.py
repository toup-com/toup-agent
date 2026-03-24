"""Streaming credentials API — store/retrieve/delete per-user channel credentials."""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.api.auth import get_current_user

router = APIRouter(prefix="/streaming", tags=["Streaming"])


class CredentialIn(BaseModel):
    channel: str
    email: str
    password: str


class CredentialOut(BaseModel):
    channel: str
    email: str
    has_password: bool
    connected_at: Optional[str] = None


@router.get("/credentials")
async def list_credentials(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all connected streaming channels (passwords redacted)."""
    from app.db.models import StreamingCredential

    result = await db.execute(
        select(StreamingCredential).where(StreamingCredential.user_id == user.id)
    )
    creds = result.scalars().all()
    return [
        CredentialOut(
            channel=c.channel,
            email=c.email,
            has_password=bool(c.password),
            connected_at=c.created_at.isoformat() if c.created_at else None,
        )
        for c in creds
    ]


@router.put("/credentials")
async def upsert_credential(
    body: CredentialIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Connect or update a streaming channel."""
    from app.db.models import StreamingCredential

    result = await db.execute(
        select(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user.id,
                StreamingCredential.channel == body.channel,
            )
        )
    )
    cred = result.scalar_one_or_none()

    if cred:
        cred.email = body.email.strip()
        cred.password = body.password
        cred.updated_at = datetime.utcnow()
    else:
        cred = StreamingCredential(
            id=str(uuid.uuid4()),
            user_id=user.id,
            channel=body.channel,
            email=body.email.strip(),
            password=body.password,
        )
        db.add(cred)

    await db.commit()
    return {"status": "connected", "channel": body.channel}


@router.delete("/credentials/{channel}")
async def disconnect_credential(
    channel: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disconnect a streaming channel."""
    from app.db.models import StreamingCredential

    await db.execute(
        delete(StreamingCredential).where(
            and_(
                StreamingCredential.user_id == user.id,
                StreamingCredential.channel == channel,
            )
        )
    )
    await db.commit()
    return {"status": "disconnected", "channel": channel}
