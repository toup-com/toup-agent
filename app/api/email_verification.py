"""Email verification endpoints (F13).

Lives outside ``auth.py`` to keep the auth surface focused on token /
session lifecycle. Two endpoints:

* ``POST /api/auth/send-verification`` — auth-required. Rotates the
  token, sends the verification email. Throttled to one per 60s per user.
* ``POST /api/auth/verify-email``      — no auth. Accepts the token from
  the email link, sets ``users.email_verified_at`` and clears the token
  (single-use).
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db
from app.db.models import User
from app.services.email_service import render_verification_email, send_email


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Email Verification"])


VERIFICATION_RESEND_COOLDOWN_S = 60


async def _issue_and_send_verification(
    db: AsyncSession, user: User, request: Optional[Request] = None,
) -> None:
    """Rotate token, persist, send email. Public so /auth/register can
    call fire-and-forget."""
    token = secrets.token_urlsafe(32)
    user.email_verification_token = token
    user.email_verification_sent_at = datetime.utcnow()
    await db.commit()

    base = (settings.app_public_base_url or "").strip()
    if not base and request is not None:
        base = f"{request.url.scheme}://{request.headers.get('host', 'toup.ai')}"
    if not base:
        base = "https://toup.ai"
    verify_url = f"{base.rstrip('/')}/verify-email?token={token}"

    subject, html, text = render_verification_email(
        name=getattr(user, "name", None), verify_url=verify_url,
    )
    await send_email(to=user.email, subject=subject, html=html, text=text)


async def post_register_send_verification(
    db: AsyncSession, user: User, request: Optional[Request] = None,
) -> None:
    """Fire-and-forget wrapper for /auth/register. Never raises."""
    try:
        await _issue_and_send_verification(db, user, request)
    except Exception as e:
        logger.warning(
            "[email_verification] post-register send failed for %s: %s", user.email, e,
        )


@router.post("/send-verification")
async def send_verification(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Resend the verification email. 60s throttle, rotates token."""
    if current_user.email_verified_at is not None:
        return {"already_verified": True}
    if current_user.email_verification_sent_at is not None:
        elapsed = (datetime.utcnow() - current_user.email_verification_sent_at).total_seconds()
        if elapsed < VERIFICATION_RESEND_COOLDOWN_S:
            retry_after = int(VERIFICATION_RESEND_COOLDOWN_S - elapsed)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "verification_cooldown",
                    "message": f"Wait {retry_after}s before requesting another verification email.",
                    "retry_after_s": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
    await _issue_and_send_verification(db, current_user, request)
    return {"sent": True}


class VerifyEmailRequest(BaseModel):
    token: str


@router.post("/verify-email")
async def verify_email(
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Confirm an email-verification token. No auth required: the
    token authorizes itself. Single-use."""
    token = (payload.token or "").strip()
    if not token:
        raise HTTPException(400, "Token is required")
    res = await db.execute(select(User).where(User.email_verification_token == token))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(404, "Invalid or expired verification token")
    user.email_verified_at = datetime.utcnow()
    user.email_verification_token = None
    await db.commit()
    return {"verified": True, "email": user.email}
