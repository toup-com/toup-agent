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


# Strong references to in-flight background sends. asyncio.create_task only
# keeps a WEAK reference to the task, so without this the event loop can
# garbage-collect a send mid-flight — silently stranding a password user
# with no verification email and therefore no grant. We hold the task here
# until it completes (add_done_callback discards it). See test_email_*.
_PENDING_SENDS: set = set()


def schedule_post_register_verification(user_id: str) -> None:
    """Send the verification email OFF the signup hot path.

    /auth/register must not pay an SMTP/Resend round-trip synchronously
    (friction budget). This schedules the send on its own session — the
    request's session closes once register returns — so the grant-unlock
    link is on its way without adding latency to signup. Best-effort:
    a missing event loop or send failure is logged, never raised.

    The task is pinned in ``_PENDING_SENDS`` so it cannot be GC'd before
    it finishes (the create_task weak-reference footgun).
    """
    import asyncio
    try:
        task = asyncio.create_task(_run_post_register_verification(user_id))
    except RuntimeError:
        logger.warning(
            "[email_verification] no event loop to schedule post-register send for %s",
            str(user_id)[:8],
        )
        return
    _PENDING_SENDS.add(task)
    task.add_done_callback(_PENDING_SENDS.discard)


async def _run_post_register_verification(user_id: str) -> None:
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            if user is not None and user.email_verified_at is None:
                await _issue_and_send_verification(db, user)
    except Exception as e:
        logger.warning(
            "[email_verification] background post-register send failed for %s: %s",
            str(user_id)[:8], e,
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

    # Unlock the DEFERRED one-time free grant now that the email is verified
    # (no-op when the grant already fired at signup, e.g. require_verified_
    # email_for_grant is off). Best-effort: a grant hiccup must not fail the
    # verification itself.
    try:
        from app.services.credit_service import CreditService
        granted = await CreditService().grant_initial_free_credits(
            db, str(user.id), email_verified=True,
        )
        if granted:
            await db.commit()
    except Exception as e:
        logger.warning(
            "[email_verification] deferred grant on verify failed for %s: %s",
            str(user.id)[:8], e,
        )

    return {"verified": True, "email": user.email}
