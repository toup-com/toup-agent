"""Signup IP rate-limit guard — persistent or in-memory.

A single async surface used by /auth/register and the OAuth/Apple
new-account branches so every signup vector shares ONE cap:

* When ``settings.signup_rate_limit_persistent`` is True, slots are counted
  in the ``signup_attempts`` table — cross-replica and deploy-surviving.
* Otherwise it delegates to the legacy per-process in-memory
  ``signup_rate_limiter`` (unchanged default behavior).

The cap is ``settings.signup_rate_limit_per_hour`` (0 disables). Records
are written on their OWN session and committed independently, so a limiter
write never entangles with — or fails — the signup transaction. Reads use
the caller's session.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete as sa_delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import SignupAttempt
from app.services.rate_limiter import signup_rate_limiter

logger = logging.getLogger(__name__)

_WINDOW_S = 3600


def _max_per_window() -> int:
    return int(getattr(settings, "signup_rate_limit_per_hour", 5) or 0)


def _persistent_enabled() -> bool:
    return bool(getattr(settings, "signup_rate_limit_persistent", False))


async def signup_retry_after(db: AsyncSession, ip: str) -> Optional[int]:
    """None if the IP may sign up, else seconds until a slot frees.

    Does NOT consume a slot — call :func:`record_signup` after a successful
    create. Fail-open: a limiter read error never blocks a real signup.
    """
    if not _persistent_enabled():
        return signup_rate_limiter.check(ip)
    maxn = _max_per_window()
    if maxn <= 0:
        return None
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=_WINDOW_S)
        count = (await db.execute(
            select(func.count()).select_from(SignupAttempt).where(
                SignupAttempt.ip == ip, SignupAttempt.created_at >= cutoff,
            )
        )).scalar() or 0
        if count < maxn:
            return None
        oldest = (await db.execute(
            select(func.min(SignupAttempt.created_at)).where(
                SignupAttempt.ip == ip, SignupAttempt.created_at >= cutoff,
            )
        )).scalar()
        if oldest is None:
            return _WINDOW_S
        retry = int((oldest + timedelta(seconds=_WINDOW_S) - datetime.utcnow()).total_seconds()) + 1
        return max(retry, 1)
    except Exception as e:
        logger.warning("[signup_guard] persistent check failed (%s); failing open", e)
        return None


async def record_signup(ip: str) -> None:
    """Consume one signup slot for ``ip``. Best-effort, own session +
    commit; prunes this IP's expired rows so the table stays bounded."""
    if not _persistent_enabled():
        signup_rate_limiter.record(ip)
        return
    if _max_per_window() <= 0:
        return
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as db:
            cutoff = datetime.utcnow() - timedelta(seconds=_WINDOW_S)
            await db.execute(sa_delete(SignupAttempt).where(
                SignupAttempt.ip == ip, SignupAttempt.created_at < cutoff,
            ))
            db.add(SignupAttempt(ip=ip, created_at=datetime.utcnow()))
            await db.commit()
    except Exception as e:
        logger.warning("[signup_guard] persistent record failed for ip=%s: %s", ip, e)
