"""Per-device session tracking.

Backs the account-page "Active sessions" surface. Every login mints a
JWT with a `jti` claim AND inserts a `UserSession` row keyed by that
jti. `get_current_user` then validates the jti against this table on
each authenticated request:

  - jti not found        → revoked or never-issued token, 401
  - row is_revoked=True  → user explicitly signed this device out
  - row last_seen_at     → bumped on each hit (rate-limited) so the
                           "last active 2m ago" display works without
                           a separate heartbeat

The lookup adds one indexed query per authenticated request. Index
`ix_user_sessions_jti` (unique) keeps it < 1ms.

Cancellation is server-side: revoking the row immediately invalidates
the bearer token AND the SSO cookie because both encode the same jti.
The cookie itself isn't deleted (we can't reach the browser), but the
next request using it fails auth and the frontend redirects to login.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Request
from jose import jwt, JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import UserSession

logger = logging.getLogger(__name__)


# Updating last_seen_at on every authenticated request is wasteful —
# the column exists to give the user a rough "last active" timestamp
# on the account page, not as a precise heartbeat. Bumping once per
# 60s of activity is plenty granular.
LAST_SEEN_BUMP_INTERVAL_SECONDS = 60

# Grace window after a JWT is issued before the session lookup is
# required to succeed. Protects against the race where the JWT is
# returned to the client but the UserSession row's commit hasn't
# replicated to the read connection yet. Tiny — 5s is more than
# enough on the same Postgres instance.
JTI_REVOCATION_GRACE_SECONDS = 5


def parse_device_label(user_agent: Optional[str]) -> str:
    """Derive a short human-readable device label from a User-Agent
    string. "Chrome on macOS", "Safari on iPhone", "API client", or
    "Unknown device" if the UA is empty / unparseable.

    Deliberately simple — full UA parsing is its own library
    (ua-parser) and the labels here are just to help users tell their
    own devices apart on the sessions list. Edge before Chrome because
    Edge UAs contain both substrings."""
    ua = (user_agent or "").strip()
    if not ua:
        return "Unknown device"
    ua_lower = ua.lower()

    if "edg/" in ua_lower:
        browser = "Edge"
    elif "firefox/" in ua_lower:
        browser = "Firefox"
    elif "chrome/" in ua_lower:
        browser = "Chrome"
    elif "safari/" in ua_lower:
        browser = "Safari"
    elif "curl/" in ua_lower or "python-requests" in ua_lower or "httpx" in ua_lower:
        return "API client"
    else:
        browser = "Browser"

    if "iphone" in ua_lower:
        os_name = "iPhone"
    elif "ipad" in ua_lower:
        os_name = "iPad"
    elif "android" in ua_lower:
        os_name = "Android"
    elif "macintosh" in ua_lower or "mac os" in ua_lower or "mac_os" in ua_lower:
        os_name = "macOS"
    elif "windows" in ua_lower:
        os_name = "Windows"
    elif "linux" in ua_lower:
        os_name = "Linux"
    else:
        os_name = "Unknown OS"

    return f"{browser} on {os_name}"


def _client_ip(request: Optional[Request]) -> Optional[str]:
    """Best-effort client IP. Prefers X-Forwarded-For first hop
    (Railway sets this), falls back to request.client.host. Returns
    None if we have neither — the IP is informational on the sessions
    page, not load-bearing."""
    if not request:
        return None
    xff = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if xff:
        return xff[:45]  # IPv6 max length
    if request.client and request.client.host:
        return str(request.client.host)[:45]
    return None


def _decode_jti(token: str) -> Optional[str]:
    """Pull the `jti` claim out of a JWT without raising on bad
    signatures. Used by record_login_session — the token was just
    minted server-side so we know it's valid, this is just claim
    extraction."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload.get("jti")
    except JWTError:
        return None


async def record_login_session(
    db: AsyncSession,
    user_id: str,
    token: str,
    request: Optional[Request] = None,
) -> Optional[UserSession]:
    """Insert a UserSession row matching the freshly-minted token.
    Safe to call from any login path (password login, SSO, agent
    session token) — caller passes the token they just gave the
    client and we decode the jti out of it.

    Returns the row on success, None on any failure (we don't want
    session tracking issues to bring down login).
    """
    jti = _decode_jti(token)
    if not jti:
        # Should be unreachable — every token we issue has a jti — but
        # if for some reason we can't decode it, just skip recording.
        # The user still gets logged in; the device just won't appear
        # on the sessions page.
        logger.warning("record_login_session: token had no jti claim, skipping")
        return None

    ua = request.headers.get("user-agent", "") if request else ""

    try:
        row = UserSession(
            user_id=user_id,
            jti=jti,
            device_label=parse_device_label(ua),
            user_agent=ua[:500] if ua else None,
            ip_address=_client_ip(request),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row
    except Exception as e:
        # Don't block login on a bookkeeping failure. The session
        # validation in get_current_user will let the token through
        # via the grace window (jti not found within first 5s of
        # issuance is treated as "race, not revoked").
        logger.warning("record_login_session failed: %s", e)
        await db.rollback()
        return None


async def record_login_session_async(
    user_id: str, token: str, ua: str = "", ip: Optional[str] = None,
) -> None:
    """Background variant of record_login_session: opens its OWN DB session and
    inserts the UserSession row, so the OAuth/Apple response doesn't wait on the
    INSERT+commit+refresh (a cross-region round-trip). Safe to defer: the
    get_current_user JTI check tolerates a not-yet-written row for
    JTI_REVOCATION_GRACE_SECONDS, and the client's first authed call is well
    inside that window. Caller pre-extracts ua/ip from the request (which can't
    cross into a background task). Best-effort; failures only log.
    """
    jti = _decode_jti(token)
    if not jti:
        return
    try:
        from app.db import async_session_maker
        async with async_session_maker() as db:
            row = UserSession(
                user_id=user_id,
                jti=jti,
                device_label=parse_device_label(ua),
                user_agent=ua[:500] if ua else None,
                ip_address=ip,
            )
            db.add(row)
            await db.commit()
    except Exception as e:
        logger.warning("record_login_session_async failed: %s", e)


async def get_session_by_jti(db: AsyncSession, jti: str) -> Optional[UserSession]:
    """Lookup helper used by get_current_user. Returns the session
    row even if revoked — the caller checks is_revoked."""
    result = await db.execute(select(UserSession).where(UserSession.jti == jti))
    return result.scalar_one_or_none()


async def maybe_bump_last_seen(
    db: AsyncSession, session: UserSession
) -> None:
    """Bump last_seen_at if it's been more than
    LAST_SEEN_BUMP_INTERVAL_SECONDS since the last bump. Cheap
    optimisation so we don't write the table on every request. Best
    effort — failures are swallowed since the timestamp is purely
    informational."""
    now = datetime.utcnow()
    if (now - session.last_seen_at).total_seconds() < LAST_SEEN_BUMP_INTERVAL_SECONDS:
        return
    try:
        session.last_seen_at = now
        await db.commit()
    except Exception as e:
        logger.debug("maybe_bump_last_seen swallowed: %s", e)
        await db.rollback()


async def revoke_all_user_sessions_except(
    db: AsyncSession, user_id: str, except_jti: Optional[str] = None
) -> int:
    """Mark every non-revoked session for the user as revoked, except
    the one matching `except_jti` (if any). Used by:
      - change-password (revoke everyone, current session keeps its
        new token from the response body so doesn't need preserving)
      - the account-page "sign out other devices" CTA (preserves the
        current session)
    Returns the number of rows revoked."""
    stmt = select(UserSession).where(
        UserSession.user_id == user_id,
        UserSession.is_revoked == False,  # noqa: E712
    )
    if except_jti:
        stmt = stmt.where(UserSession.jti != except_jti)
    result = await db.execute(stmt)
    rows = result.scalars().all()
    count = 0
    for r in rows:
        r.is_revoked = True
        count += 1
    if count:
        await db.commit()
    return count
